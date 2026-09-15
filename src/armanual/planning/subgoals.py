"""Execute a high-level instruction as a sequence of language subgoals.

This is where "set the dinner table" becomes something a policy trained on a few hundred
demonstrations can actually do. The instruction is decomposed into the sentences the policy was
trained on, and each one is executed, verified against a fresh camera observation, and only then
followed by the next. The loop stays closed at the subgoal boundary: if the cup was knocked over
while the fork was being placed, the next observation sees it.

Each subgoal can be executed by either the learned policy or the scripted controller, behind one
interface, which makes three things directly comparable on the same episodes:

* **scripted** — the analytical baseline,
* **policy** — the VLA alone, and
* **policy with fallback** — the VLA first, the scripted controller only for subgoals it fails.

The third is what a deployed system would do, and reporting all three keeps the comparison honest
rather than quietly crediting the policy with the fallback's successes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from armanual.planning.executor import ClosedLoopExecutor
from armanual.policy.tasks import decompose
from armanual.task.grounding import Grounder
from armanual.task.parser import parse_instruction

#: How close an object must land to its commanded destination for a subgoal to count as done.
SUBGOAL_TOLERANCE = 0.07


@dataclass
class SubgoalResult:
    subgoal: str
    executor: str  # "policy" | "scripted" | "policy+scripted"
    success: bool
    seconds: float = 0.0
    detail: str = ""
    inference: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "subgoal": self.subgoal,
            "executor": self.executor,
            "success": self.success,
            "seconds": round(self.seconds, 2),
            "detail": self.detail,
            "inference": self.inference,
        }


@dataclass
class SubgoalEpisode:
    instruction: str
    style: str | None
    seed: int
    mode: str
    results: list[SubgoalResult] = field(default_factory=list)
    wall_seconds: float = 0.0
    sim_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        return (
            sum(r.success for r in self.results) / len(self.results) if self.results else 0.0
        )

    @property
    def all_succeeded(self) -> bool:
        return bool(self.results) and all(r.success for r in self.results)

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "style": self.style,
            "seed": self.seed,
            "mode": self.mode,
            "subgoal_success_rate": round(self.success_rate, 3),
            "all_succeeded": self.all_succeeded,
            "wall_seconds": round(self.wall_seconds, 2),
            "sim_seconds": round(self.sim_seconds, 2),
            "subgoals": [r.to_dict() for r in self.results],
        }


def subgoal_target(world, observer, sentence: str, grounder: Grounder | None = None):
    """Where the subgoal says something should end up, and which category it is.

    Returns ``(category, target_xy)`` or ``(None, None)`` when the sentence has no placement —
    "open the drawer" and "pour water into the cup" are verified by their own effects instead.
    """
    grounder = grounder or Grounder()
    observation = observer.observe()
    request = parse_instruction(sentence)
    task = grounder.ground(request, observation)
    for action in task.actions:
        if action.spec.verb in ("place", "handoff") and action.target is not None:
            category = action.target.category
            if action.destination_xy is not None:
                return category, np.asarray(action.destination_xy, dtype=float)
            from armanual.planning.place_setting import build_setting

            setting = build_setting(None, (0.0, -0.11))
            role = {"cup": "cup", "plate": "plate", "utensil": "fork"}.get(category, "plate")
            target = setting.target_for(role)
            if target is not None:
                return category, np.asarray(target, dtype=float)
    return None, None


#: A placement subgoal must move its object at least this far to count. Without it, a scene that
#: happens to start with the plate near the middle scores a pass for a robot that did nothing.
MIN_TRAVEL = 0.015


def subgoal_baseline(world, observer, sentence: str) -> dict:
    """Snapshot what the scene looks like before a subgoal, for honest verification afterwards."""
    text = sentence.lower()
    if "drawer" in text and "open" in text:
        return {"drawer": world.drawer_opening()}
    if "pour" in text or "fill" in text:
        from armanual.eval.harness import count_liquid_in_cups

        return {"liquid": count_liquid_in_cups(world)}
    category, target = subgoal_target(world, observer, sentence)
    if target is None:
        return {}
    observation = observer.observe()
    candidates = [
        d for d in observation.detections
        if d.category not in ("furniture", "unknown")
        and (category in ("", None) or d.category == category)
    ]
    if not candidates:
        return {"nearest_before": None}
    nearest = min(candidates, key=lambda d: float(np.linalg.norm(d.position[:2] - target)))
    return {
        "nearest_before": float(np.linalg.norm(nearest.position[:2] - target)),
        "position_before": nearest.position[:2].copy(),
    }


def verify_subgoal(world, observer, sentence: str, *, before=None) -> tuple[bool, str]:
    """Check a subgoal's effect against a fresh observation.

    Deliberately checks the *world*, not the controller's opinion of the world: a policy that
    believes it placed the cup and a cup that is actually on the floor must not score the same.

    ``before`` is a :func:`subgoal_baseline` snapshot. With it, a placement only counts when the
    object actually travelled — a scene that starts with the plate near the middle must not give
    a robot credit for standing still.
    """
    text = sentence.lower()
    if "drawer" in text and "open" in text:
        opening = world.drawer_opening()
        threshold = world.scene.drawer.open_threshold
        return opening >= threshold, f"drawer open {opening * 1000:.0f}mm (need {threshold * 1000:.0f})"

    if "pour" in text or "fill" in text:
        from armanual.eval.harness import count_liquid_in_cups

        inside = count_liquid_in_cups(world)
        started_with = (before or {}).get("liquid", 0)
        return inside > started_with, f"{inside} liquid particles in a cup (was {started_with})"

    category, target = subgoal_target(world, observer, sentence)
    if target is None:
        return False, "subgoal has no checkable placement target"

    observation = observer.observe()
    candidates = [
        d
        for d in observation.detections
        if d.category not in ("furniture", "unknown")
        and (category in ("", None) or d.category == category)
    ]
    if not candidates:
        return False, f"no {category or 'object'} visible after the subgoal"
    best = min(candidates, key=lambda d: float(np.linalg.norm(d.position[:2] - target)))
    distance = float(np.linalg.norm(best.position[:2] - target))
    placed = distance <= SUBGOAL_TOLERANCE
    detail = f"nearest {best.category} {distance * 1000:.0f}mm from target"

    if placed and before and before.get("position_before") is not None:
        travelled = float(np.linalg.norm(best.position[:2] - np.asarray(before["position_before"])))
        started_placed = (before.get("nearest_before") or 1e9) <= SUBGOAL_TOLERANCE
        if started_placed and travelled < MIN_TRAVEL:
            return False, (
                f"{detail}, but it was already there and only moved "
                f"{travelled * 1000:.0f}mm — the robot did not place it"
            )
        detail += f" (travelled {travelled * 1000:.0f}mm)"
    return placed, detail


class SubgoalRunner:
    """Runs an instruction as a subgoal sequence, with a policy, the scripted stack, or both."""

    def __init__(self, world, observer, *, backend=None, fallback: bool = True,
                 policy_seconds: float = 20.0, dt: float = 0.05):
        self.world = world
        self.observer = observer
        self.backend = backend
        self.fallback = fallback
        self.policy_seconds = policy_seconds
        self.dt = dt
        self.grounder = Grounder()
        self.on_frame = []
        self._executor: ClosedLoopExecutor | None = None

    @property
    def mode(self) -> str:
        if self.backend is None:
            return "scripted"
        return "policy+fallback" if self.fallback else "policy"

    def park_arms(self, seconds: float = 12.0) -> None:
        """Return both arms to their home pose.

        Done before every check: an arm left stretched over the table hides the very object the
        verification is looking for, and "I cannot see it" would be scored as "it is not there".
        It also gives the next subgoal a known starting configuration.
        """
        from armanual.control import primitives as prim
        from armanual.control.executor import Scheduler

        scheduler = Scheduler(self.world, dt=self.dt)
        scheduler.on_tick.extend(self.on_frame)
        scheduler.run(
            {arm: prim.home(self.world, arm, dt=self.dt) for arm in self.world.arms},
            max_seconds=seconds,
        )

    def run(self, instruction: str, *, style: str | None = None, seed: int = 0) -> SubgoalEpisode:
        plan = decompose(instruction, style=style)
        episode = SubgoalEpisode(instruction=instruction, style=style, seed=seed, mode=self.mode)
        wall_started, sim_started = time.perf_counter(), self.world.time

        for sentence in plan.subgoals:
            started = time.perf_counter()
            used = "scripted"
            inference: dict = {}
            baseline = subgoal_baseline(self.world, self.observer, sentence)

            if self.backend is not None:
                from armanual.policy.runtime import PolicyRunner

                runner = PolicyRunner(self.world, self.backend, dt=self.dt)
                runner.on_tick.extend(self.on_frame)
                record = runner.run(sentence, max_seconds=self.policy_seconds)
                inference = record.get("inference", {})
                used = "policy"
                self.park_arms()
                ok, detail = verify_subgoal(self.world, self.observer, sentence, before=baseline)
                if not ok and self.fallback:
                    ok, detail = self._run_scripted(sentence, baseline)
                    used = "policy+scripted"
            else:
                ok, detail = self._run_scripted(sentence, baseline)

            episode.results.append(
                SubgoalResult(
                    subgoal=sentence,
                    executor=used,
                    success=ok,
                    seconds=time.perf_counter() - started,
                    detail=detail,
                    inference=inference,
                )
            )

        episode.wall_seconds = time.perf_counter() - wall_started
        episode.sim_seconds = self.world.time - sim_started
        return episode

    def _run_scripted(self, sentence: str, baseline: dict | None = None) -> tuple[bool, str]:
        # One executor for the whole episode. Each one owns a wrist-camera refiner with its own GL
        # renderers, so building a fresh executor per subgoal leaks contexts across a ten-seed
        # sweep and eventually exhausts them.
        if self._executor is None:
            self._executor = ClosedLoopExecutor(self.world, self.observer, max_steps=4, dt=self.dt)
            self._executor.on_frame.extend(self.on_frame)
        self._executor.run(sentence)
        self.park_arms()
        return verify_subgoal(self.world, self.observer, sentence, before=baseline)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.refiner.close()
            self._executor = None
