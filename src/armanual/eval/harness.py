"""Run benchmark tasks over seeds and score them, reproducibly.

Design rules this harness follows, because they are what make the numbers worth anything:

* **Determinism.** A run is identified by (task, seed, observer, policy). The same tuple rebuilds
  the same scene and replays the same episode, so a regression is reproducible from the results
  file alone.
* **Scoring from the final state, not from the log.** Success is checked by looking at the table
  afterwards, so a step that reports success but leaves the cup somewhere else still fails.
* **Failure attribution.** Every failure is tagged: perception, grounding, planning, reachability,
  grasp, placement, coordination, or timeout. "It failed" is not a useful benchmark result.
* **Three levels of success.** Task-level (all criteria met), criterion-level (how much of the
  table is right) and subtask-level (fraction of executed steps that worked). A system that gets
  four of five items right should not score the same as one that gets none.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from armanual.eval.tiers import TASKS, SuccessCriterion, TaskDefinition
from armanual.perception.observer import CameraObserver, PrivilegedObserver
from armanual.planning.executor import ClosedLoopExecutor, EpisodeRecord
from armanual.planning.place_setting import ROLE_CATEGORIES, build_setting
from armanual.sim.randomize import sample_scene
from armanual.sim.world import World

#: Distance from a setting slot at which an object counts as placed there.
SLOT_TOLERANCE = 0.06


@dataclass
class CriterionResult:
    criterion: str
    passed: bool
    detail: str = ""


@dataclass
class EpisodeResult:
    task_id: str
    tier: int
    seed: int
    instruction: str
    modality: str
    observer: str
    success: bool
    criteria: list[CriterionResult] = field(default_factory=list)
    episode: dict = field(default_factory=dict)
    failure_kinds: list[str] = field(default_factory=list)
    wall_seconds: float = 0.0
    sim_seconds: float = 0.0

    @property
    def criterion_score(self) -> float:
        if not self.criteria:
            return float(self.success)
        return sum(c.passed for c in self.criteria) / len(self.criteria)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["criterion_score"] = round(self.criterion_score, 3)
        return data


def _object_positions(world) -> dict[str, np.ndarray]:
    """Final ground-truth positions. Used *only* for scoring, never by the controller."""
    return {obj.name: world.object_pos(obj.name) for obj in world.scene.objects}


def _category_of(world, name: str) -> str:
    obj = world.scene.object_by_name(name)
    return "utensil" if obj.is_utensil else obj.category


def check_criterion(world, criterion: SuccessCriterion, task: TaskDefinition,
                    start_positions: dict[str, np.ndarray]) -> CriterionResult:
    """Evaluate one success criterion against the table's final state."""
    label = criterion.describe()
    if criterion.kind == "drawer_open":
        opening = world.drawer_opening()
        return CriterionResult(label, opening >= criterion.minimum,
                               f"drawer open {opening * 1000:.0f}mm")

    if criterion.kind == "liquid_in_cup":
        inside = count_liquid_in_cups(world)
        return CriterionResult(label, inside >= criterion.minimum,
                               f"{inside} particles in a cup")

    positions = _object_positions(world)
    if criterion.kind == "object_moved":
        moved = [
            name
            for name, position in positions.items()
            if _category_of(world, name) == criterion.category
            and float(np.linalg.norm(position[:2] - start_positions[name][:2])) > criterion.tolerance
        ]
        return CriterionResult(label, bool(moved), f"moved: {moved}")

    if criterion.kind == "object_at_slot":
        setting = build_setting(task.style, (0.0, -0.11))
        target = setting.target_for(criterion.role)
        if target is None:
            return CriterionResult(label, False, f"style has no {criterion.role} slot")
        categories = ROLE_CATEGORIES.get(criterion.role, (criterion.role,))
        best_name, best_distance = None, float("inf")
        for name, position in positions.items():
            category = _category_of(world, name)
            if category not in categories and criterion.category not in (category, ""):
                continue
            distance = float(np.linalg.norm(position[:2] - np.asarray(target)))
            if distance < best_distance:
                best_name, best_distance = name, distance
        passed = best_distance <= criterion.tolerance
        detail = (
            f"nearest {criterion.role}: {best_name} at {best_distance * 1000:.0f}mm"
            if best_name
            else f"no {criterion.role} on the table"
        )
        if passed and criterion.require_moved and best_name is not None:
            travelled = float(
                np.linalg.norm(positions[best_name][:2] - start_positions[best_name][:2])
            )
            if travelled < criterion.moved_threshold:
                passed = False
                detail += f" but it only travelled {travelled * 1000:.0f}mm — the robot did not place it"
        return CriterionResult(label, passed, detail)

    return CriterionResult(label, False, f"unknown criterion kind {criterion.kind!r}")


def count_liquid_in_cups(world, margin: float = 0.004) -> int:
    """Water particles whose position lies inside any cup's interior."""
    particles = world.water_positions()
    if not len(particles):
        return 0
    inside = 0
    for obj in world.scene.objects:
        if obj.category != "cup":
            continue
        centre = world.object_pos(obj.name)
        radius = obj.size[0] + margin
        top = centre[2] + 2 * obj.size[2] + 0.01
        within = (
            (np.linalg.norm(particles[:, :2] - centre[:2], axis=1) < radius)
            & (particles[:, 2] > centre[2])
            & (particles[:, 2] < top)
        )
        inside += int(within.sum())
    return inside


def run_episode(task: TaskDefinition, seed: int, *, privileged: bool = False,
                instruction: str | None = None, modality: str = "text",
                capture=None, max_steps: int | None = None,
                backend=None, fallback: bool = False) -> EpisodeResult:
    """Run one (task, seed) episode and score it.

    With ``backend`` set, the episode runs through the subgoal runner and the learned policy;
    without it, the analytical controller runs the same instruction on the same seed. That is
    what makes the policy-versus-scripted comparison an apples-to-apples one.
    """
    scene = sample_scene(seed, task.randomization)
    world = World(scene)
    observer = PrivilegedObserver(world) if privileged else CameraObserver(world)

    text = instruction or task.instruction
    if task.style and "style" not in text:
        text = f"{text} in the {task.style} style"
    start_positions = _object_positions(world)
    started = time.perf_counter()

    if backend is not None:
        from armanual.planning.subgoals import SubgoalRunner

        runner = SubgoalRunner(world, observer, backend=backend, fallback=fallback)
        if capture is not None:
            runner.on_frame.append(capture)
        subgoal_episode = runner.run(text, style=task.style, seed=seed)
        runner.close()
        record_dict = subgoal_episode.to_dict()
        record_dict["subtask_success_rate"] = subgoal_episode.success_rate
        record_dict["steps_attempted"] = len(subgoal_episode.results)
        record_dict["steps_succeeded"] = sum(r.success for r in subgoal_episode.results)
        record_dict["replans"] = 0
        failure_kinds = sorted(
            {"policy" if r.executor == "policy" else "fallback"
             for r in subgoal_episode.results if not r.success}
        )
        sim_seconds = subgoal_episode.sim_seconds
        observer_label = f"{'privileged' if privileged else 'camera'}+{runner.mode}"
    else:
        executor = ClosedLoopExecutor(world, observer, max_steps=max_steps or task.max_steps)
        if capture is not None:
            executor.on_frame.append(capture)
        record: EpisodeRecord = executor.run(text, modality=modality, seed=seed)
        record_dict = record.to_dict()
        failure_kinds = sorted({s.failure_kind for s in record.steps if s.failure_kind})
        sim_seconds = record.sim_seconds
        observer_label = "privileged" if privileged else "camera"

    criteria = [check_criterion(world, c, task, start_positions) for c in task.criteria]

    result = EpisodeResult(
        task_id=task.id,
        tier=task.tier,
        seed=seed,
        instruction=text,
        modality=modality,
        observer=observer_label,
        success=all(c.passed for c in criteria) if criteria else False,
        criteria=criteria,
        episode=record_dict,
        failure_kinds=failure_kinds,
        wall_seconds=time.perf_counter() - started,
        sim_seconds=sim_seconds,
    )
    observer.close()
    world.close()
    return result


def summarize(results: list[EpisodeResult]) -> dict:
    """Aggregate results into the table that goes in the README."""
    if not results:
        return {}
    by_tier: dict[int, list[EpisodeResult]] = {}
    by_task: dict[str, list[EpisodeResult]] = {}
    for result in results:
        by_tier.setdefault(result.tier, []).append(result)
        by_task.setdefault(result.task_id, []).append(result)

    def block(group: list[EpisodeResult]) -> dict:
        subtask = [r.episode.get("subtask_success_rate", 0.0) for r in group]
        return {
            "episodes": len(group),
            "task_success": round(sum(r.success for r in group) / len(group), 3),
            "criterion_score": round(float(np.mean([r.criterion_score for r in group])), 3),
            "subtask_success": round(float(np.mean(subtask)), 3),
            "mean_sim_seconds": round(float(np.mean([r.sim_seconds for r in group])), 1),
            "mean_wall_seconds": round(float(np.mean([r.wall_seconds for r in group])), 1),
        }

    failure_counts: dict[str, int] = {}
    for result in results:
        for kind in result.failure_kinds:
            failure_counts[kind] = failure_counts.get(kind, 0) + 1

    return {
        "overall": block(results),
        "by_tier": {str(tier): block(group) for tier, group in sorted(by_tier.items())},
        "by_task": {task: block(group) for task, group in sorted(by_task.items())},
        "failure_kinds": dict(sorted(failure_counts.items(), key=lambda kv: -kv[1])),
    }


def environment_info() -> dict:
    """Machine facts recorded with every results file, so a number can be placed in context."""
    import mujoco

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "mujoco": mujoco.__version__,
    }
    try:
        import openvino as ov

        info["openvino"] = ov.__version__
        info["openvino_devices"] = ov.Core().available_devices
    except Exception:
        info["openvino"] = None
    return info


def _flushing_print(*args, **kwargs):
    """Progress that shows up in a redirected log immediately.

    A long evaluation writing to a file is invisible for minutes at a time otherwise, because
    Python block-buffers stdout when it is not a terminal — which looks exactly like a hang.
    """
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def run_suite(tasks: list[TaskDefinition], seeds: list[int], *, privileged: bool = False,
              out_dir: Path | None = None, progress=_flushing_print, backend=None,
              fallback: bool = False) -> dict:
    """Run every (task, seed) pair and write machine-readable results."""
    results: list[EpisodeResult] = []
    for task in tasks:
        for seed in seeds:
            result = run_episode(task, seed, privileged=privileged, backend=backend,
                                 fallback=fallback)
            results.append(result)
            if progress:
                marks = "".join("+" if c.passed else "-" for c in result.criteria)
                progress(
                    f"{task.id:28s} seed {seed:2d}  "
                    f"{'PASS' if result.success else 'fail'}  [{marks}]  "
                    f"sub={result.episode.get('subtask_success_rate', 0):.2f}  "
                    f"{result.wall_seconds:5.1f}s  {','.join(result.failure_kinds)}"
                )

    payload = {
        "environment": environment_info(),
        "observer": "privileged" if privileged else "camera",
        "controller": getattr(backend, "name", "scripted"),
        "seeds": seeds,
        "tasks": [t.id for t in tasks],
        "summary": summarize(results),
        "episodes": [r.to_dict() for r in results],
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
        _write_csv(out_dir / "results.csv", results)
        if progress:
            progress(f"\nwrote {out_dir / 'results.json'} and results.csv")
    return payload


def _write_csv(path: Path, results: list[EpisodeResult]) -> None:
    import csv

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["task_id", "tier", "seed", "observer", "success", "criterion_score",
             "subtask_success", "steps_attempted", "steps_succeeded", "replans",
             "failure_kinds", "sim_seconds", "wall_seconds", "instruction"]
        )
        for r in results:
            writer.writerow([
                r.task_id, r.tier, r.seed, r.observer, int(r.success),
                round(r.criterion_score, 3),
                r.episode.get("subtask_success_rate", 0.0),
                r.episode.get("steps_attempted", 0),
                r.episode.get("steps_succeeded", 0),
                r.episode.get("replans", 0),
                "|".join(r.failure_kinds), round(r.sim_seconds, 2), round(r.wall_seconds, 2),
                r.instruction,
            ])


def all_tasks() -> list[TaskDefinition]:
    return list(TASKS)
