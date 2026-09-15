"""The closed loop: understand, act, observe, update, replan.

One episode runs as:

    observe -> ground the instruction in what was seen -> plan -> execute ONE step ->
    observe again -> check the step did what it was supposed to -> re-plan -> ...

Re-planning after every step is what makes the rest of the system worth having. The plan built
before the drawer is opened cannot include the utensils, because nothing has seen them yet; after
the drawer opens they appear, and the next plan places them. If a cup is knocked over, the next
observation says so and the new plan deals with the cup where it *is*. Nothing in the loop reads
simulator state — steps are executed against detections, verified against detections, and the
episode record says exactly which stage failed when one does.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from armanual.control import primitives as P
from armanual.control.executor import Scheduler, SkillFailure, Status
from armanual.control.primitives import Rendezvous
from armanual.control.grasp import grasp_from_detection
from armanual.control.gripper import pinch_offset
from armanual.perception.detector import Detection, SceneObservation
from armanual.perception.refine import WristRefiner
from armanual.planning.planner import Plan, Planner, Step
from armanual.task.grounding import Grounder
from armanual.task.parser import parse_instruction
from armanual.task.schema import TaskRequest

#: A step counts as done when the object ends up this close to its target.
PLACEMENT_TOLERANCE = 0.055


@dataclass
class StepRecord:
    step_id: str
    verb: str
    arm: str
    object_name: str | None
    target_xy: tuple[float, float] | None
    status: str
    error: str | None = None
    failure_kind: str | None = None
    sim_seconds: float = 0.0
    wall_seconds: float = 0.0
    placement_error_mm: float | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.step_id,
            "verb": self.verb,
            "arm": self.arm,
            "object": self.object_name,
            "target_xy": list(self.target_xy) if self.target_xy else None,
            "status": self.status,
            "error": self.error,
            "failure_kind": self.failure_kind,
            "sim_seconds": round(self.sim_seconds, 2),
            "wall_seconds": round(self.wall_seconds, 2),
            "placement_error_mm": (
                round(self.placement_error_mm, 1) if self.placement_error_mm is not None else None
            ),
        }


@dataclass
class EpisodeRecord:
    instruction: str
    modality: str = "text"
    seed: int = 0
    style: str | None = None
    steps: list[StepRecord] = field(default_factory=list)
    replans: int = 0
    ambiguities: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    sim_seconds: float = 0.0
    wall_seconds: float = 0.0
    finished_reason: str = ""

    @property
    def steps_attempted(self) -> int:
        return len(self.steps)

    @property
    def steps_succeeded(self) -> int:
        return sum(1 for s in self.steps if s.status == "success")

    @property
    def subtask_success_rate(self) -> float:
        return self.steps_succeeded / self.steps_attempted if self.steps_attempted else 0.0

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "modality": self.modality,
            "seed": self.seed,
            "style": self.style,
            "replans": self.replans,
            "ambiguities": self.ambiguities,
            "unresolved": self.unresolved,
            "notes": self.notes,
            "sim_seconds": round(self.sim_seconds, 2),
            "wall_seconds": round(self.wall_seconds, 2),
            "finished_reason": self.finished_reason,
            "steps_attempted": self.steps_attempted,
            "steps_succeeded": self.steps_succeeded,
            "subtask_success_rate": round(self.subtask_success_rate, 3),
            "steps": [s.to_dict() for s in self.steps],
        }


class ClosedLoopExecutor:
    """Runs one instruction to completion (or to a reported failure)."""

    def __init__(self, world, observer, *, dt: float = 0.05, max_steps: int = 14,
                 max_replans: int = 10, step_budget_seconds: float = 45.0,
                 planner: Planner | None = None, grounder: Grounder | None = None):
        self.world = world
        self.observer = observer
        self.scheduler = Scheduler(world, dt=dt)
        self.planner = planner or Planner(world)
        self.grounder = grounder or Grounder()
        self.dt = dt
        self.max_steps = max_steps
        self.max_replans = max_replans
        self.step_budget = step_budget_seconds
        # Eye-in-hand refinement: every pick re-localizes its target from the wrist camera at the
        # pre-grasp hover, where the object is hundreds of pixels across instead of dozens.
        self.refiner = WristRefiner(world)
        self.on_frame = []  # callbacks(world) invoked every control tick, for video capture
        self.scheduler.on_tick.append(lambda _t: [cb(world) for cb in self.on_frame])

    # ----------------------------------------------------------------------------- helpers
    def _detection_near(self, observation: SceneObservation, point,
                        radius: float = 0.07) -> Detection | None:
        point = np.asarray(point, dtype=float)[:2]
        best, best_distance = None, radius
        for detection in observation.detections:
            if detection.category in ("furniture", "unknown"):
                continue
            distance = float(np.linalg.norm(detection.position[:2] - point))
            if distance < best_distance:
                best, best_distance = detection, distance
        return best

    def _execute_step(self, step: Step, observation: SceneObservation) -> tuple[Status, str, str]:
        """Run one plan step. Returns (status, error, failure_kind)."""
        try:
            if step.verb == "open_drawer":
                result = self.scheduler.run(
                    {step.arm: P.open_drawer(self.world, step.arm, dt=self.dt)},
                    max_seconds=self.step_budget,
                )
            elif step.verb == "pick_place":
                detection = self._detection_near(observation, step.target_source or step.pick_xy)
                if detection is None:
                    return Status.FAILURE, "object no longer visible", "perception"
                grasp = grasp_from_detection(detection, self.world.base_pos(step.arm)[:2],
                                             obj_yaw=_detection_yaw(detection))
                centre_offset = grasp.pos[:2] - detection.position[:2]
                result = self.scheduler.run(
                    {step.arm: _pick_then_place(self.world, step.arm, grasp, step.target_xy,
                                                dt=self.dt, refiner=self.refiner,
                                                centre_offset=centre_offset)},
                    max_seconds=self.step_budget * 1.6,
                )
            elif step.verb == "handoff":
                detection = self._detection_near(observation, step.pick_xy)
                if detection is None:
                    return Status.FAILURE, "object no longer visible", "perception"
                grasp = grasp_from_detection(detection, self.world.base_pos(step.giver)[:2],
                                             obj_yaw=_detection_yaw(detection))
                result = self._run_handoff(step, grasp)
            elif step.verb == "place_held":
                result = self.scheduler.run(
                    {step.arm: P.place_held(self.world, step.arm, step.target_xy, dt=self.dt)},
                    max_seconds=self.step_budget,
                )
            elif step.verb == "pour":
                result = self._run_pour(step, observation)
            else:
                return Status.FAILURE, f"unknown step verb {step.verb!r}", "planning"
        except SkillFailure as exc:  # raised outside the scheduler, e.g. while building a skill
            return Status.FAILURE, str(exc), exc.kind
        return result.status, result.error or "", result.kind or ""

    def _run_handoff(self, step: Step, grasp) -> object:
        """Pick with the giver, then run both arms together through the exchange."""
        pick = self.scheduler.run(
            {step.giver: P.pick_detected(self.world, step.giver, grasp, dt=self.dt,
                                         refiner=self.refiner)},
            max_seconds=self.step_budget,
        )
        if not pick.ok:
            return pick
        sync = Rendezvous(("giver_ready", "receiver_gripped", "giver_released"))
        meeting = np.array([step.target_xy[0], step.target_xy[1], 0.13])
        return self.scheduler.run(
            {
                step.giver: P.handoff_give(self.world, step.giver, meeting, grasp.width, sync,
                                           dt=self.dt),
                step.receiver: P.handoff_take(self.world, step.receiver, meeting, grasp.width,
                                              sync, dt=self.dt),
            },
            max_seconds=self.step_budget * 2,
        )

    def _run_pour(self, step: Step, observation: SceneObservation) -> object:
        """Pick up the bottle, optionally have the other arm steady the cup, then pour."""
        bottles = [d for d in observation.detections if d.category == "bottle"]
        if not bottles:
            from armanual.control.executor import RunResult

            return RunResult(Status.FAILURE, 0.0, error="no bottle visible", kind="perception")
        grasp = grasp_from_detection(bottles[0], self.world.base_pos(step.arm)[:2])
        pick = self.scheduler.run(
            {step.arm: P.pick_detected(self.world, step.arm, grasp, dt=self.dt,
                                       refiner=self.refiner)},
            max_seconds=self.step_budget,
        )
        if not pick.ok:
            return pick

        cup_point = np.array([step.target_xy[0], step.target_xy[1], 0.0])
        skills = {step.arm: P.pour_over(self.world, step.arm, cup_point, dt=self.dt)}
        if step.receiver:
            # Complementary action: the other arm holds the cup steady while this one pours.
            cup = self._detection_near(observation, cup_point)
            if cup is not None:
                hold_grasp = grasp_from_detection(cup, self.world.base_pos(step.receiver)[:2])
                skills[step.receiver] = P.steady_and_hold(
                    self.world, step.receiver, hold_grasp, seconds=9.0, dt=self.dt
                )
        return self.scheduler.run(skills, max_seconds=self.step_budget * 2.2)

    def _verify(self, step: Step, observation: SceneObservation) -> float | None:
        """Placement error in millimetres, or None when the step has no placement target."""
        if step.verb in ("open_drawer", "pour") or step.target_xy is None:
            return None
        detection = self._detection_near(observation, step.target_xy, radius=0.12)
        if detection is None:
            return None
        return float(np.linalg.norm(detection.position[:2] - np.asarray(step.target_xy))) * 1000

    # --------------------------------------------------------------------------------- run
    def run(self, instruction: str, *, modality: str = "text", seed: int = 0,
            pointing_xy=None, request: TaskRequest | None = None) -> EpisodeRecord:
        """Execute an instruction, re-observing and re-planning between steps."""
        started_sim, started_wall = self.world.time, time.perf_counter()
        request = request or parse_instruction(instruction, modality=modality)
        record = EpisodeRecord(instruction=instruction, modality=modality, seed=seed,
                               style=request.style)
        record.notes.extend(request.warnings)

        done_targets: list[tuple[float, float]] = []
        for iteration in range(self.max_steps):
            observation = self.observer.observe()
            if pointing_xy is not None:
                for action in request.actions:
                    if action.target is not None and action.target.pointing_xy is None:
                        action.target.pointing_xy = pointing_xy
            task = self.grounder.ground(request, observation)
            for note in task.ambiguities:
                if note not in record.ambiguities:
                    record.ambiguities.append(note)
            for referent in task.unresolved:
                text = referent.describe()
                if text not in record.unresolved:
                    record.unresolved.append(text)

            plan = self.planner.plan(task, observation)
            _annotate_pick_points(plan, observation)
            remaining = [s for s in plan.steps if not _already_done(s, done_targets)]
            for note in plan.notes:
                if note not in record.notes:
                    record.notes.append(note)
            if not remaining:
                record.finished_reason = "plan empty — nothing left to do"
                break
            if iteration:
                record.replans += 1

            step = remaining[0]
            step_sim, step_wall = self.world.time, time.perf_counter()
            status, error, kind = self._execute_step(step, observation)
            after = self.observer.observe()
            placement_error = self._verify(step, after)
            if status is Status.SUCCESS and placement_error is not None:
                if placement_error > PLACEMENT_TOLERANCE * 1000:
                    status, error, kind = Status.FAILURE, (
                        f"placed {placement_error:.0f}mm from target"
                    ), "placement"

            record.steps.append(
                StepRecord(
                    step_id=step.step_id,
                    verb=step.verb,
                    arm=step.arm or f"{step.giver}->{step.receiver}",
                    object_name=step.object_name,
                    target_xy=step.target_xy,
                    status=status.value,
                    error=error or None,
                    failure_kind=kind or None,
                    sim_seconds=self.world.time - step_sim,
                    wall_seconds=time.perf_counter() - step_wall,
                    placement_error_mm=placement_error,
                )
            )
            if status is Status.SUCCESS and step.target_xy is not None:
                done_targets.append(step.target_xy)
            elif status is not Status.SUCCESS:
                # Park the arm out of the way so the next observation is not blocked by it, then
                # let the loop re-plan from what it can now see.
                self.scheduler.run(
                    {step.arm or step.giver: P.home(self.world, step.arm or step.giver, dt=self.dt)},
                    max_seconds=10.0,
                )
                if len([s for s in record.steps if s.status != "success"]) > self.max_replans:
                    record.finished_reason = "too many failed steps"
                    break
                done_targets.append(step.target_xy) if step.target_xy else None
        else:
            record.finished_reason = "step budget exhausted"

        record.sim_seconds = self.world.time - started_sim
        record.wall_seconds = time.perf_counter() - started_wall
        if not record.finished_reason:
            record.finished_reason = "completed"
        return record


def _pick_then_place(world, arm: str, grasp, target_xy, *, dt: float = 0.05, refiner=None,
                     centre_offset=(0.0, 0.0)):
    """Composite skill: grasp a detected object, then place it at a table position.

    ``centre_offset`` is the vector from the object's centre to the point the jaws hold. A plate
    is held by its rim, five centimetres off centre, so placing "the grasped feature" at the
    target would leave the plate itself half a setting away — which is exactly what the placement
    error metric was reporting before this was accounted for.
    """
    yield from P.pick_detected(world, arm, grasp, dt=dt, refiner=refiner)
    target = (float(target_xy[0]) + float(centre_offset[0]),
              float(target_xy[1]) + float(centre_offset[1]))
    yield from P.place_held(world, arm, target, height=0.045, dt=dt, jaw=grasp.jaw,
                            tool_offset=pinch_offset(grasp.width))


def _detection_yaw(detection: Detection) -> float:
    """Orientation of an elongated detection, used to grasp utensils across their handle."""
    return float(getattr(detection, "orientation", 0.0))


def _annotate_pick_points(plan: Plan, observation: SceneObservation) -> None:
    """Refresh each step's pick point against the latest observation.

    The planner already recorded *where* it intends to pick from. This re-snaps that point to the
    nearest current detection, so a plan made one observation ago still aims at the object where
    it is now rather than where it was — and an object that has vanished is detected as missing
    here instead of halfway through a reach.
    """
    for step in plan.steps:
        if step.pick_xy is None:
            step.pick_xy = step.target_xy
        if step.pick_xy is None:
            continue
        anchor = np.asarray(step.pick_xy, dtype=float)
        best, best_distance = None, 0.06
        for detection in observation.detections:
            if detection.category in ("furniture", "unknown"):
                continue
            distance = float(np.linalg.norm(detection.position[:2] - anchor))
            if distance < best_distance:
                best, best_distance = detection, distance
        if best is not None:
            step.pick_xy = (float(best.position[0]), float(best.position[1]))
        step.target_source = step.pick_xy


def _already_done(step: Step, done_targets: list[tuple[float, float]]) -> bool:
    if step.target_xy is None:
        return False
    return any(
        float(np.linalg.norm(np.asarray(step.target_xy) - np.asarray(t))) < 0.02
        for t in done_targets
    )
