"""Generate bimanual demonstrations by letting the existing closed-loop system do the task.

The demonstration generator *is* the analytical stack: an instruction sentence goes in, the
grounder resolves it against the cameras, the planner assigns arms, and the scripted controller
executes it — while every control tick is recorded. Two things follow from that, and both matter:

* **The label is the instruction.** Each episode is annotated with the exact sentence that
  produced it, so the policy learns language → behaviour rather than state → behaviour.
* **The demonstrations are as good as the system that is already measured.** No separate
  teleoperation path that behaves differently from what the benchmark reports.

Actions are **12-dimensional — both arms at once**. Recording only the moving arm would make
hand-offs and hold-while-pouring structurally unlearnable, since those are precisely the moments
when what one arm should do depends on what the other is doing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from armanual.perception.observer import CameraObserver
from armanual.planning.executor import ClosedLoopExecutor
from armanual.sim.randomize import RandomizationConfig, sample_scene
from armanual.sim.world import World

#: Cameras recorded for every frame, in the order the policy receives them.
CAMERAS: tuple[tuple[str, str], ...] = (
    ("observation.images.top", "cam_overhead"),
    ("observation.images.left_wrist", "left/wrist_cam"),
    ("observation.images.right_wrist", "right/wrist_cam"),
)
IMAGE_SIZE = (224, 224)
ARMS = ("left", "right")
STATE_DIM = 12  # 5 joints + gripper, per arm
ACTION_DIM = 12


@dataclass
class DemoFrame:
    images: dict[str, np.ndarray]
    state: np.ndarray
    action: np.ndarray


@dataclass
class DemoEpisode:
    """One recorded demonstration, labelled with the sentence that produced it."""

    task: str  # the natural-language instruction — the policy's conditioning
    seed: int
    skill: str
    frames: list[DemoFrame] = field(default_factory=list)
    success: bool = False
    notes: str = ""
    steps: list[dict] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frames)


class BimanualRecorder:
    """Scheduler tick hook that captures both arms and all three cameras."""

    def __init__(self, world: World, size: tuple[int, int] = IMAGE_SIZE):
        self.world = world
        self.size = size
        self.frames: list[DemoFrame] = []
        self.enabled = True

    def state_vector(self) -> np.ndarray:
        """Joint angles and gripper opening for both arms, left first."""
        parts = []
        for arm in ARMS:
            parts.append(self.world.arm_qpos(arm)[:5])
            parts.append([self.world.gripper_opening(arm)])
        return np.concatenate(parts).astype(np.float32)

    def action_vector(self) -> np.ndarray:
        """What the controller commanded this tick, for both arms."""
        parts = []
        for arm in ARMS:
            handle = self.world.arms[arm]
            commanded = np.array(self.world.data.ctrl[handle.actuator_ids], dtype=np.float32)
            parts.append(commanded[:5])
            # Normalize the gripper command to [0, 1] so it matches the state convention.
            from armanual.sim.builder import GRIPPER_CLOSED, GRIPPER_OPEN

            parts.append([(commanded[5] - GRIPPER_CLOSED) / (GRIPPER_OPEN - GRIPPER_CLOSED)])
        return np.concatenate(parts).astype(np.float32)

    def capture(self, *_args) -> None:
        if not self.enabled:
            return
        images = {
            key: self.world.render(camera, size=self.size) for key, camera in CAMERAS
        }
        self.frames.append(
            DemoFrame(images=images, state=self.state_vector(), action=self.action_vector())
        )


def collect_episode(
    instruction: str,
    seed: int,
    *,
    skill: str = "place_object",
    randomization: RandomizationConfig | None = None,
    max_steps: int = 3,
    pre_open_drawer: bool = False,
    image_size: tuple[int, int] = IMAGE_SIZE,
    fast_render: bool = True,
) -> DemoEpisode:
    """Run one instruction with the scripted stack and return the recorded demonstration."""
    scene = sample_scene(seed, randomization or RandomizationConfig.placement_only())
    world = World(scene, fast_render=fast_render)
    observer = CameraObserver(world)
    episode = DemoEpisode(task=instruction, seed=seed, skill=skill)

    if pre_open_drawer:
        # For skills that depend on the drawer being open, open it *before* recording starts so
        # the demonstration contains only the skill being taught.
        from armanual.control import primitives as prim
        from armanual.control.executor import Scheduler

        scheduler = Scheduler(world, dt=0.05)
        arm = "left" if world.scene.drawer.pos[0] < 0 else "right"
        scheduler.run({arm: prim.open_drawer(world, arm)}, max_seconds=45)
        scheduler.run({arm: prim.home(world, arm)}, max_seconds=10)

    recorder = BimanualRecorder(world, size=image_size)
    executor = ClosedLoopExecutor(world, observer, max_steps=max_steps)
    executor.on_frame.append(recorder.capture)

    record = executor.run(instruction, seed=seed)
    episode.frames = recorder.frames
    episode.steps = [s.to_dict() for s in record.steps]
    episode.success = bool(record.steps) and all(s.status == "success" for s in record.steps)
    episode.notes = "; ".join(
        f"{s.step_id}:{s.error}" for s in record.steps if s.error
    ) or record.finished_reason

    observer.close()
    world.close()
    return episode


def _best_arm(world, point, exclude: str | None = None) -> str | None:
    """Arm that can reach a point most cheaply, or None when neither can."""
    from armanual.control.kinematics import PLANNING_OPENING, solve_reach

    best, best_cost = None, float("inf")
    for arm in ARMS:
        if arm == exclude:
            continue
        target = np.array([point[0], point[1], max(0.045, float(point[2]) if len(point) > 2 else 0.045)])
        solution = solve_reach(world, arm, target, gripper=PLANNING_OPENING,
                               ignore_geoms=world.movable_geom_ids())
        if solution.feasible and solution.cost < best_cost:
            best, best_cost = arm, solution.cost
    return best


def collect_skill_episode(
    instruction: str,
    seed: int,
    *,
    skill: str,
    randomization: RandomizationConfig | None = None,
    pre_open_drawer: bool = False,
    image_size: tuple[int, int] = IMAGE_SIZE,
    fast_render: bool = True,
) -> DemoEpisode:
    """Record one *skill* demonstration by driving the primitives directly.

    The closed-loop planner is deliberately not in this path. It contributes failure modes that
    teach the policy nothing — an empty plan, a hand-off inserted because a destination was a
    centimetre out of reach — and every episode it loses is a training example lost. Grounding
    still resolves the instruction against the cameras, so the language label means exactly what
    it says; only the sequencing is taken out.

    The planner remains in the loop at *evaluation* time, where its decisions are what is being
    measured.
    """
    from armanual.control import primitives as prim
    from armanual.control.executor import Scheduler
    from armanual.control.grasp import grasp_from_detection
    from armanual.perception.refine import WristRefiner
    from armanual.planning.subgoals import subgoal_target

    scene = sample_scene(seed, randomization or RandomizationConfig.placement_only())
    world = World(scene, fast_render=fast_render)
    observer = CameraObserver(world)
    episode = DemoEpisode(task=instruction, seed=seed, skill=skill)
    recorder = BimanualRecorder(world, size=image_size)
    scheduler = Scheduler(world, dt=0.05)

    try:
        if pre_open_drawer:
            arm = "left" if world.scene.drawer.pos[0] < 0 else "right"
            scheduler.run({arm: prim.open_drawer(world, arm)}, max_seconds=45)
            scheduler.run({arm: prim.home(world, arm)}, max_seconds=10)

        scheduler.on_tick.append(recorder.capture)
        refiner = WristRefiner(world)

        if skill == "open_drawer":
            arm = "left" if world.scene.drawer.pos[0] < 0 else "right"
            result = scheduler.run({arm: prim.open_drawer(world, arm)}, max_seconds=45)
            episode.success = result.ok and world.drawer_opening() >= world.scene.drawer.open_threshold
            episode.notes = result.error or ""
        elif skill == "pour":
            episode.success, episode.notes = _record_pour(world, observer, scheduler, refiner)
        else:
            episode.success, episode.notes = _record_place(
                world, observer, scheduler, refiner, instruction
            )
    except Exception as exc:  # noqa: BLE001 - a failed demo is data about the expert, not a crash
        episode.success, episode.notes = False, f"{type(exc).__name__}: {exc}"

    episode.frames = recorder.frames
    observer.close()
    world.close()
    return episode


def _record_place(world, observer, scheduler, refiner, instruction: str) -> tuple[bool, str]:
    """Pick the object the sentence refers to and put it where the sentence says."""
    from armanual.control import primitives as prim
    from armanual.control.grasp import grasp_from_detection
    from armanual.planning.subgoals import subgoal_target

    category, target_xy = subgoal_target(world, observer, instruction)
    if target_xy is None:
        return False, "instruction has no resolvable destination"

    observation = observer.observe()
    candidates = [
        d for d in observation.detections
        if d.category not in ("furniture", "unknown")
        and (category in (None, "") or d.category == category)
    ]
    if not candidates:
        return False, f"no {category or 'object'} detected"
    detection = max(candidates, key=lambda d: d.pixel_area)

    arm = _best_arm(world, detection.position)
    if arm is None:
        return False, "object out of reach of both arms"
    grasp = grasp_from_detection(detection, world.base_pos(arm)[:2],
                                 obj_yaw=float(detection.orientation))
    result = scheduler.run(
        {arm: prim.pick_detected(world, arm, grasp, refiner=refiner, attempts=2)},
        max_seconds=40,
    )
    if not result.ok:
        return False, result.error or "pick failed"

    offset = grasp.pos[:2] - detection.position[:2]
    place_xy = (float(target_xy[0] + offset[0]), float(target_xy[1] + offset[1]))
    result = scheduler.run(
        {arm: prim.place_held(world, arm, place_xy, height=0.045, jaw=grasp.jaw)},
        max_seconds=35,
    )
    if not result.ok:
        return False, result.error or "place failed"

    after = observer.observe()
    near = [
        d for d in after.detections
        if float(np.linalg.norm(d.position[:2] - np.asarray(target_xy))) < 0.07
    ]
    return bool(near), "" if near else "object did not end up at the destination"


def _record_pour(world, observer, scheduler, refiner) -> tuple[bool, str]:
    """One arm steadies the cup while the other tips the bottle over it."""
    from armanual.control import primitives as prim
    from armanual.control.grasp import grasp_from_detection
    from armanual.eval.harness import count_liquid_in_cups

    observation = observer.observe()
    bottles = sorted((d for d in observation.detections if d.category == "bottle"),
                     key=lambda d: (-d.confidence, -d.pixel_area))
    cups = [d for d in observation.detections if d.category == "cup"]
    if not bottles or not cups:
        return False, "pour needs a visible bottle and cup"
    bottle, cup = bottles[0], max(cups, key=lambda d: d.pixel_area)

    pour_arm = _best_arm(world, bottle.position)
    if pour_arm is None:
        return False, "bottle out of reach"
    hold_arm = next((a for a in ARMS if a != pour_arm), None)

    grasp = grasp_from_detection(bottle, world.base_pos(pour_arm)[:2])
    result = scheduler.run(
        {pour_arm: prim.pick_detected(world, pour_arm, grasp, refiner=refiner, attempts=2)},
        max_seconds=40,
    )
    if not result.ok:
        return False, result.error or "bottle pick failed"

    skills = {pour_arm: prim.pour_over(world, pour_arm, cup.position)}
    if hold_arm and _best_arm(world, cup.position, exclude=pour_arm) == hold_arm:
        hold_grasp = grasp_from_detection(cup, world.base_pos(hold_arm)[:2])
        skills[hold_arm] = prim.steady_and_hold(world, hold_arm, hold_grasp, seconds=9.0)
    result = scheduler.run(skills, max_seconds=60)
    inside = count_liquid_in_cups(world)
    return inside >= 1, "" if inside else f"no liquid landed in a cup ({result.error or 'pour missed'})"
