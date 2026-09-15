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
