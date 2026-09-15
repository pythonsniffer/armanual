"""Manipulation primitives for the SO-101 arms, written as scheduler-driven generators.

Every primitive is a generator that writes actuator targets and yields once per control tick.
They compose: ``pick`` is ``move above -> open -> descend -> close -> lift``, and the bimanual
skills (hand-off, hold-and-pour, two-arm carry) are the same primitives run on two arms under one
scheduler, synchronized by shared state rather than by sleeping.

Failure is explicit: a primitive raises :class:`SkillFailure` with a *kind* so the evaluation
harness can attribute a failure to perception, manipulation or planning instead of logging one
undifferentiated "episode failed".
"""

from __future__ import annotations

import numpy as np

from armanual.control.executor import Skill, SkillFailure
from armanual.control.grasp import GraspSpec, grasp_for, tray_handle_grasp
from armanual.control.kinematics import solve_reach

UP = np.array([0.0, 0.0, 1.0])


def _smoothstep(alpha: float) -> float:
    """Ease-in/ease-out. Position actuators track a smooth ramp far better than a step."""
    a = float(np.clip(alpha, 0.0, 1.0))
    return a * a * (3.0 - 2.0 * a)


# --------------------------------------------------------------------------- joint-space moves
#: Joint-space convergence band. The SO-101's position actuators lag a moving setpoint, so every
#: move ends by holding its target until the joints actually arrive — measured, not assumed.
SETTLE_TOL = 0.012  # rad
SETTLE_MAX = 1.2  # seconds


def goto_qpos(world, arm: str, q_target: np.ndarray, duration: float = 1.2, dt: float = 0.05,
              settle: bool = True, settle_tol: float = SETTLE_TOL) -> Skill:
    """Interpolate the five positioning joints to ``q_target``, then wait for them to arrive."""
    handle = world.arms[arm]
    q0 = np.array(world.data.ctrl[handle.arm_actuators], dtype=float)
    q_target = np.asarray(q_target, dtype=float)[:5]
    ticks = max(1, int(round(duration / dt)))
    for i in range(1, ticks + 1):
        world.set_arm_target(arm, q0 + (q_target - q0) * _smoothstep(i / ticks))
        yield
    if not settle:
        return
    for _ in range(int(round(SETTLE_MAX / dt))):
        world.set_arm_target(arm, q_target)
        if float(np.max(np.abs(world.arm_qpos(arm)[:5] - q_target))) < settle_tol:
            return
        yield


def goto_pose(
    world,
    arm: str,
    pos: np.ndarray,
    *,
    jaw: np.ndarray | None = None,
    duration: float = 1.2,
    dt: float = 0.05,
    tolerance: float = 0.012,
    label: str = "pose",
) -> Skill:
    """Solve IK for ``pos`` and drive the arm there. Raises if the pose is unreachable."""
    solution = solve_reach(world, arm, pos, jaw=jaw)
    if not solution.ik.ok(pos_tol=tolerance, rot_tol=0.6):
        raise SkillFailure(
            f"{arm}: {label} target {np.round(pos, 3).tolist()} unreachable "
            f"(pos_err={solution.ik.pos_error * 1000:.0f}mm, tilt={solution.tilt})",
            kind="reachability",
        )
    yield from goto_qpos(world, arm, solution.qpos, duration=duration, dt=dt)


def set_gripper(world, arm: str, opening: float, settle: float = 0.35, dt: float = 0.05) -> Skill:
    """Command the gripper and hold for ``settle`` seconds so contacts can establish."""
    world.set_gripper(arm, opening)
    for _ in range(max(1, int(round(settle / dt)))):
        yield


def wait(seconds: float, dt: float = 0.05) -> Skill:
    for _ in range(max(1, int(round(seconds / dt)))):
        yield


def home(world, arm: str, duration: float = 1.2, dt: float = 0.05) -> Skill:
    spec = next(a for a in world.scene.arms if a.name == arm)
    yield from goto_qpos(world, arm, np.asarray(spec.home_qpos), duration=duration, dt=dt)


def retreat(world, arm: str, height: float = 0.12, duration: float = 0.8, dt: float = 0.05) -> Skill:
    """Lift straight up from wherever the tool centre point currently is."""
    target = world.tcp_pos(arm) + UP * height
    yield from goto_pose(world, arm, target, duration=duration, dt=dt, tolerance=0.03,
                         label="retreat")


# ------------------------------------------------------------------------------ pick and place
def grasp_at(world, arm: str, grasp: GraspSpec, *, dt: float = 0.05) -> Skill:
    """Execute a grasp: pre-open, approach from above, descend, close, lift."""
    yield from set_gripper(world, arm, grasp.pre_open, settle=0.15, dt=dt)
    yield from goto_pose(world, arm, grasp.approach_pos(), jaw=grasp.jaw, duration=1.2, dt=dt,
                         tolerance=0.02, label="pre-grasp")
    yield from goto_pose(world, arm, grasp.pos - UP * grasp.seat, jaw=grasp.jaw, duration=0.9,
                         dt=dt, label="grasp")
    yield from set_gripper(world, arm, grasp.close_to, settle=0.45, dt=dt)
    lift_to = world.tcp_pos(arm) + UP * grasp.lift
    yield from goto_pose(world, arm, lift_to, jaw=grasp.jaw, duration=0.9, dt=dt, tolerance=0.03,
                         label="lift")


def pick(world, arm: str, obj_name: str, *, dt: float = 0.05, verify: bool = True) -> Skill:
    """Pick up a named object, verifying afterwards that it actually left the table."""
    obj = world.scene.object_by_name(obj_name)
    start_z = float(world.object_pos(obj_name)[2])
    grasp = grasp_for(obj, world.object_pos(obj_name), world.base_pos(arm)[:2],
                      obj_yaw=_object_yaw(world, obj_name))
    yield from grasp_at(world, arm, grasp, dt=dt)
    if verify:
        lifted = float(world.object_pos(obj_name)[2]) - start_z
        if lifted < grasp.lift * 0.45:
            raise SkillFailure(
                f"{arm}: grasp of {obj_name} did not lift it (+{lifted * 1000:.0f}mm)",
                kind="grasp",
            )


def place(world, arm: str, position: np.ndarray, *, release: float = 0.5, dt: float = 0.05,
          approach_height: float = 0.09, jaw: np.ndarray | None = None) -> Skill:
    """Place whatever is held at ``position`` (the object's resting point) and retreat."""
    position = np.asarray(position, dtype=float)
    yield from goto_pose(world, arm, position + UP * approach_height, jaw=jaw, duration=1.2,
                         dt=dt, tolerance=0.02, label="pre-place")
    yield from goto_pose(world, arm, position, jaw=jaw, duration=0.9, dt=dt, tolerance=0.015,
                         label="place")
    yield from set_gripper(world, arm, release, settle=0.35, dt=dt)
    yield from goto_pose(world, arm, position + UP * approach_height, jaw=jaw, duration=0.7,
                         dt=dt, tolerance=0.03, label="post-place")


def pick_and_place(world, arm: str, obj_name: str, target: np.ndarray, *, dt: float = 0.05,
                   place_clearance: float = 0.012) -> Skill:
    """Convenience composition used by the task planner."""
    obj = world.scene.object_by_name(obj_name)
    yield from pick(world, arm, obj_name, dt=dt)
    grasp = grasp_for(obj, world.object_pos(obj_name), world.base_pos(arm)[:2])
    offset = world.tcp_pos(arm) - world.object_pos(obj_name)
    target = np.asarray(target, dtype=float)
    yield from place(world, arm, target + offset + UP * place_clearance, jaw=grasp.jaw, dt=dt)


# --------------------------------------------------------------------------------- drawer skill
def open_drawer(world, arm: str, *, distance: float | None = None, dt: float = 0.05) -> Skill:
    """Grasp the drawer handle and pull it open along -y."""
    spec = world.scene.drawer
    if not spec.present:
        raise SkillFailure("scene has no drawer", kind="planning")
    handle = world.site_pos("drawer_handle")
    jaw = np.array([1.0, 0.0, 0.0])  # jaws close across the handle's long (x) axis
    travel = spec.travel if distance is None else distance

    yield from set_gripper(world, arm, 0.5, settle=0.15, dt=dt)
    yield from goto_pose(world, arm, handle + UP * 0.08, jaw=jaw, duration=1.2, dt=dt,
                         tolerance=0.02, label="pre-handle")
    yield from goto_pose(world, arm, handle, jaw=jaw, duration=0.9, dt=dt, label="handle")
    yield from set_gripper(world, arm, 0.0, settle=0.45, dt=dt)

    # Pull in small increments: a single long IK jump would swing the wrist through the drawer.
    steps = 5
    for i in range(1, steps + 1):
        pulled = world.site_pos("drawer_handle").copy()
        pulled[1] -= travel / steps
        yield from goto_pose(world, arm, pulled, jaw=jaw, duration=0.45, dt=dt, tolerance=0.025,
                             label="pull")
    yield from set_gripper(world, arm, 0.6, settle=0.25, dt=dt)
    yield from retreat(world, arm, height=0.10, duration=0.7, dt=dt)

    if world.drawer_opening() < spec.open_threshold:
        raise SkillFailure(
            f"{arm}: drawer only opened {world.drawer_opening() * 1000:.0f}mm "
            f"(need {spec.open_threshold * 1000:.0f}mm)",
            kind="manipulation",
        )


# ------------------------------------------------------------------------------ bimanual skills
class Rendezvous:
    """Tiny shared latch two arms use to synchronize inside one scheduler run."""

    def __init__(self, stages: tuple[str, ...]):
        self._reached: set[str] = set()
        self.stages = stages

    def mark(self, stage: str) -> None:
        self._reached.add(stage)

    def reached(self, stage: str) -> bool:
        return stage in self._reached

    def wait_for(self, stage: str, timeout_ticks: int = 400) -> Skill:
        for _ in range(timeout_ticks):
            if self.reached(stage):
                return
            yield
        raise SkillFailure(f"timed out waiting for stage {stage!r}", kind="coordination")


def handoff_giver(world, giver: str, obj_name: str, meeting: np.ndarray, sync: Rendezvous,
                  *, dt: float = 0.05) -> Skill:
    """Giver side of an arm-to-arm hand-off: bring the object to the meeting point and wait."""
    obj = world.scene.object_by_name(obj_name)
    yield from pick(world, giver, obj_name, dt=dt)
    grasp = grasp_for(obj, world.object_pos(obj_name), world.base_pos(giver)[:2])
    offset = world.tcp_pos(giver) - world.object_pos(obj_name)
    yield from goto_pose(world, giver, np.asarray(meeting) + offset, jaw=grasp.jaw, duration=1.4,
                         dt=dt, tolerance=0.02, label="handoff-meet")
    sync.mark("giver_ready")
    yield from sync.wait_for("receiver_gripped")
    yield from set_gripper(world, giver, 0.55, settle=0.35, dt=dt)
    sync.mark("giver_released")
    yield from retreat(world, giver, height=0.10, duration=0.9, dt=dt)


def handoff_receiver(world, receiver: str, obj_name: str, meeting: np.ndarray, sync: Rendezvous,
                     *, dt: float = 0.05) -> Skill:
    """Receiver side: wait for the object to arrive, close on it, then take it away."""
    yield from set_gripper(world, receiver, 0.6, settle=0.1, dt=dt)
    stage = np.asarray(meeting, dtype=float) + np.array([0.0, 0.0, 0.10])
    yield from goto_pose(world, receiver, stage, duration=1.2, dt=dt, tolerance=0.025,
                         label="handoff-stage")
    yield from sync.wait_for("giver_ready")

    obj = world.scene.object_by_name(obj_name)
    grasp = grasp_for(obj, world.object_pos(obj_name), world.base_pos(receiver)[:2])
    yield from goto_pose(world, receiver, grasp.pos, jaw=grasp.jaw, duration=1.0, dt=dt,
                         tolerance=0.015, label="handoff-approach")
    yield from set_gripper(world, receiver, grasp.close_to, settle=0.45, dt=dt)
    sync.mark("receiver_gripped")
    yield from sync.wait_for("giver_released")
    yield from retreat(world, receiver, height=0.08, duration=0.8, dt=dt)


def hold(world, arm: str, seconds: float, *, dt: float = 0.05) -> Skill:
    """Actively hold the current configuration — the stabilizing half of a complementary action."""
    handle = world.arms[arm]
    held = np.array(world.data.ctrl[handle.arm_actuators], dtype=float)
    for _ in range(max(1, int(round(seconds / dt)))):
        world.set_arm_target(arm, held)
        yield


def pour(world, arm: str, target_cup: str, *, tilt_angle: float = 1.9, hold_seconds: float = 1.6,
         dt: float = 0.05, height: float = 0.16) -> Skill:
    """Pour from the held bottle into ``target_cup`` by rolling the wrist over its mouth."""
    handle = world.arms[arm]
    cup_pos = world.object_pos(target_cup)
    cup = world.scene.object_by_name(target_cup)
    above = cup_pos + UP * height
    # Offset the wrist to the near side so the bottle mouth sits over the cup when tilted.
    radial = above[:2] - world.base_pos(arm)[:2]
    radial = radial / (np.linalg.norm(radial) + 1e-9)
    spout = np.array([above[0] - radial[0] * 0.035, above[1] - radial[1] * 0.035, above[2]])
    yield from goto_pose(world, arm, spout, duration=1.5, dt=dt, tolerance=0.03, label="pour-above")

    roll_actuator = handle.actuator_ids[4]
    start_roll = float(world.data.ctrl[roll_actuator])
    target_roll = start_roll + tilt_angle
    ticks = max(1, int(round(1.1 / dt)))
    for i in range(1, ticks + 1):
        world.data.ctrl[roll_actuator] = start_roll + (target_roll - start_roll) * _smoothstep(i / ticks)
        yield
    yield from wait(hold_seconds, dt=dt)
    for i in range(1, ticks + 1):
        world.data.ctrl[roll_actuator] = target_roll + (start_roll - target_roll) * _smoothstep(i / ticks)
        yield
    _ = cup


def carry_tray_arm(world, arm: str, side: str, tray_name: str, target: np.ndarray,
                   sync: Rendezvous, *, dt: float = 0.05) -> Skill:
    """One arm's half of a two-arm tray carry. Both arms grasp, then move in lockstep."""
    tray = world.scene.object_by_name(tray_name)
    grasp = tray_handle_grasp(tray, world.object_pos(tray_name), side)
    yield from set_gripper(world, arm, grasp.pre_open, settle=0.1, dt=dt)
    yield from goto_pose(world, arm, grasp.approach_pos(), jaw=grasp.jaw, duration=1.3, dt=dt,
                         tolerance=0.02, label="tray-pre")
    yield from goto_pose(world, arm, grasp.pos, jaw=grasp.jaw, duration=0.8, dt=dt,
                         tolerance=0.015, label="tray-grasp")
    yield from set_gripper(world, arm, grasp.close_to, settle=0.4, dt=dt)
    sync.mark(f"{side}_gripped")
    yield from sync.wait_for("left_gripped" if side == "right" else "right_gripped")

    offset = world.tcp_pos(arm) - world.object_pos(tray_name)
    yield from goto_pose(world, arm, world.tcp_pos(arm) + UP * 0.05, jaw=grasp.jaw, duration=0.8,
                         dt=dt, tolerance=0.025, label="tray-lift")
    yield from goto_pose(world, arm, np.asarray(target) + offset + UP * 0.05, jaw=grasp.jaw,
                         duration=1.6, dt=dt, tolerance=0.03, label="tray-move")
    yield from goto_pose(world, arm, np.asarray(target) + offset, jaw=grasp.jaw, duration=0.8,
                         dt=dt, tolerance=0.03, label="tray-down")
    yield from set_gripper(world, arm, 0.5, settle=0.3, dt=dt)
    yield from retreat(world, arm, height=0.09, duration=0.8, dt=dt)


def _object_yaw(world, obj_name: str) -> float:
    quat = world.object_quat(obj_name)
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
