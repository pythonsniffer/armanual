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

#: How far from the tool centre point the poured stream lands, along the arm's radial direction,
#: when the wrist is flexed by ~1.2 rad. Measured in simulation (scripts/measure_pour.py), not
#: derived: the bottle is held by its body, so the mouth swings well clear of the wrist.
SPILL_REACH = 0.108


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
    tool_offset: np.ndarray | None = None,
    payload: float = 0.0,
    ignore_geoms: set[int] | None = None,
) -> Skill:
    """Solve IK for ``pos`` and drive the arm there. Raises if the pose is unreachable.

    Pass ``tool_offset`` (from :attr:`~armanual.control.grasp.GraspSpec.tool_offset`) when ``pos``
    is where the *object* should sit between the jaws rather than where the tool site should go;
    pass ``ignore_geoms`` for geometry the arm is allowed to touch, such as the object it holds.
    """
    solution = solve_reach(world, arm, pos, jaw=jaw, payload=payload, tool_offset=tool_offset,
                           ignore_geoms=ignore_geoms)
    if not solution.ik.ok(pos_tol=tolerance, rot_tol=0.6):
        raise SkillFailure(
            f"{arm}: {label} target {np.round(pos, 3).tolist()} unreachable "
            f"(pos_err={solution.ik.pos_error * 1000:.0f}mm, tilt={solution.tilt})",
            kind="reachability",
        )
    yield from goto_qpos(world, arm, solution.qpos, duration=duration, dt=dt)


def goto_pose_linear(
    world,
    arm: str,
    pos: np.ndarray,
    *,
    jaw: np.ndarray | None = None,
    duration: float = 1.0,
    dt: float = 0.05,
    steps: int = 6,
    tolerance: float = 0.012,
    label: str = "linear",
    tool_offset: np.ndarray | None = None,
    ignore_geoms: set[int] | None = None,
) -> Skill:
    """Move to ``pos`` along a straight line in space, not a straight line in joint space.

    Interpolating joint angles between two poses bows the path sideways, which is how a descent
    onto a cup ends with the jaw pressed into its wall instead of around it. Solving IK at
    intermediate Cartesian waypoints keeps the tool on the line the grasp assumes.
    """
    start = world.tcp_pos(arm)
    if tool_offset is not None:
        # Track the *pinch point* rather than the site, so the interpolation is a straight line
        # for the thing being grasped.
        start = start + world.tcp_mat(arm) @ np.asarray(tool_offset)
    target = np.asarray(pos, dtype=float)
    for i in range(1, steps + 1):
        waypoint = start + (target - start) * (i / steps)
        yield from goto_pose(
            world, arm, waypoint, jaw=jaw, duration=duration / steps, dt=dt,
            tolerance=tolerance if i == steps else max(tolerance, 0.02),
            label=f"{label}[{i}/{steps}]", tool_offset=tool_offset, ignore_geoms=ignore_geoms,
        )


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
def grasp_at(world, arm: str, grasp: GraspSpec, *, dt: float = 0.05,
             ignore_geoms: set[int] | None = None, refiner=None) -> Skill:
    """Execute a grasp: pre-open, approach from above, look again, descend, close, lift.

    ``ignore_geoms`` should hold the target object's geoms: the jaws are *meant* to touch it, so
    the collision screen must not treat that contact as a reason to reject the pose.

    ``refiner`` (a :class:`~armanual.perception.refine.WristRefiner`) re-localizes the object from
    the wrist camera once the arm is hovering above it. At that range the object is a few hundred
    pixels across instead of a few dozen, which is the difference between closing on a plate's rim
    and closing on air.
    """
    offset = grasp.tool_offset
    yield from set_gripper(world, arm, grasp.pre_open, settle=0.15, dt=dt)
    yield from goto_pose(world, arm, grasp.approach_pos(), jaw=grasp.jaw, duration=1.2, dt=dt,
                         tolerance=0.02, label=f"pre-grasp({grasp.label})", tool_offset=offset,
                         ignore_geoms=ignore_geoms)
    if refiner is not None:
        from armanual.perception.refine import refine_grasp

        refined, result = refine_grasp(refiner, arm, grasp)
        if result is not None:
            grasp = refined
            offset = grasp.tool_offset
    yield from goto_pose_linear(world, arm, grasp.pos - UP * grasp.seat, jaw=grasp.jaw,
                                duration=0.9, dt=dt, steps=5, label=f"grasp({grasp.label})",
                                tool_offset=offset, ignore_geoms=ignore_geoms)
    yield from set_gripper(world, arm, grasp.close_to, settle=0.45, dt=dt)
    lift_to = grasp.pos + UP * grasp.lift
    yield from goto_pose_linear(world, arm, lift_to, jaw=grasp.jaw, duration=0.9, dt=dt, steps=4,
                                tolerance=0.03, label="lift", tool_offset=offset,
                                ignore_geoms=ignore_geoms)


def pick(world, arm: str, obj_name: str, *, dt: float = 0.05, verify: bool = True) -> Skill:
    """Pick up a named object, verifying afterwards that it actually left the table."""
    obj = world.scene.object_by_name(obj_name)
    start_z = float(world.object_pos(obj_name)[2])
    grasp = grasp_for(obj, world.object_pos(obj_name), world.base_pos(arm)[:2],
                      obj_yaw=_object_yaw(world, obj_name))
    yield from grasp_at(world, arm, grasp, dt=dt, ignore_geoms=world.object_geom_ids(obj_name))
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
    # The jaws must close *across* the bar, i.e. along y — closing along the bar's own x axis
    # grips nothing at all.
    jaw = np.array([0.0, 1.0, 0.0])
    travel = spec.travel if distance is None else distance

    # Pre-open just wide enough for the 14 mm handle bar: jaws opened further sweep a much
    # larger volume and the collision screen (correctly) rejects every approach to the cabinet.
    yield from set_gripper(world, arm, 0.25, settle=0.15, dt=dt)
    yield from goto_pose(world, arm, handle + UP * 0.08, jaw=jaw, duration=1.2, dt=dt,
                         tolerance=0.02, label="pre-handle")
    yield from goto_pose(world, arm, handle, jaw=jaw, duration=0.9, dt=dt, label="handle")
    yield from set_gripper(world, arm, 0.0, settle=0.45, dt=dt)

    # Pull in small increments, re-reading the handle each time and stopping on the *measured*
    # opening rather than after a fixed number of steps: the servos lag, so a fixed count
    # systematically under-opens the drawer.
    target_opening = min(travel, spec.travel) * 0.97
    for _ in range(12):
        if world.drawer_opening() >= target_opening:
            break
        pulled = world.site_pos("drawer_handle").copy()
        pulled[1] -= 0.022
        yield from goto_pose(world, arm, pulled, jaw=jaw, duration=0.4, dt=dt, tolerance=0.03,
                             label="pull")
    yield from set_gripper(world, arm, 0.35, settle=0.25, dt=dt)
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


# ------------------------------------------------------- proprioceptive grasp verification
def held_width(world, arm: str) -> float:
    """Distance between the jaw tips right now, in metres.

    A real robot checks a grasp with its own gripper feedback, not by asking the simulator where
    the object went. After closing on an object the jaws stall at its width; closing on nothing
    bottoms out near zero. That difference is the whole check.
    """
    from armanual.control.gripper import FIXED_JAW, moving_jaw_z

    return float(abs(moving_jaw_z(world.gripper_opening(arm)) - FIXED_JAW[2]))


def holding_something(world, arm: str, expected_width: float = 0.0,
                      tolerance: float = 0.016) -> bool:
    """True when the jaws are stalled at a plausible object width."""
    width = held_width(world, arm)
    if width < 0.005:
        return False
    if expected_width <= 0.0:
        return True
    return abs(width - expected_width) < max(tolerance, 0.5 * expected_width)


def pick_detected(world, arm: str, grasp: GraspSpec, *, dt: float = 0.05,
                  verify: bool = True, refiner=None, attempts: int = 3) -> Skill:
    """Grasp a perceived object, verified with gripper feedback, re-grasping if the jaws close empty.

    Contacts with whatever sits at the grasp point are expected — that is the object being
    picked — so those geoms are excluded from the collision screen. Everything else on the table
    still counts, so a grasp that would sweep a neighbouring cup off the table is still rejected.

    The retry matters more than it looks. A thin feature like a plate's rim is about as wide as
    the combined perception and tracking error, so a first attempt lands slightly high or slightly
    outside perhaps half the time. Detecting that from the gripper's own feedback and trying again
    a couple of millimetres lower — exactly what a person does — turns a coin flip into a
    dependable pick, and it is honest about *why* it succeeded, because every attempt is logged.
    """
    from dataclasses import replace

    dither = (0.0, -0.0025, 0.0035)
    last_gap = 0.0
    for attempt in range(max(1, attempts)):
        candidate = grasp
        if attempt:
            adjusted = grasp.pos.copy()
            adjusted[2] = max(0.005, adjusted[2] + dither[attempt % len(dither)])
            candidate = replace(grasp, pos=adjusted)
        yield from grasp_at(world, arm, candidate, dt=dt, refiner=refiner if attempt == 0 else None,
                            ignore_geoms=world.geoms_near(candidate.pos, radius=0.07))
        if not verify or holding_something(world, arm, candidate.width):
            return
        last_gap = held_width(world, arm)
        if attempt < attempts - 1:
            # Open up and clear the object before trying again, so the retry starts from a known
            # pose rather than from wherever the failed close left the jaws.
            yield from set_gripper(world, arm, candidate.pre_open, settle=0.2, dt=dt)
            yield from goto_pose(world, arm, candidate.approach_pos(), jaw=candidate.jaw,
                                 duration=0.7, dt=dt, tolerance=0.03, label="regrasp-clear",
                                 tool_offset=candidate.tool_offset,
                                 ignore_geoms=world.geoms_near(candidate.pos, radius=0.07))
    raise SkillFailure(
        f"{arm}: closed on nothing at {np.round(grasp.pos, 3).tolist()} after {attempts} attempts "
        f"(jaw gap {last_gap * 1000:.0f}mm, expected ~{grasp.width * 1000:.0f}mm)",
        kind="grasp",
    )


def place_held(world, arm: str, target_xy, *, height: float = 0.035, dt: float = 0.05,
               jaw: np.ndarray | None = None, release: float = 0.5,
               tool_offset: np.ndarray | None = None, hover: float = 0.05) -> Skill:
    """Put whatever the arm is holding down at a table position, then back off.

    ``hover`` is deliberately modest: the arm's reachable envelope shrinks quickly with height
    (docs/WORKSPACE.md), so a tall approach hover turns reachable placements into unreachable
    ones near the workspace edge.
    """
    from armanual.control.kinematics import solve_reach

    target = np.array([float(target_xy[0]), float(target_xy[1]), height])
    # Pick the highest hover the arm can actually reach: approaching from above is safer, but at
    # the edge of the workspace a high hover is simply not reachable and a low one is.
    chosen = hover
    for candidate in (hover, hover * 0.7, hover * 0.45):
        if solve_reach(world, arm, target + UP * candidate, jaw=jaw, tool_offset=tool_offset,
                       ignore_geoms=world.geoms_near(target, radius=0.09)).feasible:
            chosen = candidate
            break
    hover = chosen
    yield from goto_pose(world, arm, target + UP * hover, jaw=jaw, duration=1.3, dt=dt,
                         tolerance=0.03, label="pre-place", tool_offset=tool_offset,
                         ignore_geoms=world.geoms_near(target, radius=0.09))
    yield from goto_pose_linear(world, arm, target, jaw=jaw, duration=0.8, dt=dt, steps=4,
                                tolerance=0.025, label="place", tool_offset=tool_offset,
                                ignore_geoms=world.geoms_near(target, radius=0.09))
    yield from set_gripper(world, arm, release, settle=0.35, dt=dt)
    yield from goto_pose_linear(world, arm, target + UP * hover, jaw=jaw, duration=0.7, dt=dt,
                                steps=3, tolerance=0.04, label="post-place")


def handoff_give(world, giver: str, meeting, width: float, sync: "Rendezvous",
                 *, dt: float = 0.05) -> Skill:
    """Giver half of a hand-off: present the held object at the meeting point, then let go."""
    meeting = np.asarray(meeting, dtype=float)
    jaw = np.array([1.0, 0.0, 0.0])
    yield from goto_pose(world, giver, meeting, jaw=jaw, duration=1.5, dt=dt, tolerance=0.03,
                         label="handoff-present")
    sync.mark("giver_ready")
    yield from sync.wait_for("receiver_gripped")
    yield from set_gripper(world, giver, 0.55, settle=0.4, dt=dt)
    sync.mark("giver_released")
    yield from retreat(world, giver, height=0.10, duration=0.9, dt=dt)


def handoff_take(world, receiver: str, meeting, width: float, sync: "Rendezvous",
                 *, dt: float = 0.05) -> Skill:
    """Receiver half: wait for the object to be presented, close on it, take it away.

    The receiver approaches across the table rather than from above, so the two grippers meet
    jaw-to-jaw instead of one descending onto the other.
    """
    from armanual.control.gripper import opening_for_width, pinch_offset

    meeting = np.asarray(meeting, dtype=float)
    jaw = np.array([0.0, 1.0, 0.0])
    offset = pinch_offset(width)
    yield from set_gripper(world, receiver, opening_for_width(width, clearance=0.020),
                           settle=0.1, dt=dt)
    stage = meeting + np.array([0.0, 0.0, 0.10])
    yield from goto_pose(world, receiver, stage, jaw=jaw, duration=1.3, dt=dt, tolerance=0.035,
                         label="handoff-stage")
    yield from sync.wait_for("giver_ready")
    yield from goto_pose_linear(world, receiver, meeting, jaw=jaw, duration=1.0, dt=dt, steps=4,
                                tolerance=0.025, label="handoff-approach", tool_offset=offset)
    yield from set_gripper(world, receiver, 0.0, settle=0.5, dt=dt)
    sync.mark("receiver_gripped")
    yield from sync.wait_for("giver_released")
    yield from goto_pose_linear(world, receiver, meeting + np.array([0.0, -0.05, 0.06]), jaw=jaw,
                                duration=0.9, dt=dt, steps=3, tolerance=0.045,
                                label="handoff-clear")


def pour_over(world, arm: str, cup_xyz, *, tilt_angle: float = 1.2, hold_seconds: float = 2.0,
              height: float = 0.165, dt: float = 0.05) -> Skill:
    """Pour the held bottle into a cup by tipping the wrist over it.

    Two things here are measured rather than assumed, because the obvious version pours onto the
    table:

    * **Which joint tips the bottle.** The gripper holds the bottle from above, so its *roll* axis
      is vertical and rolling merely spins the bottle in place. It is ``wrist_flex`` that swings
      the bottle's mouth over, so that is the joint this drives.
    * **Where the liquid lands.** With the wrist flexed by ~1.2 rad the stream leaves the bottle
      about 11 cm from the tool centre point, along the arm's own radial direction. So the arm
      stands *back* from the cup by that much before tipping, instead of hovering over it.
    """
    handle = world.arms[arm]
    cup = np.asarray(cup_xyz, dtype=float)
    radial = cup[:2] - world.base_pos(arm)[:2]
    radial = radial / (np.linalg.norm(radial) + 1e-9)
    stand_off = SPILL_REACH * float(np.sin(min(tilt_angle, 1.4)) / np.sin(1.2))
    station = np.array([cup[0] - radial[0] * stand_off, cup[1] - radial[1] * stand_off, height])

    yield from goto_pose(world, arm, station, duration=1.6, dt=dt, tolerance=0.04,
                         label="pour-station")
    flex = handle.actuator_ids[3]
    start = float(world.data.ctrl[flex])
    target = start - tilt_angle
    ticks = max(1, int(round(1.4 / dt)))
    for i in range(1, ticks + 1):
        world.data.ctrl[flex] = start + (target - start) * _smoothstep(i / ticks)
        yield
    yield from wait(hold_seconds, dt=dt)
    for i in range(1, ticks + 1):
        world.data.ctrl[flex] = target + (start - target) * _smoothstep(i / ticks)
        yield


def steady_and_hold(world, arm: str, grasp: GraspSpec, *, seconds: float = 8.0,
                    dt: float = 0.05) -> Skill:
    """Complementary action: grasp an object and hold it still while the other arm works.

    Used for hold-the-cup-while-pouring. The hold is active — the arm keeps commanding its
    configuration — so the cup resists the nudge of the pour instead of being merely adjacent.
    """
    yield from grasp_at(world, arm, grasp, dt=dt)
    if not holding_something(world, arm, grasp.width):
        raise SkillFailure(f"{arm}: could not take hold of the object to steady it", kind="grasp")
    yield from hold(world, arm, seconds, dt=dt)
    yield from set_gripper(world, arm, 0.55, settle=0.3, dt=dt)
    yield from retreat(world, arm, height=0.08, duration=0.8, dt=dt)
