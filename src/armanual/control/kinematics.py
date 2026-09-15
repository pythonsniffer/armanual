"""Grasp-pose generation and reachability scoring for the SO-101 arms.

A 5-DOF arm cannot approach every point from every direction, so a grasp is specified as
*position + a radial approach tilt* and the tilt is searched, smallest first: a straight-down
grasp when the object is close, a progressively more slanted one as it gets further away. The
same search doubles as the reachability oracle used by dynamic arm assignment — an arm "can
serve" a point exactly when some tilt produces a converged IK solution, which is a measured fact
rather than a hand-drawn workspace box.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from armanual.control.ik import IKResult, grasp_frame, solve_ik

#: Searched in order. 0.0 is a pure top-down grasp; larger values lean the wrist outward, which
#: is what buys reach near the workspace boundary (see docs/WORKSPACE.md).
TILT_CANDIDATES: tuple[float, ...] = (0.0, 0.3, 0.6, 0.9, 1.3)

#: Gripper command that puts the jaw midpoint closest to the TCP site (~46 mm aperture).
PREGRASP_OPENING = 0.30
UP = np.array([0.0, 0.0, 1.0])

#: Where a pinched object actually ends up, expressed in the ``gripperframe`` site frame (metres).
#: Measured with scripts/measure_workspace.py: the *fixed* jaw tip sits at (2.9, 0.2, -20.1) mm
#: and never moves, while the moving jaw sweeps along +z. So an object is pinched against the
#: fixed jaw, about 14 mm along -z from the site — not at the site itself. Targeting the site at
#: the object (the obvious thing to do) misses every thin object by that much.
PINCH_OFFSET = np.array([0.003, 0.0, -0.014])


@dataclass
class ReachSolution:
    """One feasible way for an arm to put its tool centre point at a target."""

    arm: str
    qpos: np.ndarray
    tilt: float
    ik: IKResult
    target: np.ndarray

    @property
    def feasible(self) -> bool:
        return self.ik.ok()

    @property
    def cost(self) -> float:
        """Lower is better: IK residual, wrist lean, and a hard penalty for infeasibility.

        The penalty matters — without it a low-residual pose that the collision or torque screen
        rejected outranks a slightly less accurate pose the arm can actually execute.
        """
        penalty = 0.0 if self.feasible else 100.0
        return penalty + self.ik.pos_error * 20.0 + self.ik.rot_error * 0.5 + self.tilt * 0.35


def radial_frame(base_xy: np.ndarray, target_xy: np.ndarray, tilt: float) -> np.ndarray:
    """Grasp frame whose approach leans outward in the arm's own vertical plane.

    Leaning *radially* matters: the shoulder pan fixes that plane, so an approach tilted in any
    other direction is unreachable no matter how the remaining joints move.
    """
    radial = np.asarray(target_xy, dtype=float) - np.asarray(base_xy, dtype=float)
    norm = float(np.linalg.norm(radial))
    radial3 = np.array([radial[0] / norm, radial[1] / norm, 0.0]) if norm > 1e-6 else np.array([0.0, 1.0, 0.0])
    approach = -UP + tilt * radial3
    jaw = np.cross(UP, radial3)  # jaws close across the radial direction
    return grasp_frame(approach, jaw)


def solve_reach(
    world,
    arm: str,
    target: np.ndarray,
    *,
    tilts: tuple[float, ...] = TILT_CANDIDATES,
    jaw: np.ndarray | None = None,
    q_init: np.ndarray | None = None,
    restarts: int = 3,
    payload: float = 0.0,
    check_torque: bool = True,
    check_collision: bool = True,
    ignore_geoms: set[int] | None = None,
    tool_offset: np.ndarray | None = None,
) -> ReachSolution:
    """Best (lowest-cost) way for ``arm`` to reach ``target``; check ``.feasible`` before using.

    ``tool_offset`` (a vector in the TCP frame, e.g. :data:`PINCH_OFFSET`) makes ``target`` mean
    "put *this point of the tool* here" instead of "put the site here".
    """
    handle = world.arms[arm]
    base_xy = world.base_pos(arm)[:2]
    target = np.asarray(target, dtype=float)
    q_init = world.arm_qpos(arm)[:5] if q_init is None else q_init
    best: ReachSolution | None = None
    # A jaw axis is a *line*: closing along +j and along -j are the same grasp, but they put the
    # moving jaw on opposite sides, which decides whether it sweeps into nearby geometry. Try both.
    jaw_options: tuple[np.ndarray | None, ...] = (
        (None,) if jaw is None
        else (np.asarray(jaw, dtype=float), -np.asarray(jaw, dtype=float))
    )

    def accept(q: np.ndarray) -> bool:
        if check_torque and not holdable(world, arm, q, payload=payload):
            return False
        if check_collision and not collision_free(world, arm, q, ignore_geoms):
            return False
        return True

    for tilt, jaw_dir in itertools.product(tilts, jaw_options):
        if jaw_dir is None:
            mat = radial_frame(base_xy, target[:2], tilt)
        else:
            radial3 = np.array([target[0] - base_xy[0], target[1] - base_xy[1], 0.0])
            radial3 /= np.linalg.norm(radial3) + 1e-9
            mat = grasp_frame(-UP + tilt * radial3, jaw_dir)
        site_target = target if tool_offset is None else target - mat @ np.asarray(tool_offset)
        ik = solve_ik(
            world.model,
            handle.tcp_site,
            handle.qpos_adr[:5],
            handle.qvel_adr[:5],
            site_target,
            mat,
            q_init=q_init,
            restarts=restarts,
            accept=accept,
        )
        candidate = ReachSolution(arm=arm, qpos=ik.qpos, tilt=tilt, ik=ik, target=target)
        if best is None or candidate.cost < best.cost:
            best = candidate
        if candidate.feasible:
            break
    assert best is not None
    return best


def can_reach(world, arm: str, target: np.ndarray, **kwargs) -> bool:
    return solve_reach(world, arm, target, **kwargs).feasible


def reach_distance(world, arm: str, target: np.ndarray) -> float:
    """Planar distance from the arm's base to a target — a cheap pre-filter before IK."""
    return float(np.linalg.norm(np.asarray(target)[:2] - world.base_pos(arm)[:2]))


# ------------------------------------------------------------------- static torque feasibility
#: Fraction of an actuator's force limit a pose may demand just to hold itself up. The SO-101's
#: STS3215 servos are modelled with a ±2.94 N·m force range, and the shoulder saturates well
#: inside the kinematically reachable workspace — a pose past this margin sags instead of
#: tracking, which looks like a mysterious grasp failure if it is not checked for.
TORQUE_MARGIN = 0.80


def hold_torque(world, arm: str, qpos: np.ndarray, payload: float = 0.0) -> np.ndarray:
    """Joint torques needed to hold ``qpos`` against gravity, in N·m.

    Computed with MuJoCo's own inverse dynamics at zero velocity and acceleration, so it accounts
    for the real link masses and inertias rather than a hand-rolled arm model.
    """
    import mujoco

    handle = world.arms[arm]
    scratch = world.scratch_data()  # cached: allocating MjData per IK restart is expensive
    scratch.qpos[:] = world.data.qpos
    scratch.qpos[handle.qpos_adr[:5]] = np.asarray(qpos, dtype=float)[:5]
    scratch.qvel[:] = 0.0
    scratch.qacc[:] = 0.0
    mujoco.mj_forward(world.model, scratch)
    torque = np.abs(scratch.qfrc_bias[handle.qvel_adr[:5]])
    if payload > 0.0:
        # First-order payload allowance: extra mass at the tool centre point adds torque
        # proportional to its moment arm about each joint axis.
        jacp = np.zeros((3, world.model.nv))
        mujoco.mj_jacSite(world.model, scratch, jacp, None, handle.tcp_site)
        torque = torque + np.abs(jacp[2, handle.qvel_adr[:5]]) * payload * 9.81
    return torque


def holdable(world, arm: str, qpos: np.ndarray, *, payload: float = 0.0,
             margin: float = TORQUE_MARGIN) -> bool:
    """True when every joint can statically hold ``qpos`` within ``margin`` of its force limit."""
    handle = world.arms[arm]
    limits = world.model.actuator_forcerange[handle.actuator_ids[:5], 1]
    return bool(np.all(hold_torque(world, arm, qpos, payload=payload) <= limits * margin))


# ------------------------------------------------------------------------- collision screening
#: Penetration depth (metres) below which a contact counts as a real collision rather than the
#: light touch a grasp is supposed to make.
COLLISION_DEPTH = 0.0015


def colliding_pairs(world, arm: str, qpos: np.ndarray, ignore_geoms: set[int] | None = None):
    """Contacts between ``arm`` at ``qpos`` and anything it should not be pressing into.

    Uses the simulator's own broad/narrow phase on a scratch state, so the check sees exactly the
    geometry the episode will see — including the other arm, which is what makes this usable as a
    shared-workspace screen and not only a self-collision test.
    """
    import mujoco

    handle = world.arms[arm]
    scratch = world.scratch_data()
    scratch.qpos[:] = world.data.qpos
    scratch.qpos[handle.qpos_adr[:5]] = np.asarray(qpos, dtype=float)[:5]
    mujoco.mj_kinematics(world.model, scratch)
    mujoco.mj_collision(world.model, scratch)

    own = world.arm_geom_ids(arm)
    ignore = ignore_geoms or set()
    hits = []
    for i in range(scratch.ncon):
        contact = scratch.contact[i]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if contact.dist > -COLLISION_DEPTH:
            continue
        involved_own = (g1 in own, g2 in own)
        if not any(involved_own) or all(involved_own):
            continue  # not this arm, or an internal self-contact the model already excludes
        other = g2 if involved_own[0] else g1
        if other in ignore:
            continue
        hits.append((g1, g2, float(contact.dist)))
    return hits


def collision_free(world, arm: str, qpos: np.ndarray, ignore_geoms: set[int] | None = None) -> bool:
    return not colliding_pairs(world, arm, qpos, ignore_geoms)
