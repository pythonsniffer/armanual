"""Grasp-pose generation and reachability scoring for the SO-101 arms.

A 5-DOF arm cannot approach every point from every direction, so a grasp is specified as
*position + a radial approach tilt* and the tilt is searched, smallest first: a straight-down
grasp when the object is close, a progressively more slanted one as it gets further away. The
same search doubles as the reachability oracle used by dynamic arm assignment — an arm "can
serve" a point exactly when some tilt produces a converged IK solution, which is a measured fact
rather than a hand-drawn workspace box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from armanual.control.ik import IKResult, grasp_frame, solve_ik

#: Searched in order. 0.0 is a pure top-down grasp; larger values lean the wrist outward, which
#: is what buys reach near the workspace boundary (see docs/WORKSPACE.md).
TILT_CANDIDATES: tuple[float, ...] = (0.0, 0.3, 0.6, 0.9, 1.3)

#: Gripper command that puts the jaw midpoint closest to the TCP site (~46 mm aperture).
PREGRASP_OPENING = 0.30
UP = np.array([0.0, 0.0, 1.0])


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
        """Lower is better. Penalizes IK residual and how much the wrist had to lean."""
        return self.ik.pos_error * 20.0 + self.ik.rot_error * 0.5 + self.tilt * 0.35


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
) -> ReachSolution:
    """Best (lowest-cost) way for ``arm`` to reach ``target``; check ``.feasible`` before using."""
    handle = world.arms[arm]
    base_xy = world.base_pos(arm)[:2]
    target = np.asarray(target, dtype=float)
    q_init = world.arm_qpos(arm)[:5] if q_init is None else q_init
    best: ReachSolution | None = None
    for tilt in tilts:
        mat = radial_frame(base_xy, target[:2], tilt)
        if jaw is not None:
            radial = target[:2] - base_xy
            radial3 = np.array([radial[0], radial[1], 0.0])
            radial3 /= np.linalg.norm(radial3) + 1e-9
            mat = grasp_frame(-UP + tilt * radial3, jaw)
        ik = solve_ik(
            world.model,
            handle.tcp_site,
            handle.qpos_adr[:5],
            handle.qvel_adr[:5],
            target,
            mat,
            q_init=q_init,
            restarts=restarts,
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
