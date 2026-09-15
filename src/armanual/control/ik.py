"""Damped least-squares inverse kinematics for one SO-101 arm.

The SO-101 has five positioning joints, so a full 6-DOF pose is generally unreachable: the
shoulder pan fixes the vertical plane the arm works in, and within that plane we can choose
position plus the *elevation* of the approach direction, with ``wrist_roll`` free to spin the jaws
about that approach. The solver therefore treats position as a hard objective and orientation as a
weighted one, and reports the residual so a caller can reject an infeasible grasp instead of
silently commanding a bad pose.

Frames (measured from the vendered MJCF, not assumed):

* ``gripperframe`` site **x-axis** points out of the jaws — the *approach* direction,
* its **z-axis** is the jaw open/close direction,
* its origin sits between the jaw tips.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class IKResult:
    qpos: np.ndarray
    pos_error: float
    rot_error: float
    iterations: int
    converged: bool

    def ok(self, pos_tol: float = 0.006, rot_tol: float = 0.45) -> bool:
        return self.pos_error <= pos_tol and self.rot_error <= rot_tol


def grasp_frame(approach: np.ndarray, jaw: np.ndarray | None = None) -> np.ndarray:
    """Build a TCP rotation matrix from an approach direction and an optional jaw axis.

    ``approach`` becomes the frame's x-axis (out of the jaws); ``jaw`` is orthogonalized against
    it and becomes the z-axis (the direction the jaws open along).
    """
    x = np.asarray(approach, dtype=float)
    x = x / (np.linalg.norm(x) + 1e-12)
    if jaw is None:
        jaw = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    z = np.asarray(jaw, dtype=float)
    z = z - x * float(z @ x)
    norm = np.linalg.norm(z)
    if norm < 1e-8:
        z = np.array([0.0, 0.0, 1.0]) - x * float(x[2])
        norm = np.linalg.norm(z)
    z = z / norm
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def _rot_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation error as a 3-vector in world frame (axis * angle of target @ current^T)."""
    err_mat = target @ current.T
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, err_mat.flatten())
    vel = np.empty(3)
    mujoco.mju_quat2Vel(vel, quat, 1.0)
    return vel


def solve_ik(
    model: mujoco.MjModel,
    site_id: int,
    qpos_adr: np.ndarray,
    dof_adr: np.ndarray,
    target_pos: np.ndarray,
    target_mat: np.ndarray | None = None,
    *,
    q_init: np.ndarray | None = None,
    rot_weight: float = 0.35,
    damping: float = 0.12,
    max_iters: int = 180,
    pos_tol: float = 0.002,
    rot_tol: float = 0.08,
    step_scale: float = 0.6,
    seed: int | None = None,
    restarts: int = 4,
) -> IKResult:
    """Solve for joint angles putting ``site_id`` at ``target_pos`` (and near ``target_mat``).

    Runs on a scratch ``MjData`` so the caller's simulation state is never disturbed. Random
    restarts handle the SO-101's elbow-up / elbow-down ambiguity and the plain fact that a
    5-DOF arm has a lot of local minima near the workspace boundary.
    """
    scratch = mujoco.MjData(model)
    lower = model.jnt_range[model.dof_jntid[dof_adr], 0]
    upper = model.jnt_range[model.dof_jntid[dof_adr], 1]
    rng = np.random.default_rng(seed if seed is not None else 0)
    n = len(dof_adr)
    best: IKResult | None = None

    for attempt in range(max(1, restarts)):
        if attempt == 0 and q_init is not None:
            q = np.clip(np.asarray(q_init, dtype=float)[:n], lower, upper)
        elif attempt == 0:
            q = np.clip((lower + upper) / 2, lower, upper)
        else:
            q = rng.uniform(lower, upper)

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        iters = 0
        for iters in range(1, max_iters + 1):
            scratch.qpos[qpos_adr] = q
            mujoco.mj_kinematics(model, scratch)
            mujoco.mj_comPos(model, scratch)
            pos = scratch.site_xpos[site_id]
            err = np.asarray(target_pos, dtype=float) - pos
            mujoco.mj_jacSite(model, scratch, jacp, jacr, site_id)
            J = jacp[:, dof_adr]
            e = err
            if target_mat is not None:
                rot_err = _rot_error(scratch.site_xmat[site_id].reshape(3, 3), target_mat)
                J = np.vstack([J, rot_weight * jacr[:, dof_adr]])
                e = np.concatenate([err, rot_weight * rot_err])
            pos_error = float(np.linalg.norm(err))
            rot_error = (
                float(np.linalg.norm(_rot_error(scratch.site_xmat[site_id].reshape(3, 3), target_mat)))
                if target_mat is not None
                else 0.0
            )
            if pos_error < pos_tol and rot_error < rot_tol:
                break
            JJt = J @ J.T
            dq = J.T @ np.linalg.solve(JJt + (damping**2) * np.eye(JJt.shape[0]), e)
            q = np.clip(q + step_scale * dq, lower, upper)

        scratch.qpos[qpos_adr] = q
        mujoco.mj_kinematics(model, scratch)
        pos_error = float(np.linalg.norm(np.asarray(target_pos) - scratch.site_xpos[site_id]))
        rot_error = (
            float(np.linalg.norm(_rot_error(scratch.site_xmat[site_id].reshape(3, 3), target_mat)))
            if target_mat is not None
            else 0.0
        )
        result = IKResult(
            qpos=q.copy(),
            pos_error=pos_error,
            rot_error=rot_error,
            iterations=iters,
            converged=pos_error < pos_tol and rot_error < rot_tol,
        )
        score = pos_error + 0.05 * rot_error
        if best is None or score < best.pos_error + 0.05 * best.rot_error:
            best = result
        if result.converged:
            break

    assert best is not None
    return best
