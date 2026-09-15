"""Where to put the jaws on each kind of dinner-table object.

Grasp geometry is derived from the object's own spec (radius, wall height, handle length), not
hard-coded per episode, so a randomized larger plate or a smaller cup is grasped correctly without
re-tuning. Each grasp says: the TCP target, the direction the jaws must close along, how wide to
pre-open, and how high to lift afterwards.

Measured gripper facts this relies on (``scripts/measure_workspace.py``):
jaw aperture spans ~4 mm (closed) to ~133 mm (open), and at command ~0.30 the jaw midpoint
coincides with the ``gripperframe`` site to within a few millimetres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from armanual.sim.spec import ObjectSpec

UP = np.array([0.0, 0.0, 1.0])


@dataclass
class GraspSpec:
    """A concrete grasp: where the tool centre point goes and how the jaws are oriented."""

    pos: np.ndarray
    jaw: np.ndarray  # world direction the jaws open/close along
    pre_open: float = 0.45
    close_to: float = 0.0
    lift: float = 0.09
    approach_height: float = 0.07
    #: Extra downward push applied at the grasp point; small values seat the jaws on thin objects.
    seat: float = 0.002

    def approach_pos(self) -> np.ndarray:
        return self.pos + UP * self.approach_height


def _toward(from_xy: np.ndarray, to_xy: np.ndarray) -> np.ndarray:
    """Unit vector in the xy-plane from one point to another, as a 3-vector."""
    delta = np.asarray(to_xy, dtype=float)[:2] - np.asarray(from_xy, dtype=float)[:2]
    norm = float(np.linalg.norm(delta))
    if norm < 1e-9:
        return np.array([0.0, 1.0, 0.0])
    return np.array([delta[0] / norm, delta[1] / norm, 0.0])


def grasp_for(obj: ObjectSpec, obj_pos: np.ndarray, base_xy: np.ndarray, obj_yaw: float = 0.0) -> GraspSpec:
    """Grasp for ``obj`` currently at ``obj_pos``, for an arm based at ``base_xy``."""
    obj_pos = np.asarray(obj_pos, dtype=float)
    radial = _toward(base_xy, obj_pos)  # points from the arm toward the object
    sx, _sy, sz = obj.size

    if obj.category == "plate":
        # Pinch the rim on the near side: jaws straddle the 8 mm rim wall.
        rim = obj_pos - radial * sx
        rim[2] = obj_pos[2] + sz * 1.6
        return GraspSpec(pos=rim, jaw=radial, pre_open=0.40, close_to=0.02, lift=0.075,
                         approach_height=0.075, seat=0.004)

    if obj.category in ("cup", "bottle"):
        # Straddle the near wall just below the rim; a full-diameter grasp would need the jaws
        # to clear the far wall as well and collides with the cup on descent.
        wall = obj_pos - radial * sx
        wall[2] = obj_pos[2] + sz * 1.45
        # A tall object needs a *lower* pre-grasp hover: approaching a 10 cm bottle from 8.5 cm
        # above its rim puts the wrist outside the arm's folded workspace near the base.
        tall = obj.category == "bottle"
        return GraspSpec(pos=wall, jaw=radial, pre_open=0.42, close_to=0.0,
                         lift=0.12 if tall else 0.10,
                         approach_height=0.055 if tall else 0.085, seat=0.003)

    if obj.is_utensil:
        # Grasp the raised cylindrical handle across its axis.
        axis = np.array([np.cos(obj_yaw), np.sin(obj_yaw), 0.0])
        handle = obj_pos - axis * (sx * 0.9)
        handle[2] = obj_pos[2] + 0.0065
        jaw = np.cross(UP, axis)
        return GraspSpec(pos=handle, jaw=jaw, pre_open=0.35, close_to=0.0, lift=0.07,
                         approach_height=0.065, seat=0.002)

    if obj.category == "tray":
        # Handled by the two-arm carry primitive, which grasps the side handles explicitly.
        handle = obj_pos.copy()
        handle[2] = obj_pos[2] + sz + 0.010
        return GraspSpec(pos=handle, jaw=np.array([0.0, 1.0, 0.0]), pre_open=0.40, close_to=0.0,
                         lift=0.06, approach_height=0.07)

    # Flat objects (napkin): pinch the near edge.
    edge = obj_pos - radial * sx * 0.6
    edge[2] = obj_pos[2] + sz
    return GraspSpec(pos=edge, jaw=radial, pre_open=0.30, close_to=0.0, lift=0.05,
                     approach_height=0.05, seat=0.003)


def tray_handle_grasp(obj: ObjectSpec, obj_pos: np.ndarray, side: str) -> GraspSpec:
    """Grasp one of the tray's two side handles ('left' = -x handle, 'right' = +x handle)."""
    sx, sy, sz = obj.size
    sign = -1.0 if side == "left" else 1.0
    pos = np.array([obj_pos[0] + sign * (sx + 0.012), obj_pos[1], obj_pos[2] + sz + 0.010])
    return GraspSpec(pos=pos, jaw=np.array([0.0, 1.0, 0.0]), pre_open=0.40, close_to=0.0,
                     lift=0.05, approach_height=0.07, seat=0.002)
