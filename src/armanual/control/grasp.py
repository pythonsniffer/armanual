"""Where to put the jaws on each kind of dinner-table object.

Grasp geometry is derived from the object's own spec (radius, wall height, handle length), not
hard-coded per episode, so a randomized larger plate or a smaller cup is grasped correctly without
re-tuning. Each grasp states four things:

* ``pos`` — where the *object* should end up between the jaws (not where the tool site goes; see
  :mod:`armanual.control.gripper`),
* ``jaw`` — the world direction the jaws close along,
* ``width`` — how thick the pinched feature is, which sets both the pre-grasp aperture and the
  tool offset, and
* how far to lift and from how high to approach.

Two grasp families are used: a **straddle** grasp, where the jaws close across a thin feature
(a plate rim, a utensil handle, a drawer bar), and a **span** grasp, where they close across a
whole object (a cup or bottle body). Which one applies is a property of the object, so it is
decided here rather than by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from armanual.control.gripper import opening_for_width, pinch_offset
from armanual.sim.spec import ObjectSpec

UP = np.array([0.0, 0.0, 1.0])


@dataclass
class GraspSpec:
    """A concrete grasp: where the object goes between the jaws and how they are oriented."""

    pos: np.ndarray
    jaw: np.ndarray  # world direction the jaws open/close along
    width: float  # object dimension across the jaws, metres
    close_to: float = 0.0
    lift: float = 0.09
    approach_height: float = 0.07
    #: Extra downward push applied at the grasp point; small values seat the jaws on thin objects.
    seat: float = 0.002
    label: str = ""
    pre_open: float = field(default=0.0)

    def __post_init__(self) -> None:
        if not self.pre_open:
            self.pre_open = opening_for_width(self.width)

    @property
    def tool_offset(self) -> np.ndarray:
        """TCP-frame offset that makes :attr:`pos` mean "the object goes here"."""
        return pinch_offset(self.width)

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
        # Straddle the rim on the near side: the jaws close across the 8 mm rim wall.
        rim = obj_pos - radial * sx
        rim[2] = obj_pos[2] + sz * 1.7
        return GraspSpec(pos=rim, jaw=radial, width=0.009, close_to=0.0, lift=0.075,
                         approach_height=0.075, seat=0.003, label="plate-rim")

    if obj.category in ("cup", "bottle"):
        # Span the whole body: the jaws close across the diameter, which is both the sturdiest
        # grasp and the only one that survives the cup being knocked slightly off-centre.
        # Grasp height was swept empirically (scripts/grasp_sweep.py): gripping a cup near the
        # top of its wall lifts reliably, while gripping at mid-height lets the jaws slip off the
        # taper. A bottle is grasped lower, nearer its centre of mass.
        if obj.category == "bottle":
            # Grip the neck, not the body: half the width, and the spout ends up above the cup
            # when the wrist rolls over to pour.
            neck = obj_pos.copy()
            neck[2] = obj_pos[2] + 2 * sz + 0.024
            return GraspSpec(pos=neck, jaw=radial, width=sx, close_to=0.0, lift=0.11,
                             approach_height=0.075, seat=0.0, label="bottle-neck")
        body = obj_pos.copy()
        body[2] = obj_pos[2] + sz * 1.5
        return GraspSpec(pos=body, jaw=radial, width=2 * sx, close_to=0.0, lift=0.10,
                         approach_height=0.085, seat=0.0, label="cup-span")

    if obj.is_utensil:
        # Straddle the raised cylindrical handle across its axis.
        axis = np.array([np.cos(obj_yaw), np.sin(obj_yaw), 0.0])
        handle = obj_pos - axis * (sx * 0.9)
        handle[2] = obj_pos[2] + 0.0065
        jaw = np.cross(UP, axis)
        return GraspSpec(pos=handle, jaw=jaw, width=0.013, close_to=0.0, lift=0.07,
                         approach_height=0.065, seat=0.0, label="utensil-handle")

    if obj.category == "tray":
        handle = obj_pos.copy()
        handle[2] = obj_pos[2] + sz + 0.010
        return GraspSpec(pos=handle, jaw=np.array([0.0, 1.0, 0.0]), width=0.014, close_to=0.0,
                         lift=0.06, approach_height=0.07, label="tray")

    # Flat objects (napkin): pinch the near edge.
    edge = obj_pos - radial * sx * 0.6
    edge[2] = obj_pos[2] + sz
    return GraspSpec(pos=edge, jaw=radial, width=max(0.006, 2 * sz), close_to=0.0, lift=0.05,
                     approach_height=0.05, seat=0.002, label="flat-edge")


def tray_handle_grasp(obj: ObjectSpec, obj_pos: np.ndarray, side: str) -> GraspSpec:
    """Grasp one of the tray's two side handles ('left' = -x handle, 'right' = +x handle)."""
    sx, sy, sz = obj.size
    sign = -1.0 if side == "left" else 1.0
    pos = np.array([obj_pos[0] + sign * (sx + 0.012), obj_pos[1], obj_pos[2] + sz + 0.010])
    return GraspSpec(pos=pos, jaw=np.array([0.0, 1.0, 0.0]), width=0.014, close_to=0.0,
                     lift=0.05, approach_height=0.07, seat=0.0, label=f"tray-handle-{side}")


def drawer_handle_grasp(handle_pos: np.ndarray) -> GraspSpec:
    """Grasp the drawer's horizontal bar: jaws close across it, along y."""
    return GraspSpec(pos=np.asarray(handle_pos, dtype=float), jaw=np.array([0.0, 1.0, 0.0]),
                     width=0.015, close_to=0.0, lift=0.0, approach_height=0.08, seat=0.0,
                     label="drawer-handle")
