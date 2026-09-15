"""Measured model of the SO-101 jaw geometry.

Everything here comes from probing the compiled model (``scripts/measure_workspace.py``), not
from the datasheet, because the numbers that matter are in the ``gripperframe`` site's frame:

* the **fixed** jaw tip sits at ``(2.9, 0.2, -20.1) mm`` in that frame and never moves;
* the **moving** jaw tip slides along the frame's +z axis, essentially linearly with the gripper
  command: ``z_mm ≈ -17.2 + 147.8 * opening`` for ``opening`` in [0, 1];
* so an object is gripped in the band between ``z = -20.1 mm`` and the moving jaw.

The practical consequence, and the reason this file exists: the tool site is *not* between the
jaws. Aiming the site at an object misses it by roughly two centimetres, which silently turns
into "the grasp closed on nothing" — the single most expensive bug in this codebase's history.
"""

from __future__ import annotations

import numpy as np

#: Fixed jaw tip position in the TCP frame (metres).
FIXED_JAW = np.array([0.0029, 0.0002, -0.0201])
#: Moving jaw tip z in the TCP frame: ``A + B * opening`` (metres). Accurate to ~1 mm for
#: commands up to about 0.45 (aperture ~68 mm); beyond that the jaw's arc bends away from the
#: line and the fit over-predicts. Every grasp in this project stays inside the accurate range —
#: see the measured table in docs/WORKSPACE.md.
MOVING_JAW_Z0, MOVING_JAW_SLOPE = -0.0172, 0.1478
MODEL_VALID_TO = 0.45
#: Clearance left on each side of an object when choosing a pre-grasp aperture.
DEFAULT_CLEARANCE = 0.006
#: Command range accepted by :meth:`armanual.sim.world.World.set_gripper`.
MIN_OPENING, MAX_OPENING = 0.0, 1.0


def moving_jaw_z(opening: float) -> float:
    """Where the moving jaw tip sits along the frame's z axis for a gripper command."""
    return MOVING_JAW_Z0 + MOVING_JAW_SLOPE * float(opening)


def opening_for_width(width: float, clearance: float = DEFAULT_CLEARANCE) -> float:
    """Gripper command that leaves ``clearance`` around an object of ``width`` across the jaws."""
    needed_z = FIXED_JAW[2] + width + clearance
    opening = (needed_z - MOVING_JAW_Z0) / MOVING_JAW_SLOPE
    return float(np.clip(opening + 0.06, 0.10, 0.95))


def pinch_offset(width: float, seat: float = 0.002) -> np.ndarray:
    """Where the centre of a ``width``-thick object ends up, in the TCP frame.

    The object is squeezed against the stationary jaw, so its centre sits half its width from
    the fixed jaw tip — not at the tool site. Pass this as ``tool_offset`` to
    :func:`armanual.control.kinematics.solve_reach` and the IK will place the *object*, not the
    site, where you asked.
    """
    return np.array([FIXED_JAW[0], 0.0, FIXED_JAW[2] + width / 2.0 + seat])


def grip_span(opening: float) -> tuple[float, float]:
    """(near, far) extent of the open jaws along the frame z axis, for collision reasoning."""
    return float(FIXED_JAW[2]), moving_jaw_z(opening)
