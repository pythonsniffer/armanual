"""Offscreen rendering backend selection.

MuJoCo picks its GL backend from ``MUJOCO_GL`` at first import. Machines differ: a headless
server wants ``egl`` or ``osmesa``, a WSL2/desktop session has a working ``glfw`` context. We
probe once, cache the choice, and let ``MUJOCO_GL`` override it so a deployment (for example the
Intel Core Ultra target) can pin a backend without touching code.
"""

from __future__ import annotations

import os
import warnings

_BACKENDS = ("egl", "glfw", "osmesa")
_chosen: str | None = None


def select_backend() -> str:
    """Return a usable MuJoCo GL backend, setting ``MUJOCO_GL`` before mujoco is imported."""
    global _chosen
    if _chosen:
        return _chosen
    forced = os.environ.get("MUJOCO_GL")
    candidates = (forced,) if forced else _BACKENDS
    for backend in candidates:
        os.environ["MUJOCO_GL"] = backend
        try:
            import mujoco

            model = mujoco.MjModel.from_xml_string(
                '<mujoco><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>'
            )
            with mujoco.Renderer(model, 32, 32) as renderer:
                renderer.update_scene(mujoco.MjData(model))
                renderer.render()
        except Exception as exc:
            if forced:
                raise RuntimeError(f"MUJOCO_GL={forced!r} is not usable here: {exc}") from exc
            warnings.warn(f"GL backend {backend!r} unavailable: {exc}", stacklevel=2)
            continue
        _chosen = backend
        return backend
    raise RuntimeError(
        "No usable MuJoCo GL backend. Install one of: libegl1 + mesa drivers (egl), "
        "libosmesa6 (osmesa), or run with a display (glfw)."
    )
