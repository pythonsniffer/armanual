"""Camera intrinsics, extrinsics and back-projection for the simulated cameras.

The perception stack is deliberately built on the *rendered images*, not on simulator state, so
the pipeline that runs in evaluation is the same one a real camera would feed. This module is the
bridge: it reads each camera's true intrinsics from the compiled model and turns an RGB-D pair
into metric 3D points in world coordinates.

MuJoCo's camera frame convention: +x right, +y up, and the camera looks down its **-z** axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class CameraModel:
    """Pinhole model for one MuJoCo camera at a particular simulation state."""

    name: str
    width: int
    height: int
    fovy_deg: float
    position: np.ndarray  # (3,) world
    rotation: np.ndarray  # (3, 3) camera-to-world

    @property
    def focal_px(self) -> float:
        return 0.5 * self.height / np.tan(np.deg2rad(self.fovy_deg) / 2.0)

    @property
    def principal_point(self) -> tuple[float, float]:
        return (self.width - 1) / 2.0, (self.height - 1) / 2.0

    @property
    def intrinsics(self) -> np.ndarray:
        f = self.focal_px
        cx, cy = self.principal_point
        return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])

    def unproject(self, u: np.ndarray, v: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Pixel coordinates + metric depth -> world points, shape (..., 3)."""
        f = self.focal_px
        cx, cy = self.principal_point
        x = (np.asarray(u, dtype=float) - cx) / f
        y = -(np.asarray(v, dtype=float) - cy) / f  # image v grows downward, camera y grows up
        depth = np.asarray(depth, dtype=float)
        cam = np.stack([x * depth, y * depth, -depth], axis=-1)
        return cam @ self.rotation.T + self.position

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points -> (pixel uv, depth). Points behind the camera get negative depth."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        cam = (points - self.position) @ self.rotation
        depth = -cam[:, 2]
        f = self.focal_px
        cx, cy = self.principal_point
        with np.errstate(divide="ignore", invalid="ignore"):
            u = cx + f * cam[:, 0] / depth
            v = cy - f * cam[:, 1] / depth
        return np.stack([u, v], axis=-1), depth


def camera_model(world, name: str, size: tuple[int, int]) -> CameraModel:
    """Read a camera's current intrinsics and pose out of the compiled model."""
    cam_id = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if cam_id < 0:
        raise KeyError(f"camera {name!r} not in model")
    width, height = size
    return CameraModel(
        name=name,
        width=width,
        height=height,
        fovy_deg=float(world.model.cam_fovy[cam_id]),
        position=world.data.cam_xpos[cam_id].copy(),
        rotation=world.data.cam_xmat[cam_id].reshape(3, 3).copy(),
    )
