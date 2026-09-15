"""Close-range pose refinement with the wrist camera.

The overhead camera localizes objects to about a centimetre (docs/PERCEPTION.md). That is fine for
deciding *what* to pick and *which arm* should do it, but it is not fine for closing a 9 mm-wide
pair of jaws around the rim of a plate: the error and the target are the same size.

So the pick does what a person does — it looks again from close up. At the pre-grasp pose the
wrist camera is already directly above the object at about 7 cm, where one pixel covers a fraction
of a millimetre instead of several. Re-detecting from there and correcting the grasp point turns a
coin-flip grasp into a repeatable one, and it costs no extra motion.

This is eye-in-hand sensing, not privileged state: the camera pose comes from the arm's own
forward kinematics, exactly as it would from encoders on hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from armanual.perception.camera import camera_model
from armanual.perception.detector import Detection, DetectorConfig, TabletopDetector


@dataclass
class RefinementResult:
    position: np.ndarray
    radius: float
    height: float
    shift_mm: float
    pixels: int

    @property
    def usable(self) -> bool:
        # A correction larger than this means the wrist view found a *different* object; trust
        # the original detection rather than lunging at something else.
        return self.shift_mm < 45.0 and self.pixels >= 30


class WristRefiner:
    """Re-localizes an object from the wrist camera of a given arm."""

    def __init__(self, world, size: tuple[int, int] = (200, 150)):
        self.world = world
        self.size = size
        self._renderers: dict[str, tuple[mujoco.Renderer, mujoco.Renderer]] = {}
        table = world.scene.table
        self.detector = TabletopDetector(
            DetectorConfig(
                table_half_x=table.half_x + 0.05,
                table_half_y=table.half_y + 0.05,
                table_center_y=table.center_y,
                min_pixels=20,
                # The wrist camera is mounted *on* the arm, so the arm fills much of the frame and
                # the base-exclusion discs are meaningless here; rely on the colour mask instead.
                base_exclusion=0.0,
                base_positions=(),
            )
        )

    def _renderer(self, arm: str) -> tuple[mujoco.Renderer, mujoco.Renderer]:
        if arm not in self._renderers:
            width, height = self.size
            rgb = mujoco.Renderer(self.world.model, height=height, width=width)
            depth = mujoco.Renderer(self.world.model, height=height, width=width)
            depth.enable_depth_rendering()
            self._renderers[arm] = (rgb, depth)
        return self._renderers[arm]

    def refine(self, arm: str, expected: np.ndarray, *, search_radius: float = 0.06
               ) -> RefinementResult | None:
        """Re-detect the object nearest ``expected`` from ``arm``'s wrist camera."""
        rgb_renderer, depth_renderer = self._renderer(arm)
        camera = f"{arm}/wrist_cam"
        rgb_renderer.update_scene(self.world.data, camera=camera)
        depth_renderer.update_scene(self.world.data, camera=camera)
        rgb = rgb_renderer.render()
        depth = np.asarray(depth_renderer.render(), dtype=np.float32)

        model = camera_model(self.world, camera, self.size)
        height_px, width_px = depth.shape
        vs, us = np.mgrid[0:height_px, 0:width_px]
        points = model.unproject(us, vs, depth)
        detections = self.detector.detect_from_points(rgb, points)
        if not detections:
            return None

        expected = np.asarray(expected, dtype=float)
        best: Detection | None = None
        best_distance = search_radius
        for detection in detections:
            distance = float(np.linalg.norm(detection.position[:2] - expected[:2]))
            if distance < best_distance:
                best, best_distance = detection, distance
        if best is None:
            return None
        return RefinementResult(
            position=best.position.copy(),
            radius=float(best.radius),
            height=float(best.height),
            shift_mm=best_distance * 1000,
            pixels=best.pixel_area,
        )

    def close(self) -> None:
        for rgb, depth in self._renderers.values():
            rgb.close()
            depth.close()
        self._renderers.clear()


def refine_grasp(refiner: WristRefiner, arm: str, grasp, *, keep_height: bool = False):
    """Return a copy of ``grasp`` shifted onto the wrist camera's closer look at the object.

    The xy correction is always applied when the refinement is usable. The z is only taken from
    the wrist view when asked for, because at close range a curved rim's apparent top varies more
    than its position does.
    """
    from dataclasses import replace

    result = refiner.refine(arm, grasp.pos)
    if result is None or not result.usable:
        return grasp, None
    shift = result.position[:2] - grasp.pos[:2]
    position = grasp.pos.copy()
    position[:2] += shift
    if keep_height:
        position[2] = max(0.006, result.height * 0.62)
    return replace(grasp, pos=position), result
