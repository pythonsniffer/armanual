"""Turn the simulator's cameras into a :class:`SceneObservation` the planner can use.

The observer owns the RGB-D rendering and the association step that gives detections stable
names across time. Association is by *appearance and position*, not by identity: two detections
match the same track when they agree on category and colour and are close in the plane. That is
what lets the closed loop notice that "the blue cup" has moved, rather than silently following a
privileged pose.

A second, privileged observer is provided for ablation and for labelling training data. It is
never used in evaluation runs unless explicitly requested, and every result records which
observer produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import mujoco
import numpy as np

from armanual.perception.camera import CameraModel, camera_model
from armanual.perception.detector import Detection, DetectorConfig, SceneObservation, TabletopDetector


@dataclass
class ObserverConfig:
    camera: str = "cam_overhead"
    #: Cameras fused for a full observation. Overhead gives position, front recovers objects the
    #: arms hide from above.
    cameras: tuple[str, ...] = ("cam_overhead", "cam_front")
    size: tuple[int, int] = (320, 240)
    association_radius: float = 0.07
    merge_radius: float = 0.05
    #: Drawer front's y position when fully closed, measured once from the scene spec.
    drawer_closed_y: float | None = None


class CameraObserver:
    """Camera-only perception: renders RGB + depth and detects tabletop objects."""

    def __init__(self, world, config: ObserverConfig | None = None,
                 detector: TabletopDetector | None = None):
        self.world = world
        self.config = config or ObserverConfig()
        table = world.scene.table
        self.detector = detector or TabletopDetector(
            DetectorConfig(
                table_half_x=table.half_x + 0.02,
                table_half_y=table.half_y + 0.02,
                table_center_y=table.center_y,
                base_positions=tuple((a.base_pos[0], a.base_pos[1]) for a in world.scene.arms),
                furniture_boxes=(
                    ((world.scene.drawer.pos[0], world.scene.drawer.pos[1], 0.11, 0.11),)
                    if world.scene.drawer.present
                    else ()
                ),
            )
        )
        self._depth_renderer: mujoco.Renderer | None = None
        self._tracks: dict[int, Detection] = {}
        self._next_track = 1

    # ------------------------------------------------------------------------------ rendering
    def _render_rgbd(self, camera: str) -> tuple[np.ndarray, np.ndarray, CameraModel]:
        width, height = self.config.size
        rgb = self.world.render(camera, size=(width, height))
        if self._depth_renderer is None:
            self._depth_renderer = mujoco.Renderer(self.world.model, height=height, width=width)
            self._depth_renderer.enable_depth_rendering()
        self._depth_renderer.update_scene(self.world.data, camera=camera)
        depth = np.asarray(self._depth_renderer.render(), dtype=np.float32)
        return rgb, depth, camera_model(self.world, camera, (width, height))

    # ---------------------------------------------------------------------------- observation
    def observe(self, camera: str | None = None) -> SceneObservation:
        """Observe from one camera, or fuse several when ``config.cameras`` lists more than one.

        Fusing the overhead and front views matters in practice: an arm parked over the table
        hides whatever is under it from above, and a single-camera observation would report that
        object as gone.
        """
        cameras = [camera] if camera else list(self.config.cameras)
        detections: list[Detection] = []
        points_by_camera: dict[str, np.ndarray] = {}
        for order, cam in enumerate(cameras):
            rgb, depth, model = self._render_rgbd(cam)
            height_px, width_px = depth.shape
            vs, us = np.mgrid[0:height_px, 0:width_px]
            points = model.unproject(us, vs, depth)
            points_by_camera[cam] = points
            self._set_drawer_region(points)
            found = self.detector.detect_from_points(rgb, points)
            detections = self._merge(detections, found, primary=order == 0)
        self._assign_tracks(detections)
        return SceneObservation(
            detections=detections,
            drawer_open=self._estimate_drawer(points_by_camera[cameras[0]]),
            camera="+".join(cameras),
            stamp=self.world.time,
        )

    def _set_drawer_region(self, points: np.ndarray) -> None:
        """Raise the height threshold inside the drawer, wherever the drawer currently is."""
        spec = self.world.scene.drawer
        if not spec.present:
            return
        opening = self._estimate_drawer(points)
        centre_y = spec.pos[1] - opening
        # Clearance sits just above the drawer's floor: high enough to cut the floor itself out of
        # the "above the table" mask, low enough that a 17 mm-tall utensil still shows several
        # millimetres of height and survives the minimum-blob-size test.
        self.detector.config = replace(
            self.detector.config,
            region_slabs=((spec.pos[0], centre_y, 0.085, 0.085, 0.0125, 0.021),),
            furniture_boxes=(
                (spec.pos[0], spec.pos[1], 0.11, 0.11),
                (spec.pos[0], centre_y, 0.10, 0.10),
            ),
        )

    def _merge(self, existing: list[Detection], new: list[Detection], *,
               primary: bool) -> list[Detection]:
        """Fold another view's detections into the running list.

        The overhead camera is authoritative: it sees the table plane square-on, so its position,
        radius and colour estimates are the ones to keep. A secondary view may only *add* objects
        the primary could not see — typically something an arm was standing over. It is not
        allowed to overwrite, because from a shallow angle several objects often melt into one
        blob, and letting that blob win turns two correct detections into one wrong one.
        """
        if primary:
            return list(new)
        merged = list(existing)
        for candidate in new:
            if candidate.category in ("furniture", "unknown"):
                continue
            if candidate.radius > 0.09:  # implausibly wide: a merged blob, not a table object
                continue
            near = min(
                (float(np.linalg.norm(c.position[:2] - candidate.position[:2])) for c in merged),
                default=1e9,
            )
            if near > self.config.merge_radius * 2:
                candidate.confidence *= 0.8  # seen only from a shallow angle
                merged.append(candidate)
        return merged

    def _assign_tracks(self, detections: list[Detection]) -> None:
        """Give each detection the id of the nearest compatible track, or a fresh one."""
        used: set[int] = set()
        for detection in detections:
            best_id, best_dist = None, self.config.association_radius
            for track_id, previous in self._tracks.items():
                if track_id in used or previous.category != detection.category:
                    continue
                distance = float(np.linalg.norm(previous.position[:2] - detection.position[:2]))
                if distance < best_dist:
                    best_id, best_dist = track_id, distance
            if best_id is None:
                best_id = self._next_track
                self._next_track += 1
            detection.track_id = best_id
            detection.name = self._tracks[best_id].name if best_id in self._tracks else None
            used.add(best_id)
            self._tracks[best_id] = detection

    def _estimate_drawer(self, points: np.ndarray) -> float:
        """How far the drawer is open, in metres, measured from the depth image.

        Rather than relying on blob labelling, this reads the point cloud directly: inside the
        cabinet's x band and below its top panel, the front-most (smallest y) surface *is* the
        drawer front, and how far it has travelled from the closed position is the opening.
        """
        spec = self.world.scene.drawer
        if not spec.present:
            return 0.0
        closed_y = (
            self.config.drawer_closed_y
            if self.config.drawer_closed_y is not None
            else spec.pos[1] - 0.076
        )
        band = (
            (np.abs(points[..., 0] - spec.pos[0]) < 0.07)
            & (points[..., 2] > 0.012)
            & (points[..., 2] < 0.048)
            & (points[..., 1] < spec.pos[1])
            & (points[..., 1] > spec.pos[1] - 0.30)
        )
        if not band.any():
            return 0.0
        front_y = float(np.percentile(points[..., 1][band], 2))
        return float(np.clip(closed_y - front_y, 0.0, spec.travel + 0.02))

    def close(self) -> None:
        if self._depth_renderer is not None:
            self._depth_renderer.close()
            self._depth_renderer = None


class PrivilegedObserver:
    """Ground-truth 'perception' straight from simulator state.

    Used for two things only: labelling demonstration data, and ablations that separate
    perception error from manipulation error. Results produced with it are tagged as privileged
    so they are never mistaken for camera-based performance.
    """

    privileged = True

    def __init__(self, world):
        self.world = world

    def observe(self, camera: str | None = None) -> SceneObservation:
        detections = []
        for obj in self.world.scene.objects:
            position = self.world.object_pos(obj.name)
            category = "utensil" if obj.is_utensil else obj.category
            detections.append(
                Detection(
                    position=position,
                    radius=float(max(obj.size[0], obj.size[1])),
                    height=float(position[2] + obj.size[2]),
                    color_name=obj.color_name,
                    color_rgb=np.asarray(obj.rgba[:3]),
                    category=category,
                    pixel_area=1000,
                    elongation=1.0 if category != "utensil" else 4.0,
                    confidence=1.0,
                    name=obj.name,
                )
            )
        return SceneObservation(
            detections=detections,
            drawer_open=self.world.drawer_opening(),
            camera="privileged",
            stamp=self.world.time,
        )

    def close(self) -> None:  # symmetry with CameraObserver
        return
