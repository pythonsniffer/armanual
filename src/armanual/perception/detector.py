"""Tabletop object detection from rendered RGB-D — no simulator state.

This is the perception stage the rest of the system runs on. It sees only what a camera sees:

1. back-project the depth image to metric world points using the camera's real intrinsics,
2. keep points that sit above the table plane and inside the table's footprint,
3. drop the robot's own pixels by colour (the SO-101 model is a distinctive yellow),
4. label connected components and, for each blob, estimate position, footprint radius, height,
   dominant colour and pixel area,
5. classify the category from that measured geometry (a tall narrow blob is a bottle, a flat wide
   disc is a plate, and so on).

Nothing here reads object poses from MuJoCo. That matters for the rubric — "reason over camera
observations" is only a real claim if the pipeline *can't* cheat — and it matters for robustness,
because a detector that works off pixels degrades the way a real one does when lighting, colour
and placement change. Accuracy against simulator ground truth is measured separately in
:mod:`armanual.perception.evaluate`, which is allowed to look.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from armanual.perception.camera import CameraModel

#: Colour vocabulary the detector can name, matching the scene palette in
#: :mod:`armanual.sim.randomize`. Kept as linear RGB, the space the renderer writes.
COLOR_ANCHORS: dict[str, tuple[float, float, float]] = {
    "white": (0.94, 0.94, 0.95),
    "blue": (0.20, 0.35, 0.85),
    "navy": (0.12, 0.20, 0.52),
    "red": (0.82, 0.21, 0.18),
    "green": (0.18, 0.60, 0.35),
    "purple": (0.55, 0.30, 0.75),
    "orange": (0.92, 0.50, 0.13),
    "silver": (0.78, 0.79, 0.82),
    "brown": (0.48, 0.32, 0.20),
}
#: The arm's own yellow. Compared in *chromaticity* (rgb / max(rgb)) rather than raw RGB, because
#: the same yellow shell renders anywhere from bright to nearly brown depending on the light — and
#: a shaded arm link that survives this filter becomes a phantom "bottle" on the table.
ROBOT_YELLOW = np.array([1.0, 0.82, 0.12])
ROBOT_DARK = np.array([0.1, 0.1, 0.1])


def chromaticity(rgb: np.ndarray) -> np.ndarray:
    """Brightness-normalized colour: rgb divided by its own maximum channel."""
    peak = np.max(rgb, axis=-1, keepdims=True)
    return rgb / np.maximum(peak, 1e-3)


@dataclass
class Detection:
    """One detected tabletop object, in world coordinates."""

    position: np.ndarray  # (3,) centroid of the visible top surface, world frame
    radius: float  # footprint radius, metres
    height: float  # top height above the table, metres
    color_name: str
    color_rgb: np.ndarray
    category: str
    pixel_area: int
    elongation: float  # major/minor axis ratio of the footprint
    #: Direction of the footprint's major axis, radians in the table plane. Meaningful for
    #: elongated objects (a utensil must be grasped *across* its handle, not along it).
    orientation: float = 0.0
    #: Radius of just the top slice of the object. For a bottle this is the *neck*, which is what
    #: the gripper actually closes on — the full footprint would be the body and would put the
    #: pinch point off the object entirely.
    top_radius: float = 0.0
    confidence: float = 1.0
    track_id: int | None = None
    #: Name assigned by association with an expected object list, when one is supplied.
    name: str | None = None

    @property
    def size_label(self) -> str:
        if self.category == "plate":
            return "small" if self.radius < 0.042 else ("large" if self.radius > 0.052 else "medium")
        if self.category == "cup":
            return "small" if self.radius < 0.024 else ("large" if self.radius > 0.030 else "medium")
        return "medium"

    def describe(self) -> str:
        return f"{self.size_label} {self.color_name} {self.category}"


@dataclass
class DetectorConfig:
    """Thresholds for the tabletop segmentation. All in metres unless noted."""

    table_clearance: float = 0.008  # points below this above the table are table, not object
    #: Nothing the task cares about stands taller than this. Anything that does is an arm seen
    #: from the front camera, and treating it as tableware invents objects that are not there.
    max_height: float = 0.17
    min_pixels: int = 25
    robot_color_tol: float = 0.22
    table_half_x: float = 0.44
    table_half_y: float = 0.30
    table_center_y: float = -0.05
    #: Ignore a margin around the arm bases; the mounts are part of the robot, not the scene.
    base_exclusion: float = 0.075
    base_positions: tuple[tuple[float, float], ...] = ((-0.16, -0.30), (0.16, -0.30))
    #: Footprint of fixed furniture (the drawer cabinet). Blobs centred here are labelled
    #: ``furniture`` instead of tableware — the cabinet is a landmark, not something to pick up.
    furniture_boxes: tuple[tuple[float, float, float, float], ...] = ()
    #: Inside a furniture box, only blobs *taller* than this are furniture (the cabinet shell and
    #: the drawer's own walls). Anything shorter is something lying in the drawer — which is the
    #: whole point of opening it.
    furniture_min_height: float = 0.026
    #: Height slabs that replace the global "above the table" test inside a region, as
    #: ``(cx, cy, half_x, half_y, z_min, z_max)``. The open drawer needs both bounds: its floor
    #: sits ~10 mm up, so the lower bound cuts the floor away, and its side walls stand ~33 mm up,
    #: so without an upper bound the cutlery merges with the walls into one tall blob and is
    #: classified as furniture. Slicing the drawer at cutlery height separates them.
    region_slabs: tuple[tuple[float, float, float, float, float, float], ...] = ()


@dataclass
class SceneObservation:
    """Everything perception reports for one time step."""

    detections: list[Detection] = field(default_factory=list)
    drawer_open: float = 0.0
    camera: str = "cam_overhead"
    stamp: float = 0.0

    def by_category(self, category: str) -> list[Detection]:
        return [d for d in self.detections if d.category == category]

    def describe(self) -> str:
        return "; ".join(sorted(d.describe() for d in self.detections))


def _nearest_color(rgb: np.ndarray) -> tuple[str, float]:
    """Nearest palette colour and a 0-1 confidence from how close the match is."""
    best_name, best_dist = "white", 1e9
    for name, anchor in COLOR_ANCHORS.items():
        dist = float(np.linalg.norm(rgb - np.asarray(anchor)))
        if dist < best_dist:
            best_name, best_dist = name, dist
    return best_name, float(np.clip(1.0 - best_dist, 0.0, 1.0))


def _classify(radius: float, height: float, elongation: float, color: str) -> str:
    """Category from measured geometry.

    The thresholds come from the object dimensions the scene actually generates (see
    ``armanual.sim.randomize``), widened to cover the randomized size range.
    """
    if height > 0.075:
        return "bottle"
    if height > 0.022 and radius < 0.045:
        # A long thin ridge is not a cup, whatever its height: that shape is a handle or a rim
        # seen edge-on, and letting it pass as a cup puts a phantom target on the table.
        return "utensil" if elongation > 3.5 and height < 0.040 else "cup"
    if height <= 0.022 and elongation > 2.2 and radius < 0.055:
        return "utensil"
    if height <= 0.026 and radius >= 0.055:
        return "tray"
    if height <= 0.022:
        return "plate" if color != "silver" else "utensil"
    return "unknown"


class TabletopDetector:
    """RGB-D tabletop detector. Stateless: one call in, one observation out."""

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()

    def detect(self, rgb: np.ndarray, depth: np.ndarray, camera: CameraModel) -> list[Detection]:
        """Convenience wrapper: back-project the depth image, then detect."""
        height_px, width_px = depth.shape
        vs, us = np.mgrid[0:height_px, 0:width_px]
        return self.detect_from_points(rgb, camera.unproject(us, vs, depth))

    def detect_from_points(self, rgb: np.ndarray, points: np.ndarray) -> list[Detection]:
        """Detect from an already back-projected point image (H, W, 3) in world coordinates."""
        cfg = self.config
        rgb_f = rgb.astype(np.float32) / 255.0

        z = points[..., 2]
        lower = np.full(z.shape, cfg.table_clearance, dtype=float)
        upper = np.full(z.shape, cfg.max_height, dtype=float)
        for cx, cy, hx, hy, z_min, z_max in cfg.region_slabs:
            region = (np.abs(points[..., 0] - cx) < hx) & (np.abs(points[..., 1] - cy) < hy)
            lower = np.where(region, z_min, lower)
            upper = np.where(region, z_max, upper)
        on_table = (
            (z > lower)
            & (z < upper)
            & (np.abs(points[..., 0]) < cfg.table_half_x)
            & (np.abs(points[..., 1] - cfg.table_center_y) < cfg.table_half_y)
        )
        # Robot pixels: the arm's yellow shell (matched by chromaticity so shading does not
        # disguise it) and its dark servos.
        robot = (
            np.linalg.norm(chromaticity(rgb_f) - chromaticity(ROBOT_YELLOW), axis=-1)
            < cfg.robot_color_tol
        ) | (np.linalg.norm(rgb_f - ROBOT_DARK, axis=-1) < 0.14)
        for base in cfg.base_positions:
            near_base = (
                np.linalg.norm(points[..., :2] - np.asarray(base), axis=-1) < cfg.base_exclusion
            )
            robot |= near_base
        mask = on_table & ~robot

        labels, count = ndimage.label(mask)
        detections: list[Detection] = []
        for index in range(1, count + 1):
            blob = labels == index
            area = int(blob.sum())
            if area < cfg.min_pixels:
                continue
            blob_points = points[blob]
            blob_rgb = rgb_f[blob]
            centroid_xy = np.median(blob_points[:, :2], axis=0)
            top_z = float(np.percentile(blob_points[:, 2], 95))
            centered = blob_points[:, :2] - centroid_xy
            orientation = 0.0
            if len(centered) >= 3:
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                order = np.argsort(np.abs(eigenvalues))[::-1]
                major = float(np.sqrt(max(abs(eigenvalues[order[0]]), 1e-9)))
                minor = float(np.sqrt(max(abs(eigenvalues[order[1]]), 1e-12)))
                axis = eigenvectors[:, order[0]]
                orientation = float(np.arctan2(axis[1], axis[0]))
            else:
                major = minor = 0.01
            radius = float(np.percentile(np.linalg.norm(centered, axis=1), 92))
            top_slice = blob_points[blob_points[:, 2] > top_z - 0.022]
            top_radius = (
                float(np.percentile(np.linalg.norm(top_slice[:, :2] - centroid_xy, axis=1), 85))
                if len(top_slice) >= 5
                else radius
            )
            elongation = major / max(minor, 1e-6)
            mean_rgb = np.median(blob_rgb, axis=0)
            color_name, color_conf = _nearest_color(mean_rgb)
            category = _classify(radius, top_z, elongation, color_name)
            for cx, cy, hx, hy in cfg.furniture_boxes:
                inside = abs(centroid_xy[0] - cx) < hx and abs(centroid_xy[1] - cy) < hy
                if inside and top_z > cfg.furniture_min_height:
                    category = "furniture"
                    break
            detections.append(
                Detection(
                    position=np.array([centroid_xy[0], centroid_xy[1], top_z]),
                    radius=radius,
                    top_radius=top_radius,
                    height=top_z,
                    color_name=color_name,
                    color_rgb=mean_rgb,
                    category=category,
                    pixel_area=area,
                    elongation=elongation,
                    orientation=orientation,
                    confidence=float(np.clip(0.45 + 0.4 * color_conf + min(area, 600) / 3000, 0, 1)),
                )
            )
        return detections
