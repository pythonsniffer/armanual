"""Declarative description of a dinner-table scene.

A :class:`SceneSpec` is a plain-data description of one episode's world: which arms exist,
which objects are on the table, their physical and visual properties, and the global
perturbations (lighting, friction scale, background). It is produced either by a task config
or by the randomizer from a seed, and consumed by :mod:`armanual.sim.builder` to emit MJCF.

Keeping this layer data-only means an episode is fully reproducible from a serialized spec,
and randomization is a pure function ``seed -> SceneSpec`` that we can log, diff and replay.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Literal

Vec3 = tuple[float, float, float]
Vec4 = tuple[float, float, float, float]

#: Object categories the dinner-table task understands. Each one exists to exercise a
#: capability, not to add clutter (see docs/ARCHITECTURE.md).
Category = Literal["plate", "cup", "spoon", "fork", "knife", "bottle", "tray", "napkin"]


@dataclass(frozen=True)
class ArmSpec:
    """One SO-101 arm instance: where it is mounted and what it is called."""

    name: str  # "left" | "right"
    base_pos: Vec3
    base_yaw: float  # rotation about +z, radians; 0 = facing +y
    home_qpos: tuple[float, ...] = (0.0, -1.2, 1.2, 0.6, 0.0, 1.0)


@dataclass(frozen=True)
class ObjectSpec:
    """A single manipulable table object.

    ``size`` is interpreted per category (see :mod:`armanual.sim.builder`); ``color_name``
    is the human-facing colour word used by language grounding, and must stay consistent
    with ``rgba`` so that an instruction like "the blue cup" is checkable.
    """

    name: str
    category: Category
    pos: Vec3
    yaw: float = 0.0
    size: Vec3 = (0.04, 0.04, 0.01)
    rgba: Vec4 = (0.9, 0.9, 0.9, 1.0)
    color_name: str = "white"
    mass: float = 0.05
    friction: Vec3 = (1.0, 0.005, 0.0001)
    size_label: Literal["small", "medium", "large"] = "medium"
    movable: bool = True

    @property
    def is_utensil(self) -> bool:
        return self.category in ("spoon", "fork", "knife")


@dataclass(frozen=True)
class TableSpec:
    """Table surface geometry. The table top plane is the z origin of the task frame.

    The table is deliberately sized to the *measured* SO-101 workspace (see
    docs/WORKSPACE.md): a table the arms cannot cover would put objects out of reach and turn
    robustness failures into geometry failures.
    """

    half_x: float = 0.42
    half_y: float = 0.28
    center_y: float = -0.05
    top_z: float = 0.0
    rgba: Vec4 = (0.65, 0.52, 0.39, 1.0)


@dataclass(frozen=True)
class DrawerSpec:
    """A sliding drawer at the far edge of the table, used for dependency tasks."""

    present: bool = True
    pos: Vec3 = (-0.26, 0.06, 0.0)
    travel: float = 0.11  # metres of slide
    open_threshold: float = 0.07  # opened far enough to reach inside
    contents: tuple[str, ...] = ()  # object names that start inside the drawer


@dataclass(frozen=True)
class LightingSpec:
    diffuse: float = 0.8
    ambient: float = 0.3
    specular: float = 0.3
    azimuth: float = 0.0  # light direction rotation, radians


@dataclass(frozen=True)
class SceneSpec:
    """Everything needed to rebuild one episode's world, byte-for-byte."""

    seed: int = 0
    arms: tuple[ArmSpec, ...] = ()
    objects: tuple[ObjectSpec, ...] = ()
    table: TableSpec = field(default_factory=TableSpec)
    drawer: DrawerSpec = field(default_factory=DrawerSpec)
    lighting: LightingSpec = field(default_factory=LightingSpec)
    background_rgb: Vec3 = (0.35, 0.40, 0.46)
    friction_scale: float = 1.0
    timestep: float = 0.005

    def object_by_name(self, name: str) -> ObjectSpec:
        for obj in self.objects:
            if obj.name == name:
                return obj
        raise KeyError(f"no object named {name!r} in scene (have: {[o.name for o in self.objects]})")

    def objects_of(self, category: Category) -> tuple[ObjectSpec, ...]:
        return tuple(o for o in self.objects if o.category == category)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, sort_keys=True)


#: Both arms are mounted at the near edge of the table, rotated a quarter turn so their
#: zero-configuration reach direction (+x in the arm's own frame) points across the table (+y).
#: The 0.32 m separation gives each arm its own side plus a shared middle band for hand-offs.
DEFAULT_ARMS: tuple[ArmSpec, ...] = (
    ArmSpec(name="left", base_pos=(-0.16, -0.30, 0.0), base_yaw=math.pi / 2),
    ArmSpec(name="right", base_pos=(0.16, -0.30, 0.0), base_yaw=math.pi / 2),
)
