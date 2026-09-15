"""Seeded scene randomization: ``seed -> SceneSpec``.

Randomization is the whole of the Robustness & Generalization rubric dimension, so it is a pure,
inspectable function: the same seed and the same :class:`RandomizationConfig` always produce the
same :class:`~armanual.sim.spec.SceneSpec`, which is serialized with every evaluation result.

What varies: object placement, object dimensions and size labels, colours (including deliberately
similar colours), mass, friction, lighting, background, and which distractors are present.
What never varies: object *names*, so a task definition written against ``cup_target`` keeps
meaning across seeds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from armanual.sim.spec import (
    DEFAULT_ARMS,
    DrawerSpec,
    LightingSpec,
    ObjectSpec,
    SceneSpec,
    TableSpec,
)

#: Colour vocabulary shared with language grounding: the word a user may say, and the RGB the
#: renderer shows. Ambiguity is created by *near* colours (two blues), never by mislabelling.
COLORS: dict[str, tuple[float, float, float]] = {
    "white": (0.94, 0.94, 0.95),
    "blue": (0.20, 0.35, 0.85),
    "navy": (0.12, 0.20, 0.52),
    "red": (0.82, 0.21, 0.18),
    "green": (0.18, 0.60, 0.35),
    "yellow": (0.92, 0.78, 0.20),
    "black": (0.12, 0.12, 0.14),
    "silver": (0.78, 0.79, 0.82),
    "brown": (0.48, 0.32, 0.20),
}

# Table-frame reach model for one SO-101, measured with the IK solver over a grid of top-down and
# radially-tilted grasp poses (see docs/WORKSPACE.md and scripts/measure_workspace.py). Objects are
# always sampled inside this annulus so that an out-of-reach object is never mistaken for a
# manipulation failure.
REACH_MIN, REACH_MAX = 0.15, 0.32


@dataclass(frozen=True)
class RandomizationConfig:
    """Which perturbation axes are active, and how strong they are.

    Tier 1 evaluation runs with everything off (fixed nominal scene); Tier 5 turns everything on.
    Each flag maps to one line in the robustness table in docs/EVALUATION.md.
    """

    placement: bool = True
    placement_jitter: float = 0.05  # metres, per-axis uniform
    sizes: bool = True
    size_range: tuple[float, float] = (0.85, 1.18)
    colors: bool = True
    mass: bool = True
    mass_range: tuple[float, float] = (0.6, 1.6)
    friction: bool = True
    friction_range: tuple[float, float] = (0.7, 1.3)
    lighting: bool = True
    background: bool = True
    distractors: bool = True
    max_distractors: int = 3

    @classmethod
    def off(cls) -> "RandomizationConfig":
        return cls(
            placement=False,
            sizes=False,
            colors=False,
            mass=False,
            friction=False,
            lighting=False,
            background=False,
            distractors=False,
        )

    @classmethod
    def placement_only(cls) -> "RandomizationConfig":
        return cls(
            sizes=False,
            colors=False,
            mass=False,
            friction=False,
            lighting=False,
            background=False,
            distractors=False,
        )


# Nominal (seed-independent) layout. Positions are in table frame: +x right, +y away from the
# user, z = 0 at the table top. Both arm bases sit at y = -0.26.
_NOMINAL: tuple[dict, ...] = (
    dict(name="plate_main", category="plate", pos=(-0.04, -0.10, 0.006),
         size=(0.050, 0.050, 0.005), color="white", mass=0.085, size_label="medium"),
    dict(name="plate_side", category="plate", pos=(0.24, -0.13, 0.006),
         size=(0.037, 0.037, 0.004), color="white", mass=0.055, size_label="small"),
    dict(name="cup_target", category="cup", pos=(0.11, -0.03, 0.0),
         size=(0.027, 0.027, 0.032), color="blue", mass=0.055, size_label="medium"),
    dict(name="cup_distract", category="cup", pos=(-0.12, -0.03, 0.0),
         size=(0.026, 0.026, 0.030), color="navy", mass=0.050, size_label="medium"),
    dict(name="bottle_water", category="bottle", pos=(-0.35, -0.19, 0.0),
         size=(0.029, 0.029, 0.052), color="green", mass=0.16, size_label="large"),
    dict(name="tray_1", category="tray", pos=(0.30, -0.17, 0.008),
         size=(0.065, 0.048, 0.005), color="brown", mass=0.13, size_label="large"),
)

#: Utensils start inside the drawer — retrieving them is the multi-step dependency.
_DRAWER_ITEMS: tuple[dict, ...] = (
    dict(name="spoon_1", category="spoon", pos=(-0.048, -0.005, 0.022), yaw0=math.pi / 2,
         size=(0.026, 0.011, 0.0028), color="silver", mass=0.022, size_label="medium"),
    dict(name="fork_1", category="fork", pos=(0.002, -0.005, 0.022), yaw0=math.pi / 2,
         size=(0.026, 0.010, 0.0028), color="silver", mass=0.022, size_label="medium"),
    dict(name="knife_1", category="knife", pos=(0.050, -0.005, 0.022), yaw0=math.pi / 2,
         size=(0.030, 0.008, 0.0030), color="silver", mass=0.026, size_label="medium"),
)

_DISTRACTORS: tuple[dict, ...] = (
    dict(name="cup_extra", category="cup", pos=(0.33, -0.08, 0.0),
         size=(0.024, 0.024, 0.028), color="red", mass=0.045, size_label="small"),
    dict(name="napkin_1", category="napkin", pos=(-0.11, -0.21, 0.003),
         size=(0.040, 0.030, 0.002), color="yellow", mass=0.012, size_label="medium"),
    dict(name="plate_extra", category="plate", pos=(0.04, -0.22, 0.006),
         size=(0.042, 0.042, 0.004), color="white", mass=0.060, size_label="medium"),
)


def _clamp_reachable(pos: np.ndarray, arms) -> np.ndarray:
    """Pull a sampled position back into the reach annulus of its nearest arm."""
    best = None
    for arm in arms:
        base = np.array(arm.base_pos[:2])
        delta = pos[:2] - base
        dist = float(np.linalg.norm(delta))
        score = 0.0 if REACH_MIN <= dist <= REACH_MAX else min(
            abs(dist - REACH_MIN), abs(dist - REACH_MAX)
        )
        if best is None or score < best[0]:
            best = (score, base, delta, dist)
    score, base, delta, dist = best
    if score == 0.0:
        return pos
    target = float(np.clip(dist, REACH_MIN, REACH_MAX))
    direction = delta / (dist + 1e-9)
    out = pos.copy()
    out[:2] = base + direction * target
    return out


#: Half-extents of the drawer cabinet, from the MJCF in :mod:`armanual.sim.builder`, plus the
#: band the drawer sweeps when it opens. Table objects are kept out of it: an object spawned
#: inside the cabinet is jammed geometry, which would show up as a mysterious grasp failure.
DRAWER_KEEPOUT_HALF = (0.095, 0.085)
DRAWER_SWEEP = 0.14  # metres the open drawer + its handle occupy toward -y


def _avoid_drawer(objects: list[ObjectSpec], drawer_pos, present: bool) -> list[ObjectSpec]:
    """Push any table object out of the drawer cabinet and its opening path."""
    if not present:
        return objects
    hx, hy = DRAWER_KEEPOUT_HALF
    y_min = drawer_pos[1] - hy - DRAWER_SWEEP
    y_max = drawer_pos[1] + hy
    out = []
    for obj in objects:
        x, y = obj.pos[0], obj.pos[1]
        radius = max(obj.size[0], obj.size[1])
        inside_x = abs(x - drawer_pos[0]) < hx + radius
        inside_y = y_min - radius < y < y_max + radius
        if inside_x and inside_y:
            # Push out sideways (the cheapest direction that keeps the object on the table).
            sign = 1.0 if x >= drawer_pos[0] else -1.0
            x = drawer_pos[0] + sign * (hx + radius + 0.015)
            obj = _with_xy(obj, (x, y))
        out.append(obj)
    return out


def _separate(objects: list[ObjectSpec], min_gap: float = 0.018) -> list[ObjectSpec]:
    """Push overlapping objects apart so the episode does not start in contact."""
    items = list(objects)
    for _ in range(60):
        moved = False
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                ra = max(a.size[0], a.size[1])
                rb = max(b.size[0], b.size[1])
                pa, pb = np.array(a.pos[:2]), np.array(b.pos[:2])
                delta = pb - pa
                dist = float(np.linalg.norm(delta)) or 1e-6
                need = ra + rb + min_gap
                if dist < need:
                    push = (need - dist) / 2 + 1e-4
                    unit = delta / dist
                    items[i] = _with_xy(a, pa - unit * push)
                    items[j] = _with_xy(b, pb + unit * push)
                    moved = True
        if not moved:
            break
    return items


def _with_xy(obj: ObjectSpec, xy) -> ObjectSpec:
    return ObjectSpec(**{**obj.__dict__, "pos": (float(xy[0]), float(xy[1]), obj.pos[2])})


def _make_object(
    entry: dict, rng: np.random.Generator, cfg: RandomizationConfig, arms, in_drawer: bool
) -> ObjectSpec:
    size = np.array(entry["size"], dtype=float)
    size_label = entry["size_label"]
    if cfg.sizes:
        scale = rng.uniform(*cfg.size_range)
        size = size * scale
        size_label = "small" if scale < 0.95 else ("large" if scale > 1.08 else "medium")

    color_name = entry["color"]
    if cfg.colors and entry["category"] in ("cup", "plate", "napkin"):
        # Re-colour only tableware, and only within a palette that keeps the
        # instruction checkable (a "blue cup" may become navy — deliberately confusable).
        palette = ["blue", "navy", "red", "green", "yellow", "white"]
        color_name = str(rng.choice(palette))
    rgb = COLORS[color_name]

    pos = np.array(entry["pos"], dtype=float)
    if cfg.placement and not in_drawer:
        pos[:2] += rng.uniform(-cfg.placement_jitter, cfg.placement_jitter, size=2)
        pos = _clamp_reachable(pos, arms)
    elif cfg.placement and in_drawer:
        pos[0] += float(rng.uniform(-0.012, 0.012))

    mass = entry["mass"] * (rng.uniform(*cfg.mass_range) if cfg.mass else 1.0)
    fr = 1.0 * (rng.uniform(*cfg.friction_range) if cfg.friction else 1.0)
    yaw0 = float(entry.get("yaw0", 0.0))
    yaw = yaw0 + (float(rng.uniform(-math.pi, math.pi)) if cfg.placement else 0.0)
    if entry["category"] in ("spoon", "fork", "knife"):
        # Utensils keep their nominal orientation (in the drawer: pointing away from the user);
        # only a small jitter, because a utensil lying across its neighbours is a scene bug,
        # not a robustness test.
        yaw = yaw0 + (float(rng.uniform(-0.18, 0.18)) if cfg.placement else 0.0)

    return ObjectSpec(
        name=entry["name"],
        category=entry["category"],
        pos=(float(pos[0]), float(pos[1]), float(pos[2])),
        yaw=yaw,
        size=(float(size[0]), float(size[1]), float(size[2])),
        rgba=(*rgb, 1.0),
        color_name=color_name,
        mass=float(mass),
        friction=(float(fr), 0.005, 0.0001),
        size_label=size_label,
    )


def sample_scene(
    seed: int,
    cfg: RandomizationConfig | None = None,
    *,
    drawer: bool = True,
    include_tray: bool = True,
) -> SceneSpec:
    """Sample one reproducible dinner-table episode."""
    cfg = cfg if cfg is not None else RandomizationConfig()
    rng = np.random.default_rng(seed)
    arms = DEFAULT_ARMS

    entries = [e for e in _NOMINAL if include_tray or e["category"] != "tray"]
    objects = [_make_object(e, rng, cfg, arms, in_drawer=False) for e in entries]

    drawer_pos = (-0.26, 0.06, 0.0)
    drawer_items: list[ObjectSpec] = []
    if drawer:
        for entry in _DRAWER_ITEMS:
            item = _make_object(entry, rng, cfg, arms, in_drawer=True)
            drawer_items.append(
                ObjectSpec(
                    **{
                        **item.__dict__,
                        "pos": (
                            drawer_pos[0] + item.pos[0],
                            drawer_pos[1] + item.pos[1],
                            drawer_pos[2] + item.pos[2],
                        ),
                    }
                )
            )

    if cfg.distractors:
        n = int(rng.integers(1, cfg.max_distractors + 1))
        chosen = rng.choice(len(_DISTRACTORS), size=n, replace=False)
        objects += [_make_object(_DISTRACTORS[i], rng, cfg, arms, in_drawer=False) for i in chosen]

    objects = _avoid_drawer(objects, drawer_pos, drawer)
    objects = _separate(objects)
    objects = _avoid_drawer(objects, drawer_pos, drawer)

    lighting = LightingSpec()
    if cfg.lighting:
        lighting = LightingSpec(
            diffuse=float(rng.uniform(0.55, 1.0)),
            ambient=float(rng.uniform(0.18, 0.42)),
            specular=float(rng.uniform(0.15, 0.5)),
            azimuth=float(rng.uniform(-1.0, 1.0)),
        )
    background = (0.35, 0.40, 0.46)
    if cfg.background:
        background = tuple(float(v) for v in rng.uniform(0.15, 0.6, size=3))

    return SceneSpec(
        seed=seed,
        arms=arms,
        objects=tuple(objects + drawer_items),
        table=TableSpec(),
        drawer=DrawerSpec(
            present=drawer,
            pos=drawer_pos,
            contents=tuple(o.name for o in drawer_items),
        ),
        lighting=lighting,
        background_rgb=background,
        friction_scale=float(rng.uniform(*cfg.friction_range)) if cfg.friction else 1.0,
    )
