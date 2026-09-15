"""Measure the SO-101's real capabilities in this scene, and write them to docs/WORKSPACE.md.

Every geometry decision in this repository (table size, where objects may spawn, how a grasp is
aimed, how wide the jaws pre-open) is derived from the numbers this script produces rather than
from the datasheet or from guesswork. Re-run it after any change to the arm model, the mount
positions or the table, and commit the regenerated document:

    python scripts/measure_workspace.py

It measures four things:

1. **Jaw geometry** — where each jaw tip sits in the tool frame, and the aperture curve.
2. **Kinematic reach** — where IK converges for a top-down and for a radially tilted grasp.
3. **Static reach** — where the servos can actually *hold* the pose. At table height this costs
   little coverage, but it decides *which IK branch* is usable: some branches saturate the
   shoulder at its 2.94 N·m limit and sag instead of tracking.
4. **Collision-screened reach** — what survives once the scene's own furniture is considered.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from armanual.control.gripper import moving_jaw_z  # noqa: E402
from armanual.control.kinematics import (  # noqa: E402
    colliding_pairs,
    hold_torque,
    holdable,
    solve_reach,
)
from armanual.paths import REPO_ROOT  # noqa: E402
from armanual.sim.randomize import RandomizationConfig, sample_scene  # noqa: E402
from armanual.sim.world import World  # noqa: E402

GRASP_HEIGHT = 0.05


def measure_jaws(world: World, arm: str = "left") -> dict:
    """Jaw tip positions in the tool frame across the gripper's command range."""
    handle = world.arms[arm]
    rows = []
    for opening in np.linspace(0.0, 1.0, 11):
        q = np.array([0.0, -0.6, 0.8, 0.6, 0.3])
        world.data.qpos[handle.qpos_adr[:5]] = q
        world.data.qpos[handle.qpos_adr[5]] = -0.17453 + opening * (1.74533 + 0.17453)
        mujoco.mj_kinematics(world.model, world.data)
        rot = world.data.site_xmat[handle.tcp_site].reshape(3, 3)
        site = world.data.site_xpos[handle.tcp_site]

        def local(geom: str, rot=rot, site=site) -> np.ndarray:
            gid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_GEOM, f"{arm}/{geom}")
            return rot.T @ (world.data.geom_xpos[gid] - site)

        fixed = local("fixed_jaw_sph_tip1")
        moving = local("moving_jaw_sph_tip1")
        rows.append(
            {
                "opening": round(float(opening), 3),
                "fixed_mm": [round(float(v) * 1000, 1) for v in fixed],
                "moving_mm": [round(float(v) * 1000, 1) for v in moving],
                "aperture_mm": round(float(np.linalg.norm(moving - fixed)) * 1000, 1),
                "model_moving_z_mm": round(moving_jaw_z(float(opening)) * 1000, 1),
            }
        )
    world.reset()
    return {"rows": rows}


def measure_reach(world: World, arm: str = "left") -> dict:
    """Grid-scan the table for feasible grasps under three progressively stricter screens."""
    xs = np.arange(-0.40, 0.41, 0.05)
    ys = np.arange(-0.26, 0.19, 0.04)
    grids = {"kinematic": [], "static": [], "collision_free": []}
    counts = dict.fromkeys(grids, 0)
    for x in xs:
        rows = {key: "" for key in grids}
        for y in ys:
            target = np.array([x, y, GRASP_HEIGHT])
            raw = solve_reach(world, arm, target, check_torque=False, check_collision=False)
            kin = raw.ik.pos_error < 0.006
            stat = kin and holdable(world, arm, raw.qpos)
            screened = solve_reach(world, arm, target).feasible
            for key, ok in (("kinematic", kin), ("static", stat), ("collision_free", screened)):
                rows[key] += "X" if ok else "."
                counts[key] += int(ok)
        for key in grids:
            grids[key].append(f"x={x:+.2f} {rows[key]}")
    total = len(xs) * len(ys)
    return {
        "x_range": [float(xs[0]), float(xs[-1])],
        "y_range": [float(ys[0]), float(ys[-1])],
        "height": GRASP_HEIGHT,
        "grids": grids,
        "coverage": {k: round(v / total, 3) for k, v in counts.items()},
    }


def measure_torque(world: World, arm: str = "left") -> dict:
    """Holding torque along a line away from the base, at a fixed grasp height."""
    limit = float(world.model.actuator_forcerange[world.arms[arm].actuator_ids[0], 1])
    base = world.base_pos(arm)
    rows = []
    for distance in np.arange(0.12, 0.42, 0.04):
        target = np.array([base[0], base[1] + distance, GRASP_HEIGHT])
        solution = solve_reach(world, arm, target, check_torque=False, check_collision=False)
        if solution.ik.pos_error > 0.01:
            continue
        torque = hold_torque(world, arm, solution.qpos)
        rows.append(
            {
                "distance_m": round(float(distance), 3),
                "max_torque_Nm": round(float(np.max(torque)), 3),
                "holdable": bool(np.all(torque <= limit * 0.8)),
            }
        )
    return {"actuator_limit_Nm": limit, "rows": rows}


def measure_furniture_clearance(world: World, arm: str = "left") -> dict:
    """How much of the drawer's surroundings the arm can enter without hitting the cabinet."""
    handle = world.site_pos("drawer_handle")
    blocked, free = 0, 0
    for dy in (-0.02, 0.0, 0.02):
        for dz in (0.0, 0.03, 0.06):
            target = handle + np.array([0.0, dy, dz])
            solution = solve_reach(world, arm, target, check_collision=False, check_torque=False)
            if solution.ik.pos_error > 0.01:
                continue
            if colliding_pairs(world, arm, solution.qpos):
                blocked += 1
            else:
                free += 1
    return {"drawer_handle_poses_free": free, "drawer_handle_poses_blocked": blocked}


def render_markdown(data: dict) -> str:
    jaws = data["jaws"]["rows"]
    reach = data["reach"]
    torque = data["torque"]
    lines = [
        "# Measured SO-101 workspace",
        "",
        "Generated by `scripts/measure_workspace.py` — **do not hand-edit**. Every number here is",
        "read out of the compiled MuJoCo model of the actual dinner-table scene.",
        "",
        "## 1. Jaw geometry (tool frame, millimetres)",
        "",
        "The `gripperframe` site is *not* between the jaws. The fixed jaw sits ~20 mm along the",
        "frame's -z axis and never moves; the moving jaw sweeps along +z. An object is therefore",
        "pinched against the fixed jaw, and a grasp aimed at the site misses by about that much.",
        "",
        "| command | fixed tip (x,y,z) | moving tip (x,y,z) | aperture | model z |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in jaws:
        lines.append(
            f"| {row['opening']:.2f} | {tuple(row['fixed_mm'])} | {tuple(row['moving_mm'])} | "
            f"{row['aperture_mm']} mm | {row['model_moving_z_mm']} mm |"
        )
    lines += [
        "",
        "`model z` is the linear fit used in `armanual.control.gripper.moving_jaw_z`. It tracks",
        "the measurement to about a millimetre up to command ~0.45 and then drifts, because the",
        "moving jaw swings on an arc rather than a line. Grasps in this project need apertures",
        "well below that, so the fit is used only in its accurate range.",
        "",
        f"## 2. Reach at grasp height z = {reach['height']} m",
        "",
        "Three screens, strictest last. `X` = a grasp is possible at that table position.",
        "",
    ]
    for key, title in (
        ("kinematic", "Kinematic only (IK converges)"),
        ("static", "Kinematic + the servos can hold the pose"),
        ("collision_free", "…and the arm does not hit the scene"),
    ):
        lines += [
            f"### {title} — coverage {reach['coverage'][key] * 100:.0f}%",
            "",
            "```",
            f"y from {reach['y_range'][0]:+.2f} to {reach['y_range'][1]:+.2f} (step 0.04)",
        ]
        lines += reach["grids"][key]
        lines += ["```", ""]
    kin_cov = reach["coverage"]["kinematic"]
    stat_cov = reach["coverage"]["static"]
    col_cov = reach["coverage"]["collision_free"]
    lines += [
        f"Coverage drops from {kin_cov * 100:.0f}% to {stat_cov * 100:.0f}% once holding torque is",
        f"required, and to {col_cov * 100:.0f}% once the scene's furniture is taken into account.",
        "",
        "Note what the torque screen does and does not do. Position-by-position it removes little",
        "at this height — but for a *given* position the 5-DOF arm has several IK branches, and",
        "some of them saturate the shoulder at its limit and sag about 2 cm instead of tracking.",
        "The screen's real job is to reject those branches during the search, which is why it is",
        "wired into IK acceptance rather than applied as a post-filter.",
        "",
        "## 3. Holding torque vs. distance (straight ahead of the base)",
        "",
        f"Actuator limit: **{torque['actuator_limit_Nm']} N·m**; the screen allows 80% of it.",
        "",
        "| distance | peak joint torque | holdable |",
        "| --- | --- | --- |",
    ]
    for row in torque["rows"]:
        lines.append(
            f"| {row['distance_m']:.2f} m | {row['max_torque_Nm']:.2f} N·m | "
            f"{'yes' if row['holdable'] else 'no'} |"
        )
    lines += [
        "",
        "## 4. Drawer clearance",
        "",
        f"Of the sampled poses around the drawer handle, "
        f"{data['furniture']['drawer_handle_poses_free']} are collision-free and "
        f"{data['furniture']['drawer_handle_poses_blocked']} are blocked by the cabinet. "
        "This is why the cabinet is low-profile: with a taller one, *every* top-down handle grasp "
        "is blocked and the drawer simply cannot be opened.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="left")
    parser.add_argument("--json", type=Path, default=None, help="also write raw measurements here")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "WORKSPACE.md")
    args = parser.parse_args()

    world = World(sample_scene(0, RandomizationConfig.off()))
    data = {
        "arm": args.arm,
        "jaws": measure_jaws(world, args.arm),
        "reach": measure_reach(world, args.arm),
        "torque": measure_torque(world, args.arm),
        "furniture": measure_furniture_clearance(world, args.arm),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(data))
    print(f"wrote {args.out}")
    if args.json:
        args.json.write_text(json.dumps(data, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
