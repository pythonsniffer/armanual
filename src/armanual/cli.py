"""Command-line entry points, including the environment check judges should run first.

    python -m armanual.cli verify      # does this machine have everything, and does it work?
    python -m armanual.cli devices     # what can OpenVINO see here?
    python -m armanual.cli scene       # render the dinner-table scene to a PNG

``verify`` is deliberately more than an import check: it builds the scene, renders a frame, runs
the detector, solves an IK target and reports what is missing. A setup that imports cleanly but
cannot render is the most common way a robotics repository fails on someone else's machine.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _status(label: str, ok: bool, detail: str = "") -> bool:
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {label}{(' — ' + detail) if detail else ''}")
    return ok


def verify(args) -> int:
    """Check the environment end to end and report what works."""
    import numpy as np

    print("armanual environment check\n")
    results = []

    print("dependencies")
    try:
        import mujoco

        results.append(_status("mujoco", True, mujoco.__version__))
    except Exception as exc:  # noqa: BLE001
        results.append(_status("mujoco", False, str(exc)))
        print("\ncannot continue without mujoco")
        return 1
    try:
        import scipy

        results.append(_status("scipy", True, scipy.__version__))
    except Exception as exc:  # noqa: BLE001
        results.append(_status("scipy", False, str(exc)))
    for optional, note in (("torch", "policy training/inference"),
                           ("lerobot", "dataset + SmolVLA"),
                           ("openvino", "Intel deployment")):
        try:
            module = __import__(optional)
            version = getattr(module, "__version__", "installed")
            _status(optional, True, f"{version} ({note})")
        except Exception:  # noqa: BLE001
            _status(optional, True, f"not installed — optional, needed for {note}")

    print("\nassets")
    from armanual.paths import SO101_XML

    results.append(_status("SO-101 MJCF", SO101_XML.exists(), str(SO101_XML)))

    print("\nsimulation")
    started = time.perf_counter()
    from armanual.sim.randomize import RandomizationConfig, sample_scene
    from armanual.sim.world import World

    world = World(sample_scene(0, RandomizationConfig.placement_only()), fast_render=True)
    results.append(
        _status("scene builds", world.model.nu == 12,
                f"{world.model.nu} actuators, {world.model.ngeom} geoms, "
                f"{time.perf_counter() - started:.1f}s")
    )

    started = time.perf_counter()
    try:
        frame = world.render("cam_overhead", (128, 128))
        results.append(
            _status("rendering", frame.shape == (128, 128, 3),
                    f"{(time.perf_counter() - started) * 1000:.0f} ms/frame")
        )
    except Exception as exc:  # noqa: BLE001
        results.append(_status("rendering", False, f"{exc} — try MUJOCO_GL=egl or osmesa"))

    print("\nperception")
    try:
        from armanual.perception.observer import CameraObserver

        observer = CameraObserver(world)
        observation = observer.observe()
        tableware = [d for d in observation.detections if d.category not in ("furniture", "unknown")]
        results.append(_status("detector", len(tableware) >= 3, f"{len(tableware)} objects detected"))
        observer.close()
    except Exception as exc:  # noqa: BLE001
        results.append(_status("detector", False, str(exc)))

    print("\ncontrol")
    try:
        from armanual.control.kinematics import solve_reach

        solution = solve_reach(world, "left", np.array([-0.15, -0.14, 0.06]))
        results.append(
            _status("inverse kinematics", solution.feasible,
                    f"residual {solution.ik.pos_error * 1000:.1f} mm")
        )
    except Exception as exc:  # noqa: BLE001
        results.append(_status("inverse kinematics", False, str(exc)))

    print("\nlanguage")
    try:
        from armanual.task.parser import parse_instruction

        request = parse_instruction("put the blue cup to the right of the plate")
        results.append(_status("instruction parser", len(request.actions) == 1,
                               request.actions[0].describe() if request.actions else "no actions"))
    except Exception as exc:  # noqa: BLE001
        results.append(_status("instruction parser", False, str(exc)))

    world.close()
    passed = sum(bool(r) for r in results)
    print(f"\n{passed}/{len(results)} checks passed")
    if passed == len(results):
        print("\nready. Try:  python scripts/run_task.py --instruction 'set the table'")
        return 0
    return 1


def devices(args) -> int:
    """Print what OpenVINO can see, and whether this is the Intel deployment target."""
    import json

    from armanual.policy.benchmark import host_info

    info = host_info()
    print(json.dumps(info, indent=2))
    if not info.get("is_intel_core_ultra"):
        print("\nThis is not an Intel Core Ultra system: numbers measured here are a development "
              "baseline, not the submission's headline figures.")
    return 0


def scene(args) -> int:
    """Render the scene to a PNG so the environment can be eyeballed."""
    import numpy as np
    from PIL import Image

    from armanual.sim.randomize import RandomizationConfig, sample_scene
    from armanual.sim.world import World

    config = RandomizationConfig() if args.randomize else RandomizationConfig.placement_only()
    world = World(sample_scene(args.seed, config))
    views = [world.render(camera, (480, 360)) for camera in ("cam_overhead", "cam_front")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(views, axis=1)).save(out)
    print(f"wrote {out}")
    world.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="armanual", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("verify", help="check the environment end to end").set_defaults(func=verify)
    subparsers.add_parser("devices", help="list OpenVINO devices").set_defaults(func=devices)

    scene_parser = subparsers.add_parser("scene", help="render the scene to a PNG")
    scene_parser.add_argument("--seed", type=int, default=0)
    scene_parser.add_argument("--randomize", action="store_true")
    scene_parser.add_argument("--out", default="outputs/scene.png")
    scene_parser.set_defaults(func=scene)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
