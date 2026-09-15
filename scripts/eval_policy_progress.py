"""Measure how well a policy checkpoint executes individual language subgoals.

This is the metric that decides when to stop training — not the loss. A falling loss only says
the policy reproduces the demonstrations' actions; what matters is whether the arms end up putting
the object where the sentence said.

    python scripts/eval_policy_progress.py --checkpoint outputs/train/armanual_smolvla/checkpoints/004000/pretrained_model
    python scripts/eval_policy_progress.py --checkpoint <ckpt> --seeds 5 --device cuda

Runs the policy **alone** (no scripted fallback) so the number is the policy's own.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from armanual.perception.observer import CameraObserver  # noqa: E402
from armanual.planning.subgoals import (  # noqa: E402
    SubgoalRunner,
    subgoal_baseline,
    verify_subgoal,
)
from armanual.policy.runtime import LeRobotBackend, PolicyRunner  # noqa: E402
from armanual.sim.randomize import RandomizationConfig, sample_scene  # noqa: E402
from armanual.sim.world import World  # noqa: E402

#: The subgoals the policy was trained on, one per skill family it has data for.
SUBGOALS = (
    "put the plate in the middle of the table",
    "put the blue cup to the right of the plate",
    "open the drawer",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seconds", type=float, default=15.0,
                        help="sim seconds allowed per subgoal")
    parser.add_argument("--subgoals", nargs="*", default=list(SUBGOALS))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    backend = LeRobotBackend(args.checkpoint, device=args.device)
    results = []
    started = time.perf_counter()

    for subgoal in args.subgoals:
        for seed in range(args.seeds):
            world = World(sample_scene(seed, RandomizationConfig.placement_only()),
                          fast_render=True)
            observer = CameraObserver(world)
            runner = SubgoalRunner(world, observer, backend=backend, fallback=False)
            baseline = subgoal_baseline(world, observer, subgoal)

            policy_runner = PolicyRunner(world, backend, dt=0.05)
            record = policy_runner.run(subgoal, max_seconds=args.seconds)
            runner.park_arms()
            ok, detail = verify_subgoal(world, observer, subgoal, before=baseline)

            results.append({
                "subgoal": subgoal,
                "seed": seed,
                "success": ok,
                "detail": detail,
                "inference": record.get("inference", {}),
            })
            print(f"{'PASS' if ok else 'fail'}  seed {seed}  {subgoal!r}\n      {detail}",
                  flush=True)
            runner.close()
            observer.close()
            world.close()

    by_subgoal: dict[str, list[bool]] = {}
    for row in results:
        by_subgoal.setdefault(row["subgoal"], []).append(row["success"])
    summary = {
        "checkpoint": str(args.checkpoint),
        "episodes": len(results),
        "policy_only_success": round(sum(r["success"] for r in results) / max(len(results), 1), 3),
        "by_subgoal": {k: round(sum(v) / len(v), 3) for k, v in by_subgoal.items()},
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    print("\n" + json.dumps(summary, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summary, "episodes": results}, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
