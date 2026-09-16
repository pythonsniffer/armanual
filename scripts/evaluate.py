"""Run the benchmark suite and write machine-readable results.

    python scripts/evaluate.py --tier 1 --seeds 3
    python scripts/evaluate.py --seeds 10 --out outputs/eval_full      # the 10-seed submission run
    python scripts/evaluate.py --task t4_pour_with_hold --seeds 5
    python scripts/evaluate.py --seeds 10 --workers 6                  # shard episodes across processes

An episode costs minutes of wall-clock because MuJoCo renders on the CPU, so the full 11-task,
10-seed suite is a day's work serially. ``--workers`` shards it; episodes are a pure function of
(task, seed) so the results file is unchanged. With ``--policy``, each worker loads its own copy
of the network, so keep the count to what the GPU's memory allows (3 is safe for SmolVLA on 8 GB).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from armanual.eval.harness import BackendSpec, all_tasks, run_suite, run_suite_parallel
from armanual.eval.tiers import task_by_id, tasks_for_tier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", type=int, default=None, help="run only this tier")
    parser.add_argument("--task", type=str, default=None, help="run only this task id")
    parser.add_argument("--seeds", type=int, default=3, help="number of seeds (0..N-1)")
    parser.add_argument("--seed-list", type=int, nargs="*", default=None)
    parser.add_argument("--privileged", action="store_true",
                        help="use ground-truth perception (ablation only — never for headline numbers)")
    parser.add_argument("--out", type=Path, default=None, help="directory for results.json/csv")
    parser.add_argument("--policy", type=Path, default=None,
                        help="policy checkpoint; without it the analytical controller runs")
    parser.add_argument("--device", default="cuda", help="torch device for the policy")
    parser.add_argument("--ov-device", default=None,
                        help="run the policy through OpenVINO on this device (CPU/GPU/NPU)")
    parser.add_argument("--ir-dir", type=Path, default=None, help="exported IR directory")
    parser.add_argument("--workers", type=int, default=1,
                        help="run this many episodes in parallel (each worker loads its own policy)")
    parser.add_argument("--fallback", action="store_true",
                        help="allow the scripted controller to rescue subgoals the policy fails "
                             "(off by default: the deployed system is pure VLA)")
    args = parser.parse_args()

    if args.task:
        tasks = [task_by_id(args.task)]
    elif args.tier:
        tasks = list(tasks_for_tier(args.tier))
    else:
        tasks = all_tasks()
    seeds = args.seed_list if args.seed_list is not None else list(range(args.seeds))

    if args.policy and args.ov_device and args.ir_dir is None:
        parser.error("--ov-device requires --ir-dir (export with scripts/benchmark_intel.py)")

    spec = None
    if args.policy:
        spec = BackendSpec(checkpoint=args.policy, device=args.device,
                           ov_device=args.ov_device, ir_dir=args.ir_dir)

    if args.workers > 1:
        print(f"{len(tasks) * len(seeds)} episodes across {args.workers} workers")
        payload = run_suite_parallel(tasks, seeds, workers=args.workers,
                                     privileged=args.privileged, out_dir=args.out,
                                     backend_spec=spec, fallback=args.fallback)
    else:
        backend = spec.build() if spec is not None else None
        if backend is not None:
            print(f"policy on {backend.name}")
        payload = run_suite(tasks, seeds, privileged=args.privileged, out_dir=args.out,
                            backend=backend, fallback=args.fallback)
    print("\n" + json.dumps(payload["summary"]["overall"], indent=2))
    print("by tier:", json.dumps(payload["summary"]["by_tier"], indent=2))
    if payload["summary"].get("failure_kinds"):
        print("failure kinds:", json.dumps(payload["summary"]["failure_kinds"]))


if __name__ == "__main__":
    main()
