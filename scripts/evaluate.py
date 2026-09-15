"""Run the benchmark suite and write machine-readable results.

    python scripts/evaluate.py --tier 1 --seeds 3
    python scripts/evaluate.py --seeds 10 --out outputs/eval_full      # the 10-seed submission run
    python scripts/evaluate.py --task t4_pour_with_hold --seeds 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from armanual.eval.harness import all_tasks, run_suite  # noqa: E402
from armanual.eval.tiers import task_by_id, tasks_for_tier  # noqa: E402


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
    args = parser.parse_args()

    if args.task:
        tasks = [task_by_id(args.task)]
    elif args.tier:
        tasks = list(tasks_for_tier(args.tier))
    else:
        tasks = all_tasks()
    seeds = args.seed_list if args.seed_list is not None else list(range(args.seeds))

    payload = run_suite(tasks, seeds, privileged=args.privileged, out_dir=args.out)
    print("\n" + json.dumps(payload["summary"]["overall"], indent=2))
    print("by tier:", json.dumps(payload["summary"]["by_tier"], indent=2))
    if payload["summary"].get("failure_kinds"):
        print("failure kinds:", json.dumps(payload["summary"]["failure_kinds"]))


if __name__ == "__main__":
    main()
