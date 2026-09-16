"""How repeatable is a policy episode?

The analytical controller is deterministic: the same ``(task, seed)`` replays exactly. SmolVLA is
not. It is a flow-matching policy that samples noise to produce each action chunk, so the same
scene can succeed on one run and fail on the next. The benchmark in ``scripts/evaluate.py`` runs
each ``(task, seed)`` pair once, which is the right unit for a deterministic controller and an
under-specified one for a stochastic policy.

This script runs a whole tier several times over the same seeds and reports the spread, so a
headline policy number can be quoted with an idea of how much of it is capability and how much is
the draw:

    python scripts/policy_variance.py --policy <ckpt> --tier 1 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from armanual.eval.harness import _object_positions, check_criterion  # noqa: E402
from armanual.eval.tiers import tasks_for_tier  # noqa: E402
from armanual.perception.observer import CameraObserver  # noqa: E402
from armanual.planning.subgoals import SubgoalRunner  # noqa: E402
from armanual.sim.randomize import sample_scene  # noqa: E402
from armanual.sim.world import World  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--tier", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    from armanual.policy.runtime import LeRobotBackend

    backend = LeRobotBackend(args.policy, device=args.device)
    tasks = list(tasks_for_tier(args.tier))
    rates, per_episode = [], {}
    for rep in range(args.repeats):
        ok = total = 0
        for task in tasks:
            for seed in range(args.seeds):
                world = World(sample_scene(seed, task.randomization))
                observer = CameraObserver(world)
                start = _object_positions(world)
                runner = SubgoalRunner(world, observer, backend=backend, fallback=False)
                runner.run(task.instruction, style=task.style, seed=seed)
                runner.close()
                good = all(check_criterion(world, c, task, start).passed for c in task.criteria)
                per_episode.setdefault(f"{task.id}:{seed}", []).append(bool(good))
                ok += int(good)
                total += 1
                observer.close()
                world.close()
        rates.append(ok / total)
        print(f"repeat {rep + 1}: {ok}/{total} = {ok / total:.3f}", flush=True)

    # An episode that flips between runs is one the policy neither reliably does nor reliably
    # fails; counting them separates "cannot" from "sometimes".
    flaky = [k for k, v in per_episode.items() if any(v) and not all(v)]
    always = [k for k, v in per_episode.items() if all(v)]
    payload = {
        "tier": args.tier, "seeds": args.seeds, "repeats": args.repeats,
        "rates": rates,
        "mean": statistics.fmean(rates),
        "min": min(rates), "max": max(rates),
        "stdev": statistics.stdev(rates) if len(rates) > 1 else 0.0,
        "always_pass": sorted(always),
        "flaky": sorted(flaky),
        "per_episode": per_episode,
    }
    print(f"\n=== tier {args.tier}, {args.seeds} seeds, {args.repeats} runs ===")
    print("  rates:", [f"{r:.3f}" for r in rates])
    print(f"  mean {payload['mean']:.3f}  min {payload['min']:.3f}  max {payload['max']:.3f}")
    print(f"  always pass: {len(always)}   flaky (pass sometimes): {len(flaky)}")
    if flaky:
        print("  flaky episodes:", ", ".join(sorted(flaky)))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
