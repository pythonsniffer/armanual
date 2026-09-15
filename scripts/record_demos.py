"""Record scripted-expert demonstrations for policy training.

    python scripts/record_demos.py --episodes 40 --out data/demos
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from armanual.policy.dataset import save_episodes  # noqa: E402
from armanual.policy.record import record_pick_place  # noqa: E402
from armanual.sim.randomize import RandomizationConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--category", default="cup", choices=["cup", "plate", "bottle"])
    parser.add_argument("--out", type=Path, default=Path("data/demos"))
    parser.add_argument("--randomize", action="store_true",
                        help="vary size/mass/friction as well as placement")
    args = parser.parse_args()

    config = RandomizationConfig(
        colors=False, lighting=False, background=False, distractors=True, max_distractors=1
    ) if args.randomize else RandomizationConfig.placement_only()

    episodes, kept, started = [], 0, time.perf_counter()
    for index in range(args.episodes):
        seed = args.start_seed + index
        buffer = record_pick_place(seed, category=args.category, randomization=config)
        if buffer is None:
            print(f"seed {seed}: no {args.category} detected, skipped")
            continue
        episodes.append(buffer)
        kept += int(buffer.success)
        print(f"seed {seed}: {len(buffer):4d} frames  "
              f"{'ok' if buffer.success else 'FAILED: ' + buffer.notes[:48]}")

    card = save_episodes(episodes, args.out)
    elapsed = time.perf_counter() - started
    print(
        f"\n{card['episodes']} episodes ({kept} successful), {card['frames']} frames "
        f"in {elapsed:.0f}s -> {args.out}"
    )


if __name__ == "__main__":
    main()
