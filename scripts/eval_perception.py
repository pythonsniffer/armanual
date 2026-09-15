"""Measure detector accuracy against ground truth over randomized seeds.

    python scripts/eval_perception.py --seeds 10 --json outputs/perception.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from armanual.perception.evaluate import aggregate, score_observation  # noqa: E402
from armanual.perception.observer import CameraObserver  # noqa: E402
from armanual.sim.randomize import RandomizationConfig, sample_scene  # noqa: E402
from armanual.sim.world import World  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--full-random", action="store_true",
                        help="randomize lighting, colour, size and distractors too")
    args = parser.parse_args()

    config = RandomizationConfig() if args.full_random else RandomizationConfig.placement_only()
    scores = []
    for seed in range(args.seeds):
        world = World(sample_scene(seed, config))
        observer = CameraObserver(world)
        score = score_observation(world, observer.observe(), seed=seed)
        scores.append(score)
        print(
            f"seed {seed:2d}  recall {score.recall:.2f}  precision {score.precision:.2f}  "
            f"pos err {score.mean_position_error_mm:5.1f}mm  "
            f"cat {score.category_accuracy:.2f}  col {score.color_accuracy:.2f}  "
            f"missed={score.missed} fp={len(score.false_positives)}"
        )
        observer.close()
        world.close()

    summary = aggregate(scores)
    print("\nsummary:", json.dumps(summary, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "randomization": "full" if args.full_random else "placement_only",
                    "summary": summary,
                    "per_seed": [s.to_dict() for s in scores],
                },
                indent=2,
            )
        )
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
