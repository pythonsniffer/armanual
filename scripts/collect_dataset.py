"""Collect a bimanual demonstration dataset in LeRobot format.

Episodes are recorded by running the scripted stack on generated instructions, and are streamed
into the dataset one at a time — a single episode is ~90 MB of frames, so batching them in memory
would exhaust RAM long before the dataset is large enough to train on.

    python scripts/collect_dataset.py --episodes-per-skill 60 --repo-id pythonsniffer/armanual-dinner-table
    python scripts/collect_dataset.py --episodes-per-skill 3 --dry-run     # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from armanual.policy.collect import collect_episode  # noqa: E402
from armanual.policy.tasks import SKILLS  # noqa: E402
from armanual.sim.randomize import RandomizationConfig  # noqa: E402

#: Object descriptions used in generated instructions, paired with where they may be sent.
PLACE_TARGETS: tuple[tuple[str, str, bool], ...] = (
    # (object description, placement phrase, needs the drawer open first)
    ("plate", "in the middle of the table", False),
    ("white plate", "in the middle of the table", False),
    ("blue cup", "to the right of the plate", False),
    ("cup", "to the right of the plate", False),
    ("navy cup", "next to the plate", False),
    ("fork", "to the left of the plate", True),
    ("spoon", "to the right of the plate", True),
    ("knife", "to the right of the plate", True),
)


def instruction_plan(episodes_per_skill: int, rng: random.Random) -> list[tuple[str, str, bool]]:
    """Build the (instruction, skill, needs_open_drawer) list to record."""
    plan: list[tuple[str, str, bool]] = []
    for _ in range(episodes_per_skill):
        plan.append((rng.choice(SKILLS["open_drawer"].templates), "open_drawer", False))
    for index in range(episodes_per_skill * 2):  # the workhorse skill gets twice the data
        description, where, needs_drawer = PLACE_TARGETS[index % len(PLACE_TARGETS)]
        template = rng.choice(SKILLS["place_object"].templates)
        plan.append((template.format(description=description, where=where), "place_object",
                     needs_drawer))
    for _ in range(episodes_per_skill):
        template = rng.choice(SKILLS["pour"].templates)
        plan.append((template.format(description=rng.choice(("blue cup", "cup"))), "pour", False))
    for _ in range(episodes_per_skill):
        template = rng.choice(SKILLS["handoff"].templates)
        plan.append((template.format(description=rng.choice(("cup", "plate"))), "handoff", False))
    rng.shuffle(plan)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes-per-skill", type=int, default=60)
    parser.add_argument("--start-seed", type=int, default=2000)
    parser.add_argument("--repo-id", default="pythonsniffer/armanual-dinner-table")
    parser.add_argument("--root", type=Path, default=Path("data/armanual-dinner-table"))
    parser.add_argument("--randomize", action="store_true", default=True,
                        help="vary sizes, mass, friction and distractors (default on)")
    parser.add_argument("--no-videos", action="store_true", help="store raw images instead of video")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-failures", action="store_true",
                        help="record unsuccessful episodes too (off by default)")
    parser.add_argument("--dry-run", action="store_true",
                        help="record but do not write a dataset; prints per-episode outcomes")
    args = parser.parse_args()

    rng = random.Random(args.start_seed)
    plan = instruction_plan(args.episodes_per_skill, rng)
    config = (
        RandomizationConfig(colors=True, lighting=True, background=True, distractors=True,
                            max_distractors=2)
        if args.randomize
        else RandomizationConfig.placement_only()
    )

    dataset = None
    if not args.dry_run:
        from armanual.policy.lerobot_export import append_episode, create_dataset

        dataset = create_dataset(args.repo_id, args.root, use_videos=not args.no_videos,
                                 overwrite=args.overwrite)

    stats = {"recorded": 0, "kept": 0, "frames": 0, "per_skill": {}, "failures": {}}
    started = time.perf_counter()
    for index, (instruction, skill, needs_drawer) in enumerate(plan):
        seed = args.start_seed + index
        episode = collect_episode(
            instruction, seed, skill=skill, randomization=config,
            pre_open_drawer=needs_drawer,
            max_steps=2 if skill != "open_drawer" else 1,
        )
        stats["recorded"] += 1
        keep = episode.success or args.keep_failures
        flag = "ok " if episode.success else "FAIL"
        print(
            f"[{index + 1:4d}/{len(plan)}] seed {seed} {flag} {len(episode):4d} frames  "
            f"{skill:12s} {instruction!r} {('' if episode.success else episode.notes[:60])}",
            flush=True,
        )
        if not episode.success:
            stats["failures"][skill] = stats["failures"].get(skill, 0) + 1
        if keep and len(episode) and dataset is not None:
            from armanual.policy.lerobot_export import append_episode

            stats["frames"] += append_episode(dataset, episode)
            stats["kept"] += 1
            stats["per_skill"][skill] = stats["per_skill"].get(skill, 0) + 1
        elif keep and len(episode):
            stats["kept"] += 1
            stats["frames"] += len(episode)
            stats["per_skill"][skill] = stats["per_skill"].get(skill, 0) + 1

    stats["wall_seconds"] = round(time.perf_counter() - started, 1)
    print("\n" + json.dumps(stats, indent=2))
    if dataset is not None:
        print(f"\ndataset written to {args.root}")
        print(f"push with: hf upload {args.repo_id} {args.root} --repo-type dataset")


if __name__ == "__main__":
    main()
