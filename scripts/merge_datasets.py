"""Append one LeRobotDataset into another.

Demonstration collection is incremental: a skill that came out thin on the first pass gets a
second, targeted pass, and those episodes have to end up in the same dataset the policy trains on.
Collecting directly into the live dataset is possible but fragile — an interrupted write leaves a
truncated parquet shard and the whole dataset stops loading — so passes are collected into their
own directory and merged here, when nothing else is writing.

    python scripts/merge_datasets.py --source data/armanual-extra --target data/armanual-dinner-table
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from armanual.policy.collect import CAMERAS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--repo-id", default="pythonsniffer/armanual-dinner-table")
    parser.add_argument("--source-repo-id", default=None)
    parser.add_argument("--limit", type=int, default=None, help="merge at most this many episodes")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source = LeRobotDataset(args.source_repo_id or args.repo_id, root=args.source)
    target = LeRobotDataset(args.repo_id, root=args.target)
    print(f"source: {source.num_episodes} episodes / {source.num_frames} frames")
    print(f"target: {target.num_episodes} episodes / {target.num_frames} frames")

    episodes = range(source.num_episodes)
    if args.limit:
        episodes = range(min(args.limit, source.num_episodes))

    # LeRobotDataset v3.0 keeps each episode's frame range in the episode metadata table rather
    # than in an `episode_data_index` attribute, which earlier versions exposed on the dataset.
    episode_meta = source.meta.episodes

    merged_frames = 0
    for episode_index in episodes:
        row = episode_meta[int(episode_index)]
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])
        tasks = row.get("tasks") or []
        task = tasks[0] if isinstance(tasks, list) and tasks else (tasks or None)
        for frame_index in range(start, end):
            item = source[frame_index]
            task = item.get("task", task)
            payload = {}
            for key, _camera in CAMERAS:
                image = item[key]
                # LeRobot returns CHW float tensors in [0, 1]; the writer wants HWC uint8.
                array = (image.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                payload[key] = array
            payload["observation.state"] = item["observation.state"].numpy().astype(np.float32)
            payload["action"] = item["action"].numpy().astype(np.float32)
            payload["task"] = task
            target.add_frame(payload)
        target.save_episode()
        merged_frames += end - start
        print(f"  merged episode {episode_index + 1}/{len(episodes)} ({end - start} frames) "
              f"task={task!r}", flush=True)

    print(f"\nmerged {len(episodes)} episodes / {merged_frames} frames into {args.target}")


if __name__ == "__main__":
    main()
