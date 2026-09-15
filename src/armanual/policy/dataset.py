"""Recording and loading demonstrations for policy learning.

Demonstrations come from the scripted expert in :mod:`armanual.control.primitives` running inside
the same closed loop the evaluation uses, so every frame is labelled with exactly what the
controller commanded at that instant. Each sample holds:

* ``observation.images.wrist`` — the acting arm's wrist camera, 96x96 RGB,
* ``observation.images.scene`` — the overhead camera, 96x96 RGB,
* ``observation.state`` — the arm's six joint angles plus its gripper opening,
* ``observation.goal`` — the grounded target, in the *arm's own base frame*: where to go and
  what to do there. This is the channel language arrives through: the grounder turns "the blue
  cup" into a position, and the policy is conditioned on that rather than on raw text,
* ``action`` — the commanded joint targets for the next control tick.

Column names follow the LeRobot convention so a dataset written here can be converted without
renaming anything; the storage format is plain compressed ``.npz`` shards plus a JSON card, which
keeps the repository dependency-free for anyone who just wants to inspect the data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

IMAGE_SIZE = (96, 96)
#: Joint targets + gripper command. The gripper is included so the policy learns *when* to close.
ACTION_DIM = 6
STATE_DIM = 7
GOAL_DIM = 8


@dataclass
class Frame:
    wrist: np.ndarray  # (H, W, 3) uint8
    scene: np.ndarray  # (H, W, 3) uint8
    state: np.ndarray  # (STATE_DIM,)
    goal: np.ndarray  # (GOAL_DIM,)
    action: np.ndarray  # (ACTION_DIM,)
    phase: str = ""


@dataclass
class EpisodeBuffer:
    """Frames for one demonstration, plus the metadata needed to reproduce it."""

    seed: int
    task: str
    arm: str
    frames: list[Frame] = field(default_factory=list)
    success: bool = False
    notes: str = ""

    def add(self, frame: Frame) -> None:
        self.frames.append(frame)

    def __len__(self) -> int:
        return len(self.frames)


def goal_vector(target_xyz, place_xyz, base_xy, phase: str) -> np.ndarray:
    """Encode 'what to do' relative to the arm's base.

    Base-relative coordinates matter: it makes the same policy usable by either arm, and it is
    the frame the arm's own kinematics live in, so the network does not have to learn the mount
    offset as a special case.
    """
    base = np.asarray(base_xy, dtype=float)
    target = np.asarray(target_xyz, dtype=float)
    place = np.asarray(place_xyz, dtype=float) if place_xyz is not None else target
    grasp_phase = 1.0 if phase in ("approach", "descend", "close") else 0.0
    return np.array(
        [
            target[0] - base[0], target[1] - base[1], target[2],
            place[0] - base[0], place[1] - base[1], place[2],
            grasp_phase,
            1.0 - grasp_phase,
        ],
        dtype=np.float32,
    )


def save_episodes(episodes: list[EpisodeBuffer], out_dir: Path, *, shard_size: int = 8) -> dict:
    """Write episodes as compressed shards plus a dataset card."""
    out_dir.mkdir(parents=True, exist_ok=True)
    kept = [e for e in episodes if len(e) > 0]
    total_frames = sum(len(e) for e in kept)
    index = []
    for shard_id, start in enumerate(range(0, len(kept), shard_size)):
        chunk = kept[start : start + shard_size]
        arrays: dict[str, np.ndarray] = {}
        for local, episode in enumerate(chunk):
            prefix = f"ep{local:02d}"
            arrays[f"{prefix}.wrist"] = np.stack([f.wrist for f in episode.frames])
            arrays[f"{prefix}.scene"] = np.stack([f.scene for f in episode.frames])
            arrays[f"{prefix}.state"] = np.stack([f.state for f in episode.frames]).astype(np.float32)
            arrays[f"{prefix}.goal"] = np.stack([f.goal for f in episode.frames]).astype(np.float32)
            arrays[f"{prefix}.action"] = np.stack([f.action for f in episode.frames]).astype(np.float32)
            index.append(
                {
                    "shard": shard_id,
                    "key": prefix,
                    "seed": episode.seed,
                    "task": episode.task,
                    "arm": episode.arm,
                    "frames": len(episode),
                    "success": episode.success,
                    "notes": episode.notes,
                }
            )
        np.savez_compressed(out_dir / f"shard_{shard_id:03d}.npz", **arrays)

    card = {
        "format": "armanual-demos-v1",
        "lerobot_columns": {
            "observation.images.wrist": f"uint8 {IMAGE_SIZE[1]}x{IMAGE_SIZE[0]}x3",
            "observation.images.scene": f"uint8 {IMAGE_SIZE[1]}x{IMAGE_SIZE[0]}x3",
            "observation.state": f"float32 [{STATE_DIM}] (6 joint angles + gripper opening)",
            "observation.goal": f"float32 [{GOAL_DIM}] (target xyz, place xyz, phase one-hot), base-relative",
            "action": f"float32 [{ACTION_DIM}] (commanded joint targets incl. gripper)",
        },
        "episodes": len(kept),
        "frames": total_frames,
        "control_hz": 20,
        "index": index,
    }
    (out_dir / "dataset_card.json").write_text(json.dumps(card, indent=2))
    return card


def load_dataset(data_dir: Path, *, successful_only: bool = True) -> dict[str, np.ndarray]:
    """Load all shards into flat arrays ready for training."""
    data_dir = Path(data_dir)
    card = json.loads((data_dir / "dataset_card.json").read_text())
    buckets: dict[str, list[np.ndarray]] = {k: [] for k in ("wrist", "scene", "state", "goal", "action")}
    episode_ids: list[int] = []

    for entry in card["index"]:
        if successful_only and not entry["success"]:
            continue
        shard = np.load(data_dir / f"shard_{entry['shard']:03d}.npz")
        for key in buckets:
            buckets[key].append(shard[f"{entry['key']}.{key}"])
        episode_ids.append(len(episode_ids))

    if not episode_ids:
        raise RuntimeError(f"no episodes loaded from {data_dir} (successful_only={successful_only})")
    out = {key: np.concatenate(values) for key, values in buckets.items()}
    out["episode_lengths"] = np.array([len(v) for v in buckets["state"]])
    return out


def chunk_indices(episode_lengths: np.ndarray, horizon: int) -> np.ndarray:
    """Start indices whose action chunk of length ``horizon`` stays inside one episode.

    Action chunking is what makes a small policy usable at 20 Hz: predicting a short burst of
    future actions smooths the output and lets inference run less often than control.
    """
    starts = []
    offset = 0
    for length in episode_lengths:
        for i in range(max(0, length - horizon + 1)):
            starts.append(offset + i)
        offset += length
    return np.array(starts, dtype=np.int64)
