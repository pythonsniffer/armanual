"""Write recorded demonstrations as a LeRobotDataset.

Using LeRobot's own dataset format rather than a bespoke one buys three things that matter for
this submission: `lerobot-train` can consume it directly (so the SmolVLA fine-tune is the stock
recipe, not a custom loop nobody can reproduce), the dataset visualiser works, and pushing to the
Hub gives judges the exact data the policy was trained on.

Schema, for a bimanual SO-101 pair:

===================================== ========================================================
Column                                Contents
===================================== ========================================================
``observation.images.top``            overhead camera, 224x224 RGB, video-encoded
``observation.images.left_wrist``     left arm wrist camera
``observation.images.right_wrist``    right arm wrist camera
``observation.state``                 12 floats: 5 joints + gripper, per arm (left first)
``action``                            12 floats: commanded joint targets, same layout
``task``                              the instruction sentence that produced the episode
===================================== ========================================================
"""

from __future__ import annotations

import inspect
import shutil
from pathlib import Path

import numpy as np

from armanual.policy.collect import ACTION_DIM, CAMERAS, IMAGE_SIZE, STATE_DIM, DemoEpisode

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
STATE_NAMES = tuple(f"{arm}_{joint}" for arm in ("left", "right") for joint in (*JOINT_NAMES, "gripper"))
FPS = 20


def build_features(image_size: tuple[int, int] = IMAGE_SIZE, use_videos: bool = True) -> dict:
    """The LeRobot feature dictionary for this robot."""
    width, height = image_size
    image_dtype = "video" if use_videos else "image"
    features = {
        key: {
            "dtype": image_dtype,
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
        for key, _camera in CAMERAS
    }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (STATE_DIM,),
        "names": list(STATE_NAMES),
    }
    features["action"] = {
        "dtype": "float32",
        "shape": (ACTION_DIM,),
        "names": list(STATE_NAMES),
    }
    return features


def open_dataset(repo_id: str, root: Path):
    """Reopen an existing dataset so more episodes can be appended to it.

    Collection is naturally incremental — a skill that produced no usable demonstrations on the
    first pass gets another, easier pass — and rebuilding the whole dataset each time would throw
    away an hour of simulation for no reason.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id, root=Path(root))


def create_dataset(repo_id: str, root: Path, *, image_size: tuple[int, int] = IMAGE_SIZE,
                   use_videos: bool = True, overwrite: bool = False, append: bool = False):
    """Create (or recreate, or reopen) a LeRobotDataset ready to receive frames."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    if append and (root / "meta" / "info.json").exists():
        return open_dataset(repo_id, root)
    if overwrite and root.exists():
        shutil.rmtree(root)
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        root=root,
        robot_type="so101_bimanual",
        features=build_features(image_size, use_videos),
        use_videos=use_videos,
    )


def append_episode(dataset, episode: DemoEpisode) -> int:
    """Append one demonstration. Returns the number of frames written."""
    for frame in episode.frames:
        payload: dict[str, np.ndarray] = {
            key: frame.images[key].astype(np.uint8) for key, _camera in CAMERAS
        }
        payload["observation.state"] = frame.state.astype(np.float32)
        payload["action"] = frame.action.astype(np.float32)
        _add_frame(dataset, payload, episode.task)
    _save_episode(dataset, episode.task)
    return len(episode.frames)


def _add_frame(dataset, payload: dict, task: str) -> None:
    """Call ``add_frame`` across LeRobot versions (the ``task`` argument moved between them)."""
    signature = inspect.signature(dataset.add_frame)
    if "task" in signature.parameters:
        dataset.add_frame(payload, task=task)
    else:
        payload = {**payload, "task": task}
        dataset.add_frame(payload)


def _save_episode(dataset, task: str) -> None:
    signature = inspect.signature(dataset.save_episode)
    if "task" in signature.parameters:
        dataset.save_episode(task=task)
    else:
        dataset.save_episode()


def write_dataset(episodes: list[DemoEpisode], repo_id: str, root: Path, *,
                  image_size: tuple[int, int] = IMAGE_SIZE, use_videos: bool = True,
                  overwrite: bool = False, successful_only: bool = True) -> dict:
    """Write a whole set of demonstrations and return a small summary."""
    kept = [e for e in episodes if len(e) > 0 and (e.success or not successful_only)]
    dataset = create_dataset(repo_id, root, image_size=image_size, use_videos=use_videos,
                             overwrite=overwrite)
    frames = 0
    per_skill: dict[str, int] = {}
    for episode in kept:
        frames += append_episode(dataset, episode)
        per_skill[episode.skill] = per_skill.get(episode.skill, 0) + 1
    return {
        "repo_id": repo_id,
        "root": str(root),
        "episodes": len(kept),
        "frames": frames,
        "per_skill": per_skill,
        "dropped": len(episodes) - len(kept),
    }
