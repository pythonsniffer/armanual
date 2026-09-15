"""Collect demonstrations across several processes.

Under WSL there is no GPU OpenGL, so MuJoCo renders on the CPU: about 100 ms per 224x224 frame
with shadows disabled, and an episode needs three cameras at every control tick. One process
therefore produces roughly one episode per 20 seconds, which is far too slow for a dataset of
several hundred.

Rendering is single-threaded and CPU-bound, which is the one case where processes scale almost
linearly. Workers each simulate their own episode and send it back as **JPEG-encoded frames** —
an episode is ~90 MB raw but ~2 MB encoded, and the parent has to encode video anyway, so the
compression costs nothing and keeps the pipe from becoming the new bottleneck.
"""

from __future__ import annotations

import io
import multiprocessing as mp
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class EncodedEpisode:
    """An episode in transit between processes: frames as JPEG bytes."""

    task: str
    seed: int
    skill: str
    success: bool
    notes: str
    frames: list[dict]  # {"images": {key: bytes}, "state": list, "action": list}

    def __len__(self) -> int:
        return len(self.frames)

    def decode(self):
        """Rebuild the in-memory episode with numpy images."""
        from PIL import Image

        from armanual.policy.collect import DemoEpisode, DemoFrame

        episode = DemoEpisode(task=self.task, seed=self.seed, skill=self.skill,
                              success=self.success, notes=self.notes)
        for frame in self.frames:
            images = {
                key: np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
                for key, blob in frame["images"].items()
            }
            episode.frames.append(
                DemoFrame(
                    images=images,
                    state=np.asarray(frame["state"], dtype=np.float32),
                    action=np.asarray(frame["action"], dtype=np.float32),
                )
            )
        return episode


def _encode(episode, quality: int = 92) -> EncodedEpisode:
    from PIL import Image

    frames = []
    for frame in episode.frames:
        images = {}
        for key, array in frame.images.items():
            buffer = io.BytesIO()
            Image.fromarray(array).save(buffer, format="JPEG", quality=quality)
            images[key] = buffer.getvalue()
        frames.append(
            {"images": images, "state": frame.state.tolist(), "action": frame.action.tolist()}
        )
    return EncodedEpisode(
        task=episode.task, seed=episode.seed, skill=episode.skill,
        success=episode.success, notes=episode.notes, frames=frames,
    )


def _worker(job: dict) -> EncodedEpisode | None:
    """Run one episode in this process and return it encoded."""
    import warnings

    warnings.filterwarnings("ignore")
    # Keep the software rasterizer from spawning its own thread pool in every worker; with a
    # dozen workers that oversubscribes the machine and makes everything slower.
    os.environ.setdefault("LP_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MUJOCO_GL", "glfw")

    from armanual.policy.collect import collect_episode, collect_skill_episode
    from armanual.sim.randomize import RandomizationConfig

    config = RandomizationConfig(**job["randomization"]) if job.get("randomization") else None
    try:
        if job.get("direct", True):
            # Skill-level recording: the primitives are driven straight from the grounded
            # instruction. Measured 5/8 -> 7/8 success against routing demonstrations through the
            # planner, and every episode the planner loses is a training example lost.
            episode = collect_skill_episode(
                job["instruction"], job["seed"], skill=job["skill"], randomization=config,
                pre_open_drawer=job.get("pre_open_drawer", False), fast_render=True,
            )
        else:
            episode = collect_episode(
                job["instruction"], job["seed"], skill=job["skill"], randomization=config,
                pre_open_drawer=job.get("pre_open_drawer", False),
                max_steps=job.get("max_steps", 2), fast_render=True,
            )
    except Exception as exc:  # noqa: BLE001 - one bad episode must not kill the collection
        return EncodedEpisode(task=job["instruction"], seed=job["seed"], skill=job["skill"],
                              success=False, notes=f"worker error: {exc}", frames=[])
    if not len(episode):
        return None
    return _encode(episode)


def collect_parallel(jobs: list[dict], workers: int = 8, chunksize: int = 1):
    """Yield encoded episodes as workers finish them.

    Uses ``spawn`` rather than ``fork``: MuJoCo's GL context does not survive a fork, and a forked
    child inherits a broken one that fails on first render.
    """
    context = mp.get_context("spawn")
    with context.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(_worker, jobs, chunksize=chunksize):
            if result is not None:
                yield result
