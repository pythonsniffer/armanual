"""Record episodes to video with an on-screen overlay of what the system is doing.

The submission video has to let a judge verify claims, not just watch a robot move. So every frame
carries the things that would otherwise have to be taken on trust: the instruction, the current
subgoal, which controller is driving (policy or scripted), the inference device, and the measured
latency. If the overlay says ``NPU 8.3 ms`` while the arms move, that is evidence; a voice-over
saying "this runs on the NPU" is not.

Frames are composed from several cameras side by side, so the same recording shows the scene view
and what the robot itself sees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class OverlayState:
    """What the HUD should currently say. Mutated by the runner as the episode progresses."""

    instruction: str = ""
    subgoal: str = ""
    controller: str = ""
    device: str = ""
    latency_ms: float = 0.0
    seed: int | None = None
    extra: dict[str, str] = field(default_factory=dict)

    def lines(self) -> list[str]:
        out = [f'"{self.instruction}"'] if self.instruction else []
        if self.subgoal:
            out.append(f"subgoal: {self.subgoal}")
        status = []
        if self.controller:
            status.append(self.controller)
        if self.device:
            status.append(self.device)
        if self.latency_ms:
            status.append(f"{self.latency_ms:.1f} ms/inference")
        if self.seed is not None:
            status.append(f"seed {self.seed}")
        if status:
            out.append("  |  ".join(status))
        for key, value in self.extra.items():
            out.append(f"{key}: {value}")
        return out


class VideoRecorder:
    """Collects frames during an episode and writes an mp4."""

    def __init__(self, world, path: Path, *, cameras=("cam_front", "cam_overhead"),
                 size: tuple[int, int] = (480, 360), fps: int = 20, every: int = 1,
                 overlay: OverlayState | None = None):
        self.world = world
        self.path = Path(path)
        self.cameras = cameras
        self.size = size
        self.fps = fps
        self.every = max(1, every)
        self.overlay = overlay or OverlayState()
        self.frames: list[np.ndarray] = []
        self._tick = 0

    def capture(self, *_args) -> None:
        """Scheduler tick hook."""
        self._tick += 1
        if self._tick % self.every:
            return
        views = [self.world.render(camera, size=self.size) for camera in self.cameras]
        frame = np.concatenate(views, axis=1)
        self.frames.append(self._draw_overlay(frame))

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        try:
            from PIL import Image, ImageDraw
        except ImportError:  # pragma: no cover - overlay is optional
            return frame

        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image, "RGBA")
        lines = self.overlay.lines()
        if not lines:
            return np.asarray(image)
        height = 14 * len(lines) + 10
        draw.rectangle([(0, 0), (image.width, height)], fill=(0, 0, 0, 165))
        for index, line in enumerate(lines):
            draw.text((8, 5 + index * 14), line[:150], fill=(255, 255, 255, 255))
        return np.asarray(image)

    def save(self) -> Path | None:
        """Write the mp4. Returns the path, or None when there is nothing to write."""
        if not self.frames:
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import imageio.v2 as imageio

            with imageio.get_writer(self.path, fps=self.fps, quality=8, macro_block_size=1) as writer:
                for frame in self.frames:
                    writer.append_data(frame)
        except Exception:
            # No encoder available: fall back to a strip of stills so the run still leaves
            # visual evidence rather than nothing at all.
            from PIL import Image

            fallback = self.path.with_suffix(".png")
            picks = np.linspace(0, len(self.frames) - 1, min(6, len(self.frames))).astype(int)
            strip = np.concatenate([self.frames[i] for i in picks], axis=0)
            Image.fromarray(strip).save(fallback)
            return fallback
        return self.path

    def reset(self) -> None:
        self.frames.clear()
        self._tick = 0
