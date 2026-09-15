"""Running a trained policy in the loop, on any of the backends we deploy to.

Everything above this file — the planner, the evaluation harness, the demo script — talks to a
:class:`PolicyBackend`. Swapping PyTorch for OpenVINO, or CPU for iGPU or NPU, is a constructor
argument, so the *same* closed loop produces the before/after numbers the Intel rubric asks for
and there is no second code path that might behave differently.

The observation the policy receives is exactly what the dataset recorded: three camera images,
the twelve-dimensional bimanual state, and the instruction sentence. The policy returns an action
chunk; the runner executes it open-loop at the control rate and asks for the next one when it runs
out, which is what keeps inference off the critical path at 20 Hz.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from armanual.policy.collect import ACTION_DIM, ARMS, CAMERAS, IMAGE_SIZE
from armanual.sim.builder import GRIPPER_CLOSED, GRIPPER_OPEN


@dataclass
class InferenceStats:
    """Latency bookkeeping, kept per run so benchmarks and evaluations agree on the numbers."""

    calls: int = 0
    total_ms: float = 0.0
    samples_ms: list[float] = field(default_factory=list)

    def record(self, milliseconds: float) -> None:
        self.calls += 1
        self.total_ms += milliseconds
        self.samples_ms.append(milliseconds)

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0

    def percentile(self, q: float) -> float:
        return float(np.percentile(self.samples_ms, q)) if self.samples_ms else 0.0

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "mean_ms": round(self.mean_ms, 2),
            "p50_ms": round(self.percentile(50), 2),
            "p95_ms": round(self.percentile(95), 2),
            "max_ms": round(max(self.samples_ms), 2) if self.samples_ms else 0.0,
        }


class PolicyBackend(Protocol):
    """Anything that can turn one observation into an action chunk."""

    name: str
    action_dim: int

    def predict(self, images: dict[str, np.ndarray], state: np.ndarray, task: str) -> np.ndarray:
        """Return an (H, action_dim) chunk, or (1, action_dim) for single-step policies."""
        ...

    def reset(self) -> None:
        ...


class LeRobotBackend:
    """A LeRobot policy (SmolVLA, ACT, ...) running under PyTorch."""

    def __init__(self, checkpoint: str | Path, device: str = "cuda", *, dtype: str = "float32"):
        import torch
        from lerobot.policies.factory import get_policy_class

        self.torch = torch
        self.device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        checkpoint = str(checkpoint)
        self.policy = self._load(checkpoint, get_policy_class)
        self.policy.to(self.device)
        self.policy.eval()
        if dtype == "bfloat16" and self.device.type == "cuda":
            self.policy = self.policy.to(torch.bfloat16)
        self.dtype = getattr(torch, dtype)
        self.name = f"torch:{self.device.type}"
        self.action_dim = ACTION_DIM
        self.stats = InferenceStats()

    @staticmethod
    def _load(checkpoint: str, get_policy_class):
        """Load a checkpoint without needing to know which policy class wrote it."""
        import json

        config_path = Path(checkpoint) / "config.json"
        policy_type = "smolvla"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            policy_type = config.get("type", config.get("policy_type", policy_type))
        return get_policy_class(policy_type).from_pretrained(checkpoint)

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def predict(self, images: dict[str, np.ndarray], state: np.ndarray, task: str) -> np.ndarray:
        torch = self.torch
        batch = {}
        for key, _camera in CAMERAS:
            image = torch.from_numpy(images[key]).to(self.device)
            image = image.permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255.0
            batch[key] = image
        batch["observation.state"] = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        batch["task"] = [task]

        started = time.perf_counter()
        with torch.inference_mode():
            action = self.policy.select_action(batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.stats.record((time.perf_counter() - started) * 1000)
        action = action.detach().to("cpu").to(torch.float32).numpy()
        return action.reshape(-1, self.action_dim)


class ScriptedBackend:
    """The analytical controller, exposed through the same interface.

    Used as the baseline in every comparison and as the fallback when a learned policy fails a
    subgoal. Having both behind one interface is what makes "VLA vs scripted" an apples-to-apples
    number rather than two differently-instrumented runs.
    """

    name = "scripted"
    action_dim = ACTION_DIM

    def __init__(self):
        self.stats = InferenceStats()

    def reset(self) -> None:
        return

    def predict(self, images, state, task):  # pragma: no cover - never used as a policy
        raise NotImplementedError(
            "ScriptedBackend does not predict actions; the executor runs its primitives directly"
        )


class PolicyRunner:
    """Drives a world with a policy backend for one language subgoal."""

    def __init__(self, world, backend: PolicyBackend, *, dt: float = 0.05,
                 chunk_reuse: int | None = None, image_size: tuple[int, int] = IMAGE_SIZE):
        self.world = world
        self.backend = backend
        self.dt = dt
        self.image_size = image_size
        self.steps_per_tick = max(1, int(round(dt / world.model.opt.timestep)))
        #: How many actions of a chunk to execute before asking for a new one. Executing the whole
        #: chunk is cheapest; executing a prefix is more reactive. Default: half.
        self.chunk_reuse = chunk_reuse
        self.on_tick = []

    def observation(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        images = {
            key: self.world.render(camera, size=self.image_size) for key, camera in CAMERAS
        }
        state = []
        for arm in ARMS:
            state.append(self.world.arm_qpos(arm)[:5])
            state.append([self.world.gripper_opening(arm)])
        return images, np.concatenate(state).astype(np.float32)

    def apply(self, action: np.ndarray) -> None:
        """Send one 12-dim action to both arms."""
        action = np.asarray(action, dtype=float).reshape(-1)
        for index, arm in enumerate(ARMS):
            handle = self.world.arms[arm]
            joints = action[index * 6 : index * 6 + 5]
            gripper = float(np.clip(action[index * 6 + 5], 0.0, 1.0))
            self.world.data.ctrl[handle.arm_actuators] = joints
            self.world.data.ctrl[handle.gripper_actuator] = (
                GRIPPER_CLOSED + gripper * (GRIPPER_OPEN - GRIPPER_CLOSED)
            )

    def run(self, task: str, *, max_seconds: float = 25.0) -> dict:
        """Execute one language subgoal under the policy. Returns a small run record."""
        self.backend.reset()
        started = self.world.time
        ticks = 0
        chunk: np.ndarray | None = None
        cursor = 0
        while self.world.time - started < max_seconds:
            if chunk is None or cursor >= len(chunk):
                images, state = self.observation()
                chunk = self.backend.predict(images, state, task)
                limit = self.chunk_reuse or max(1, len(chunk) // 2)
                chunk = chunk[:limit]
                cursor = 0
            self.apply(chunk[cursor])
            cursor += 1
            self.world.step(self.steps_per_tick)
            ticks += 1
            for hook in self.on_tick:
                hook(self.world)
        return {
            "task": task,
            "backend": self.backend.name,
            "ticks": ticks,
            "sim_seconds": round(self.world.time - started, 2),
            "inference": getattr(self.backend, "stats", InferenceStats()).summary(),
        }
