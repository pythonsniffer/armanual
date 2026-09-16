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
    """A LeRobot policy (SmolVLA, ACT, ...) running under PyTorch.

    Inference goes through the **processor pipeline saved with the checkpoint**, not a hand-built
    batch. That pipeline is what renames the cameras to the names the policy was trained with,
    adds the batch dimension, tokenizes the instruction into ``observation.language.tokens`` and
    applies the dataset normalization. Reconstructing any of that by hand is how an inference path
    silently diverges from training — SmolVLA raises a `KeyError` for the missing language tokens,
    but a normalization mismatch would just produce quietly wrong actions.
    """

    def __init__(self, checkpoint: str | Path, device: str = "cuda", *, dtype: str = "float32"):
        import torch
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.torch = torch
        self.device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        checkpoint = str(checkpoint)
        self.policy = self._load(checkpoint, get_policy_class)
        self.policy.to(self.device)
        self.policy.eval()
        # The saved pipeline carries a device_processor pinned to whatever trained the policy
        # (cuda here). Override it so the pipeline follows the device this backend runs on —
        # otherwise inference on CPU feeds CUDA tensors to CPU weights, or vice versa.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": self.device.type}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
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

    def _batch(self, images: dict[str, np.ndarray], state: np.ndarray, task: str) -> dict:
        """Unbatched tensors under the *dataset's* key names; the pipeline batches and renames."""
        torch = self.torch
        batch = {
            key: torch.from_numpy(np.ascontiguousarray(images[key])).permute(2, 0, 1).to(
                torch.float32
            )
            / 255.0
            for key, _camera in CAMERAS
        }
        batch["observation.state"] = torch.from_numpy(np.asarray(state, dtype=np.float32))
        batch["task"] = task
        return batch

    def predict(self, images: dict[str, np.ndarray], state: np.ndarray, task: str) -> np.ndarray:
        torch = self.torch
        batch = self._batch(images, state, task)
        started = time.perf_counter()
        with torch.inference_mode():
            processed = self.preprocessor(batch)
            action = self.policy.select_action(processed)
            action = self.postprocessor(action)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.stats.record((time.perf_counter() - started) * 1000)
        action = action.detach().to("cpu").to(torch.float32).numpy()
        return action.reshape(-1, self.action_dim)

    def predict_chunk(self, images: dict[str, np.ndarray], state: np.ndarray,
                      task: str) -> np.ndarray:
        """The whole action chunk in one call, rather than one action at a time.

        ``select_action`` returns a *single* action and keeps the rest of the chunk in a queue
        inside the policy, refilling it by running the network every ``n_action_steps`` ticks.
        That is efficient for the network but not for the caller: a runner that gets one action
        back has to re-observe every tick, and under this project's CPU renderer three camera
        frames cost ~290 ms — so 50 ticks of a 50-step chunk render 150 frames to feed one
        inference. Asking for the chunk directly makes observation cost track inference cost,
        which is the difference between a six-minute evaluation episode and a twenty-second one.
        """
        torch = self.torch
        batch = self._batch(images, state, task)
        started = time.perf_counter()
        with torch.inference_mode():
            processed = self.preprocessor(batch)
            chunk = self.policy.predict_action_chunk(processed)
            chunk = self.postprocessor(chunk)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.stats.record((time.perf_counter() - started) * 1000)
        chunk = chunk.detach().to("cpu").to(torch.float32).numpy()
        return chunk.reshape(-1, self.action_dim)


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

    def ask(self, images, state, task: str) -> np.ndarray:
        """One inference, returning as many actions as the backend will give.

        A backend that exposes ``predict_chunk`` hands back the whole horizon, so the runner
        re-observes once per chunk instead of once per tick — the cameras are the expensive part
        of this loop, not the network.
        """
        ask = getattr(self.backend, "predict_chunk", None) or self.backend.predict
        return ask(images, state, task)

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
                chunk = self.ask(images, state, task)
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


class OpenVINOBackend:
    """Run a policy with its convertible components on OpenVINO and the rest on PyTorch.

    A VLA does not convert as one graph (see :mod:`armanual.policy.openvino_export`), so this
    backend is deliberately *hybrid*: each component that produced an IR is executed by OpenVINO
    on the chosen Intel device, and anything that did not convert stays on PyTorch. The split is
    recorded in :attr:`placement` and written into every benchmark and evaluation record, because
    "runs on the NPU" means nothing without saying which part.

    Substitution works by wrapping the module's ``forward``: the surrounding policy code, the
    preprocessing and the action decoding are untouched, so a difference in behaviour between this
    backend and the PyTorch one is a real numerical difference and not a different pipeline.
    """

    def __init__(self, checkpoint: str | Path, ir_dir: str | Path, *, device: str = "CPU",
                 torch_device: str = "cpu"):
        import openvino as ov
        import torch

        self.torch = torch
        self.core = ov.Core()
        self.device = device
        if device not in self.core.available_devices:
            raise RuntimeError(
                f"OpenVINO device {device!r} is not available here "
                f"(available: {self.core.available_devices})"
            )

        self.base = LeRobotBackend(checkpoint, device=torch_device)
        self.policy = self.base.policy
        self.placement: dict[str, str] = {}
        self.compiled: dict[str, object] = {}
        self._patch_components(Path(ir_dir))

        self.name = f"openvino:{device}"
        self.action_dim = ACTION_DIM
        self.stats = InferenceStats()

    def _patch_components(self, ir_dir: Path) -> None:
        """Replace each converted module's forward with its compiled IR."""
        from armanual.policy.openvino_export import _find_components

        components = {name: module for name, module, _example in _find_components(self.policy)}
        for ir_path in sorted(Path(ir_dir).glob("*.xml")):
            name = ir_path.stem
            module = components.get(name)
            if module is None:
                self.placement[name] = "ir present but module not found; skipped"
                continue
            compiled = self.core.compile_model(self.core.read_model(ir_path), self.device)
            self.compiled[name] = compiled
            module.forward = self._make_forward(compiled)
            self.placement[name] = f"openvino:{self.device}"
        for name in components:
            self.placement.setdefault(name, f"torch:{self.base.device.type}")

    def _make_forward(self, compiled):
        torch = self.torch

        def forward(*args, **kwargs):
            inputs = [a.detach().cpu().numpy() for a in args if hasattr(a, "detach")]
            result = compiled(inputs)
            outputs = [torch.from_numpy(np.asarray(result[port])) for port in compiled.outputs]
            return outputs[0] if len(outputs) == 1 else tuple(outputs)

        return forward

    def reset(self) -> None:
        self.base.reset()

    def predict(self, images: dict[str, np.ndarray], state: np.ndarray, task: str) -> np.ndarray:
        started = time.perf_counter()
        action = self.base.predict(images, state, task)
        self.stats.record((time.perf_counter() - started) * 1000)
        return action

    def predict_chunk(self, images: dict[str, np.ndarray], state: np.ndarray,
                      task: str) -> np.ndarray:
        """Same chunked path as the PyTorch backend, so the two are timed on equal terms."""
        started = time.perf_counter()
        chunk = self.base.predict_chunk(images, state, task)
        self.stats.record((time.perf_counter() - started) * 1000)
        return chunk

    def describe(self) -> dict:
        return {"device": self.device, "placement": dict(self.placement)}
