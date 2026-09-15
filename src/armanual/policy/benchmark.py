"""Measure inference performance on Intel hardware, honestly.

Rules this module follows, because they are what separate a benchmark from a marketing number:

* **Warm-up is excluded and reported separately.** The first inference on an iGPU or NPU includes
  compilation and driver setup and can be a hundred times the steady-state latency. Folding it
  into the mean makes a device look bad; hiding it entirely hides a real cost of deployment.
* **A device is only reported if it actually ran.** Device availability is read from the runtime,
  each requested device is compiled for, and a failure is recorded with its error rather than
  quietly falling back to CPU — which is what ``AUTO`` would do, and is how benchmarks end up
  claiming NPU numbers that were measured on a CPU.
* **Latency distribution, not just a mean.** p50/p95/max, because a control loop lives or dies by
  its tail.
* **The end-to-end loop is timed too.** Policy latency is only part of the story: rendering and
  preprocessing are on the same critical path, and optimizing the model while ignoring them
  produces a "3x faster" claim that the robot never sees.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class DeviceInfo:
    name: str
    full_name: str = ""
    available: bool = True
    properties: dict = field(default_factory=dict)


@dataclass
class BenchmarkRun:
    """One (component, device, precision) measurement."""

    component: str
    device: str
    precision: str
    ok: bool
    warmup_ms: float = 0.0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    max_ms: float = 0.0
    throughput_hz: float = 0.0
    iterations: int = 0
    compile_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def list_devices() -> list[DeviceInfo]:
    """Enumerate what OpenVINO can actually see on this machine."""
    import openvino as ov

    core = ov.Core()
    devices = []
    for name in core.available_devices:
        info = DeviceInfo(name=name)
        for key in ("FULL_DEVICE_NAME", "DEVICE_ARCHITECTURE", "OPTIMIZATION_CAPABILITIES"):
            try:
                value = core.get_property(name, key)
                if key == "FULL_DEVICE_NAME":
                    info.full_name = str(value)
                else:
                    info.properties[key] = str(value)
            except Exception:  # noqa: BLE001 - not every plugin exposes every property
                continue
        devices.append(info)
    return devices


def benchmark_ir(ir_path: Path, device: str, *, precision: str = "fp16", iterations: int = 200,
                 warmup: int = 10, component: str = "") -> BenchmarkRun:
    """Compile an IR for one device and measure steady-state latency."""
    import openvino as ov

    component = component or Path(ir_path).stem
    run = BenchmarkRun(component=component, device=device, precision=precision, ok=False,
                       iterations=iterations)
    try:
        core = ov.Core()
        model = core.read_model(ir_path)
        started = time.perf_counter()
        compiled = core.compile_model(model, device)
        run.compile_seconds = round(time.perf_counter() - started, 2)

        inputs = {}
        for port in compiled.inputs:
            shape = [int(d.get_length()) if d.is_static else 1 for d in port.partial_shape]
            inputs[port.any_name] = np.random.rand(*shape).astype(np.float32)

        request = compiled.create_infer_request()
        started = time.perf_counter()
        request.infer(inputs)
        run.warmup_ms = round((time.perf_counter() - started) * 1000, 2)
        for _ in range(max(0, warmup - 1)):
            request.infer(inputs)

        samples = []
        for _ in range(iterations):
            started = time.perf_counter()
            request.infer(inputs)
            samples.append((time.perf_counter() - started) * 1000)

        run.ok = True
        run.mean_ms = round(statistics.fmean(samples), 3)
        run.p50_ms = round(float(np.percentile(samples, 50)), 3)
        run.p95_ms = round(float(np.percentile(samples, 95)), 3)
        run.max_ms = round(max(samples), 3)
        run.throughput_hz = round(1000.0 / run.mean_ms, 1) if run.mean_ms else 0.0
    except Exception as exc:  # noqa: BLE001 - a device that cannot run it is a real result
        run.error = f"{type(exc).__name__}: {exc}"
    return run


def benchmark_torch(callable_fn, *, iterations: int = 100, warmup: int = 10,
                    component: str = "torch_baseline", device: str = "cpu") -> BenchmarkRun:
    """Time the PyTorch path, so the OpenVINO numbers have something to be compared against."""
    run = BenchmarkRun(component=component, device=f"torch:{device}", precision="fp32", ok=False,
                       iterations=iterations)
    try:
        started = time.perf_counter()
        callable_fn()
        run.warmup_ms = round((time.perf_counter() - started) * 1000, 2)
        for _ in range(max(0, warmup - 1)):
            callable_fn()
        samples = []
        for _ in range(iterations):
            started = time.perf_counter()
            callable_fn()
            samples.append((time.perf_counter() - started) * 1000)
        run.ok = True
        run.mean_ms = round(statistics.fmean(samples), 3)
        run.p50_ms = round(float(np.percentile(samples, 50)), 3)
        run.p95_ms = round(float(np.percentile(samples, 95)), 3)
        run.max_ms = round(max(samples), 3)
        run.throughput_hz = round(1000.0 / run.mean_ms, 1) if run.mean_ms else 0.0
    except Exception as exc:  # noqa: BLE001
        run.error = f"{type(exc).__name__}: {exc}"
    return run


def host_info() -> dict:
    """Everything a reader needs to know to judge whether a number is meaningful."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }
    try:
        import openvino as ov

        info["openvino"] = ov.__version__
        info["devices"] = [asdict(d) for d in list_devices()]
    except Exception as exc:  # noqa: BLE001
        info["openvino_error"] = str(exc)
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    # Intel Core Ultra check: the deployment target must be identifiable in the results file.
    processor = str(info.get("processor", "")).lower()
    devices = " ".join(str(d.get("full_name", "")).lower() for d in info.get("devices", []))
    info["is_intel_core_ultra"] = "core(tm) ultra" in devices or "core ultra" in processor
    return info


def write_report(path: Path, runs: list[BenchmarkRun], extra: dict | None = None) -> dict:
    """Write a benchmark report with host context and a comparison table."""
    payload = {
        "host": host_info(),
        "runs": [r.to_dict() for r in runs],
        "summary": _summarize(runs),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return payload


def _summarize(runs: list[BenchmarkRun]) -> dict:
    ok = [r for r in runs if r.ok]
    if not ok:
        return {"note": "no successful runs"}
    best = min(ok, key=lambda r: r.mean_ms)
    baseline = next((r for r in ok if r.device.startswith("torch")), None)
    summary = {
        "fastest": {"component": best.component, "device": best.device,
                    "precision": best.precision, "mean_ms": best.mean_ms},
        "devices_measured": sorted({r.device for r in ok}),
        "devices_failed": {r.device: r.error for r in runs if not r.ok},
    }
    if baseline and baseline.mean_ms:
        summary["speedup_vs_torch"] = {
            f"{r.device}/{r.precision}": round(baseline.mean_ms / r.mean_ms, 2)
            for r in ok
            if not r.device.startswith("torch") and r.mean_ms
        }
    return summary
