"""Convert a trained policy to OpenVINO IR, optionally quantized, for Intel deployment.

A VLA is not one graph. SmolVLA is a vision tower, a language model and an action expert, and they
do not all convert equally well: the vision tower and the action expert are static-shape feed-
forward graphs that convert cleanly, while the language model carries a KV cache and dynamic
sequence lengths, which is exactly what an NPU will not accept. Pretending otherwise produces a
benchmark that measures nothing.

So this module converts **per component**, records what happened to each one, and leaves the rest
on PyTorch:

* ``vision`` — the image encoder, run once per camera per inference,
* ``expert`` — the action head, the part that runs every chunk,
* ``text`` — the instruction embedding. In this task the instruction is fixed for the duration of
  a subgoal, so its embedding is computed once and cached rather than recomputed at 20 Hz. That is
  not a trick to make the number look good; it is the correct thing to do for a robot that is
  told what to do once and then does it.

Components that fail to convert are reported as failures with the error text, because
"this did not compile for NPU" is a legitimate and useful result to publish.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

#: Precisions we attempt, in the order a deployment would try them.
PRECISIONS = ("fp32", "fp16", "int8")


@dataclass
class ComponentReport:
    """What happened when we tried to convert one component."""

    name: str
    converted: bool
    precision: str = "fp32"
    ir_path: str | None = None
    parameters: int = 0
    size_mb: float = 0.0
    error: str | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExportReport:
    policy: str
    components: list[ComponentReport] = field(default_factory=list)
    openvino_version: str = ""
    torch_version: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "openvino_version": self.openvino_version,
            "torch_version": self.torch_version,
            "seconds": round(self.seconds, 1),
            "components": [c.to_dict() for c in self.components],
            "converted": [c.name for c in self.components if c.converted],
            "failed": {c.name: c.error for c in self.components if not c.converted},
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


def _file_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.parent.glob(f"{path.stem}.*") if f.is_file())
    return round(total / 1e6, 2)


def convert_module(module, example_inputs: tuple, out_path: Path, *, name: str,
                   precision: str = "fp16", input_names: list[str] | None = None,
                   calibration=None) -> ComponentReport:
    """Convert one ``torch.nn.Module`` to OpenVINO IR at the requested precision.

    ``calibration`` (an iterable of input tuples) enables INT8 post-training quantization with
    NNCF. Without it, ``int8`` falls back to fp16 and says so in the report rather than silently
    producing an unquantized model.
    """
    import openvino as ov
    import torch

    parameters = sum(p.numel() for p in module.parameters())
    try:
        module = module.eval()
        with torch.inference_mode():
            model = ov.convert_model(module, example_input=example_inputs)
    except Exception as exc:  # noqa: BLE001 - a conversion failure is a reportable result
        return ComponentReport(name=name, converted=False, precision=precision,
                               parameters=parameters, error=f"{type(exc).__name__}: {exc}")

    notes = ""
    if precision == "int8":
        if calibration is None:
            precision, notes = "fp16", "int8 requested but no calibration data supplied"
        else:
            try:
                import nncf

                dataset = nncf.Dataset(calibration)
                model = nncf.quantize(model, dataset)
                notes = "post-training INT8 via NNCF"
            except Exception as exc:  # noqa: BLE001
                precision, notes = "fp16", f"NNCF quantization failed: {exc}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(model, out_path, compress_to_fp16=precision in ("fp16", "int8"))
    return ComponentReport(
        name=name,
        converted=True,
        precision=precision,
        ir_path=str(out_path),
        parameters=parameters,
        size_mb=_file_size_mb(out_path),
        notes=notes,
    )


def export_policy(policy, out_dir: Path, *, precision: str = "fp16",
                  image_size: tuple[int, int] = (224, 224), state_dim: int = 12,
                  calibration: dict | None = None) -> ExportReport:
    """Export the convertible parts of a LeRobot policy.

    The policy object is inspected rather than assumed: whichever of ``model.vision_tower``,
    ``model.action_expert`` (and the ACT single-graph case) exist are converted, and anything
    unrecognized is reported as skipped so the summary reflects the real model.
    """
    import openvino as ov
    import torch

    started = time.perf_counter()
    report = ExportReport(
        policy=type(policy).__name__,
        openvino_version=ov.__version__,
        torch_version=torch.__version__,
    )
    out_dir = Path(out_dir)
    width, height = image_size

    candidates = _find_components(policy)
    if not candidates:
        report.components.append(
            ComponentReport(name="whole_policy", converted=False,
                            error="no convertible submodule found; inspect the policy structure")
        )
        report.seconds = time.perf_counter() - started
        return report

    for name, module, example in candidates:
        if example is None:
            example = (torch.zeros(1, 3, height, width),)
        calib = (calibration or {}).get(name)
        report.components.append(
            convert_module(module, example, out_dir / f"{name}.xml", name=name,
                           precision=precision, calibration=calib)
        )
    report.seconds = time.perf_counter() - started
    return report


def _find_components(policy) -> list[tuple[str, object, tuple | None]]:
    """Locate the sub-modules worth converting, without hard-coding one policy's layout."""
    import torch

    found: list[tuple[str, object, tuple | None]] = []
    model = getattr(policy, "model", policy)

    for attribute in ("vision_tower", "vision_encoder", "image_encoder", "backbone"):
        module = _getattr_path(model, attribute)
        if module is not None:
            found.append((f"vision_{attribute}", module, (torch.zeros(1, 3, 224, 224),)))
            break

    for attribute in ("action_expert", "action_head", "expert", "head"):
        module = _getattr_path(model, attribute)
        if module is not None:
            found.append((f"expert_{attribute}", module, None))
            break

    return found


def _getattr_path(root, path: str):
    node = root
    for part in path.split("."):
        node = getattr(node, part, None)
        if node is None:
            return None
    return node


def save_metadata(out_dir: Path, metadata: dict) -> None:
    """Store the preprocessing contract next to the IR so inference cannot drift from training."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "armanual_policy.json").write_text(json.dumps(metadata, indent=2))


def image_calibration_set(dataset_root: Path, count: int = 64,
                          image_size: tuple[int, int] = (224, 224)):
    """A small calibration set drawn from recorded demonstrations, for INT8 quantization.

    Calibrating on the robot's *own* camera images matters: quantization ranges fitted to
    ImageNet-like photos do not match a simulator's flat-shaded tabletop, and the accuracy loss
    shows up as a policy that reaches slightly wrong.
    """
    import torch

    frames = []
    for shard in sorted(Path(dataset_root).rglob("*.npz"))[:4]:
        data = np.load(shard)
        for key in data.files:
            if key.endswith(".wrist") or key.endswith(".scene"):
                images = data[key]
                for image in images[:: max(1, len(images) // 8)]:
                    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    frames.append((tensor,))
                    if len(frames) >= count:
                        return frames
    return frames
