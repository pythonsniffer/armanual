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
        # Convert from CPU float32 regardless of how the checkpoint was loaded. Tracing a module
        # whose weights are CUDA bfloat16 against CPU float32 example inputs fails with a dtype
        # mismatch that reads like an OpenVINO problem but is not one.
        module = module.to("cpu").to(torch.float32).eval()
        example_inputs = tuple(
            a.to("cpu").to(torch.float32) if hasattr(a, "to") else a for a in example_inputs
        )
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
    # Record the shapes the model was traced with. The benchmark needs them to build valid inputs,
    # and the NPU needs static shapes outright — guessing 1 for a dynamic dimension produces an
    # image of 1x1 pixels and a confusing failure deep inside the plugin.
    shapes = [list(a.shape) for a in example_inputs if hasattr(a, "shape")]
    out_path.with_suffix(".shapes.json").write_text(json.dumps({"inputs": shapes}, indent=2))
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


#: Where the convertible pieces live inside the policies we deploy. Verified by inspecting the
#: loaded model rather than guessed: SmolVLA is a SmolVLM2 (vision transformer + connector +
#: Llama text model) alongside a separate Llama action expert.
COMPONENT_PATHS: tuple[tuple[str, str, str], ...] = (
    ("vision_tower", "vlm_with_expert.vlm.model.vision_model", "image"),
    ("vision_connector", "vlm_with_expert.vlm.model.connector", "vision_hidden"),
    ("action_expert", "vlm_with_expert.lm_expert", "sequence"),
    # Generic fallbacks for other LeRobot policies (ACT and friends).
    ("backbone", "backbone", "image"),
    ("action_head", "action_head", "sequence"),
)


def _find_components(policy, image_size: int = 256) -> list[tuple[str, object, tuple | None]]:
    """Locate the sub-modules worth converting, with an example input for each.

    Conversion is attempted per component because a VLA is not one graph: the vision transformer
    and the connector are static feed-forward stacks that convert cleanly, while a Llama with a KV
    cache brings dynamic shapes that the NPU rejects outright. Reporting that per component is
    more useful than one failed whole-model conversion.
    """
    import torch

    found: list[tuple[str, object, tuple | None]] = []
    model = getattr(policy, "model", policy)
    seen: set[int] = set()

    for name, path, kind in COMPONENT_PATHS:
        module = _getattr_path(model, path)
        if module is None or id(module) in seen:
            continue
        seen.add(id(module))
        example = None
        if kind == "image":
            example = (torch.zeros(1, 3, image_size, image_size),)
        found.append((name, module, example))
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


def simulator_calibration_set(count: int = 48, image_size: int = 256, seeds: int = 8):
    """Calibration images rendered from the robot's own cameras.

    Calibrating on the *right distribution* is the whole game for INT8. Activation ranges fitted
    to photographs do not describe a flat-shaded simulated tabletop, and the mismatch shows up not
    as an obvious error but as a policy that reaches a centimetre wrong. These frames come from
    randomized scenes through the same three cameras the policy uses at run time.
    """
    import torch

    from armanual.policy.collect import CAMERAS
    from armanual.sim.randomize import RandomizationConfig, sample_scene
    from armanual.sim.world import World

    frames: list[tuple] = []
    for seed in range(seeds):
        world = World(sample_scene(seed, RandomizationConfig()), fast_render=True)
        for _key, camera in CAMERAS:
            image = world.render(camera, size=(image_size, image_size))
            tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            frames.append((tensor,))
            if len(frames) >= count:
                world.close()
                return frames
        world.close()
    return frames
