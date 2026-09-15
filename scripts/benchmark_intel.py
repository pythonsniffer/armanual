"""Benchmark policy inference on Intel hardware with OpenVINO.

    python scripts/benchmark_intel.py --list-devices
    python scripts/benchmark_intel.py --ir-dir outputs/openvino/smolvla --devices CPU GPU NPU
    python scripts/benchmark_intel.py --checkpoint outputs/train/armanual_smolvla/checkpoints/last/pretrained_model \\
        --export --precisions fp16 int8 --devices CPU GPU NPU

What it reports, per component x device x precision: compile time, first-inference (warm-up)
latency, steady-state p50/p95/max, throughput, and — where a torch baseline is available — the
speed-up over it. Devices that fail to compile are listed with the reason; nothing silently falls
back to CPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from armanual.policy.benchmark import (
    benchmark_ir,
    host_info,
    list_devices,
    write_report,
)

DEFAULT_DEVICES = ("CPU", "GPU", "NPU")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--ir-dir", type=Path, default=Path("outputs/openvino"))
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="policy checkpoint to export before benchmarking")
    parser.add_argument("--export", action="store_true", help="export the checkpoint first")
    parser.add_argument("--devices", nargs="*", default=None,
                        help=f"default: every available device among {DEFAULT_DEVICES}")
    parser.add_argument("--precisions", nargs="*", default=["fp16"])
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("outputs/benchmarks/intel_benchmark.json"))
    args = parser.parse_args()

    if args.list_devices:
        info = host_info()
        print(json.dumps(info, indent=2))
        if not info.get("is_intel_core_ultra"):
            print(
                "\nNOTE: this machine does not look like an Intel Core Ultra system. "
                "Numbers collected here are a development baseline; the submission's headline "
                "figures must come from the Core Ultra target."
            )
        return

    available = {d.name for d in list_devices()}
    requested = args.devices or [d for d in DEFAULT_DEVICES if d in available]
    missing = [d for d in requested if d not in available]
    if missing:
        print(f"requested but not available on this host: {missing} (available: {sorted(available)})")

    if args.export:
        if args.checkpoint is None:
            parser.error("--export requires --checkpoint")
        _export(args.checkpoint, args.ir_dir, args.precisions)

    ir_files = sorted(Path(args.ir_dir).rglob("*.xml"))
    if not ir_files:
        parser.error(f"no IR files found under {args.ir_dir}; run with --export first")

    runs = []
    for ir_path in ir_files:
        for device in requested:
            if device not in available:
                continue
            precision = _precision_of(ir_path, args.precisions)
            print(f"benchmarking {ir_path.stem} on {device} ({precision}) ...", flush=True)
            run = benchmark_ir(ir_path, device, precision=precision, iterations=args.iterations)
            runs.append(run)
            if run.ok:
                print(
                    f"  compile {run.compile_seconds:5.2f}s  warmup {run.warmup_ms:8.2f}ms  "
                    f"p50 {run.p50_ms:7.3f}ms  p95 {run.p95_ms:7.3f}ms  "
                    f"{run.throughput_hz:7.1f} Hz"
                )
            else:
                print(f"  FAILED: {run.error}")

    payload = write_report(args.out, runs)
    print("\n" + json.dumps(payload["summary"], indent=2))
    print(f"\nwrote {args.out}")


def _precision_of(ir_path: Path, precisions: list[str]) -> str:
    for precision in precisions:
        if precision in ir_path.stem or precision in ir_path.parent.name:
            return precision
    return precisions[0] if precisions else "fp16"


def _export(checkpoint: Path, ir_dir: Path, precisions: list[str]) -> None:
    from armanual.policy.openvino_export import export_policy
    from armanual.policy.runtime import LeRobotBackend

    backend = LeRobotBackend(checkpoint, device="cpu")
    for precision in precisions:
        out_dir = ir_dir / precision
        report = export_policy(backend.policy, out_dir, precision=precision)
        report.save(out_dir / "export_report.json")
        print(f"\n=== export ({precision}) ===")
        print(json.dumps(report.to_dict()["converted"], indent=2))
        if report.to_dict()["failed"]:
            print("failed components:", json.dumps(report.to_dict()["failed"], indent=2))


if __name__ == "__main__":
    main()
