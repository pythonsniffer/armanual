"""Fine-tune SmolVLA on the collected dataset.

This wraps LeRobot's own trainer rather than reimplementing one: the recipe stays the published
one, and anyone can reproduce it from the printed command alone.

    python scripts/train_policy.py --steps 20000
    python scripts/train_policy.py --steps 40000 --image-size 224   # smaller images, more steps

Memory notes for an 8 GB card (RTX 5050 and similar):

* SmolVLA base is ~450 M parameters. With the vision encoder frozen and bf16 activations, the
  trainable action expert fits comfortably; a full fine-tune of everything does not.
* Batch size 16 fits in 8 GB with the vision encoder frozen at 256x256. Measured: batch 2, 4, 8
  and 16 all run at ~2 steps/s, because video decoding in the dataloader — not the GPU — is the
  bottleneck, so the larger batch costs nothing.
* Gradient accumulation is only available when the optimizer is configured from the CLI; with
  ``--policy.path`` the policy supplies its own, so increase ``--batch-size`` instead.
* If you still hit an out-of-memory error, halve ``--batch-size`` and double
  ``--grad-accum`` before touching anything else — that trade is free apart from wall-clock.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: SmolVLA's pretrained config names its cameras camera1..camera3; our dataset names them by
#: where they are mounted. LeRobot maps them with an explicit rename rather than by position,
#: which also documents which physical camera the policy treats as primary.
RENAME_MAP = {
    "observation.images.top": "observation.images.camera1",
    "observation.images.left_wrist": "observation.images.camera2",
    "observation.images.right_wrist": "observation.images.camera3",
}

#: torchcodec decodes the dataset's videos and needs real FFmpeg shared libraries. This repo
#: installs them into .venv/ffmpeg via conda (no root required); PyAV's bundled copies have
#: mangled sonames and do not satisfy it.
FFMPEG_LIB = REPO_ROOT / ".venv" / "ffmpeg" / "lib"
FFMPEG_BIN = REPO_ROOT / ".venv" / "ffmpeg" / "bin"

POLICY_PRESETS = {
    "smolvla": {
        "path": "lerobot/smolvla_base",
        # Measured on an 8 GB RTX 5050: batch 2, 4, 8 and 16 all run at ~2 steps/s, because the
        # bottleneck is video decoding in the dataloader, not the GPU. A larger batch is therefore
        # nearly free, and 16 is where this dataset stops gaining from it.
        "batch_size": 16,
        "grad_accum": 1,
        "steps": 20000,
        "extra": [
            # The published defaults, kept because they are what makes 450M parameters fit in
            # 8 GB: the SigLIP tower stays frozen and only the action expert trains.
            "--policy.freeze_vision_encoder=true",
            "--policy.train_expert_only=true",
            # Our cameras record at 224x224, so padding up to the pretrained 512x512 adds no
            # information and costs ~4x the vision compute per sample. 256 keeps the patch grid
            # meaningful and roughly quadruples training throughput on a laptop GPU.
            "--policy.resize_imgs_with_padding=[256,256]",
            "--policy.push_to_hub=false",
        ],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", choices=sorted(POLICY_PRESETS), default="smolvla")
    parser.add_argument("--dataset", default="pythonsniffer/armanual-dinner-table")
    parser.add_argument("--dataset-root", type=Path,
                        default=REPO_ROOT / "data" / "armanual-dinner-table")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2, dest="num_workers",
                        help="dataloader workers; each one decodes video, so keep it modest on a "
                             "memory-constrained machine")
    parser.add_argument("--print-only", action="store_true", help="show the command and exit")
    args = parser.parse_args()

    preset = POLICY_PRESETS[args.policy]
    steps = args.steps or preset["steps"]
    batch_size = args.batch_size or preset["batch_size"]
    grad_accum = args.grad_accum or preset["grad_accum"]
    job_name = args.job_name or f"armanual_{args.policy}"
    output = args.output or REPO_ROOT / "outputs" / "train" / job_name

    command = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={args.dataset}",
        f"--dataset.root={args.dataset_root}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--output_dir={output}",
        f"--job_name={job_name}",
        "--policy.device=cuda",
        f"--wandb.enable={'true' if args.wandb else 'false'}",
        "--save_freq=2000",
        "--log_freq=100",
        f"--rename_map={json.dumps(RENAME_MAP)}",
        f"--num_workers={args.num_workers}",
    ]
    if preset.get("path"):
        command.append(f"--policy.path={preset['path']}")
    else:
        command.append(f"--policy.type={preset['type']}")
    if grad_accum > 1:
        # Only valid when the optimizer is configured from the CLI. With --policy.path the policy
        # brings its own optimizer config and LeRobot rejects the sub-flag, so raising the batch
        # size is the supported way to increase the effective batch here.
        command.append("--optimizer.type=adamw")
        command.append(f"--optimizer.grad_accumulation_steps={grad_accum}")
    command.extend(preset["extra"])
    if args.resume:
        command.append("--resume=true")

    printable = " \\\n    ".join(command)
    print(f"# effective batch = {batch_size} x {grad_accum} = {batch_size * grad_accum}\n")
    print(printable + "\n")
    if args.print_only:
        return

    env = dict(os.environ)
    # Fragmentation is the usual cause of a late OOM on a small card; this keeps the allocator
    # from holding on to unusable blocks.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if FFMPEG_LIB.exists():
        env["LD_LIBRARY_PATH"] = f"{FFMPEG_LIB}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
        env["PATH"] = f"{FFMPEG_BIN}:{env.get('PATH', '')}"
    else:
        print(
            f"warning: {FFMPEG_LIB} not found. torchcodec needs FFmpeg shared libraries to read "
            "the dataset's videos. Install them with:\n"
            "  conda create -y -p .venv/ffmpeg -c conda-forge 'ffmpeg=7.*'"
        )
    raise SystemExit(subprocess.call(command, env=env))


if __name__ == "__main__":
    main()
