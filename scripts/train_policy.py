"""Fine-tune SmolVLA (or train ACT) on the collected dataset.

This wraps LeRobot's own trainer rather than reimplementing one: the recipe stays the published
one, and anyone can reproduce it from the printed command alone.

    python scripts/train_policy.py --policy smolvla --steps 20000
    python scripts/train_policy.py --policy act --steps 40000 --batch-size 16

Memory notes for an 8 GB card (RTX 5050 and similar):

* SmolVLA base is ~450 M parameters. With the vision encoder frozen and bf16 activations, the
  trainable action expert fits comfortably; a full fine-tune of everything does not.
* Batch size 4 with gradient accumulation reaches the same effective batch as the reference
  recipe's 64 without the memory spike.
* If you still hit an out-of-memory error, halve ``--batch-size`` and double
  ``--grad-accum`` before touching anything else — that trade is free apart from wall-clock.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

POLICY_PRESETS = {
    "smolvla": {
        "path": "lerobot/smolvla_base",
        "batch_size": 4,
        "grad_accum": 16,
        "steps": 20000,
        "extra": [
            "--policy.freeze_vision_encoder=true",
            "--policy.train_expert_only=true",
            "--policy.push_to_hub=false",
        ],
    },
    "act": {
        "path": None,  # trained from scratch
        "type": "act",
        "batch_size": 16,
        "grad_accum": 1,
        "steps": 40000,
        "extra": ["--policy.push_to_hub=false"],
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
    ]
    if preset.get("path"):
        command.append(f"--policy.path={preset['path']}")
    else:
        command.append(f"--policy.type={preset['type']}")
    if grad_accum > 1:
        # LeRobot exposes this as an optimizer setting; name differs across versions, so pass the
        # one the installed version knows and let it fail loudly rather than silently ignoring it.
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
    raise SystemExit(subprocess.call(command, env=env))


if __name__ == "__main__":
    main()
