# Setup

Two paths: the **evaluation setup** (everything except training a policy — no GPU needed) and the
**full setup** (adds dataset collection, SmolVLA training and OpenVINO export).

## Evaluation setup

```bash
git clone https://github.com/pythonsniffer/armanual.git && cd armanual
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m armanual.cli verify
```

`verify` builds the scene, renders a frame, runs the detector, solves an IK target and parses an
instruction. It prints what works and what is missing rather than failing on the first import.

### Rendering

MuJoCo needs a GL backend. The repo probes `egl`, then `glfw`, then `osmesa`, and `MUJOCO_GL`
overrides the choice. On a headless server install `libegl1` plus Mesa drivers, or `libosmesa6`.

Under WSL there is **no GPU OpenGL**: rendering falls back to `llvmpipe` on the CPU, costing about
285 ms per 224×224 frame with shadows and 96 ms without. `World(..., fast_render=True)` disables
shadows and reflections, and dataset collection runs several worker processes to compensate.

## Full setup

### 1. PyTorch — mind the version window

```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
```

Two constraints meet here and it is worth stating both:

* **LeRobot 0.4.4 requires `torch<2.11`.** Installing 2.11 first sends pip into dependency
  backtracking that can run for half an hour before failing.
* **Blackwell GPUs (RTX 50-series, compute capability 12.0) need CUDA 12.8 wheels.** Older builds
  have no kernels for `sm_120` and fail at the first matmul.

Verify with a real operation, not just an import:

```bash
python -c "import torch; a=torch.randn(1024,1024,device='cuda',dtype=torch.bfloat16); print((a@a).shape)"
```

### 2. LeRobot

```bash
pip install "lerobot[smolvla]"
```

### 3. FFmpeg shared libraries

The dataset stores camera streams as video, and `torchcodec` decodes them using **FFmpeg shared
libraries** — not the `ffmpeg` binary, and not PyAV's bundled copies (their sonames are
hash-mangled, so torchcodec cannot find them). Without root:

```bash
conda create -y -p .venv/ffmpeg -c conda-forge "ffmpeg=7.*"
```

`scripts/train_policy.py` adds `.venv/ffmpeg/lib` to `LD_LIBRARY_PATH` automatically. For manual
`lerobot-train` invocations, export it yourself:

```bash
export LD_LIBRARY_PATH="$PWD/.venv/ffmpeg/lib:$LD_LIBRARY_PATH"
```

Check it:

```bash
python -c "from torchcodec.decoders import VideoDecoder; print('ok')"
```

### 4. OpenVINO (for the Intel target)

```bash
pip install "openvino>=2026.3" nncf
python -m armanual.cli devices
```

On an Intel Core Ultra machine this should list `CPU`, `GPU` and `NPU`. If `GPU` or `NPU` is
missing, install Intel's compute runtime and NPU driver packages and reboot; see
[INTEL.md](INTEL.md).

## Memory

Dataset collection runs one MuJoCo world per worker, each with three renderers — roughly 500 MB
each. On a 15 GiB WSL allocation, 10 workers plus the parent's video encoder is enough to invoke
the OOM killer; 6 workers is comfortable. To raise the ceiling, set in `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=24GB
processors=16
```

then `wsl --shutdown`. Note this restarts the WSL instance and kills anything running inside it.

## Reproducing the pipeline

```bash
# 1. demonstrations (~50 min at 6 workers for ~140 successful episodes)
python scripts/collect_dataset.py --episodes-per-skill 60 --workers 6

# 2. fine-tune SmolVLA (~75 min for 20k steps on an 8 GB laptop GPU)
python scripts/train_policy.py --policy smolvla --steps 20000

# 3. evaluate: analytical baseline, then the policy, on the same seeds
python scripts/evaluate.py --seeds 10 --out outputs/eval
python scripts/evaluate.py --seeds 10 --policy outputs/train/armanual_smolvla/checkpoints/last/pretrained_model \
    --out outputs/eval_policy

# 4. export and benchmark on the Intel target
python scripts/benchmark_intel.py --checkpoint <ckpt> --export --devices CPU GPU NPU
```

Every script takes `--help` and prints the exact command it runs.
