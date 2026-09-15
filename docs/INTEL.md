# Intel deployment with OpenVINO

## What has to run on Intel, and what does not

The challenge requires the **final simulation and AI inference** to run on an Intel Core Ultra
Series 2/3 system, and explicitly permits training anywhere. This project follows that split:

| Stage | Hardware | Why |
| --- | --- | --- |
| Demonstration collection | any CPU | MuJoCo physics + software rendering, parallelized |
| Policy training | NVIDIA CUDA (RTX 5050, 8 GB) | OpenVINO is an inference runtime; the NPU has no training path at all, and an Intel iGPU would be slower than the discrete card |
| **Policy inference** | **Intel Core Ultra — CPU / iGPU / NPU** | This is the part the rubric measures |
| Simulation during the demo | Intel Core Ultra CPU | MuJoCo runs on the CPU; the final demo runs entirely on the target machine |

A note worth stating plainly, because it is a common misconception: **OpenVINO cannot accelerate
training, and the NPU cannot train.** There is no autograd path for either. Asking for "OpenVINO
to speed up training on the NPU" has no implementation — the speed-up OpenVINO offers is at
inference, which is exactly where the submission needs it.

## Why the model is exported per component

A VLA is three networks in a trenchcoat. SmolVLA is a SigLIP vision tower, a SmolLM2 language
model and a flow-matching action expert. They do not convert equally:

Measured, not predicted — `scripts/benchmark_intel.py --export` on `lerobot/smolvla_base`:

| Component | Params | Converts to IR? | Size (fp16) | Notes |
| --- | --- | --- | --- | --- |
| Vision tower (SmolVLMVisionTransformer) | 86.4 M | **yes** | 173.4 MB | Static shapes, pure feed-forward — the ideal OpenVINO workload |
| Vision connector (SmolVLMConnector) | 11.8 M | **yes** | 23.6 MB | Also static once given the tower's output shape |
| Action expert (LlamaModel) | 98.2 M | **no** | — | `torch.jit.trace` fails (`unordered_map::at`) even with `use_cache=False` and eager attention |
| Language model (LlamaModel) | 204.6 M | not attempted | — | KV cache and dynamic sequence length; the NPU rejects dynamic shapes outright |

So the deployable split today is: **vision tower and connector on the Intel device, the rest on
PyTorch**. That is not a disappointing result — the vision tower runs *once per camera* and the
policy uses three, so it is the largest single term in the inference budget.

So `armanual/policy/openvino_export.py` converts **each component separately** and records the
outcome of each, and `OpenVINOBackend` runs the converted ones on the Intel device while leaving
the rest on PyTorch. The placement is written into every results file. A claim like "the VLA runs
on the NPU" would be false for any current VLA; "the vision tower runs on the NPU, the action
expert on the iGPU, and the language model on the CPU" is both true and useful.

There is one further optimization that is not a trick: **the instruction is fixed for the duration
of a subgoal**, so its text embedding is computed once and cached rather than recomputed at 20 Hz.
That is what a deployed robot should do.

## Setting up the Core Ultra machine

```bash
# 1. Runtime + drivers
pip install "openvino>=2026.3" nncf
python -m armanual.cli devices          # must list CPU, GPU and NPU

# 2. If GPU or NPU is missing
#    iGPU: install intel-opencl-icd (Level Zero + compute runtime)
#    NPU:  install intel-driver-compiler-npu and intel-level-zero-npu, then reboot
#    On Ubuntu 24.04 both come from Intel's apt repository.

# 3. Verify the model converts and runs
python scripts/benchmark_intel.py --checkpoint <checkpoint> --export \
    --precisions fp16 int8 --devices CPU GPU NPU --out outputs/benchmarks/core_ultra.json
```

`python -m armanual.cli devices` prints `is_intel_core_ultra`. Every benchmark file records it, so
a number measured on a development machine can never be mistaken for a target-hardware result.

## What the benchmark reports, and why

| Metric | Why it is reported |
| --- | --- |
| Compile time | On iGPU/NPU this is seconds, and it is a real deployment cost |
| Warm-up latency (first inference) | Often 10–100× steady state; folding it into the mean is misleading, hiding it is dishonest |
| p50 / p95 / max latency | A control loop lives on its tail, not its mean |
| Throughput | Inferences per second at batch 1 — the only batch a robot has |
| Precision | fp32 / fp16 / INT8, with INT8 calibrated on the robot's *own* camera frames |
| Model size on disk | Edge deployments care |
| Device failures | A device that cannot compile the model is listed with its error, never silently replaced by CPU |
| **Task success before and after** | The rubric asks that optimization preserve behaviour; latency alone cannot show that |

The last one matters most: `scripts/evaluate.py` runs the same benchmark tiers with the PyTorch
backend and with each OpenVINO device, so the comparison is success rate *and* latency on the same
episodes and seeds.

## Quantization

INT8 uses NNCF post-training quantization with a calibration set drawn from recorded
demonstrations — the robot's own camera frames, not a generic image corpus. Simulated tabletop
renders have very different statistics from photographs, and calibrating on the wrong distribution
shows up as a policy that reaches slightly wrong rather than as an obvious error.

If NNCF is unavailable or quantization fails, the exporter falls back to fp16 **and says so in the
report** rather than shipping an unquantized model labelled INT8.

## Expected shape of the results

Filled in from the Core Ultra run; the table below is the format, not a claim.

| Component | Device | Precision | Compile | Warm-up | p50 | p95 | Throughput |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vision tower | CPU | fp16 | | | | | |
| vision tower | GPU | fp16 | | | | | |
| vision tower | NPU | fp16 | | | | | |
| action expert | CPU | int8 | | | | | |
| … | | | | | | | |

Plus, on the same seeds: task success with PyTorch vs with each device, and the end-to-end control
loop rate (which includes rendering and preprocessing, not just the model).
