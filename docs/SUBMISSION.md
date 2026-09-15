# Submission checklist

The challenge names five required deliverables. This is where each one lives and how to verify it.

## 1. Reproducible GitHub repository

| Requirement | Where | Verify |
| --- | --- | --- |
| Setup instructions | [SETUP.md](SETUP.md) | `python -m armanual.cli verify` |
| Environment dependencies | `pyproject.toml`, pinned constraints in SETUP.md | `pip install -e ".[dev]"` |
| MuJoCo scene and assets | `assets/so101/` (Apache-2.0, unmodified) + `src/armanual/sim/` | `python -m armanual.cli scene` |
| Policy training code | `scripts/train_policy.py`, `src/armanual/policy/` | `--print-only` shows the exact command |
| Evaluation code | `scripts/evaluate.py`, `src/armanual/eval/` | `--tier 1 --seeds 2` |
| Inference code | `scripts/run_task.py`, `src/armanual/policy/runtime.py` | runs with or without a policy |
| Commands to reproduce the demo | [DEMO.md](DEMO.md) | each shot lists its command |

## 2. Reproducible MuJoCo simulation

- Scene is generated from a `SceneSpec`, which is a pure function of `(seed, RandomizationConfig)`.
- Randomization axes: placement, size, mass, friction, lighting, background, distractors.
- Evaluation configuration: six benchmark tiers in `src/armanual/eval/tiers.py`, data-defined.
- Determinism is tested: `pytest tests/test_simulation.py -k deterministic`.

```bash
python scripts/evaluate.py --seeds 10 --out outputs/eval    # results.json + results.csv
```

## 3. Intel inference benchmark script

```bash
python scripts/benchmark_intel.py --list-devices
python scripts/benchmark_intel.py --checkpoint <ckpt> --export \
    --precisions fp16 int8 --devices CPU GPU NPU --out outputs/benchmarks/core_ultra.json
```

Reports per component × device × precision: compile time, warm-up, p50/p95/max latency,
throughput, model size, and device failures with their reasons. Every report stamps
`is_intel_core_ultra` so a development-machine number cannot be mistaken for a target one.

## 4. Demonstration video across 10 randomized seeds

```bash
for s in 0 1 2 3 4 5 6 7 8 9; do
  python scripts/run_task.py --instruction "set the table" --seed $s --randomize \
      --record outputs/videos/seed_$s.mp4 --json outputs/videos/seed_$s.json
done
```

Each frame carries the instruction, current subgoal, controller, device and latency. Shot list and
the rubric mapping: [DEMO.md](DEMO.md).

## 5. Technical README / architecture summary

[README.md](../README.md) and [ARCHITECTURE.md](ARCHITECTURE.md) cover solution architecture,
VLA/VLM choice, bimanual coordination strategy, training approach, robustness methods, OpenVINO
optimization and Intel hardware mapping.

---

## Pre-submission audit

```bash
python -m armanual.cli verify            # environment, end to end
pytest tests/ -q                         # unit + simulation tests
python scripts/measure_workspace.py      # regenerate measured docs
python scripts/eval_perception.py --seeds 10 --json outputs/perception.json
python scripts/evaluate.py --seeds 10 --out outputs/eval
git status --short                       # no stray artifacts, no secrets
```

Checklist:

- [ ] `.env`, API keys and tokens are absent from the repository (`SPEECHMATICS_API_KEY` is read
      from the environment and never written to a results file)
- [ ] `data/`, `outputs/` and checkpoints are gitignored; the dataset lives on the Hub
- [ ] Every number in README and RESULTS has a command next to it
- [ ] Anything unverified is in [LIMITATIONS.md](LIMITATIONS.md), not omitted
- [ ] Intel figures come from Core Ultra hardware, with `is_intel_core_ultra: true` in the file
- [ ] Vendored assets keep their licences ([ATTRIBUTION.md](../assets/ATTRIBUTION.md))

## Artifacts on Hugging Face

| Artifact | Repo |
| --- | --- |
| Demonstration dataset | `pythonsniffer/armanual-dinner-table` |
| Fine-tuned policy | `pythonsniffer/armanual-smolvla` |

```bash
hf upload pythonsniffer/armanual-dinner-table data/armanual-dinner-table --repo-type dataset
hf upload pythonsniffer/armanual-smolvla outputs/train/armanual_smolvla/checkpoints/last/pretrained_model
```
