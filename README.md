# armanual — Bimanual VLA Manipulation with Multi-Modal Reasoning

Setting a dinner table with **two simulated SO-101 arms in MuJoCo**, driven by natural-language
instructions, closed-loop camera perception and a vision-language-action policy — with inference
optimized through **OpenVINO** for **Intel Core Ultra**.

Submission for the Intel Physical AI Online Challenge.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m armanual.cli verify                       # checks sim, rendering, perception, IK, parser
python scripts/run_task.py --instruction "set the table and pour water into the blue cup"
```

> **Honest status.** Every number in this repository comes from a script you can re-run, and the
> things that do not work yet are listed in [docs/LIMITATIONS.md](docs/LIMITATIONS.md) rather than
> omitted. Where a claim needs Intel Core Ultra hardware to be meaningful, it says so.

---

## What it does

One instruction — typed, spoken, or accompanied by a pointing gesture — is parsed, **grounded in
what the cameras currently see**, decomposed into language subgoals, and executed by two arms that
decide between themselves who does what.

```
"set the table and pour water into the blue cup"
        │
        ├─ open the drawer                      ← the cutlery is not visible until it is open
        ├─ put the plate in the middle           ← arms costed per step; the cheaper one takes it
        ├─ put the fork to the left of the plate ← appears in the plan only after the drawer opens
        ├─ put the cup to the right of the plate
        └─ pour water into the blue cup          ← one arm steadies the cup, the other tips the bottle
```

Between every subgoal the system looks again, re-grounds the words against the new scene and
re-plans. "The blue cup" means whichever cup is blue *now*.

## What makes it more than a script

| | |
| --- | --- |
| **Perception is real** | Objects are found by segmenting an RGB-D point cloud from the simulated cameras. No simulator state is read anywhere in the control path. Measured: recall 0.92, precision 0.84, position error 8.6 mm |
| **Control is the policy** | In a run, all twelve joint commands come from SmolVLA reading three cameras and the instruction text. The analytical stack generated the training data and is the baseline it is compared against — it does not drive the robot |
| **Hand-offs happen for a reason** | When the pick is in one arm's workspace and the place is in the other's, the planner inserts a hand-off — it is not scripted into the demo |
| **Ambiguity is measured** | "The blue cup" next to a navy cup produces a recorded ambiguity with both candidates and their scores |
| **Failures are attributed** | Every failure is tagged perception / grounding / planning / reachability / grasp / placement / coordination / timeout |
| **The geometry was measured, not guessed** | Workspace, holding torque and jaw geometry were probed from the compiled model and drive every design choice ([docs/WORKSPACE.md](docs/WORKSPACE.md)) |

## Repository layout

```
src/armanual/
  sim/          MuJoCo scene construction, seeded randomization, runtime
  perception/   RGB-D detection, multi-camera fusion, wrist-camera refinement
  task/         instruction parsing, referent grounding, the shared task schema
  planning/     decomposition, dynamic arm assignment, closed-loop execution
  control/      IK with torque + collision screening, grasps, bimanual primitives
  policy/       demonstration collection, LeRobot dataset, SmolVLA runtime, OpenVINO export
  eval/         benchmark tiers, harness, video recording
scripts/        every command a judge needs, each self-documenting with --help
docs/           architecture, requirements traceability, measurements, limitations
tests/          fast unit tests (language, grounding) and slower simulation tests
```

## Commands

| Task | Command |
| --- | --- |
| Check the environment | `python -m armanual.cli verify` |
| Render the scene | `python -m armanual.cli scene --seed 3 --randomize` |
| Run one instruction | `python scripts/run_task.py --instruction "set the table"` |
| Benchmark suite (tiers 1–6) | `python scripts/evaluate.py --seeds 10 --workers 6 --out outputs/eval` |
| Perception accuracy | `python scripts/eval_perception.py --seeds 10` |
| Re-measure the workspace | `python scripts/measure_workspace.py` |
| Collect demonstrations | `python scripts/collect_dataset.py --episodes-per-skill 60 --workers 6` |
| Fine-tune SmolVLA | `python scripts/train_policy.py --policy smolvla --steps 20000` |
| Intel devices | `python scripts/benchmark_intel.py --list-devices` |
| Intel benchmark | `python scripts/benchmark_intel.py --checkpoint <ckpt> --export --devices CPU GPU NPU` |
| Tests | `pytest tests/ -q` |

## The policy

A single network cannot learn a minute-long, multi-object, dependency-laden task from a few
hundred demonstrations. So the instruction is decomposed into **language subgoals**, and one
policy — conditioned on the subgoal sentence — executes each of them:

- **Model**: SmolVLA (`lerobot/smolvla_base`), fine-tuned. **Every joint command in a run comes
  from the policy** — no scripted fallback, no hand-written trajectories
- **Observation**: three cameras (overhead + both wrists), 12-dim bimanual state, instruction text
- **Action**: 12-dim — *both arms* — so hand-offs and hold-and-pour are learnable rather than
  structurally impossible
- **Data**: recorded by running the analytical stack on generated instructions, successful
  episodes only — published as
  [`pythonsniffer/armanual-dinner-table`](https://huggingface.co/datasets/pythonsniffer/armanual-dinner-table)
  (617 episodes, 114k frames, 33 distinct instructions)

Training runs on CUDA; **inference** is what must run on Intel, which is what the challenge asks
for. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Intel deployment

A VLA does not convert to a single OpenVINO graph — the vision tower and action expert do, the
language model's KV cache and dynamic shapes do not, and the NPU requires static shapes. The
exporter therefore converts **per component**, reports what failed and why, and the benchmark
states which component ran on which device. Warm-up is measured and reported separately from
steady-state latency, and a device that cannot compile a model is recorded as a failure rather
than silently falling back to CPU.

See [docs/INTEL.md](docs/INTEL.md).

## Documentation

| Doc | What is in it |
| --- | --- |
| [SETUP.md](docs/SETUP.md) | Installation, including the version constraints that bite |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, data flow, why each layer exists |
| [REQUIREMENTS.md](docs/REQUIREMENTS.md) | Traceability: official / user / engineering / open, plus the rubric map |
| [WORKSPACE.md](docs/WORKSPACE.md) | Generated measurements: reach, holding torque, jaw geometry |
| [INTEL.md](docs/INTEL.md) | OpenVINO export, device placement, benchmark protocol |
| [RESULTS.md](docs/RESULTS.md) | Measured results with the commands that produced them |
| [LIMITATIONS.md](docs/LIMITATIONS.md) | What does not work, and how far off it is |
| [DEMO.md](docs/DEMO.md) | Video shot list, mapped to the rubric |
| [DATASET_CARD.md](docs/DATASET_CARD.md) | What is in the demonstration dataset and how it was made |
| [SUBMISSION.md](docs/SUBMISSION.md) | Deliverable checklist with verification commands |
| [REPO_RECON.md](docs/REPO_RECON.md) | Phase 0 survey of the environment and assets |

## License

Code: Apache-2.0. Vendored assets keep their own licences — see
[assets/ATTRIBUTION.md](assets/ATTRIBUTION.md).
