# Requirements and rubric traceability

Requirements are kept in four separate classes so that nothing invented here can be mistaken for
something the challenge asked for:

| Class | Meaning |
| --- | --- |
| **A** | Official problem statement — quoted or paraphrased from the challenge PDF |
| **B** | User-selected product requirement — asked for by the project owner, not the challenge |
| **C** | Engineering decision — our choice, made to satisfy an A or B requirement |
| **D** | Open question or assumption still to be validated |

Status values: **done** (implemented and measured), **partial** (implemented, not yet at target),
**open** (not yet built), **blocked** (waiting on something external).

---

## A — Official problem statement

| ID | Requirement | Status | Where |
| --- | --- | --- | --- |
| A1 | Two simulated SO-101 arms in MuJoCo | done | `sim/builder.py`, assets from Menagerie |
| A2 | Interpret natural-language task instructions | done | `task/parser.py`, `task/grounding.py` |
| A3 | Reason over camera observations | done | `perception/` — RGB-D only, no simulator state |
| A4 | Coordinate both manipulators | done | `control/executor.py` (both arms in one scheduler), `planning/planner.py` |
| A5 | Multi-step table-setting task | done | tier 6 in `eval/tiers.py` |
| A6 | Object hand-off between arms | done | `control/primitives.py` `handoff_give`/`handoff_take`; planner inserts it on workspace grounds |
| A7 | Collision-aware sequencing, shared-workspace reasoning | done | collision-screened IK incl. the other arm; hand-off zone |
| A8 | Complementary action (hold mug while pouring) | done | `steady_and_hold` + `pour_over` run concurrently |
| A9 | Open a drawer, retrieve cutlery | done | `open_drawer`; cutlery only becomes visible once open |
| A10 | Robustness under weight, friction, shape, lighting, background, placement | partial | all axes implemented in `sim/randomize.py`; tier 5 measures them |
| A11 | Train or fine-tune a policy with LeRobot (SmolVLA/π0.5/ACT) | partial | SmolVLA fine-tune wired via `scripts/train_policy.py`; dataset collection automated |
| A12 | Optimize inference with OpenVINO for Core Ultra (CPU/iGPU/NPU) | partial | export + benchmark harness done; **needs the Intel target** (D3) |
| A13 | Reproducible GitHub repository with setup and run commands | done | this repo; `python -m armanual.cli verify` |
| A14 | Reproducible MuJoCo simulation with randomization + eval config | done | `sim/randomize.py`, `eval/tiers.py` |
| A15 | Intel inference benchmark script reporting latency/throughput/device/precision | done | `scripts/benchmark_intel.py` |
| A16 | Demonstration video across 10 randomized seeds | open | `eval/video.py` records; the 10-seed run is the final step |
| A17 | Technical README / architecture summary | done | `README.md`, `docs/ARCHITECTURE.md` |
| A18 | Optimization must not materially degrade task success | partial | comparison harness exists; numbers pending the Intel run |

## B — User-selected product requirements

| ID | Requirement | Status | Where |
| --- | --- | --- | --- |
| B1 | Text **and** speech input through one task pathway | partial | one `TaskRequest` for all modalities; Speechmatics client pending (D1) |
| B2 | Optional pointing/gesture grounding | done | `Referent.pointing_xy`, weighted into grounding |
| B3 | Dynamic arm assignment, not fixed per object class | done | `Planner.assign_arm` costs both arms per step |
| B4 | Closed loop: understand → act → observe → update → replan | done | `planning/executor.py`, `planning/subgoals.py` |
| B5 | Meaningful visual ambiguity (similar cups, different plate sizes, distractors) | done | scene contains blue/navy cups, two plate sizes, distractors |
| B6 | Cultural / aesthetic table-setting styles | partial | `place_setting.py` — layouts implemented; sources unverified (D2) |
| B7 | Training may run on non-Intel hardware | done | CUDA training, OpenVINO inference; explicitly separated |
| B8 | Dataset and checkpoints on Hugging Face | open | `pythonsniffer/armanual-dinner-table` once collection finishes |
| B9 | Whole task from one instruction | done | `policy/tasks.py` `decompose` → subgoal sequence |

## C — Engineering decisions

| ID | Decision | Rationale | Where |
| --- | --- | --- | --- |
| C1 | Build the scene programmatically from the unmodified Menagerie MJCF | Keeps the arm diffable against upstream; randomization becomes a function of a seed | `sim/builder.py` |
| C2 | Table sized to the *measured* workspace | An unreachable object turns a geometry problem into a fake manipulation failure | `docs/WORKSPACE.md` |
| C3 | Torque **and** collision screening inside IK acceptance | The SO-101 saturates its shoulder inside the kinematic workspace; unscreened poses sag ~2 cm and read as mystery failures | `control/kinematics.py` |
| C4 | Grasps specified as *where the object sits between the jaws* | The tool site is ~20 mm off the jaw line; aiming the site at objects misses every thin feature | `control/gripper.py` |
| C5 | Perception from RGB-D only; privileged observer tagged and used only for ablation | "Reasons over camera observations" is only a real claim if the pipeline cannot cheat | `perception/observer.py` |
| C6 | Wrist-camera refinement for centroid-style grasps | Overhead error (~9 mm) is the same size as a thin grasp feature | `perception/refine.py` |
| C7 | Deterministic parser; vision does the referent resolution | Reproducible across seeds, and ambiguity becomes measurable | `task/parser.py` |
| C8 | Policy operates at the **subgoal** level, planner sequences | A few hundred demos cannot cover a minute-long multi-object task | `policy/tasks.py` |
| C9 | 12-dim bimanual action space | Hand-offs and hold-and-pour are unlearnable if only the moving arm is recorded | `policy/collect.py` |
| C10 | Per-component OpenVINO export with failures reported | A VLA is not one graph; NPU rejects dynamic shapes | `policy/openvino_export.py` |
| C11 | Liquid as particles, pour verified by counting them in the cup | MuJoCo has no fluids; a countable proxy keeps success honest | `sim/builder.py`, `eval/harness.py` |
| C12 | Demonstrations collected under reduced randomization | Expert success 25% → 75%; policy is still *evaluated* under full randomization | `scripts/collect_dataset.py` |
| C13 | Parallel collection with shadows disabled | WSL has no GPU OpenGL; rendering is CPU-bound at ~100 ms/frame | `policy/parallel.py` |

## D — Open questions and assumptions

| ID | Item | Impact | Plan |
| --- | --- | --- | --- |
| D1 | Speechmatics API details unverified (docs unreachable from this environment) | B1 incomplete | Implement against the current docs when reachable; the architecture already routes speech into `TaskRequest` |
| D2 | Cultural table-setting conventions not yet source-checked | B6 credibility | Verify each style before it appears in the video; unverified styles are flagged in `place_setting.py` |
| D3 | No Intel Core Ultra machine available yet | A12, A15, A18 blocked for headline numbers | Benchmark harness runs anywhere; headline figures must be collected on the target |
| D4 | Whether SmolVLA converges on ~200 demonstrations | A11 | Fallback is ACT on the same dataset; both are wired |
| D5 | Which SmolVLA components convert to IR, and whether NPU accepts them | A12 | `openvino_export.py` reports per component; a documented "did not compile" is an acceptable result |

---

## Rubric traceability

| Rubric line | Points | Evidence |
| --- | --- | --- |
| End-to-end task completion & bimanual manipulation | 30 | `eval/tiers.py` tiers 1–6, `outputs/eval/results.json`; hand-off and hold-and-pour are planned from geometry, not scripted into the demo |
| VLA / multi-modal reasoning | 20 | Language → grounding → subgoal decomposition → policy; ambiguity reporting; `docs/RESULTS.md` grounding accuracy |
| Robustness & generalization | 15 | `sim/randomize.py` axes, tier 5, 10-seed run |
| OpenVINO & Core Ultra optimization | 20 | `scripts/benchmark_intel.py`, `docs/INTEL.md`, per-device/per-precision results |
| Technical quality & reproducibility | 10 | Seeded determinism, `verify` command, machine-readable results, this matrix |
| Innovation | 5 | Style-conditioned table setting; measurement-driven grasping (jaw model, torque screen, wrist refinement) |
