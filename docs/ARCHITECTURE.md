# Architecture

## The shape of the system

```
                 ┌──────────────┐   ┌───────────────┐   ┌──────────────────┐
  text ─────────▶│              │   │               │   │                  │
  speech ───────▶│   parser     ├──▶│   grounder    ├──▶│  subgoal planner │
  pointing ─────▶│ (task/parser)│   │(task/grounding│   │ (policy/tasks)   │
                 └──────────────┘   └───────┬───────┘   └────────┬─────────┘
                   TaskRequest              │  GroundedTask      │ sentences
                                            │                    │
                 ┌──────────────────────────┴────────────────────▼─────────┐
                 │                    execution                            │
                 │   ┌───────────────┐            ┌────────────────────┐   │
                 │   │  VLA policy   │    or      │ analytical control │   │
                 │   │ (SmolVLA/ACT) │  fallback  │ IK + primitives    │   │
                 │   └───────┬───────┘            └─────────┬──────────┘   │
                 └───────────┼──────────────────────────────┼──────────────┘
                             │ 12-dim bimanual actions      │
                 ┌───────────▼──────────────────────────────▼──────────────┐
                 │          MuJoCo: two SO-101 arms, dinner table          │
                 └───────────┬─────────────────────────────────────────────┘
                             │ RGB-D from 3 cameras
                 ┌───────────▼──────────────┐
                 │  perception (detector)   │──────────────┐
                 └──────────────────────────┘   detections │
                             ▲                             │
                             └─────────────────────────────┘
                                   closed loop
```

Every arrow is a plain dataclass defined in `task/schema.py` or `planning/planner.py`, and every
stage is separately testable. The loop closes twice: inside a skill (the controller re-observes
between waypoints, and the wrist camera refines a grasp at close range) and between subgoals (a
full re-observation, re-grounding and re-plan).

## Layers

### 1. Simulation — `armanual/sim/`

Two SO-101 arms from MuJoCo Menagerie, attached programmatically to a generated dinner-table
scene. The scene is a pure function of a `SceneSpec`, and the spec is a pure function of a seed
plus a `RandomizationConfig`, so an episode replays exactly from its results file.

Objects exist to test capabilities, not to fill space: plates and cups for placement, two
similar-coloured cups for disambiguation, cutlery inside a drawer for dependency, a bottle and
particle "water" for pouring, a tray with two handles for two-arm carrying.

Key measured facts (see [WORKSPACE.md](WORKSPACE.md)): each arm reaches roughly 0.14–0.34 m, the
gripper spans 4–133 mm, and the tool site sits ~20 mm from the jaw line — which is why grasps are
specified as *where the object goes between the jaws*, not as a tool pose.

### 2. Perception — `armanual/perception/`

RGB-D from the simulated cameras, back-projected with each camera's real intrinsics, segmented
above the table plane, and labelled by measured geometry. **No simulator state is read.** A
`PrivilegedObserver` exists for ablations and is always tagged in results.

Measured against ground truth over 10 seeds: recall 0.92, precision 0.84, mean position error
8.6 mm ([RESULTS.md](RESULTS.md)).

Two refinements earn their place:
- **Multi-camera fusion** — the overhead view is authoritative; the front view only *adds* objects
  an arm was standing over.
- **Wrist-camera refinement** — at the pre-grasp hover the object fills the wrist view, so
  centroid-style grasps re-localize there before descending.

### 3. Language and grounding — `armanual/task/`

A deterministic parser turns an instruction into a `TaskRequest`; the grounder scores every
detection against each referent (category, colour, size, spatial relation, ordinal, pointing) and
binds it. Ambiguity is an output, not an error: when the runner-up scores within 0.15, the episode
records what it was torn between.

Speech and pointing fill the *same* `TaskRequest`, so there is exactly one task-understanding
pathway.

### 4. Planning — `armanual/planning/`

- **Decomposition**: preconditions are checked against the observation, so the drawer step exists
  only while the drawer is shut.
- **Dynamic arm assignment**: each arm is costed by whether it can reach the pick *and* the place,
  using the same torque- and collision-screened IK the controller uses. No object class is tied to
  an arm.
- **Hand-off insertion**: when one arm can reach the object and the other the destination, a
  hand-off through the shared middle band is planned — because the geometry requires it.
- **Style layer**: `place_setting.py` turns a named style into slot coordinates, so "set the table
  in the japanese style" changes where things go, not just what is said about them.

### 5. Control — `armanual/control/`

Damped-least-squares IK for a 5-DOF arm, with two screens that turn silent failures into
reportable ones: a **static torque** check (the shoulder saturates at 2.94 N·m inside the
kinematic workspace) and a **collision** check against the real scene, including the other arm.

Skills are generators advanced by a cooperative scheduler, so both arms can run different skills
in the same tick — which is what makes hold-while-pouring and hand-offs real rather than
interleaved scripts.

### 6. Policy — `armanual/policy/`

| Piece | What it does |
| --- | --- |
| `collect.py` / `parallel.py` | Records bimanual demonstrations by running the analytical stack on generated instructions; 10 worker processes because MuJoCo renders on the CPU under WSL |
| `lerobot_export.py` | Writes a LeRobotDataset (3 cameras, 12-dim state and action, instruction as the task string) |
| `tasks.py` | The skill vocabulary and the decomposition from one instruction into language subgoals |
| `runtime.py` | `LeRobotBackend` (PyTorch), `OpenVINOBackend` (hybrid IR/torch), and the runner that executes action chunks at 20 Hz |
| `openvino_export.py` | Per-component conversion to IR with optional NNCF INT8 |
| `benchmark.py` | Latency/throughput measurement with warm-up separated and failed devices reported |

**Why a subgoal-level policy rather than one end-to-end policy.** A single network trained on a
few hundred demonstrations will not perform a minute-long, multi-object, dependency-laden task.
Decomposing into language subgoals keeps the instruction end-to-end from the user's side while
giving each skill enough data to learn. The decomposition is explicit and logged, so what the
policy was asked to do at each moment is always visible.

**Why 12-dimensional actions.** Recording only the moving arm would make hand-offs and
hold-and-pour structurally unlearnable, since those are exactly the moments when one arm's correct
action depends on what the other is doing.

### 7. Evaluation — `armanual/eval/`

Six benchmark tiers, each adding one capability over the tier below. Scoring reads the **final
state of the table**, not the controller's log. Every failure is attributed to a stage —
perception, grounding, planning, reachability, grasp, placement, coordination or timeout — and
results are written as JSON and CSV with the host environment attached.

## Intel deployment

Training runs on CUDA (the challenge permits any training hardware); **inference** is what must run
on Intel Core Ultra. The policy sits behind `PolicyBackend`, so the same evaluation runs with the
PyTorch or the OpenVINO backend and produces directly comparable success rates and latencies.

A VLA does not convert to a single IR: the vision tower and action expert convert cleanly, while
the language model's KV cache and dynamic shapes do not — and NPU requires static shapes. The
exporter therefore converts per component and records what failed, and the benchmark reports which
component ran on which device. See [INTEL.md](INTEL.md).
