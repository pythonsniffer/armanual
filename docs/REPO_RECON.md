# Phase 0 — Repository Reconnaissance

Date: 2026-09-15

## 1. Repository state at start

| Item | Finding |
| --- | --- |
| Repo | `git@github.com:pythonsniffer/armanual.git`, branch `main`, **zero commits** |
| Existing implementation | None. Greenfield. |
| Prior work available | `.claude/skills/` (8 project skills) and `docs/SKILLS.md` in the parent workspace — skill/discovery phase already completed |
| Source of truth | `problem_statement` (10-page PDF, pages 1–5 text-readable), `prompt` (project brief) |

No destructive rewrite risk: nothing pre-existed.

## 2. Development machine (training / development)

```
CPU    : AMD Ryzen 7 250 w/ Radeon 780M Graphics (16 threads)
RAM    : 15 GiB
OS     : Linux 6.6.87.2-microsoft-standard-WSL2 (Ubuntu under WSL2)
Python : 3.10.12 (system), venv at .venv
torch  : 2.13.0+cpu — no CUDA, no XPU
```

Implications:
- **No GPU for training.** Any imitation-learning run must be CPU-feasible (small policy, small
  dataset, low action dimensionality) or moved to a cloud instance. The challenge explicitly
  permits non-Intel / cloud training hardware.
- **This machine is not the deployment target.** It is AMD; the final demonstration must run on
  Intel Core Ultra Series 2/3. All Intel-specific code stays behind a device abstraction so the
  same pipeline runs here (CPU) and there (CPU/iGPU/NPU).
- **WSL2 + headless**: MuJoCo rendering must use the offscreen/EGL or osmesa path, not a GUI viewer.

## 3. Simulation assets

| Asset | Source | License | Status |
| --- | --- | --- | --- |
| SO-101 MJCF + STL meshes | `google-deepmind/mujoco_menagerie` → `robotstudio_so101` | Apache-2.0 | Vendored to `assets/so101/` |

Verified model facts (read from the MJCF, not from memory):

- 6 joints / 6 position actuators per arm:
  `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`
- Gripper joint range `[-0.174533, 1.7453292]` rad (positive = open).
- End-effector reference site: `gripperframe`. Base site: `baseframe`.
- Wrist camera `wrist_cam` already defined on the gripper camera mount (1920×1080, focal 3.6 mm).
- Contact tuning for grasping already present: `condim=6`, `friction="1 5e-3 5e-4"`,
  `solref="0.01 1"`, `priority=1` on gripper collision geoms; `cone="elliptic"`, `impratio=10`.
- Default timestep `0.005 s` (200 Hz), integrator `implicitfast`.

MuJoCo version installed and verified: **3.13.0**.

## 4. Toolchain

| Tool | Version | Notes |
| --- | --- | --- |
| mujoco | 3.13.0 | verified `import mujoco` |
| numpy | installed in `.venv` | |
| torch | 2.13.0+cpu | system-level, CPU-only |
| OpenVINO | not installed yet | Phase 8; install on the Intel target |
| LeRobot | not installed yet | Phase 2/4; pulled when the dataset/policy work starts |
| Speechmatics SDK | not installed yet | Phase 5; API details unverified (gap G1) |

## 5. Carried-over gaps from the skill-discovery phase

G1 Speechmatics API unverified · G2 Intel hackathon stack unverified · G3 no dual-SO-101
dinner-table scene exists · G4 no LeRobot env for this task · G5 NPU viability for a VLA unknown ·
G6 MuJoCo has no liquids (pouring) · G7 cultural table-setting styles need sources ·
G8 Intel Core Ultra target not yet accessible.

These are tracked in [LIMITATIONS.md](LIMITATIONS.md) and closed as evidence arrives.

## 6. Decisions taken from this recon

1. Vendor the SO-101 assets into the repo (Apache-2.0, attribution kept) — judges must be able to
   reproduce without a second clone.
2. Build the bimanual scene **programmatically** from the single-arm MJCF rather than hand-editing
   a giant XML: two arm instances + table + objects, composed by a scene builder. Randomization then
   becomes a function of a seed, not of hand-edited XML.
3. Headless rendering by default; interactive viewer optional and never required by a test.
