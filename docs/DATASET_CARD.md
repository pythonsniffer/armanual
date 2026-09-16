---
license: apache-2.0
task_categories:
  - robotics
tags:
  - LeRobot
  - so101
  - bimanual
  - manipulation
  - mujoco
  - dinner-table
configs:
  - config_name: default
    data_files: data/*/*.parquet
---

# armanual — bimanual SO-101 dinner-table demonstrations

Demonstrations for **Bimanual VLA Manipulation with Multi-Modal Reasoning** (Intel Physical AI
Online Challenge). Two simulated SO-101 arms set a dinner table in MuJoCo: placing plates, cups
and cutlery, opening a drawer to reach the cutlery, handing objects between arms, and pouring.

Code: https://github.com/pythonsniffer/armanual

## What is in it

| | |
| --- | --- |
| Episodes | 307 |
| Frames | 57,844 (20 Hz) |
| Distinct instructions | 32 |
| Robot | dual SO-101 (`so101_bimanual`) |
| Format | LeRobotDataset v3.0, video-encoded |

### Features

| Column | Shape | Meaning |
| --- | --- | --- |
| `observation.images.top` | 224×224×3 | overhead camera |
| `observation.images.left_wrist` | 224×224×3 | left arm wrist camera |
| `observation.images.right_wrist` | 224×224×3 | right arm wrist camera |
| `observation.state` | 12 | 5 joints + gripper opening, per arm (left first) |
| `action` | 12 | commanded joint targets, same layout |
| `task` | text | the instruction sentence that produced the episode |

**Actions are bimanual.** Both arms are recorded at every step, not just the moving one — hand-offs
and hold-while-pouring depend on what the other arm is doing, and recording a single arm makes
those behaviours structurally unlearnable.

## Skills

| Skill | Episodes | Example instruction |
| --- | --- | --- |
| `place_object` | 87 | "put the blue cup to the right of the plate" |
| `handoff` | 14 | "pass the cup to the other arm" |
| `open_drawer` | 13 | "open the cutlery drawer" |
| `pour` | 0 | (the expert's pour succeeded too rarely to contribute demonstrations) |

Each skill has several phrasings so a policy keys on meaning rather than a memorized string.

## How it was generated

Episodes are produced by running an analytical controller — camera-based perception, referent
grounding, torque- and collision-screened IK, scripted manipulation primitives — on generated
instructions, recording every control tick. The instruction that produced an episode is its label.

**Only successful episodes are included.** Success is checked against the final state of the
table (was the object actually placed where the sentence said?), not against the controller's own
opinion. The expert succeeds about 55% of the time; the failures are discarded.

Randomization during collection: object placement, size and mass. Lighting, friction, colour and
clutter were held fixed, because enabling them drops the expert's success rate to ~25% and would
have shrunk the dataset threefold. Policies trained on this data are *evaluated* under the full
randomization, which is where generalization should be measured.

## Known limitations

- **No pour demonstrations.** The expert's bottle grasp is its weakest skill; every pour attempt
  in the collection run failed its success check and was discarded.
- **Visual domain is narrower than the evaluation domain** (see above).
- The demonstrations inherit the expert's failure modes; a policy trained on them should not be
  expected to exceed it on skills where the expert is weak.

## Reproducing it

```bash
git clone https://github.com/pythonsniffer/armanual && cd armanual
pip install -e ".[dev]" && pip install "lerobot[smolvla]"
python scripts/collect_dataset.py --episodes-per-skill 60 --workers 6

# a second, targeted pass for a skill that came out thin, merged in afterwards
python scripts/collect_dataset.py --episodes-per-skill 110 --workers 6 \
    --skills place_object --easy --root data/armanual-extra-a \
    --repo-id pythonsniffer/armanual-dinner-table-extra-a
python scripts/merge_datasets.py --source data/armanual-extra-a \
    --target data/armanual-dinner-table \
    --source-repo-id pythonsniffer/armanual-dinner-table-extra-a
```

## Licence

Apache-2.0. The SO-101 model is from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Apache-2.0).
