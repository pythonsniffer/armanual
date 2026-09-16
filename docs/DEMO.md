# Demonstration video: shot list and evidence map

The video is a technical artifact, not a trailer. Every claim in the README should have a shot a
judge can verify, and nothing should be shown in a way that makes a scripted step look autonomous
or hides a failure.

## Rules for this recording

1. **On-screen evidence beats narration.** The overlay (`armanual/eval/video.py`) shows the
   instruction, the current subgoal, which controller is driving, the inference device and the
   measured latency. If the overlay says `NPU 8.3 ms`, that is evidence; a voice-over saying the
   same thing is not.
2. **Show the seed.** Each episode's seed is on screen, and the same seed reproduces the episode
   from the results file.
3. **Do not cut failures out of the seed sweep.** The 10-seed section shows every seed, including
   the ones that fail, with the failure reason from the results file.
4. **No speed-ups without a label.** If a clip is accelerated, the overlay says so.

## What the recorded sweep actually shows

Ten seeds recorded with the final policy driving, full randomization, one instruction:
`"set the table and pour water into the blue cup"`.

```bash
CKPT=outputs/train/armanual_smolvla_v3/checkpoints/030000/pretrained_model
for s in 0 1 2 3 4 5 6 7 8 9; do
  python scripts/run_task.py --instruction "set the table and pour water into the blue cup" \
      --seed $s --randomize --policy $CKPT \
      --record outputs/videos/seed_$s.mp4 --json outputs/videos/seed_$s.json
done
```

| Subgoal | Succeeded |
| --- | --- |
| open the drawer | **4 / 10** |
| put the plate in the middle of the table | 0 / 10 |
| put the fork to the left of the plate | 0 / 10 |
| put the cup to the right of the plate | 0 / 10 |
| pour water into the blue cup | 0 / 10 |
| **all subgoals** | **4 / 50** |

Ten videos, ~54 s each. Rule 3 above applies: every seed is in the sweep, including the six where
the policy does not open the drawer either. The drawer successes are the honest highlight — a
contact-rich manipulation performed from pixels and a sentence, with no scripted fallback anywhere
in the loop — and the rest of the sweep is what a 617-episode dataset buys on a task that needs
millimetre placement. The matching per-seed JSON next to each video carries the subgoal record and
the inference latency that produced it.

## Shot list

| # | Shot | What it proves | Command |
| --- | --- | --- | --- |
| 1 | Scene tour: table, two SO-101 arms, drawer, cutlery inside it, bottle, cups with similar colours | The environment is a real dual-arm dinner-table scene with meaningful ambiguity | `python -m armanual.cli scene --seed 3 --randomize` |
| 2 | Typed instruction appears; subgoal decomposition prints | Language in, structured task out | `scripts/run_task.py --instruction "set the table and pour water into the blue cup"` |
| 3 | Perception overlay: detections with labels, positions and the drawer state | Perception runs on camera data, not simulator state | `scripts/eval_perception.py --seeds 1` |
| 4 | Drawer opens; cutlery becomes visible; the *next* plan now contains the fork | Closed loop and dependency handling — the plan grows because the world changed | same run as 2 |
| 5 | Two arms working in the same episode, each taking different objects | Dynamic arm assignment | same run as 2 |
| 6 | Hand-off across the table | Bimanual coordination that the geometry required | `--task t4_cross_table_handoff` |
| 7 | One arm steadies the cup while the other tips the bottle; liquid lands in the cup | Complementary dual-arm action, and a countable pour | `--task t4_pour_with_hold` |
| 8 | Same instruction, `--style japanese`: the layout changes | The style layer is a task constraint, not decoration | `scripts/run_task.py --style japanese` |
| 9 | Ambiguity: "the blue cup" with a navy cup present; the recorded ambiguity is shown | Grounding reports what it was torn between | `--task t2_blue_cup` |
| 10 | Ten randomized seeds, side by side, with the success/failure of each | Robustness, honestly | `scripts/evaluate.py --seeds 10 --out outputs/eval` |
| 11 | Policy vs scripted on the same seed, overlay showing which is driving | The VLA is actually in the loop | `scripts/run_task.py --policy <ckpt>` |
| 12 | `python -m armanual.cli devices` on the Core Ultra machine: CPU, GPU, NPU listed | The deployment target is the real thing | on the Intel machine |
| 13 | Benchmark table appearing: per component, per device, per precision | OpenVINO optimization measured, not asserted | `scripts/benchmark_intel.py --devices CPU GPU NPU` |
| 14 | Task success with PyTorch vs with OpenVINO on the same seeds | Optimization preserved behaviour | `scripts/evaluate.py --policy <ckpt> --ov-device NPU` |

## Rubric → shot map

| Rubric line | Shots |
| --- | --- |
| End-to-end task completion & bimanual manipulation (30) | 2, 4, 5, 6, 7, 10 |
| VLA / multi-modal reasoning (20) | 2, 3, 4, 9, 11 |
| Robustness & generalization (15) | 1, 10 |
| OpenVINO & Core Ultra optimization (20) | 12, 13, 14 |
| Technical quality & reproducibility (10) | every shot carries its command; results files are shown |
| Innovation (5) | 7, 8 |

## Recording

```bash
# One episode with overlay, mp4 out
python scripts/run_task.py --instruction "set the table and pour water into the blue cup" \
    --seed 3 --record outputs/videos/hero.mp4 --json outputs/videos/hero.json

# The 10-seed sweep, one video per seed
for s in 0 1 2 3 4 5 6 7 8 9; do
  python scripts/run_task.py --instruction "set the table" --seed $s --randomize \
      --record outputs/videos/seed_$s.mp4 --json outputs/videos/seed_$s.json
done
```

Each `--json` file holds the per-subgoal record — executor used, success, evidence, inference
latency — so the video and the numbers cannot drift apart.
