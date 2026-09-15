# Known limitations

Everything here is a real constraint of the current system, stated plainly. Where a number is
given it was measured; where something is unverified it says so.

## Physical and simulation

**The SO-101 is a small, weak arm, and the task was sized to it.** Each arm reaches about
0.14–0.34 m, and the shoulder actuator saturates at 2.94 N·m *inside* that kinematic envelope, so
poses that IK accepts can still sag about 2 cm. The IK screens for this, which is why some grasps
are reported unreachable rather than attempted. The table was sized to the measured workspace
(`docs/WORKSPACE.md`) rather than to a realistic dinner table.

**There is no liquid.** MuJoCo has no fluid phase. "Water" is twelve dense spheres inside the
bottle, and a pour succeeds when at least one ends up inside the cup. This is a deliberate,
countable proxy, not a fluid simulation — a real pour would spill differently and a real cup would
fill gradually.

**Pouring is calibrated, not derived.** The spill lands about 11 cm from the tool centre point
along the arm's radial direction when the wrist flexes ~1.2 rad. That offset was measured in
simulation and is used directly. Change the bottle geometry or the grasp height and it must be
re-measured.

**Bottle grasping is the least reliable skill.** Measured 3/4 across seeds with the body grasp,
after the neck grasp measured 0/4. It remains the most common cause of a failed pour.

## Perception

**The detector is classical, not learned.** Colour/geometry segmentation of an RGB-D point cloud:
recall 0.92, precision 0.84, mean position error 8.6 mm over 10 seeds. It degrades under colour
and lighting randomization far more than a learned detector would — which is why demonstrations
are collected with those axes fixed, and why tier-5 results are lower than tier 1–4.

**Some heuristics are scene-specific.** Robot pixels are removed by the arm's yellow chromaticity;
objects above 17 cm are assumed to be an arm rather than tableware; the open drawer is read as a
height slab in a known x-band. These are honest engineering for this scene, not general perception.

**The overhead camera is authoritative.** A shallow front view merges nearby objects, so it may
only *add* detections, never overwrite. Objects hidden from both views are simply not seen.

## Language

**The parser is deterministic, not a language model.** It handles the instruction forms listed in
`task/parser.py` — synonyms, adjectives, spatial relations, two-clause instructions and pronoun
reference. It will not handle free-form paraphrase outside that grammar, and says so (the episode
record contains the unparsed clause) rather than guessing.

**Grounding decides what the words refer to, not the parser.** That part is vision-driven and does
generalize across scenes. Ambiguity is reported when the runner-up scores within 0.15.

## Policy

**The policy operates on subgoals, not the whole task.** One instruction is decomposed into three
or four language subgoals, each executed by the policy. A single policy trained end-to-end on this
amount of data would not complete the full workflow; this is stated in the results rather than
implied away.

**Demonstrations come from the analytical expert**, so the policy inherits its failure modes and
cannot exceed it on skills where the expert is weak. It is trained on successful episodes only.

**Collection randomization is narrower than evaluation randomization.** Deliberate: the expert's
success rate falls from ~75% to ~25% with lighting, friction and clutter enabled, which would
shrink the dataset threefold. The consequence is a visual-domain gap that tier-5 numbers expose.

## Intel deployment

**Headline Intel numbers require the Core Ultra target.** The benchmark harness runs anywhere and
reports what it finds, but this development machine is an AMD Ryzen with an NVIDIA GPU. Any
number collected here is labelled a development baseline, and `host_info()` records
`is_intel_core_ultra: false` in every results file.

**A VLA does not convert to one IR.** The vision tower and action expert convert; the language
model with its KV cache and dynamic shapes does not, and the NPU requires static shapes. The
exporter converts per component and reports failures. Expect a heterogeneous placement — some
components on an Intel device, the rest on CPU — rather than "the VLA runs on the NPU".

**No training on Intel.** OpenVINO is an inference runtime and the NPU has no training path.
Training runs on CUDA, which the challenge explicitly permits.

## Environment

**No GPU OpenGL under WSL.** MuJoCo renders on the CPU (llvmpipe) at ~100 ms per 224×224 frame
with shadows off, ~285 ms with them on. This is why dataset collection is parallelized across
processes and why evaluation wall-clock is dominated by rendering rather than physics.

## Carried-over gaps

| Gap | Status |
| --- | --- |
| G1 Speechmatics API details unverified | open — architecture ready, client not written |
| G2 Intel hackathon stack unverified | partially closed — OpenVINO 2026.3 installed and working |
| G3 No dual-SO-101 dinner-table environment existed | closed — built |
| G4 No LeRobot env for this task | closed by design — LeRobot format for data/training, our own harness for closed-loop eval |
| G5 NPU viability for a VLA unknown | open — the exporter will answer it empirically |
| G6 MuJoCo has no liquids | closed by decision — particle proxy, documented above |
| G7 Cultural conventions unsourced | open — styles implemented, sources to verify before the video |
| G8 Intel Core Ultra target not accessible | open — blocks headline Intel numbers |
