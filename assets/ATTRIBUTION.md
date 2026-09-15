# Third-party assets

## SO-101 arm model (`assets/so101/`)

- **Source**: [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie),
  package `robotstudio_so101`, commit `8161bba264d7fa7c99ca301e91e7fb44737676ad` (retrieved 2026-09-15).
- **Upstream origin**: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
  `Simulation/SO101/so101_new_calib.xml`, adapted by Menagerie (see `so101/UPSTREAM_README.md`).
- **License**: Apache-2.0 — full text at `so101/LICENSE`.
- **Modifications by this project**: none to the vendored files. The bimanual scene composes
  `so101.xml` by MJCF `<include>`/attach from `src/armanual/sim/`; the file itself is unmodified so
  it can be diffed against upstream.
