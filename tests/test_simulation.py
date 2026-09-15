"""Scene, randomization and kinematics tests.

These build real MuJoCo models, so they are the slow tests in the suite — but they are the ones
that catch the failures that matter: a scene that spawns objects inside furniture, an IK solution
the arm cannot physically hold, a grasp model that drifts from the gripper it describes.
"""

from __future__ import annotations

import numpy as np
import pytest

from armanual.control.gripper import FIXED_JAW, moving_jaw_z, opening_for_width, pinch_offset
from armanual.control.kinematics import holdable, solve_reach
from armanual.sim.randomize import RandomizationConfig, sample_scene
from armanual.sim.world import World


@pytest.fixture(scope="module")
def world():
    w = World(sample_scene(0, RandomizationConfig.placement_only()), fast_render=True)
    yield w
    w.close()


class TestScene:
    def test_two_arms_with_six_actuators_each(self, world):
        assert set(world.arms) == {"left", "right"}
        assert world.model.nu == 12

    def test_cameras_exist(self, world):
        import mujoco

        names = {
            mujoco.mj_id2name(world.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(world.model.ncam)
        }
        assert {"cam_overhead", "cam_front", "left/wrist_cam", "right/wrist_cam"} <= names

    def test_randomization_is_deterministic(self):
        a = sample_scene(7, RandomizationConfig())
        b = sample_scene(7, RandomizationConfig())
        assert a.to_json() == b.to_json()

    def test_different_seeds_differ(self):
        a = sample_scene(1, RandomizationConfig())
        b = sample_scene(2, RandomizationConfig())
        assert a.to_json() != b.to_json()

    def test_objects_do_not_start_inside_each_other(self):
        for seed in range(3):
            w = World(sample_scene(seed, RandomizationConfig()), fast_render=True)
            w.step_for(0.5)
            deep = sum(1 for i in range(w.data.ncon) if w.data.contact[i].dist < -0.002)
            w.close()
            assert deep == 0, f"seed {seed} starts with {deep} interpenetrating contacts"

    def test_table_objects_are_within_reach_of_an_arm(self):
        from armanual.sim.randomize import REACH_MAX

        scene = sample_scene(3, RandomizationConfig())

        def nearest_base(position) -> float:
            return min(
                float(np.linalg.norm(np.array(position[:2]) - np.array(arm.base_pos[:2])))
                for arm in scene.arms
            )

        for obj in scene.objects:
            if obj.name in scene.drawer.contents:
                # Drawer contents are *deliberately* out of reach while the drawer is shut; that
                # is the dependency the task is built around. They must come into reach once the
                # drawer has been pulled out, or the task would be impossible rather than hard.
                opened = (obj.pos[0], obj.pos[1] - scene.drawer.travel)
                assert nearest_base(opened) <= REACH_MAX + 0.06, (
                    f"{obj.name} is still unreachable with the drawer open"
                )
                continue
            assert nearest_base(obj.pos) <= REACH_MAX + 0.06, (
                f"{obj.name} is beyond either arm's reach"
            )


class TestGripperModel:
    """The jaw model is measured from the compiled model; this keeps it honest."""

    def test_moving_jaw_matches_the_model_in_its_valid_range(self, world):
        import mujoco

        handle = world.arms["left"]
        gid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_GEOM, "left/moving_jaw_sph_tip1")
        for opening in (0.1, 0.2, 0.3, 0.4):
            world.data.qpos[handle.qpos_adr[:5]] = [0.0, -0.6, 0.8, 0.6, 0.3]
            world.set_gripper("left", opening)
            world.data.qpos[handle.qpos_adr[5]] = world.data.ctrl[handle.gripper_actuator]
            mujoco.mj_kinematics(world.model, world.data)
            rot = world.tcp_mat("left")
            local = rot.T @ (world.data.geom_xpos[gid] - world.tcp_pos("left"))
            assert abs(local[2] - moving_jaw_z(opening)) < 0.002

    def test_pinch_offset_places_the_object_against_the_fixed_jaw(self):
        offset = pinch_offset(0.02, seat=0.0)
        assert offset[2] == pytest.approx(FIXED_JAW[2] + 0.01, abs=1e-6)

    def test_aperture_grows_with_requested_width(self):
        assert opening_for_width(0.05) > opening_for_width(0.01)


class TestKinematics:
    def test_reachable_target_solves(self, world):
        solution = solve_reach(world, "left", np.array([-0.15, -0.14, 0.06]))
        assert solution.feasible
        assert solution.ik.pos_error < 0.006

    def test_far_target_is_reported_infeasible(self, world):
        solution = solve_reach(world, "left", np.array([0.55, 0.30, 0.06]))
        assert not solution.feasible

    def test_accepted_solutions_are_holdable(self, world):
        solution = solve_reach(world, "right", np.array([0.14, -0.12, 0.05]))
        assert solution.feasible
        assert holdable(world, "right", solution.qpos)

    def test_solution_forward_kinematics_matches_target(self, world):
        import mujoco

        target = np.array([-0.10, -0.16, 0.05])
        solution = solve_reach(world, "left", target)
        handle = world.arms["left"]
        scratch = mujoco.MjData(world.model)
        scratch.qpos[handle.qpos_adr[:5]] = solution.qpos
        mujoco.mj_kinematics(world.model, scratch)
        reached = scratch.site_xpos[handle.tcp_site]
        assert float(np.linalg.norm(reached - target)) < 0.006
