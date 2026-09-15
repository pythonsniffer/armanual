"""Runtime wrapper around a compiled dinner-table scene.

:class:`World` owns the ``MjModel``/``MjData`` pair and exposes the only operations the rest of
the system is allowed to perform on the simulator: stepping, commanding arm targets, reading
object poses, and rendering cameras. Everything above this layer (grounding, planning, policy)
talks in named objects and arm names, never in raw MuJoCo indices — which is what keeps the
policy swappable and the scene randomizable.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from armanual.sim import render as render_backend
from armanual.sim.builder import (
    ARM_JOINTS,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    TCP_SITE,
    build_model,
)
from armanual.sim.spec import SceneSpec


@dataclass(frozen=True)
class ArmHandle:
    """Cached MuJoCo indices for one arm. Built once at model compile time."""

    name: str
    actuator_ids: np.ndarray
    joint_ids: np.ndarray
    qpos_adr: np.ndarray
    qvel_adr: np.ndarray
    tcp_site: int
    base_body: int
    gripper_body: int
    wrist_cam: int

    @property
    def arm_actuators(self) -> np.ndarray:
        """The five positioning actuators (excludes the gripper)."""
        return self.actuator_ids[:-1]

    @property
    def gripper_actuator(self) -> int:
        return int(self.actuator_ids[-1])


class World:
    """A compiled, steppable dinner-table episode."""

    def __init__(self, scene: SceneSpec, *, render_size: tuple[int, int] = (320, 240)):
        render_backend.select_backend()
        self.scene = scene
        self.model, self.spec = build_model(scene)
        self.data = mujoco.MjData(self.model)
        self.render_size = render_size
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._scratch: mujoco.MjData | None = None
        self._arm_geoms: dict[str, set[int]] = {}
        self.arms: dict[str, ArmHandle] = {a.name: self._arm_handle(a.name) for a in scene.arms}
        self.reset()

    # ---------------------------------------------------------------- construction helpers
    def _id(self, objtype: mujoco.mjtObj, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, objtype, name)
        if idx < 0:
            raise KeyError(f"{objtype.name} {name!r} not found in compiled model")
        return idx

    def _arm_handle(self, arm: str) -> ArmHandle:
        act = np.array([self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}/{j}") for j in ARM_JOINTS])
        jnt = np.array([self._id(mujoco.mjtObj.mjOBJ_JOINT, f"{arm}/{j}") for j in ARM_JOINTS])
        return ArmHandle(
            name=arm,
            actuator_ids=act,
            joint_ids=jnt,
            qpos_adr=np.array([self.model.jnt_qposadr[j] for j in jnt]),
            qvel_adr=np.array([self.model.jnt_dofadr[j] for j in jnt]),
            tcp_site=self._id(mujoco.mjtObj.mjOBJ_SITE, f"{arm}/{TCP_SITE}"),
            base_body=self._id(mujoco.mjtObj.mjOBJ_BODY, f"{arm}/base"),
            gripper_body=self._id(mujoco.mjtObj.mjOBJ_BODY, f"{arm}/gripper"),
            wrist_cam=self._id(mujoco.mjtObj.mjOBJ_CAMERA, f"{arm}/wrist_cam"),
        )

    def arm_geom_ids(self, arm: str) -> set[int]:
        """Geom ids belonging to one arm's kinematic subtree (cached)."""
        if arm not in self._arm_geoms:
            root = self.arms[arm].base_body
            bodies = {root}
            for body in range(self.model.nbody):
                parent = body
                while parent > 0:
                    if parent == root:
                        bodies.add(body)
                        break
                    parent = self.model.body_parentid[parent]
            self._arm_geoms[arm] = {
                g
                for body in bodies
                for g in range(
                    self.model.body_geomadr[body],
                    self.model.body_geomadr[body] + self.model.body_geomnum[body],
                )
            }
        return self._arm_geoms[arm]

    def object_geom_ids(self, obj_name: str) -> set[int]:
        body = self._id(mujoco.mjtObj.mjOBJ_BODY, f"obj_{obj_name}")
        return set(
            range(
                self.model.body_geomadr[body],
                self.model.body_geomadr[body] + self.model.body_geomnum[body],
            )
        )

    def scratch_data(self) -> mujoco.MjData:
        """A reusable spare ``MjData`` for kinematics/dynamics queries off the main state."""
        if self._scratch is None:
            self._scratch = mujoco.MjData(self.model)
        return self._scratch

    # ------------------------------------------------------------------------ episode state
    def reset(self, settle_seconds: float = 0.6) -> None:
        """Reset to the scene's home configuration and let objects settle on the table."""
        mujoco.mj_resetData(self.model, self.data)
        for arm_spec in self.scene.arms:
            handle = self.arms[arm_spec.name]
            home = np.asarray(arm_spec.home_qpos, dtype=float)
            self.data.qpos[handle.qpos_adr] = home
            self.data.ctrl[handle.actuator_ids] = home
        mujoco.mj_forward(self.model, self.data)
        self.step_for(settle_seconds)

    @property
    def time(self) -> float:
        return float(self.data.time)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def step_for(self, seconds: float) -> None:
        self.step(max(1, int(round(seconds / self.model.opt.timestep))))

    # ----------------------------------------------------------------------------- commands
    def set_arm_target(self, arm: str, qpos: np.ndarray) -> None:
        """Command the five positioning joints of ``arm`` (gripper untouched)."""
        handle = self.arms[arm]
        self.data.ctrl[handle.arm_actuators] = np.asarray(qpos, dtype=float)[:5]

    def set_gripper(self, arm: str, opening: float) -> None:
        """``opening`` in [0, 1]: 0 fully closed, 1 fully open."""
        handle = self.arms[arm]
        value = GRIPPER_CLOSED + float(np.clip(opening, 0.0, 1.0)) * (GRIPPER_OPEN - GRIPPER_CLOSED)
        self.data.ctrl[handle.gripper_actuator] = value

    def gripper_opening(self, arm: str) -> float:
        handle = self.arms[arm]
        q = float(self.data.qpos[handle.qpos_adr[-1]])
        return float(np.clip((q - GRIPPER_CLOSED) / (GRIPPER_OPEN - GRIPPER_CLOSED), 0.0, 1.0))

    # -------------------------------------------------------------------------- observation
    def arm_qpos(self, arm: str) -> np.ndarray:
        return self.data.qpos[self.arms[arm].qpos_adr].copy()

    def tcp_pos(self, arm: str) -> np.ndarray:
        return self.data.site_xpos[self.arms[arm].tcp_site].copy()

    def tcp_mat(self, arm: str) -> np.ndarray:
        return self.data.site_xmat[self.arms[arm].tcp_site].reshape(3, 3).copy()

    def base_pos(self, arm: str) -> np.ndarray:
        return self.data.xpos[self.arms[arm].base_body].copy()

    def object_pos(self, name: str) -> np.ndarray:
        return self.data.xpos[self._id(mujoco.mjtObj.mjOBJ_BODY, f"obj_{name}")].copy()

    def object_quat(self, name: str) -> np.ndarray:
        return self.data.xquat[self._id(mujoco.mjtObj.mjOBJ_BODY, f"obj_{name}")].copy()

    def object_vel(self, name: str) -> np.ndarray:
        body = self._id(mujoco.mjtObj.mjOBJ_BODY, f"obj_{name}")
        return self.data.cvel[body][3:].copy()

    def set_object_pose(self, name: str, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        """Teleport an object. Used only by scene setup/randomization, never mid-episode."""
        joint = self._id(mujoco.mjtObj.mjOBJ_JOINT, f"free_{name}")
        adr = self.model.jnt_qposadr[joint]
        self.data.qpos[adr : adr + 3] = np.asarray(pos, dtype=float)
        self.data.qpos[adr + 3 : adr + 7] = np.asarray(quat, dtype=float)
        dof = self.model.jnt_dofadr[joint]
        self.data.qvel[dof : dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def drawer_opening(self) -> float:
        """Drawer extension in metres, 0 when fully closed."""
        joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
        if joint < 0:
            return 0.0
        return float(self.data.qpos[self.model.jnt_qposadr[joint]])

    def site_pos(self, name: str) -> np.ndarray:
        return self.data.site_xpos[self._id(mujoco.mjtObj.mjOBJ_SITE, name)].copy()

    def water_positions(self) -> np.ndarray:
        """World positions of the liquid particles (empty array if the scene has no bottle)."""
        out = []
        for i in range(1000):
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"water_{i}")
            if body < 0:
                break
            out.append(self.data.xpos[body].copy())
        return np.asarray(out) if out else np.empty((0, 3))

    # ------------------------------------------------------------------------------ cameras
    def render(self, camera: str = "cam_overhead", size: tuple[int, int] | None = None) -> np.ndarray:
        """Render one camera to an RGB uint8 array (H, W, 3)."""
        width, height = size or self.render_size
        key = (width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        renderer = self._renderers[key]
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()

    def observation(self, cameras=("cam_overhead", "cam_front")) -> dict[str, np.ndarray]:
        """The observation dict handed to perception / the policy."""
        obs: dict[str, np.ndarray] = {f"image.{c}": self.render(c) for c in cameras}
        for arm in self.arms:
            obs[f"state.{arm}"] = self.arm_qpos(arm)
            obs[f"tcp.{arm}"] = self.tcp_pos(arm)
        return obs

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    def __enter__(self) -> "World":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
