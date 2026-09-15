"""Turn a :class:`~armanual.sim.spec.SceneSpec` into a compiled MuJoCo model.

The dual-arm scene is assembled programmatically rather than hand-written as one large MJCF:
the world (table, drawer, objects, cameras, lights) is generated as XML from the spec, and the
two SO-101 arms are attached with ``MjSpec.attach`` from the *unmodified* vendored
``assets/so101/so101.xml``. Two consequences matter:

* the upstream arm model stays diffable against Menagerie, and
* randomization is a pure function of the spec, so an episode replays exactly.

Every element that a downstream module needs to address is given a stable name:
``<arm>/`` prefix for arm bodies/joints/actuators (e.g. ``left/gripper``), ``obj_<name>`` for
object bodies, ``cam_overhead`` / ``cam_front`` for scene cameras.
"""

from __future__ import annotations

import math
import warnings

import mujoco
import numpy as np

from armanual.paths import SO101_XML
from armanual.sim.spec import ObjectSpec, SceneSpec

#: Joint names of one SO-101 arm, in actuator order. Read from the vendored MJCF.
ARM_JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
GRIPPER_JOINT = "gripper"
#: Gripper actuator control limits from the vendored MJCF (positive = open).
GRIPPER_CLOSED, GRIPPER_OPEN = -0.17453, 1.74533
#: Site that marks the tool centre point of an arm.
TCP_SITE = "gripperframe"

WATER_PARTICLES = 12
WATER_RADIUS = 0.0055


def _q(yaw: float) -> str:
    return f"{math.cos(yaw / 2):.6f} 0 0 {math.sin(yaw / 2):.6f}"


def _rgba(v) -> str:
    return " ".join(f"{c:.3f}" for c in v)


def _object_xml(obj: ObjectSpec, scale_friction: float) -> str:
    """MJCF for one manipulable object.

    Shapes are deliberately simple primitives: they keep contact solving fast and stable, and
    every grasp affordance stays explicit. Utensils get a raised cylindrical handle because a
    flat 3 mm plate cannot be grasped off a table by the SO-101 jaws.
    """
    fr = " ".join(f"{f * s:.5f}" for f, s in zip(obj.friction, (scale_friction, 1.0, 1.0)))
    common = (
        f'material="mat_{obj.name}" friction="{fr}" condim="4" '
        f'solref="0.008 1" solimp="0.95 0.99 0.001" priority="0"'
    )
    sx, sy, sz = obj.size
    body_open = (
        f'<body name="obj_{obj.name}" pos="{obj.pos[0]:.4f} {obj.pos[1]:.4f} {obj.pos[2]:.4f}" '
        f'quat="{_q(obj.yaw)}">\n'
        f'  <freejoint name="free_{obj.name}"/>\n'
        f'  <site name="site_{obj.name}" size="0.004" rgba="0 0 0 0"/>\n'
    )
    parts: list[str] = []

    if obj.category == "plate":
        parts.append(
            f'<geom name="g_{obj.name}" type="cylinder" size="{sx:.4f} {sz:.4f}" '
            f'mass="{obj.mass:.4f}" {common}/>'
        )
        # Low rim: makes "on the plate" checkable and gives the gripper an edge to pinch.
        for i in range(12):
            a = 2 * math.pi * i / 12
            parts.append(
                f'<geom name="g_{obj.name}_rim{i}" type="box" '
                f'size="{sx * 0.28:.4f} 0.004 {sz * 1.6:.4f}" mass="0.001" '
                f'pos="{sx * math.cos(a):.4f} {sx * math.sin(a):.4f} {sz:.4f}" '
                f'quat="{_q(a + math.pi / 2)}" {common}/>'
            )
    elif obj.category in ("cup", "bottle"):
        wall_h = sz
        parts.append(
            f'<geom name="g_{obj.name}_base" type="cylinder" size="{sx:.4f} 0.004" '
            f'pos="0 0 0.004" mass="{obj.mass * 0.45:.4f}" {common}/>'
        )
        n_wall = 10
        for i in range(n_wall):
            a = 2 * math.pi * i / n_wall
            parts.append(
                f'<geom name="g_{obj.name}_w{i}" type="box" '
                f'size="{sx * math.tan(math.pi / n_wall):.4f} 0.0035 {wall_h:.4f}" '
                f'pos="{sx * math.cos(a):.4f} {sx * math.sin(a):.4f} {wall_h + 0.004:.4f}" '
                f'quat="{_q(a + math.pi / 2)}" mass="{obj.mass * 0.55 / n_wall:.5f}" {common}/>'
            )
        # Interior reference point: used to score "water landed in the cup".
        parts.append(
            f'<site name="mouth_{obj.name}" pos="0 0 {2 * wall_h:.4f}" size="0.004" rgba="0 0 0 0"/>'
        )
    elif obj.category in ("spoon", "fork", "knife"):
        parts.append(
            f'<geom name="g_{obj.name}_blade" type="box" size="{sx:.4f} {sy:.4f} {sz:.4f}" '
            f'pos="{sx:.4f} 0 {sz:.4f}" mass="{obj.mass * 0.4:.4f}" {common}/>'
        )
        parts.append(
            f'<geom name="g_{obj.name}_handle" type="cylinder" size="0.0065 {sx * 0.9:.4f}" '
            f'pos="{-sx * 0.9:.4f} 0 0.0065" quat="0.7071 0 0.7071 0" '
            f'mass="{obj.mass * 0.6:.4f}" {common}/>'
        )
    elif obj.category == "tray":
        parts.append(
            f'<geom name="g_{obj.name}" type="box" size="{sx:.4f} {sy:.4f} {sz:.4f}" '
            f'mass="{obj.mass:.4f}" {common}/>'
        )
        for side, sgn in (("l", -1.0), ("r", 1.0)):
            parts.append(
                f'<geom name="g_{obj.name}_h{side}" type="cylinder" size="0.007 {sy * 0.6:.4f}" '
                f'pos="{sgn * (sx + 0.012):.4f} 0 {sz + 0.010:.4f}" quat="0.7071 0.7071 0 0" '
                f'mass="0.01" {common}/>'
            )
            parts.append(
                f'<site name="handle_{side}_{obj.name}" '
                f'pos="{sgn * (sx + 0.012):.4f} 0 {sz + 0.010:.4f}" size="0.004" rgba="0 0 0 0"/>'
            )
    else:  # napkin and any future flat object
        parts.append(
            f'<geom name="g_{obj.name}" type="box" size="{sx:.4f} {sy:.4f} {sz:.4f}" '
            f'mass="{obj.mass:.4f}" {common}/>'
        )

    return body_open + "".join(f"  {p}\n" for p in parts) + "</body>\n"


def _water_xml(bottle: ObjectSpec) -> str:
    """Particle-based stand-in for liquid.

    MuJoCo has no fluid phase, so 'water' is a small set of dense spheres that start inside the
    bottle. Pouring success is then a countable, physical quantity (particles whose position ends
    inside the target cup) rather than a scripted flag. This is a deliberate approximation and is
    documented as such in docs/LIMITATIONS.md.
    """
    out = []
    for i in range(WATER_PARTICLES):
        ang = 2 * math.pi * i / WATER_PARTICLES
        r = 0.010
        z = bottle.pos[2] + 0.016 + 0.009 * (i % 4)
        out.append(
            f'<body name="water_{i}" pos="{bottle.pos[0] + r * math.cos(ang):.4f} '
            f'{bottle.pos[1] + r * math.sin(ang):.4f} {z:.4f}">'
            f'<freejoint name="free_water_{i}"/>'
            f'<geom name="g_water_{i}" type="sphere" size="{WATER_RADIUS}" mass="0.002" '
            f'material="mat_water" friction="0.2 0.002 0.0001" condim="4" '
            f'solref="0.006 1" contype="2" conaffinity="3"/>'
            f"</body>"
        )
    return "\n".join(out)


def scene_xml(scene: SceneSpec) -> str:
    """Generate the world MJCF (everything except the arms)."""
    t, d, li = scene.table, scene.drawer, scene.lighting
    bg = _rgba(scene.background_rgb)
    materials = "\n".join(
        f'<material name="mat_{o.name}" rgba="{_rgba(o.rgba)}" specular="0.35" shininess="0.4"/>'
        for o in scene.objects
    )
    objects = "\n".join(_object_xml(o, scene.friction_scale) for o in scene.objects)
    bottles = scene.objects_of("bottle")
    water = _water_xml(bottles[0]) if bottles else ""

    drawer = ""
    if d.present:
        dx, dy, dz = d.pos
        drawer = f"""
    <body name="drawer_case" pos="{dx:.4f} {dy:.4f} {dz + 0.001:.4f}">
      <geom name="g_case_back" type="box" size="0.085 0.006 0.045" pos="0 0.075 0.045"
            material="mat_furniture"/>
      <geom name="g_case_l" type="box" size="0.006 0.075 0.045" pos="-0.085 0 0.045"
            material="mat_furniture"/>
      <geom name="g_case_r" type="box" size="0.006 0.075 0.045" pos="0.085 0 0.045"
            material="mat_furniture"/>
      <geom name="g_case_top" type="box" size="0.091 0.081 0.006" pos="0 0 0.096"
            material="mat_furniture"/>
      <body name="drawer_box" pos="0 0 0.006">
        <joint name="drawer_slide" type="slide" axis="0 -1 0" range="0 {d.travel:.4f}"
               damping="12" frictionloss="0.8"/>
        <geom name="g_drawer_floor" type="box" size="0.076 0.070 0.004" pos="0 0 0.004"
              material="mat_furniture" friction="1.0 0.005 0.0001" condim="4"/>
        <geom name="g_drawer_front" type="box" size="0.078 0.006 0.026" pos="0 -0.070 0.030"
              material="mat_furniture"/>
        <geom name="g_drawer_bl" type="box" size="0.004 0.070 0.020" pos="-0.076 0 0.024"
              material="mat_furniture"/>
        <geom name="g_drawer_br" type="box" size="0.004 0.070 0.020" pos="0.076 0 0.024"
              material="mat_furniture"/>
        <geom name="g_drawer_handle" type="cylinder" size="0.008 0.026" pos="0 -0.084 0.030"
              quat="0.7071 0 0.7071 0" material="mat_handle" friction="1.2 0.01 0.001"
              condim="4" priority="1"/>
        <site name="drawer_handle" pos="0 -0.084 0.030" size="0.005" rgba="0 0 0 0"/>
        <site name="drawer_inside" pos="0 0 0.012" size="0.005" rgba="0 0 0 0"/>
      </body>
    </body>"""

    return f"""<mujoco model="dinner_table">
  <option timestep="{scene.timestep}" integrator="implicitfast" cone="elliptic"
          impratio="10" iterations="20" ls_iterations="30" noslip_iterations="3"/>
  <compiler angle="radian" autolimits="true"/>
  <size memory="64M"/>

  <visual>
    <global offwidth="1280" offheight="960"/>
    <quality shadowsize="2048"/>
    <map force="0.01"/>
  </visual>

  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="{bg}"
             rgb2="0.05 0.06 0.08" width="256" height="256"/>
    <texture name="tex_table" type="2d" builtin="checker" rgb1="{_rgba(t.rgba[:3])}"
             rgb2="{" ".join(f"{c * 0.82:.3f}" for c in t.rgba[:3])}" width="300" height="300"/>
    <material name="mat_table" texture="tex_table" texrepeat="6 4" specular="0.2" shininess="0.3"/>
    <material name="mat_floor" rgba="0.22 0.24 0.27 1"/>
    <material name="mat_furniture" rgba="0.45 0.36 0.28 1" specular="0.2"/>
    <material name="mat_handle" rgba="0.75 0.75 0.78 1" specular="0.6" shininess="0.7"/>
    <material name="mat_water" rgba="0.25 0.55 0.95 0.85" specular="0.8" shininess="0.9"/>
    <material name="mat_mount" rgba="0.2 0.2 0.22 1"/>
{materials}
  </asset>

  <worldbody>
    <light name="key" pos="{0.6 * math.sin(li.azimuth):.3f} {0.6 * math.cos(li.azimuth) - 0.2:.3f} 1.2"
           dir="{-math.sin(li.azimuth):.3f} {-math.cos(li.azimuth):.3f} -1"
           diffuse="{li.diffuse:.3f} {li.diffuse:.3f} {li.diffuse:.3f}"
           ambient="{li.ambient:.3f} {li.ambient:.3f} {li.ambient:.3f}"
           specular="{li.specular:.3f} {li.specular:.3f} {li.specular:.3f}"/>
    <light name="fill" pos="-0.5 -0.6 0.9" dir="0.4 0.5 -1" diffuse="0.25 0.25 0.25"
           ambient="0 0 0" specular="0.05 0.05 0.05"/>

    <geom name="floor" type="plane" size="3 3 0.05" pos="0 0 {t.top_z - 0.72:.3f}"
          material="mat_floor"/>
    <geom name="table_top" type="box" size="{t.half_x:.3f} {t.half_y:.3f} 0.02"
          pos="0 {t.center_y:.3f} {t.top_z - 0.02:.3f}" material="mat_table"
          friction="{0.9 * scene.friction_scale:.4f} 0.005 0.0001" condim="4"/>
    <geom name="table_leg_a" type="box" size="0.03 0.03 0.34" pos="{-t.half_x + 0.05:.3f} {t.center_y + (-t.half_y + 0.05):.3f} {t.top_z - 0.38:.3f}" material="mat_furniture"/>
    <geom name="table_leg_b" type="box" size="0.03 0.03 0.34" pos="{t.half_x - 0.05:.3f} {t.center_y + (-t.half_y + 0.05):.3f} {t.top_z - 0.38:.3f}" material="mat_furniture"/>
    <geom name="table_leg_c" type="box" size="0.03 0.03 0.34" pos="{-t.half_x + 0.05:.3f} {t.center_y + (t.half_y - 0.05):.3f} {t.top_z - 0.38:.3f}" material="mat_furniture"/>
    <geom name="table_leg_d" type="box" size="0.03 0.03 0.34" pos="{t.half_x - 0.05:.3f} {t.center_y + (t.half_y - 0.05):.3f} {t.top_z - 0.38:.3f}" material="mat_furniture"/>

    <camera name="cam_overhead" pos="0 {t.center_y:.3f} 0.72" xyaxes="1 0 0 0 1 0" fovy="58"/>
    <camera name="cam_front" pos="0 -0.72 0.50" xyaxes="1 0 0 0 0.55 0.84" fovy="52"/>
    <camera name="cam_side" pos="0.78 {t.center_y:.3f} 0.42" xyaxes="0 1 0 -0.6 0 0.8" fovy="50"/>
{drawer}
{objects}
{water}
  </worldbody>

  <keyframe/>
</mujoco>
"""


def build_model(scene: SceneSpec) -> tuple[mujoco.MjModel, mujoco.MjSpec]:
    """Compile the full dual-arm scene. Returns the model and the spec that produced it."""
    root = mujoco.MjSpec.from_string(scene_xml(scene))

    for arm in scene.arms:
        mount = root.worldbody.add_body(
            name=f"{arm.name}_mount",
            pos=list(arm.base_pos),
            quat=[math.cos(arm.base_yaw / 2), 0.0, 0.0, math.sin(arm.base_yaw / 2)],
        )
        mount.add_geom(
            name=f"g_{arm.name}_mount",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.045, 0.045, 0.008],
            pos=[0, 0, -0.008],
            material="mat_mount",
        )
        child = mujoco.MjSpec.from_file(str(SO101_XML))
        child.modelname = arm.name
        root.attach(child, prefix=f"{arm.name}/", frame=mount.add_frame())

    # The arm MJCF carries its own (lower) solver iteration counts; the scene's higher values win
    # on attach, which is what we want for stable grasp contacts. MuJoCo warns about the conflict.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Attach conflict.*")
        model = root.compile()
    return model, root


def arm_actuator_ids(model: mujoco.MjModel, arm: str) -> np.ndarray:
    """Actuator indices of one arm, in :data:`ARM_JOINTS` order."""
    return np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{arm}/{j}") for j in ARM_JOINTS]
    )


def arm_joint_ids(model: mujoco.MjModel, arm: str) -> np.ndarray:
    return np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{arm}/{j}") for j in ARM_JOINTS]
    )
