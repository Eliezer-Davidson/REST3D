#!/usr/bin/env python3
"""Interactive pose-editing GUI for the banner4 pick-and-pour scene.

Opens a pour-sequence USD in the Isaac GUI with the two cups DYNAMIC (real rigid
physics — they sit, tip, and get knocked like real glassware) while the robot
holds its pose on stiff drives. Physics runs continuously; no Play button needed.

A "Pour Pose Editor" window provides:
  * Pause / Resume physics    — freeze the world for careful joint editing
  * Reset cups + fluid        — snap cups (and PBD particles) back to home, zero velocities
  * Capture cup home          — adopt the CURRENT cup poses as the new home
  * Dump pose -> JSON         — write arm+hand joint angles, cup poses and palm
                                position to output/banner4/pour_seq/manual_poses/
                                (this is what the scripted pipeline reads back)
  * Save stage snapshot       — export the whole stage to manual_edits/
  * Hand collisions toggle    — collisions on just the right-hand links, so the
                                fingers can physically push the cup (rest of the
                                robot stays collision-free to avoid counter jams)
  * Arm / hand sliders        — drive-target editing per joint, in degrees

Env: g1skillgen.  All World() rules from docs/banner4-fluid-scene.md apply
(physics_dt=1/120, GPU dynamics + GPU broadphase forced at runtime).

Usage:
  setsid ~/miniconda3/envs/g1skillgen/bin/python -u scripts/g1_pose_editor.py \
      [scene.usd] > /tmp/pose_editor.log 2>&1 &
"""
import math
import json
import os
import sys
import time

import numpy as np

USD = sys.argv[1] if len(sys.argv) > 1 else \
    "output/banner4/sim/usd/banner4_g1_c0_proto_grasp_v05.usdc"

# fail BEFORE the ~60 s app boot, with a message that names the fix
if not os.path.isfile(USD):
    sys.exit(f"[editor] FATAL: scene file does not exist: {USD}\n"
             f"         pass an existing USD, e.g. "
             f"output/banner4/sim/usd/banner4_g1_c0_proto_grasp_v05.usdc")

from isaacsim import SimulationApp
app = SimulationApp({"headless": False})

# Enable the Isaac Sim MCP extension inside THIS session (localhost:8766) so the
# same live sandbox the user is looking at can also be inspected and driven over
# MCP. The upstream run_isaac_sim.sh assumes a standalone ~/isaacsim install;
# this box runs Isaac from the g1skillgen conda env, so the extension is added
# to the running Kit app instead.
_MCP_EXT_ROOT = os.environ.get("ISAAC_MCP_EXT_ROOT", "/home/user/opt/isaacsim-mcp-server")
try:
    import omni.kit.app as _kit_app
    _em = _kit_app.get_app().get_extension_manager()
    _em.add_path(_MCP_EXT_ROOT)
    _em.set_extension_enabled_immediate("isaac.sim.mcp_extension", True)
    print(f"[editor] MCP extension enabled from {_MCP_EXT_ROOT} (localhost:8766)", flush=True)
except Exception as _e:
    print(f"[editor] MCP extension NOT enabled: {_e}", flush=True)

import omni.ui as ui
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Gf
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.types import ArticulationAction
import isaacsim.core.utils.stage as stage_utils

DEG = math.pi / 180.0
ROBOT = "/World/G1"
PALM = f"{ROBOT}/right_hand_base_link"
CUPS = ["/World/labg4", "/World/labg9"]
FLUID = "/World/fluid"
ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
       "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
RHAND = ["R_index_proximal_joint", "R_index_intermediate_joint", "R_middle_proximal_joint",
         "R_middle_intermediate_joint", "R_pinky_proximal_joint", "R_pinky_intermediate_joint",
         "R_ring_proximal_joint", "R_ring_intermediate_joint", "R_thumb_proximal_yaw_joint",
         "R_thumb_proximal_pitch_joint", "R_thumb_intermediate_joint", "R_thumb_distal_joint"]
HAND_PAT = ("right_hand", "R_index", "R_middle", "R_ring", "R_pinky", "R_thumb")

# ---------------------------------------------------------------- drive modes
# The EFFORT CLAMP sets contact authority, not the stiffness. This editor used to
# ship mef = 1.0e4 N*m on every finger/thumb joint and 1.0e6 N*m on every arm
# joint. Against a 0.3 kg glass that means any link you drive with a slider wins
# against the contact solver outright and pushes through the cup -- while the
# counter (an analytic Cube, and effectively infinite mass) still stops it. That
# is exactly the "thumb/palm phases through the cups but the table is fine"
# report.
#
# Measured A/B, 2026-08-18 (scratchpad/pen_ab.py, worst penetration into the cup
# solid at Eli's open-claw pose under an identical squeeze):
#     sdf  + 1e4/1e6 clamps          2.70 mm      <- old editor
#     convexDecomposition + 1e4/1e6  4.50 mm      <- collider swap made it WORSE
#     sdf  + force-limited clamps    0.78 mm      <- this
# So the cup collider was never the fault; keep it sdf. Fix the actuators.
REAL_HAND_GAINS = {  # kp N*m/rad, kd N*m*s/rad, max effort N*m -- from g1_contact_task.py
    "R_index_proximal_joint": (20, 0.9, 0.45), "R_index_intermediate_joint": (10, 0.5, 0.35),
    "R_middle_proximal_joint": (20, 0.9, 0.45), "R_middle_intermediate_joint": (10, 0.5, 0.35),
    "R_ring_proximal_joint": (20, 0.9, 0.45), "R_ring_intermediate_joint": (10, 0.5, 0.35),
    "R_pinky_proximal_joint": (20, 0.9, 0.45), "R_pinky_intermediate_joint": (10, 0.5, 0.35),
    "R_thumb_proximal_yaw_joint": (25, 1.5, 1.2), "R_thumb_proximal_pitch_joint": (30, 1.5, 0.70),
    "R_thumb_intermediate_joint": (15, 0.8, 0.40), "R_thumb_distal_joint": (10, 0.5, 0.20),
}
REAL_ARM_EFFORT = {  # real G1 arm limits: 25 N*m major joints, 5 N*m wrists
    "right_shoulder_pitch_joint": 25.0, "right_shoulder_roll_joint": 25.0,
    "right_shoulder_yaw_joint": 25.0, "right_elbow_joint": 25.0,
    "right_wrist_roll_joint": 5.0, "right_wrist_pitch_joint": 5.0,
    "right_wrist_yaw_joint": 5.0,
}
# Eli's hand-crafted OPEN CLAW (editor sliders, 2026-08-18): fingers fully open,
# thumb swung to its yaw limit so it lies in line with them.
ELI_CLAW = {
    "R_index_proximal_joint": 0.0, "R_index_intermediate_joint": 0.49,
    "R_middle_proximal_joint": 0.49, "R_middle_intermediate_joint": 0.0,
    "R_pinky_proximal_joint": 0.0, "R_pinky_intermediate_joint": 0.0,
    "R_ring_proximal_joint": 0.0, "R_ring_intermediate_joint": 0.0,
    "R_thumb_proximal_yaw_joint": 74.48, "R_thumb_proximal_pitch_joint": -5.73,
    "R_thumb_intermediate_joint": 0.0, "R_thumb_distal_joint": 0.0,
}
POSE_DIR = "output/banner4/pour_seq/manual_poses"
SNAP_DIR = "output/banner4/sim/usd/manual_edits"
os.makedirs(POSE_DIR, exist_ok=True)
os.makedirs(SNAP_DIR, exist_ok=True)


def log(*a):
    print("[editor]", *a, flush=True)


stage_utils.open_stage(os.path.abspath(USD))
stage = omni.usd.get_context().get_stage()
if stage is None:
    log(f"FATAL: stage failed to open: {USD}")
    app.close()
    sys.exit(1)
missing = [p for p in CUPS + [ROBOT] if not stage.GetPrimAtPath(p).IsValid()]
if missing:
    log(f"FATAL: scene lacks required prims {missing} — is this a pour-sequence USD?")
    app.close()
    sys.exit(1)
log(f"opened {USD}")

# ---- make BOTH cups real dynamic rigid bodies (the sequence scripts leave them kinematic)
for cp in CUPS:
    prim = stage.GetPrimAtPath(cp)
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateRigidBodyEnabledAttr(True).Set(True)
    rb.CreateKinematicEnabledAttr(False).Set(False)
    rb.CreateVelocityAttr(Gf.Vec3f(0.0)).Set(Gf.Vec3f(0.0))
    rb.CreateAngularVelocityAttr(Gf.Vec3f(0.0)).Set(Gf.Vec3f(0.0))
log("cups set dynamic (kinematic off)")

# ---- hand collider list (collision toggle acts on these only)
hand_colliders = []
for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)):
    if p.HasAPI(UsdPhysics.CollisionAPI) and any(k in str(p.GetPath()) for k in HAND_PAT):
        hand_colliders.append(UsdPhysics.CollisionAPI(p))


def set_hand_collisions(on: bool):
    for c in hand_colliders:
        c.GetCollisionEnabledAttr().Set(bool(on))
    log(f"hand collisions {'ON' if on else 'OFF'} ({len(hand_colliders)} colliders)")


set_hand_collisions(True)

# ---- contact prep: the same settings every REAL phase of g1_contact_task.py
# applies. The editor never did this, so poses were being judged against weaker
# contact than the task actually runs with.
_grip = UsdShade.Material(stage.GetPrimAtPath("/World/PhysicsMaterials/grip"))
_nb = 0
for _p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)):
    if _p.GetName() != "collisions" or not any(k in str(_p.GetPath()) for k in HAND_PAT):
        continue
    if _grip:
        UsdShade.MaterialBindingAPI.Apply(_p).Bind(_grip, materialPurpose="physics")
    _pca = PhysxSchema.PhysxCollisionAPI.Apply(_p)
    _pca.CreateContactOffsetAttr(0.005).Set(0.005)
    _pca.CreateRestOffsetAttr(0.0).Set(0.0)
    _nb += 1
for _c in CUPS:
    _prb = PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(_c))
    _prb.CreateMaxDepenetrationVelocityAttr(1.0).Set(1.0)
    _prb.CreateSolverVelocityIterationCountAttr(8).Set(8)
    _prb.CreateSolverPositionIterationCountAttr(32).Set(32)
log(f"contact prep: grip material + 5 mm contact offset on {_nb} hand colliders, "
    f"cups depenetration 1.0 m/s, 8 vel / 32 pos iters")

# ---- world (must force: World() overrides authored physics-scene state)
world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
pc = world.get_physics_context()
pc.enable_gpu_dynamics(True)
try:
    pc.set_broadphase_type("GPU")
except Exception:
    pass

robot = SingleArticulation(prim_path=f"{ROBOT}/root_joint", name="g1")
world.scene.add(robot)
cup_prims = [SingleRigidPrim(prim_path=cp, name=os.path.basename(cp)) for cp in CUPS]
for c in cup_prims:
    world.scene.add(c)
world.reset()

names = robot.dof_names
idx = {n: i for i, n in enumerate(names)}

# stiffen the edited joints so the pose holds exactly (same recipe as g1_pour_sequence)
ac = robot.get_articulation_controller()


def ensure_views(force=False):
    """Rebuild the physics simulation view if it has been torn down.

    A raw USD write during play -- specifically `fluid_pi.GetPositionsAttr().Set()`
    in reset_cups() -- resyncs the stage, and the resync DESTROYS the physics
    simulation view. Nothing rebuilt it, so from that moment every articulation
    call returned None ('Physics Simulation View is not created yet') and each
    slider raised. The UI stayed on screen but nothing did anything, and the
    error storm eventually took the app down. Re-initialize on demand."""
    global ac
    if not force:
        try:
            q = robot.get_joint_positions()
            if q is not None and np.asarray(q).ndim == 1:
                return True
        except Exception:
            pass
    try:
        robot.initialize()
        ac = robot.get_articulation_controller()
        for c in cup_prims:
            try:
                c.initialize()
            except Exception:
                pass
        log("physics view was torn down -> re-initialized")
        return True
    except Exception as e:
        log(f"physics view re-init FAILED: {e}")
        return False
av = robot._articulation_view
kp, kd = ac.get_gains()
kp = np.asarray(kp, float).copy(); kd = np.asarray(kd, float).copy()
try:
    mef = np.asarray(av.get_max_efforts(), float).reshape(-1).copy()
except Exception:
    mef = np.zeros(robot.num_dof)
def apply_drive_mode(mode):
    """REAL  = force-limited, what the robot can actually exert. Contact wins.
       POSE  = stiff/unlimited, comfortable for shaping a pose, but the hand will
               drive straight through the cups. Never judge contact in POSE."""
    for j in ARM:
        kp[idx[j]] = 4000.0
        kd[idx[j]] = 200.0
        mef[idx[j]] = REAL_ARM_EFFORT.get(j, 25.0) if mode == "real" else 1.0e6
    for j in RHAND:
        if mode == "real":
            a, b, c = REAL_HAND_GAINS[j]
            kp[idx[j]] = a; kd[idx[j]] = b; mef[idx[j]] = c
        else:
            kp[idx[j]] = 400.0; kd[idx[j]] = 20.0; mef[idx[j]] = 1.0e4
    ac.set_gains(kps=kp, kds=kd)
    try:
        av.set_max_efforts(mef[None, :])
    except Exception as e:
        log(f"set_max_efforts warn: {e}")
    log(f"drives = {mode.upper()}  (thumb yaw clamp "
        f"{mef[idx['R_thumb_proximal_yaw_joint']]:.2f} N*m, "
        f"elbow clamp {mef[idx['right_elbow_joint']]:.0f} N*m)")


apply_drive_mode("real")

# hold the authored (loaded) pose via drives
target_q = np.asarray(robot.get_joint_positions(), float).copy()
if os.environ.get("EDITOR_START_CLAW", "1") == "1":
    for _j, _v in ELI_CLAW.items():
        if _j in idx:
            target_q[idx[_j]] = _v * DEG
    log("startup pose: Eli's open claw (fingers open, thumb yaw 74.48)")
ac.apply_action(ArticulationAction(joint_positions=target_q))
log(f"holding loaded pose; arm(deg)="
    f"{[round(float(target_q[idx[j]]) / DEG, 1) for j in ARM]}")

# joint limits for slider ranges
jlim = {}
for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)):
    if p.IsA(UsdPhysics.RevoluteJoint):
        j = UsdPhysics.RevoluteJoint(p)
        lo, hi = j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()
        jlim[p.GetName()] = (lo if lo is not None else -180.0, hi if hi is not None else 180.0)

# ---- cup + fluid home snapshot
xc = UsdGeom.XformCache(Usd.TimeCode.Default())
fluid_pi = UsdGeom.PointInstancer(stage.GetPrimAtPath(FLUID))
HOME = {"cups": {}, "fluid_pos": None, "fluid_vel": None}


def capture_home():
    for c in cup_prims:
        pos, quat = c.get_world_pose()
        HOME["cups"][c.prim_path] = (np.asarray(pos, float).copy(), np.asarray(quat, float).copy())
    if fluid_pi:
        HOME["fluid_pos"] = list(fluid_pi.GetPositionsAttr().Get())
        v = fluid_pi.GetVelocitiesAttr().Get()
        HOME["fluid_vel"] = list(v) if v else None
    log("cup/fluid home captured: " +
        "  ".join(f"{p.split('/')[-1]}@{np.round(HOME['cups'][p][0], 3)}" for p in HOME["cups"]))


def reset_cups():
    for c in cup_prims:
        pos, quat = HOME["cups"][c.prim_path]
        c.set_world_pose(position=pos, orientation=quat)
        c.set_linear_velocity(np.zeros(3))
        c.set_angular_velocity(np.zeros(3))
    if fluid_pi and HOME["fluid_pos"] is not None:
        try:
            fluid_pi.GetPositionsAttr().Set(HOME["fluid_pos"])
            if HOME["fluid_vel"] is not None:
                fluid_pi.GetVelocitiesAttr().Set(HOME["fluid_vel"])
            else:
                fluid_pi.GetVelocitiesAttr().Set([Gf.Vec3f(0.0)] * len(HOME["fluid_pos"]))
        except Exception as e:
            log(f"fluid reset warn (restart editor if particles misbehave): {e}")
        # the write above resyncs the stage and tears the physics view down --
        # rebuild it here or every control in this window goes dead
        ensure_views(force=True)
    log("cups + fluid reset to home")


capture_home()


def dump_pose():
    q = np.asarray(robot.get_joint_positions(), float)
    palm = stage.GetPrimAtPath(PALM)
    xc.Clear()
    pw = xc.GetLocalToWorldTransform(palm).ExtractTranslation()
    out = {
        "usd": USD,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arm_deg": {j: round(float(q[idx[j]]) / DEG, 2) for j in ARM},
        "hand_deg": {j: round(float(q[idx[j]]) / DEG, 2) for j in RHAND},
        "palm_world": [round(float(v), 4) for v in pw],
        "cups": {},
    }
    # NOTE frame convention: the cup PRIM origins sit ~(+0.43,+1.94,-0.98) away from
    # the visible glass (offset baked into the reconstruction's child geometry).
    # "pos" is the physics/prim origin; "center_bbox" is where the glass actually is.
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    for c in cup_prims:
        pos, quat = c.get_world_pose()
        box = bc.ComputeWorldBound(stage.GetPrimAtPath(c.prim_path)).ComputeAlignedBox()
        ctr = (np.array(box.GetMin()) + np.array(box.GetMax())) / 2
        out["cups"][c.prim_path] = {
            "pos": [round(float(v), 4) for v in np.asarray(pos)],
            "quat_wxyz": [round(float(v), 5) for v in np.asarray(quat)],
            "center_bbox": [round(float(v), 4) for v in ctr],
        }
    path = os.path.join(POSE_DIR, time.strftime("pose_%Y%m%d_%H%M%S.json"))
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    log(f"POSE DUMPED -> {path}")
    log(json.dumps(out["arm_deg"]))
    return path


def save_snapshot():
    path = os.path.join(SNAP_DIR, time.strftime("editor_snapshot_%Y%m%d_%H%M%S.usdc"))
    stage.GetRootLayer().Export(os.path.abspath(path))
    log(f"stage snapshot saved -> {path}")


# ---- UI ---------------------------------------------------------------------
state = {"paused": False, "hand_col": True, "drive": "real", "show_col": False}


def set_collider_viz(on: bool):
    """Draw PhysX's ACTUAL collision shapes over the render meshes.

    The glass is near-fully transparent, so a hand BEHIND a cup reads as a hand
    INSIDE it. Turning this on draws what physics actually collides against, so
    contact can be judged instead of guessed."""
    import carb
    st = carb.settings.get_settings()
    st.set("/persistent/physics/visualizationDisplayColliders", bool(on))
    st.set("/persistent/physics/visualizationDisplayCollidersLevel", 1 if on else 0)
    st.set("/persistent/physics/visualizationCollidersOccupancy", bool(on))
    st.set("/physics/visualizationDisplayColliders", bool(on))
    log(f"collider visualisation {'ON' if on else 'OFF'}")
sliders = {}
_slider_guard = {"busy": False}


def _on_slider(joint, model):
    if _slider_guard["busy"]:
        return
    target_q[idx[joint]] = model.get_value_as_float() * DEG
    try:
        ac.apply_action(ArticulationAction(joint_positions=target_q))
    except Exception:
        # never let a UI callback raise: an exception here is swallowed by
        # omni.ui and printed once per event, which is what produced the
        # unusable window rather than a visible failure
        if ensure_views(force=True):
            try:
                ac.apply_action(ArticulationAction(joint_positions=target_q))
            except Exception as e:
                log(f"slider apply failed after re-init: {e}")


def sync_sliders_from_robot():
    if not ensure_views():
        log("cannot sync sliders: no physics view")
        return
    q = np.asarray(robot.get_joint_positions(), float)
    target_q[:] = q
    _slider_guard["busy"] = True
    for j, m in sliders.items():
        m.set_value(float(q[idx[j]]) / DEG)
    _slider_guard["busy"] = False
    log("sliders synced from current robot pose")


window = ui.Window("Pour Pose Editor", width=400, height=760)
# Kit PERSISTS window layout between runs. A previous session left this one
# maximized over the whole app, which hides the viewport entirely and makes the
# sandbox look broken. Pin it back to a tab beside the Stage tree every launch.
try:
    window.deferred_dock_in("Stage", ui.DockPolicy.TARGET_WINDOW_IS_ACTIVE)
except Exception:
    try:
        window.undock(); window.position_x = 24; window.position_y = 90
        window.width, window.height = 400, 760
    except Exception as _e:
        print(f"[editor] window placement skipped: {_e}", flush=True)
with window.frame:
    with ui.VStack(spacing=6, height=0):
        pause_btn = ui.Button("Pause physics", height=32)

        def _toggle_pause():
            state["paused"] = not state["paused"]
            pause_btn.text = "Resume physics" if state["paused"] else "Pause physics"
            log("physics PAUSED" if state["paused"] else "physics RESUMED")

        pause_btn.set_clicked_fn(_toggle_pause)
        ui.Button("Reset cups + fluid", height=32, clicked_fn=reset_cups)
        ui.Button("Capture cup home (use current)", height=26, clicked_fn=capture_home)
        ui.Button("Dump pose -> JSON", height=32, clicked_fn=dump_pose)
        ui.Button("Save stage snapshot", height=26, clicked_fn=save_snapshot)

        col_btn = ui.Button("Hand collisions: ON", height=26)

        def _toggle_col():
            state["hand_col"] = not state["hand_col"]
            set_hand_collisions(state["hand_col"])
            col_btn.text = f"Hand collisions: {'ON' if state['hand_col'] else 'OFF'}"

        viz_btn = ui.Button("Show colliders: OFF")

        def _toggle_viz():
            state["show_col"] = not state["show_col"]
            set_collider_viz(state["show_col"])
            viz_btn.text = f"Show colliders: {'ON' if state['show_col'] else 'OFF'}"
        viz_btn.set_clicked_fn(_toggle_viz)

        col_btn.set_clicked_fn(_toggle_col)

        drv_btn = ui.Button("Drives: REAL (force-limited)", height=26)

        def _toggle_drive():
            state["drive"] = "pose" if state["drive"] == "real" else "real"
            apply_drive_mode(state["drive"])
            drv_btn.text = ("Drives: REAL (force-limited)" if state["drive"] == "real"
                            else "Drives: POSE (stiff — WILL push through cups)")

        drv_btn.set_clicked_fn(_toggle_drive)
        ui.Button("Sync sliders <- robot", height=26, clicked_fn=sync_sliders_from_robot)

        for title, joints in (("Arm (deg)", ARM), ("Hand (deg)", RHAND)):
            with ui.CollapsableFrame(title, collapsed=(title.startswith("Hand"))):
                with ui.VStack(spacing=2):
                    for j in joints:
                        lo, hi = jlim.get(j, (-180.0, 180.0))
                        with ui.HStack(height=20):
                            ui.Label(j.replace("_joint", ""), width=190)
                            s = ui.FloatSlider(min=float(lo), max=float(hi))
                            s.model.set_value(float(target_q[idx[j]]) / DEG)
                            s.model.add_value_changed_fn(
                                (lambda jj: lambda m: _on_slider(jj, m))(j))
                            sliders[j] = s.model

# Frame the viewport on the cup station. Without this the perspective camera
# opens wherever the USD left it -- for the banner4 scenes that is a blown-out
# view of nothing, which reads as "the window is broken".
try:
    from isaacsim.core.utils.viewports import set_camera_view
    _bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    _ctrs = []
    for _cp in CUPS:
        _b = _bc.ComputeWorldBound(stage.GetPrimAtPath(_cp)).ComputeAlignedBox()
        _ctrs.append((np.array(_b.GetMin()) + np.array(_b.GetMax())) / 2.0)
    _t = np.mean(_ctrs, axis=0)
    set_camera_view(eye=[_t[0] - 0.34, _t[1] - 0.62, _t[2] + 0.26],
                    target=[float(_t[0]), float(_t[1]), float(_t[2])],
                    camera_prim_path="/OmniverseKit_Persp")
    log(f"viewport framed on the cup station at {np.round(_t, 3)}")
except Exception as _e:
    log(f"viewport framing skipped: {_e}")

log("editor ready — physics running. Cups are DYNAMIC; use Reset if one escapes.")
while app.is_running():
    if state["paused"]:
        app.update()
    else:
        world.step(render=True)
app.close()
