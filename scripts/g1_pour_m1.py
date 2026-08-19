#!/usr/bin/env python3
"""M1: G1+Inspire standing (welded base) at the banner4 counter, scene untouched.

Loads banner4_fluid_v18, references the fixed-base G1+Inspire USD into the kneehole
under the counter (no open floor is within arm reach — measured), filters the robot
against the scanned cabinetry hulls (same mechanism as the cups), settles, gates on
cup stillness + fluid containment, and renders one proof still.

Env: g1skillgen. All World()-harness rules from docs/banner4-fluid-scene.md apply.
"""
from __future__ import annotations
import argparse, os

ap = argparse.ArgumentParser()
ap.add_argument("--usd", default="output/banner4/sim/usd/banner4_fluid_v20.usdc")
ap.add_argument("--robot_usd", default="/home/user/work/unitree_sim_isaaclab/"
                "assets/robots/g1-29dof-inspire-base-fix-usd/g1_29dof_with_inspire_rev_1_0.usd")
ap.add_argument("--out_dir", default="output/banner4/g1_m1")
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--save_usd", default="", help="optionally save the composed stage")
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

from isaacsim import SimulationApp
sim = SimulationApp({"headless": True, "width": 1280, "height": 720})

import numpy as np, carb, omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf, UsdLux, PhysxSchema
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
import isaacsim.core.utils.stage as stage_utils
import omni.replicator.core as rep

carb.settings.get_settings().set("/rtx/rendermode", "RaytracedLighting")

stage_utils.open_stage(os.path.abspath(args.usd))
stage = omni.usd.get_context().get_stage()

# ---- lights (scene ships none) + camera clip fix
UsdLux.DomeLight.Define(stage, "/World/ValDome").CreateIntensityAttr(800.0)
_k = UsdLux.DistantLight.Define(stage, "/World/ValKey"); _k.CreateIntensityAttr(2100.0)
_kx = UsdGeom.Xformable(_k.GetPrim()); _kx.ClearXformOpOrder(); _kx.AddRotateXYZOp().Set(Gf.Vec3f(-45, 20, 0))

# ---- robot: reference, pose, cabinetry filter
ROBOT_PATH = "/World/G1"
stage_utils.add_reference_to_stage(args.robot_usd, ROBOT_PATH)
gx = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT_PATH))
gx.ClearXformOpOrder()
gx.AddTranslateOp().Set(Gf.Vec3d(-1.30, -5.42, 0.755))  # on OPEN floor, outside the counter
gx.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(0.7071068, 0.0, 0.0, -0.7071068))  # yaw -90: face -y

# NO collision filters on the robot: it stands on open floor fully OUTSIDE the
# counter volume (Eli's law: objects never overlap — the overlap sanity gate
# enforces it). Real contacts everywhere.
print("[m1] robot referenced, unfiltered (real contacts only)", flush=True)

# The USD ships posed arms-raised (teleop tabletop rig) — fingers spawn INSIDE the
# cups' space. Author the initial joint state BEFORE physics starts: arms hang at
# zero, legs in the proven crouch. JointStateAPI angular positions are in DEGREES.
import math
# NOTE this rig's joint-zero is the teleop arms-forward pose (wrists at counter
# height, fingers inside the cups' space). shoulder_pitch +0.8 tucks the hands down
# by the thighs (measured by FK sweep); -0.8 raises them above the cups (pre-grasp).
CROUCH_RAD = {
    "left_hip_pitch_joint": -0.05, "right_hip_pitch_joint": -0.05,
    "left_knee_joint": 0.2, "right_knee_joint": 0.2,
    "left_ankle_pitch_joint": -0.15, "right_ankle_pitch_joint": -0.15,
    "left_shoulder_pitch_joint": 0.8, "right_shoulder_pitch_joint": 0.8,
}
n_state = 0
for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PATH)):
    if p.IsA(UsdPhysics.RevoluteJoint):
        deg = math.degrees(CROUCH_RAD.get(p.GetName(), 0.0))
        js = PhysxSchema.JointStateAPI.Apply(p, "angular")
        js.CreatePositionAttr(deg)
        js.CreateVelocityAttr(0.0)
        # also author the drive target so a saved stage HOLDS this pose on GUI Play
        drv = UsdPhysics.DriveAPI.Get(p, "angular")
        if drv:
            drv.CreateTargetPositionAttr(deg)
        n_state += 1
print(f"[m1] authored initial joint state on {n_state} joints (arms down, legs crouch)", flush=True)

# ---- world (mandatory forcing) + articulation
world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
_pc = world.get_physics_context()
_pc.enable_gpu_dynamics(True)
try:
    _pc.set_broadphase_type("GPU")
except Exception:
    pass

robot = SingleArticulation(prim_path=f"{ROBOT_PATH}/root_joint", name="g1")
world.scene.add(robot)

for _cp in stage.Traverse():
    if _cp.IsA(UsdGeom.Camera):
        UsdGeom.Camera(_cp).CreateClippingRangeAttr().Set(Gf.Vec2f(0.005, 100000.0))

cam = rep.create.camera(position=(-0.45, -4.85, 1.50), look_at=(-1.31, -5.78, 0.93))
for _cp in stage.Traverse():
    if _cp.IsA(UsdGeom.Camera):
        UsdGeom.Camera(_cp).CreateClippingRangeAttr().Set(Gf.Vec2f(0.005, 100000.0))
rpd = rep.create.render_product(cam, (1280, 720))
an = rep.AnnotatorRegistry.get_annotator("rgb"); an.attach([rpd])

world.reset()

# ---- standing pose (proven crouch from unitree_sim_isaaclab G129_CFG_WITH_INSPIRE_HAND)
CROUCH = dict(CROUCH_RAD)
names = robot.dof_names
targets = np.zeros(robot.num_dof, dtype=np.float32)
for j, v in CROUCH.items():
    targets[names.index(j)] = v
robot.set_joint_positions(targets)
ctrl = robot.get_articulation_controller()
ctrl.apply_action(ArticulationAction(joint_positions=targets))
print(f"[m1] articulation up: {robot.num_dof} dof, crouch pose set", flush=True)

# ---- instrumented settle: cups must stay put, fluid must stay in
xc = UsdGeom.XformCache(Usd.TimeCode.Default())
cups = {n: stage.GetPrimAtPath(f"/World/{n}") for n in ("labg4", "labg9")}
inst = UsdGeom.PointInstancer(stage.GetPrimAtPath("/World/fluid"))


def cup_pose(prim):
    xc.Clear()
    m = xc.GetLocalToWorldTransform(prim)
    t = m.ExtractTranslation()
    up = m.TransformDir(Gf.Vec3d(0, 0, 1))
    u = np.array([up[0], up[1], up[2]]); u /= np.linalg.norm(u)
    return np.array([t[0], t[1], t[2]]), float(np.degrees(np.arccos(np.clip(u[2], -1, 1))))


p0 = {n: cup_pose(p)[0] for n, p in cups.items()}
pelvis = stage.GetPrimAtPath(f"{ROBOT_PATH}/pelvis")
for f in range(args.steps):
    world.step(render=False)
    if f % 60 == 0 or f == args.steps - 1:
        line = f"[m1] step={f:4d}"
        for n, p in cups.items():
            pos, tilt = cup_pose(p)
            d = pos - p0[n]
            line += f" | {n} d=({d[0]:+.4f},{d[1]:+.4f},{d[2]:+.4f}) tilt={tilt:.2f}"
        xc.Clear()
        pt = xc.GetLocalToWorldTransform(pelvis).ExtractTranslation()
        line += f" | pelvis z={pt[2]:.3f}"
        print(line, flush=True)

P = np.array(inst.GetPositionsAttr().Get())
_b4 = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(cups["labg4"]).ComputeAlignedBox()
ax = np.array([0.5 * (_b4.GetMin()[0] + _b4.GetMax()[0]), 0.5 * (_b4.GetMin()[1] + _b4.GetMax()[1])])
r = np.linalg.norm(P[:, :2] - ax, axis=1)
inside = int(((r < 0.06) & (P[:, 2] > 0.87) & (P[:, 2] < 0.95)).sum())
print(f"[m1] fluid: {inside}/{len(P)} in labg4 pool region (zmax={P[:,2].max():.3f})", flush=True)
for n, p in cups.items():
    pos, tilt = cup_pose(p)
    d = np.linalg.norm(pos - p0[n])
    print(f"[m1] GATE {n}: drift={d*1000:.2f}mm tilt={tilt:.2f}deg -> {'PASS' if d < 0.002 and tilt < 0.5 else 'FAIL'}", flush=True)
print(f"[m1] GATE fluid: {inside}/{len(P)} -> {'PASS' if inside >= 440 else 'FAIL'}", flush=True)

# ---- proof still
world.step(render=True)
rep.orchestrator.step(rt_subframes=16, delta_time=0.0, pause_timeline=True)
from PIL import Image
a = np.asarray(an.get_data())
if a.ndim == 1:
    a = a.reshape((720, 1280, -1))
Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)[:, :, :3]).save(os.path.join(args.out_dir, "m1_standing.png"))
print(f"[m1] still -> {args.out_dir}/m1_standing.png", flush=True)

if args.save_usd:
    stage.GetRootLayer().Export(os.path.abspath(args.save_usd))
    print(f"[m1] stage saved -> {args.save_usd}", flush=True)
sim.close()
