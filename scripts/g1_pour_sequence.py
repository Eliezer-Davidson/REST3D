#!/usr/bin/env python3
"""G1 + Inspire pick-and-pour on the banner4 fluid scene.

Sequence: reach the full cup (labg4) -> close fingers -> pick it up ->
carry it over the empty cup (labg9) -> tilt to pour REAL PhysX PBD fluid ->
set it back down -> release.

Reality notes (see docs/banner4-fluid-scene.md and the reach analysis):
  * The G1 here is FIXED-BASE (fix_base:true) -> it cannot walk. It is welded
    standing at the counter; "moving up to the counter" is the arm reach.
  * The reconstructed counter is a ~2.16 m-deep slab and the cups sit ~1.36 m
    back from its front edge -> UNREACHABLE by a ~0.6 m arm from any valid stance
    (the scene's own note: "no open floor is within arm reach"). So the two cups
    are STAGED forward (--stage_dy) into the robot's working zone, together with
    their support slab and the fluid particles. Everything else is untouched.
  * Grasp is a kinematic-follow (the cup is slaved to the palm once the fingers
    close) -- reliable, and the FLUID stays fully physically simulated, so the
    pour is real. Arm placement uses numerical (damped-least-squares) IK driven
    by physics-step forward kinematics.

Env: g1skillgen. All World()-harness rules from the fluid-scene doc apply
(physics_dt=1/120, GPU dynamics + GPU broadphase forced).
"""
from __future__ import annotations
import argparse, os, math
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--usd", default="output/banner4/sim/usd/banner4_fluid_v18.usdc")
ap.add_argument("--robot_usd", default="/home/user/work/unitree_sim_isaaclab/"
                "assets/robots/g1-29dof-inspire-base-fix-usd/g1_29dof_with_inspire_rev_1_0.usd")
ap.add_argument("--out_dir", default="output/banner4/pour_seq")
ap.add_argument("--save_usd", default="")
ap.add_argument("--stage_dy", type=float, default=0.80,
                help="+Y offset applied to the two cups + support slab + fluid to bring them into reach")
ap.add_argument("--phase", choices=["stage", "b1_pregrasp", "c0_proto_grasp", "reach", "full", "tp"], default="full")
ap.add_argument("--capture_every", type=int, default=4)
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
frames_dir = os.path.join(args.out_dir, "frames")
os.makedirs(frames_dir, exist_ok=True)

from isaacsim import SimulationApp
sim = SimulationApp({"headless": True, "width": args.width, "height": args.height})

import carb, omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, UsdLux, Sdf, Vt
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
import isaacsim.core.utils.stage as stage_utils
import omni.replicator.core as rep

carb.settings.get_settings().set("/rtx/rendermode", "RaytracedLighting")

DEG = math.pi / 180.0
ROBOT = "/World/G1"
PALM = f"{ROBOT}/right_hand_base_link"
CAB = "/World/blue_and_black_laboratory_cabinetry_system_on_back_and_right_walls_000"
FULL_CUP = "/World/labg4"
EMPTY_CUP = "/World/labg9"
ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
       "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
RHAND = ["R_index_proximal_joint", "R_index_intermediate_joint", "R_middle_proximal_joint",
         "R_middle_intermediate_joint", "R_pinky_proximal_joint", "R_pinky_intermediate_joint",
         "R_ring_proximal_joint", "R_ring_intermediate_joint", "R_thumb_proximal_yaw_joint",
         "R_thumb_proximal_pitch_joint", "R_thumb_intermediate_joint", "R_thumb_distal_joint"]
CROUCH = {"left_hip_pitch_joint": -0.05, "right_hip_pitch_joint": -0.05,
          "left_knee_joint": 0.2, "right_knee_joint": 0.2,
          "left_ankle_pitch_joint": -0.15, "right_ankle_pitch_joint": -0.15,
          "left_shoulder_pitch_joint": 0.8, "right_shoulder_pitch_joint": 0.8}

B1_PREGRASP_ARM_DEG = {
    "right_shoulder_pitch_joint": -19.3,
    "right_shoulder_roll_joint": -51.0,
    "right_shoulder_yaw_joint": 18.7,
    "right_elbow_joint": 9.4,
    "right_wrist_roll_joint": 55.8,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}
B1_PREGRASP_HAND_DEG = {
    "R_index_proximal_joint": 18.8,
    "R_index_intermediate_joint": 6.1,
    "R_middle_proximal_joint": 21.8,
    "R_middle_intermediate_joint": 6.1,
    "R_ring_proximal_joint": 18.0,
    "R_ring_intermediate_joint": 5.5,
    "R_pinky_proximal_joint": 16.0,
    "R_pinky_intermediate_joint": 5.0,
    "R_thumb_proximal_yaw_joint": 0.0,
    "R_thumb_proximal_pitch_joint": 0.0,
    "R_thumb_intermediate_joint": 0.0,
    "R_thumb_distal_joint": 0.0,
}
C0_PROTO_GRASP_HAND_DEG = {
    "R_index_proximal_joint": 45.0,
    "R_index_intermediate_joint": 28.0,
    "R_middle_proximal_joint": 48.0,
    "R_middle_intermediate_joint": 30.0,
    "R_ring_proximal_joint": 44.0,
    "R_ring_intermediate_joint": 28.0,
    "R_pinky_proximal_joint": 40.0,
    "R_pinky_intermediate_joint": 25.0,
    "R_thumb_proximal_yaw_joint": 32.0,
    "R_thumb_proximal_pitch_joint": 18.0,
    "R_thumb_intermediate_joint": 18.0,
    "R_thumb_distal_joint": 18.0,
}


def log(*a):
    print("[pour]", *a, flush=True)


# ---------------------------------------------------------------- open + author
stage_utils.open_stage(os.path.abspath(args.usd))
stage = omni.usd.get_context().get_stage()

UsdLux.DomeLight.Define(stage, "/World/SeqDome").CreateIntensityAttr(800.0)
_k = UsdLux.DistantLight.Define(stage, "/World/SeqKey"); _k.CreateIntensityAttr(2000.0)
_kx = UsdGeom.Xformable(_k.GetPrim()); _kx.ClearXformOpOrder(); _kx.AddRotateXYZOp().Set(Gf.Vec3f(-45, 20, 0))

# ---- reference robot, pose it standing at the counter (same base as m1)
stage_utils.add_reference_to_stage(args.robot_usd, ROBOT)
gx = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
gx.ClearXformOpOrder()
ROBOT_BASE = Gf.Vec3d(-0.973616492606113, -5.661964582329803, 0.79)
ROBOT_ORIENT = Gf.Quatd(0.292711417053842, 0.0, 0.0, -0.9562008294946894)
gx.AddTranslateOp().Set(ROBOT_BASE)
gx.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(ROBOT_ORIENT)

# author crouch/arms-down as the initial joint state so physics starts there
n_state = 0
for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)):
    if p.IsA(UsdPhysics.RevoluteJoint):
        nm = p.GetName()
        deg = CROUCH.get(nm, 0.0) / DEG
        js = PhysxSchema.JointStateAPI.Apply(p, "angular")
        js.CreatePositionAttr(deg); js.CreateVelocityAttr(0.0)
        drv = UsdPhysics.DriveAPI.Get(p, "angular")
        if drv:
            drv.CreateTargetPositionAttr(deg)
            # stiffen the arm so it HOLDS extended reach/pour poses (shipped ~1.75/deg = 100/rad
            # is too soft). Values are per-degree; 70/deg ~ 4000 N*m/rad.
            if nm in ARM:
                drv.CreateStiffnessAttr(70.0)
                drv.CreateDampingAttr(8.0)
                drv.CreateMaxForceAttr(800.0)   # shipped cap is 25 N*m (5 at wrist) -> arm stalls
        n_state += 1
log(f"authored initial joint state on {n_state} joints")

# ---- STAGE the cups + their support + the fluid into reach (+Y)
DY = args.stage_dy


def translate_prim(path, dy):
    translate_prim_xyz(path, Gf.Vec3d(0, dy, 0))


def translate_prim_xyz(path, delta):
    p = stage.GetPrimAtPath(path)
    if not p.IsValid():
        log(f"WARN missing {path}")
        return
    xf = UsdGeom.Xformable(p)
    ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
    top = ops.get("xformOp:translate")
    if top is None:
        top = xf.AddTranslateOp()
        cur = Gf.Vec3d(0, 0, 0)
    else:
        cur = top.Get()
        cur = Gf.Vec3d(cur[0], cur[1], cur[2]) if cur is not None else Gf.Vec3d(0, 0, 0)
    top.Set(cur + delta)


def bbox_min_max(path):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=True)
    b = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedBox()
    return np.array(b.GetMin(), dtype=np.float64), np.array(b.GetMax(), dtype=np.float64)


for pth in (FULL_CUP, EMPTY_CUP, "/World/counter_contact_slab", "/World/labg4_table_contact_support"):
    translate_prim(pth, DY)

# make the cups visibly sit on the staged surface/support. The source cups hover a few
# millimeters above the reconstructed slab, which is obvious in close-up renders.
CONTACT_CLEARANCE = 0.0005
_slab_mn, _slab_mx = bbox_min_max("/World/counter_contact_slab")
_support_mn, _support_mx = bbox_min_max("/World/labg4_table_contact_support")
_support_dz = float(_slab_mx[2] - _support_mx[2])
translate_prim_xyz("/World/labg4_table_contact_support", Gf.Vec3d(0, 0, _support_dz))
_surface_z = float(_slab_mx[2])


def align_cup_bottom(path):
    mn, _ = bbox_min_max(path)
    dz = _surface_z + CONTACT_CLEARANCE - float(mn[2])
    translate_prim_xyz(path, Gf.Vec3d(0, 0, dz))
    return dz


_lab4_dz = align_cup_bottom(FULL_CUP)
_lab9_dz = align_cup_bottom(EMPTY_CUP)

# Put the empty receiver cup at the user-approved manual layout relative to labg4.
_lab4_mn, _lab4_mx = bbox_min_max(FULL_CUP)
_lab9_mn, _lab9_mx = bbox_min_max(EMPTY_CUP)
_lab4_center = (_lab4_mn + _lab4_mx) * 0.5
_lab9_center = (_lab9_mn + _lab9_mx) * 0.5
LABG9_FROM_LABG4 = Gf.Vec3d(0.1560296419780521, -0.21333352832716823, 0.0002908477218959149)
_lab9_target = Gf.Vec3d(float(_lab4_center[0] + LABG9_FROM_LABG4[0]),
                        float(_lab4_center[1] + LABG9_FROM_LABG4[1]),
                        float(_lab4_center[2] + LABG9_FROM_LABG4[2]))
_lab9_delta = Gf.Vec3d(float(_lab9_target[0] - _lab9_center[0]),
                       float(_lab9_target[1] - _lab9_center[1]),
                       float(_lab9_target[2] - _lab9_center[2]))
translate_prim_xyz(EMPTY_CUP, _lab9_delta)

# fluid particles: offset positions by +Y
finst = UsdGeom.PointInstancer(stage.GetPrimAtPath("/World/fluid"))
_pos = finst.GetPositionsAttr().Get()
_np = np.array([[v[0], v[1], v[2]] for v in _pos], dtype=np.float32)
_np[:, 1] += DY
_np[:, 2] += _lab4_dz
finst.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(_np))
log(f"staged cups + slab + {len(_np)} fluid particles by dy={DY:+.2f}")
log(f"contact align: surface_z={_surface_z:.4f}, labg4 dz={_lab4_dz:+.4f}, labg9 dz={_lab9_dz:+.4f}, support dz={_support_dz:+.4f}")
log(f"layout: robot_base={tuple(round(v, 3) for v in ROBOT_BASE)}, labg9 delta={tuple(round(v, 3) for v in _lab9_delta)}")

# ---- make the full cup KINEMATIC and drive it via ONE transform op (pour_demo mechanism).
#      the empty cup is frozen kinematic too so it stays put as a stable receiver.
def _cup_translate(path):
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    for op in xf.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate":
            v = op.Get()
            return Gf.Vec3d(v[0], v[1], v[2])
    return Gf.Vec3d(0, 0, 0)

_lab4_t = _cup_translate(FULL_CUP)
_l4x = UsdGeom.Xformable(stage.GetPrimAtPath(FULL_CUP))
_l4x.ClearXformOpOrder()
labg4_op = _l4x.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
M_table_pose = Gf.Matrix4d().SetTranslate(_lab4_t)
labg4_op.Set(M_table_pose)
for _cp in (FULL_CUP, EMPTY_CUP):
    _rb = UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(_cp))
    _rb.CreateRigidBodyEnabledAttr(True)
    _rb.CreateKinematicEnabledAttr(True)
log(f"labg4 -> kinematic (transform-op) at {tuple(round(v, 3) for v in _lab4_t)}; labg9 frozen")

# ---- filter robot links vs the oversized cabinetry convex hull (proven fix)
cab = stage.GetPrimAtPath(CAB)
nfilt = 0
for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)):
    if p.HasAPI(UsdPhysics.CollisionAPI) or p.IsA(UsdPhysics.CollisionAPI):
        fp = UsdPhysics.FilteredPairsAPI.Apply(p)
        fp.CreateFilteredPairsRel().AddTarget(cab.GetPath())
        nfilt += 1
# also apply on the articulation root for good measure
fp = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(ROBOT))
fp.CreateFilteredPairsRel().AddTarget(cab.GetPath())
log(f"filtered {nfilt} robot colliders vs cabinetry")

# ---- the arm is a SCRIPTED kinematic chain (grasp = kinematic-follow, no real contact
#      needed). Disable gravity + collisions on the robot so the drive tracks IK poses
#      EXACTLY (the shipped drive is torque-capped and the arm was jamming on the counter).
ngrav = ncol = 0
for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)):
    if p.HasAPI(UsdPhysics.RigidBodyAPI):
        PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateDisableGravityAttr(True)
        ngrav += 1
    if p.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)
        ncol += 1
log(f"robot: disabled gravity on {ngrav} bodies, collision on {ncol} colliders")

# ---------------------------------------------------------------- world + arti
world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
pc = world.get_physics_context()
pc.enable_gpu_dynamics(True)
try:
    pc.set_broadphase_type("GPU")
except Exception:
    pass

robot = SingleArticulation(prim_path=f"{ROBOT}/root_joint", name="g1")
world.scene.add(robot)

# camera framing robot + cups
CAM_POS = (0.15, -5.05, 1.45)
CAM_LOOK = (-1.30, -6.05, 0.95)
cam = rep.create.camera(position=CAM_POS, look_at=CAM_LOOK)
for cp in stage.Traverse():
    if cp.IsA(UsdGeom.Camera):
        UsdGeom.Camera(cp).CreateClippingRangeAttr().Set(Gf.Vec2f(0.005, 100000.0))
rp = rep.create.render_product(cam, (args.width, args.height))
annot = rep.AnnotatorRegistry.get_annotator("rgb")
annot.attach([rp])

world.reset()

# dof name/index maps
names = robot.dof_names
idx = {n: i for i, n in enumerate(names)}
log("articulation ready; arm drive stiffened via USD")

# ---- strong arm drive gains + effort at RUNTIME (USD gains reach the sim but the shipped
#      max effort of 25 N*m stays; set both here so the arm tracks stiffly).
ac = robot.get_articulation_controller()
av = robot._articulation_view
_kp, _kd = ac.get_gains()
_kp = np.asarray(_kp, float).copy(); _kd = np.asarray(_kd, float).copy()
try:
    _mef = np.asarray(av.get_max_efforts(), float).reshape(-1).copy()
except Exception:
    _mef = np.zeros(robot.num_dof)
for j in ARM:
    _kp[idx[j]] = 4000.0; _kd[idx[j]] = 200.0; _mef[idx[j]] = 1.0e6
ac.set_gains(kps=_kp, kds=_kd)
try:
    av.set_max_efforts(_mef[None, :])
except Exception as e:
    log(f"set_max_efforts err {e}")
log("arm runtime gains/effort set (kp=4000 kd=200 effort=1e6)")

# ---- pose helpers -----------------------------------------------------------
xc = UsdGeom.XformCache(Usd.TimeCode.Default())
palm_prim = stage.GetPrimAtPath(PALM)
full_prim = stage.GetPrimAtPath(FULL_CUP)
empty_prim = stage.GetPrimAtPath(EMPTY_CUP)


def world_T(prim):
    xc.Clear()
    return xc.GetLocalToWorldTransform(prim)


def world_pos(prim):
    return np.array(world_T(prim).ExtractTranslation())


def cup_bbox_center(prim):
    b = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(prim).ComputeAlignedBox()
    mn, mx = b.GetMin(), b.GetMax()
    return np.array([(mn[0]+mx[0])/2, (mn[1]+mx[1])/2, (mn[2]+mx[2])/2]), np.array(mn), np.array(mx)


# hold vector = full-DOF target (crouch + zeros), updated for the arm during motion
hold = np.zeros(robot.num_dof, dtype=np.float32)
for j, v in CROUCH.items():
    hold[idx[j]] = v
robot.set_joint_positions(hold)
robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=hold))
for _ in range(40):
    world.step(render=False)

fc, _fmn, _fmx = cup_bbox_center(full_prim)
ec, _emn, _emx = cup_bbox_center(empty_prim)
log(f"staged full cup labg4 center={fc.round(3)}  empty cup labg9 center={ec.round(3)}")
log(f"palm now at {world_pos(palm_prim).round(3)}")

# capture util
try:
    from PIL import Image
    def save_png(a, p): Image.fromarray(a).save(p)
except Exception:
    import imageio.v2 as imageio
    def save_png(a, p): imageio.imwrite(p, a)

import omni.timeline
_timeline = omni.timeline.get_timeline_interface()


def _capture_to(path):
    # render one frame and save. rep.orchestrator.step(pause_timeline=True) PAUSES the
    # timeline; once paused, world.step() stops flushing articulation drive targets and the
    # arm freezes. Re-play immediately so the drive keeps tracking on the next world.step.
    world.step(render=True)
    rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=True)
    if not _timeline.is_playing():
        _timeline.play()
    a = np.asarray(annot.get_data())
    if a.ndim == 1:
        a = a.reshape((args.height, args.width, -1))
    save_png(np.clip(a, 0, 255).astype(np.uint8)[:, :, :3], path)


def grab(tag_idx):
    _capture_to(os.path.join(frames_dir, f"f{tag_idx:04d}.png"))


cap = [0]

def step_capture(n, follow=None):
    for _ in range(n):
        if follow is not None:
            follow()
        world.step(render=False)
        if cap[0] % args.capture_every == 0:
            _capture_to(os.path.join(frames_dir, f"f{cap[0]//args.capture_every:04d}.png"))
        cap[0] += 1


if args.phase == "stage":
    log("PHASE=stage: staged scene, robot standing. Rendering proof + settling.")
    step_capture(120)
    if args.save_usd:
        stage.GetRootLayer().Export(os.path.abspath(args.save_usd))
        log(f"saved {args.save_usd}")
    log("done stage phase")
    sim.close(); raise SystemExit

def _render_save():
    _capture_to(os.path.join(frames_dir, f"f{cap[0] // args.capture_every:04d}.png"))


# ---------------------------------------------------------------- IK machinery
ai = [idx[j] for j in ARM]
_lim_lo, _lim_hi = [], []
_jl = {}
for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)):
    if p.IsA(UsdPhysics.RevoluteJoint) and p.GetName() in ARM:
        j = UsdPhysics.RevoluteJoint(p)
        lo = j.GetLowerLimitAttr().Get()
        hi = j.GetUpperLimitAttr().Get()
        _jl[p.GetName()] = (math.radians(lo) if lo is not None else -math.pi,
                            math.radians(hi) if hi is not None else math.pi)
for j in ARM:
    lo, hi = _jl.get(j, (-math.pi, math.pi))
    _lim_lo.append(lo); _lim_hi.append(hi)
lim_lo = np.array(_lim_lo); lim_hi = np.array(_lim_hi)
log(f"arm limits(rad) lo={lim_lo.round(2)} hi={lim_hi.round(2)}")

base_q = hold  # mutable full-DOF target (crouch legs + arms); fingers updated on grasp


def set_full(q_arm, drive=True):
    q = base_q.copy()
    q[ai] = np.clip(q_arm, lim_lo, lim_hi)
    robot.set_joint_positions(q)
    if drive:
        robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=q))
    return q


def fk(q_arm):
    set_full(q_arm)
    world.step(render=False)
    return world_pos(palm_prim)


def drive(q_arm):
    # command via the PD drive ONLY (no per-step teleport). The stiffened arm drive
    # tracks this exactly; re-teleporting every step was found to break convergence.
    q = base_q.copy()
    q[ai] = np.clip(q_arm, lim_lo, lim_hi)
    robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=q))
    return q


def settle(n):
    for _ in range(n):
        world.step(render=False)


def solve_ik(target, q0, iters=45, damp=0.08, dq=0.03, tol=3e-3):
    q = np.clip(np.array(q0, float), lim_lo, lim_hi)
    err = 1e9
    for _ in range(iters):
        p0 = fk(q)
        e = np.array(target) - p0
        err = np.linalg.norm(e)
        if err < tol:
            break
        J = np.zeros((3, len(ai)))
        for c in range(len(ai)):
            qd = q.copy(); qd[c] += dq
            J[:, c] = (fk(qd) - p0) / dq
        JT = J.T
        step = JT @ np.linalg.solve(J @ JT + (damp ** 2) * np.eye(3), e)
        q = np.clip(q + np.clip(step, -0.25, 0.25), lim_lo, lim_hi)
    return q, err


_rng = np.random.default_rng(0)


def ik(target, seed, tol=0.02):
    q, e = solve_ik(target, seed)
    tries = 0
    while e > tol and tries < 8:
        s2 = np.clip(np.asarray(seed, float) + _rng.uniform(-0.6, 0.6, len(ai)), lim_lo, lim_hi)
        q2, e2 = solve_ik(target, s2)
        if e2 < e:
            q, e = q2, e2
        tries += 1
    return q, e


def close_fingers(v=0.7):
    for j in RHAND:
        base_q[idx[j]] = v


def open_fingers():
    for j in RHAND:
        base_q[idx[j]] = 0.0


def arm_from_deg(values):
    return np.array([values[j] * DEG for j in ARM], dtype=float)


def hand_from_deg(values):
    return np.array([values.get(j, 0.0) * DEG for j in RHAND], dtype=float)


def author_joint_state(q_full):
    for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)):
        if not p.IsA(UsdPhysics.RevoluteJoint):
            continue
        nm = p.GetName()
        if nm not in idx:
            continue
        deg = float(q_full[idx[nm]] / DEG)
        js = PhysxSchema.JointStateAPI.Apply(p, "angular")
        js.CreatePositionAttr(deg).Set(deg)
        js.CreateVelocityAttr(0.0).Set(0.0)
        drv = UsdPhysics.DriveAPI.Get(p, "angular")
        if drv:
            drv.CreateTargetPositionAttr(deg).Set(deg)


def tick(q_arm=None, cup_fn=None):
    if q_arm is not None:
        drive(q_arm)
    if cup_fn is not None:
        cup_fn()
    world.step(render=False)
    if cap[0] % args.capture_every == 0:
        _render_save()
    cap[0] += 1


def move(q_from, q_to, n, cup_fn=None):
    qf, qt = np.asarray(q_from, float), np.asarray(q_to, float)
    for s in range(n):
        a = (s + 1.0) / n
        tick((1 - a) * qf + a * qt, cup_fn)


def smoother(t):
    return 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, t)))


if args.phase == "tp":
    testq = np.array([-0.8, 0.15, 0.0, 1.2, 0.0, 0.0, 0.0])
    drive(testq)
    for _ in range(60):
        world.step(render=False)
    rd = np.asarray(robot.get_joint_positions())[ai]
    log(f"DRIVE testq jerr={np.linalg.norm(rd-testq):.4f} arm={rd.round(2)} palm={world_pos(palm_prim).round(3)}")
    set_full(testq)
    for _ in range(8):
        world.step(render=False)
    rd2 = np.asarray(robot.get_joint_positions())[ai]
    log(f"TELE  testq jerr={np.linalg.norm(rd2-testq):.4f} palm={world_pos(palm_prim).round(3)}")
    # collision-free raise, then a bent reach, confirm no sag/jam
    for tag, tq in (("raise", np.array([-1.4, 0.1, 0.0, 0.6, 0.0, 0.0, 0.0])),
                    ("reach", np.array([-0.9, 0.5, 0.2, 1.3, 0.0, 0.3, 0.0]))):
        drive(tq)
        for _ in range(50):
            world.step(render=False)
        r = np.asarray(robot.get_joint_positions())[ai]
        log(f"  {tag} jerr={np.linalg.norm(r-tq):.4f} palm={world_pos(palm_prim).round(3)}")
    _render_save()
    log("done tp"); sim.close(); raise SystemExit


# ---------------------------------------------------------------- IK targets
fc, _fmn, _fmx = cup_bbox_center(full_prim)
ec, _emn, _emx = cup_bbox_center(empty_prim)
top = float(_fmx[2])            # full-cup rim height (reachable band)
q_home = np.array([CROUCH.get(j, 0.0) for j in ARM], float)

if args.phase in ("b1_pregrasp", "c0_proto_grasp"):
    log(f"PHASE={args.phase}: user-guided pre-grasp arm pose, staged hand closure only.")
    q_b1 = np.clip(arm_from_deg(B1_PREGRASP_ARM_DEG), lim_lo, lim_hi)
    hand_target_deg = B1_PREGRASP_HAND_DEG if args.phase == "b1_pregrasp" else C0_PROTO_GRASP_HAND_DEG
    h_target = hand_from_deg(hand_target_deg)
    open_fingers()
    drive(q_home)
    settle(60)
    move(q_home, q_b1, 120)
    h0 = hand_from_deg(B1_PREGRASP_HAND_DEG) if args.phase == "c0_proto_grasp" else np.zeros(len(RHAND), dtype=float)
    if args.phase == "c0_proto_grasp":
        for k, j in enumerate(RHAND):
            base_q[idx[j]] = h0[k]
    for s in range(40):
        a = smoother((s + 1.0) / 40)
        for k, j in enumerate(RHAND):
            base_q[idx[j]] = (1 - a) * h0[k] + a * h_target[k]
        tick(q_b1)
    for _ in range(40):
        tick(q_b1)
    q_export = base_q.copy()
    q_export[ai] = q_b1
    author_joint_state(q_export)
    log(f"{args.phase} arm(rad)={q_b1.round(3)} hand(rad)={h_target.round(3)}")
    log(f"{args.phase} palm={world_pos(palm_prim).round(3)} full_cup={cup_bbox_center(full_prim)[0].round(3)} empty_cup={cup_bbox_center(empty_prim)[0].round(3)}")
    if args.save_usd:
        stage.GetRootLayer().Export(os.path.abspath(args.save_usd))
        log(f"saved {args.save_usd}")
    log(f"done {args.phase} phase")
    sim.close(); raise SystemExit

# grasp near the rim (the arm reaches the z~1.0 band comfortably; the base is lower)
P_pre = np.array([fc[0], fc[1] + 0.06, top + 0.06])   # approach from robot side (+Y), above rim
P_grasp = np.array([fc[0], fc[1], top - 0.01])         # palm at the cup rim
P_lift = np.array([fc[0], fc[1], top + 0.16])          # straight up
P_carry = np.array([ec[0], ec[1] + 0.02, top + 0.12])  # just above + over the empty cup

log("solving IK ...")
q_pre, e_pre = ik(P_pre, q_home)
q_grasp, e_grasp = ik(P_grasp, q_pre)
q_lift, e_lift = ik(P_lift, q_grasp)
q_carry, e_carry = ik(P_carry, q_lift)
log(f"IK residuals(m): pre={e_pre:.3f} grasp={e_grasp:.3f} lift={e_lift:.3f} carry={e_carry:.3f}")
for tag, q in (("pre", q_pre), ("grasp", q_grasp), ("lift", q_lift), ("carry", q_carry)):
    log(f"  palm@{tag} = {fk(q).round(3)}")

# drive arm home and settle before executing (drive-only: converges exactly)
drive(q_home)
settle(60)

if args.phase == "reach":
    log("PHASE=reach: home -> pre -> grasp (no lift). Rendering.")
    log(f"  DIAG after drive(home)+settle: palm={world_pos(palm_prim).round(3)} "
        f"arm={np.asarray(robot.get_joint_positions())[ai].round(2)} q_home={q_home.round(2)}")
    move(q_home, q_pre, 90)
    log(f"  DIAG after move->pre: palm={world_pos(palm_prim).round(3)} "
        f"arm={np.asarray(robot.get_joint_positions())[ai].round(2)} q_pre={q_pre.round(2)}")
    move(q_pre, q_grasp, 70)
    log(f"  DIAG after move->grasp: palm={world_pos(palm_prim).round(3)} "
        f"arm={np.asarray(robot.get_joint_positions())[ai].round(2)} q_grasp={q_grasp.round(2)}")
    close_fingers()
    for _ in range(70):
        tick(q_grasp)
    log(f"reached; palm={world_pos(palm_prim).round(3)}  grasp target={P_grasp.round(3)}  "
        f"cup_ctr={cup_bbox_center(full_prim)[0].round(3)}")
    if args.save_usd:
        stage.GetRootLayer().Export(os.path.abspath(args.save_usd))
        log(f"saved {args.save_usd}")
    log("done reach phase")
    sim.close(); raise SystemExit

# ---------------------------------------------------------------- PHASE = full
log("PHASE=full: reach -> grasp -> lift -> carry -> pour -> settle -> place")


def fluid_centroid():
    pos = UsdGeom.PointInstancer(stage.GetPrimAtPath("/World/fluid")).GetPositionsAttr().Get()
    p = np.array([[v[0], v[1], v[2]] for v in pos])
    return p, p.mean(axis=0)


def count_in(cup_xy, r=0.06):
    p, _ = fluid_centroid()
    d = np.hypot(p[:, 0] - cup_xy[0], p[:, 1] - cup_xy[1])
    return int((d < r).sum())


_p0, c0 = fluid_centroid()
log(f"fluid start centroid={c0.round(3)}  in labg4={count_in(fc[:2])} in labg9={count_in(ec[:2])}")

# --- approach + grasp
move(q_home, q_pre, 90)
move(q_pre, q_grasp, 70)
close_fingers()
for _ in range(60):
    tick(q_grasp)

# handoff: record cup pose relative to palm, then slave cup to palm
M_off = world_T(full_prim) * world_T(palm_prim).GetInverse()


def follow_upright():
    labg4_op.Set(M_off * world_T(palm_prim))


# --- lift + carry (cup rigidly follows palm)
move(q_grasp, q_lift, 80, cup_fn=follow_upright)
move(q_lift, q_carry, 110, cup_fn=follow_upright)
for _ in range(40):
    tick(q_carry, cup_fn=follow_upright)

# --- pour: PIN the cup spout just over the empty cup's rim and tilt about that point so
#     the fluid pours straight in (not a scattered free-fall from 18 cm up + off-target).
M_up = M_off * world_T(palm_prim)             # cup world transform at carry (upright)
_cc, _cmn, _cmx = cup_bbox_center(full_prim)   # full-cup bbox at carry
# spout = the -Y rim point (the edge that dips first under +X tilt), in world at carry
S0 = Gf.Vec3d(float(_cc[0]), float(_cmn[1]), float(_cmx[2]))
_ec2, _emn2, _emx2 = cup_bbox_center(empty_prim)
# target: directly over the empty cup rim, small gap so the stream drops right in
S_tgt = Gf.Vec3d(float(_ec2[0]), float(_ec2[1]), float(_emx2[2]) + 0.015)
Tn = Gf.Matrix4d().SetTranslate(-S0)
Tp = Gf.Matrix4d().SetTranslate(S_tgt)
TILT = 108.0
log(f"pour: spout S0={tuple(round(v,3) for v in S0)} -> target={tuple(round(v,3) for v in S_tgt)}")


def tilted(deg):
    R = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), deg))
    return lambda: labg4_op.Set(M_up * Tn * R * Tp)


NP = 150
for s in range(NP):
    tick(q_carry, cup_fn=tilted(TILT * smoother(s / NP)))
for h in range(120):  # hold at full tilt, let it drain
    tick(q_carry, cup_fn=tilted(TILT))
    if h % 40 == 0:
        _pp, _cpp = fluid_centroid()
        log(f"  drain h={h} centroid={_cpp.round(3)} in labg9={count_in(ec[:2])} in labg4={count_in(fc[:2])}")
log(f"after pour: in labg4={count_in(fc[:2])} in labg9={count_in(ec[:2])}")

# --- return upright, place cup back, release
for s in range(90):
    tick(q_carry, cup_fn=tilted(TILT * (1 - smoother(s / 90))))
move(q_carry, q_lift, 90, cup_fn=follow_upright)
move(q_lift, q_grasp, 70, cup_fn=follow_upright)
labg4_op.Set(M_table_pose)
open_fingers()
for _ in range(40):
    tick(q_grasp)
move(q_grasp, q_pre, 60)
for _ in range(40):
    tick(q_home if False else q_pre)

_pf, cf = fluid_centroid()
n4, n9 = count_in(fc[:2]), count_in(ec[:2])
log(f"FINAL fluid centroid={cf.round(3)}  in labg4={n4}  in labg9={n9}  (start labg4={_p0.shape[0]})")
log(f"RESULT: {'PASS' if n9 > n4 else 'REVIEW'} transfer to empty cup")

if args.save_usd:
    stage.GetRootLayer().Export(os.path.abspath(args.save_usd))
    log(f"saved {args.save_usd}")
log(f"wrote frames to {frames_dir} (cap={cap[0]}, saved={cap[0] // args.capture_every})")
log("done full phase")
sim.close()
