#!/usr/bin/env python3
"""banner4 CONTACT sandbox: real rigid-contact physics between the G1 and the cups.

The kinematic-follow pipeline (g1_grasp_stages.py) shipped with every G1 collider
authored collisionEnabled=False and both cups kinematic — the hand passed through
the glass and the cup was tethered to the script. This sandbox is the opposite
contract: cups fully dynamic, right arm + right hand colliders live, PBD fluid
live, and the script NEVER writes a cup pose. Whatever moves, moved by contact.

  prep   : on the UNTOUCHED checkpoint (cups kinematic, colliders off) pre-solve
           IK waypoints that sweep the palm through the cup station -> JSON.
           Solving on the live scene would knock the cup around mid-solve.
  settle : enable contact + dynamic cups, hold station 2 s, gate stability,
           save the sandbox checkpoint USD.
  push   : load the sandbox, execute the pre-solved waypoints slowly; gates:
           the cup must translate (>= 2.5 cm) continuously (no teleports).

Env: g1skillgen. World() harness rules apply (physics_dt=1/120, GPU dynamics +
GPU broadphase forced). Every state probe ends on a rendered frame.
"""
from __future__ import annotations
import argparse
import json
import math
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--phase", choices=["prep", "settle", "push"], required=True)
ap.add_argument("--usd", required=True, help="input USD")
ap.add_argument("--save_usd", default="", help="output checkpoint USD (settle/push)")
ap.add_argument("--traj_json", default="output/banner4/pour_seq/contact_sandbox_v01/push_traj.json")
ap.add_argument("--out_dir", default="output/banner4/pour_seq/contact_sandbox_v01")
ap.add_argument("--capture_every", type=int, default=4)
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--push_z_drop", type=float, default=0.0,
                help="lower the palm this much below its s1 height before pushing; "
                "0 keeps the fingers at their s1 mid-glass heights (z 0.92-0.98)")
ap.add_argument("--seg_steps", type=int, default=90, help="physics steps per waypoint segment")
args = ap.parse_args()

if not os.path.isfile(args.usd):
    raise SystemExit(f"[sandbox] FATAL: input USD does not exist: {args.usd}")
phase_dir = os.path.join(args.out_dir, args.phase)
os.makedirs(phase_dir, exist_ok=True)

from isaacsim import SimulationApp
sim = SimulationApp({"headless": True, "width": args.width, "height": args.height})

import carb
import omni.timeline
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
import isaacsim.core.utils.stage as stage_utils
import omni.replicator.core as rep
from PIL import Image

carb.settings.get_settings().set("/rtx/rendermode", "RaytracedLighting")

DEG = math.pi / 180.0
ROBOT = "/World/G1"
PALM = f"{ROBOT}/right_hand_base_link"
FULL_CUP = "/World/labg4"
EMPTY_CUP = "/World/labg9"
FLUID = "/World/fluid"
ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
       "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
IK_JOINTS = ARM[:5]
RHAND = ["R_index_proximal_joint", "R_index_intermediate_joint",
         "R_middle_proximal_joint", "R_middle_intermediate_joint",
         "R_ring_proximal_joint", "R_ring_intermediate_joint",
         "R_pinky_proximal_joint", "R_pinky_intermediate_joint",
         "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint",
         "R_thumb_intermediate_joint", "R_thumb_distal_joint"]

# links whose colliders come alive in the sandbox (right arm chain + right hand)
CONTACT_LINKS = [
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link",
    "right_wrist_yaw_link", "right_hand_camera_base_link", "right_hand_base_link",
    "R_index_proximal", "R_index_intermediate", "R_middle_proximal", "R_middle_intermediate",
    "R_ring_proximal", "R_ring_intermediate", "R_pinky_proximal", "R_pinky_intermediate",
    "R_thumb_proximal_base", "R_thumb_proximal", "R_thumb_intermediate", "R_thumb_distal",
]

CAMS = {  # same verified station angles as g1_grasp_stages
    "wide":  ((0.15, -5.05, 1.45), (-1.30, -6.05, 0.95), 24.0),
    "front": ((-1.55, -6.35, 1.10), (-1.40, -5.80, 0.99), 35.0),
    "sideX": ((-1.90, -5.95, 1.05), (-1.40, -5.80, 0.99), 35.0),
    "high":  ((-1.10, -5.45, 1.45), (-1.40, -5.80, 0.99), 35.0),
}


def log(*a):
    print("[sandbox]", *a, flush=True)


metrics = {"phase": args.phase, "usd_in": args.usd, "usd_out": args.save_usd,
           "gates": {}, "probes": {}}


def gate(name, ok, detail):
    metrics["gates"][name] = {"pass": bool(ok), "detail": detail}
    log(f"GATE {name}: {'PASS' if ok else 'FAIL'} — {detail}")
    return bool(ok)


# ---------------------------------------------------------------- stage authoring
# Contact flags are authored on the OPENED STAGE BEFORE World() is constructed,
# so PhysX parses the scene with contact live from step 0 (no runtime toggling).
stage_utils.open_stage(os.path.abspath(args.usd))
stage = omni.usd.get_context().get_stage()

if args.phase in ("settle", "push"):
    n_on = 0
    for link in CONTACT_LINKS:
        cp = stage.GetPrimAtPath(f"{ROBOT}/{link}/collisions")
        if not cp:
            log(f"WARN: no collisions prim for {link}")
            continue
        UsdPhysics.CollisionAPI(cp).GetCollisionEnabledAttr().Set(True)
        n_on += 1
    for cup in (FULL_CUP, EMPTY_CUP):
        rb = UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(cup))
        rb.GetKinematicEnabledAttr().Set(False)
        rb.GetRigidBodyEnabledAttr().Set(True)
    log(f"contact authored: {n_on}/{len(CONTACT_LINKS)} G1 colliders ON, cups DYNAMIC")

world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
pc = world.get_physics_context()
pc.enable_gpu_dynamics(True)
try:
    pc.set_broadphase_type("GPU")
except Exception:
    pass
robot = SingleArticulation(prim_path=f"{ROBOT}/root_joint", name="g1")
world.scene.add(robot)

annots = {}
for name, (pos, look, f) in CAMS.items():
    cam = rep.create.camera(position=pos, look_at=look, focal_length=f)
    rp = rep.create.render_product(cam, (args.width, args.height))
    an = rep.AnnotatorRegistry.get_annotator("rgb")
    an.attach([rp])
    annots[name] = an
    os.makedirs(os.path.join(phase_dir, f"frames_{name}"), exist_ok=True)
for cp in stage.Traverse():
    if cp.IsA(UsdGeom.Camera):
        UsdGeom.Camera(cp).CreateClippingRangeAttr().Set(Gf.Vec2f(0.005, 100000.0))

world.reset()
names = robot.dof_names
idx = {n: i for i, n in enumerate(names)}

ac = robot.get_articulation_controller()
av = robot._articulation_view
kp, kd = ac.get_gains()
kp = np.asarray(kp, float).copy(); kd = np.asarray(kd, float).copy()
try:
    mef = np.asarray(av.get_max_efforts(), float).reshape(-1).copy()
except Exception:
    mef = np.zeros(robot.num_dof)
for j in ARM:
    kp[idx[j]] = 4000.0; kd[idx[j]] = 200.0; mef[idx[j]] = 1.0e6
for j in RHAND:
    kp[idx[j]] = 400.0; kd[idx[j]] = 20.0; mef[idx[j]] = 1.0e4
ac.set_gains(kps=kp, kds=kd)
try:
    av.set_max_efforts(mef[None, :])
except Exception as e:
    log(f"set_max_efforts warn: {e}")

# ---------------------------------------------------------------- helpers
xc = UsdGeom.XformCache(Usd.TimeCode.Default())
palm_prim = stage.GetPrimAtPath(PALM)
full_prim = stage.GetPrimAtPath(FULL_CUP)


def world_T(prim):
    xc.Clear()
    return xc.GetLocalToWorldTransform(prim)


def world_pos(prim):
    return np.array(world_T(prim).ExtractTranslation())


def bbox(prim):
    b = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]) \
        .ComputeWorldBound(prim).ComputeAlignedBox()
    mn, mx = np.array(b.GetMin()), np.array(b.GetMax())
    return (mn + mx) / 2, mn, mx


def cup_tilt_deg():
    r = world_T(full_prim).ExtractRotationMatrix()
    up = np.array(r.GetRow(2))  # local z in world
    return float(math.degrees(math.acos(np.clip(abs(up[2]) / np.linalg.norm(up), -1, 1))))


fluid_pi = UsdGeom.PointInstancer(stage.GetPrimAtPath(FLUID))


def fluid_pts():
    return np.array([[v[0], v[1], v[2]] for v in fluid_pi.GetPositionsAttr().Get()])


def fluid_in_cup(cup_ctr, r=0.075):
    p = fluid_pts()
    d = np.hypot(p[:, 0] - cup_ctr[0], p[:, 1] - cup_ctr[1])
    return int((d < r).sum())


_timeline = omni.timeline.get_timeline_interface()
cap = [0]


def _capture():
    world.step(render=True)
    rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=True)
    if not _timeline.is_playing():
        _timeline.play()
    n = cap[0] // args.capture_every
    for name, an in annots.items():
        a = np.asarray(an.get_data())
        if a.ndim == 1:
            a = a.reshape((args.height, args.width, -1))
        Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)[:, :, :3]).save(
            os.path.join(phase_dir, f"frames_{name}", f"f{n:04d}.png"))


base_q = np.asarray(robot.get_joint_positions(), float).copy()


def drive():
    ac.apply_action(ArticulationAction(joint_positions=base_q))


def tick():
    drive()
    world.step(render=False)
    if cap[0] % args.capture_every == 0:
        _capture()
    cap[0] += 1


def settle_steps(n):
    for _ in range(n):
        tick()


jlim = {}
for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)):
    if p.IsA(UsdPhysics.RevoluteJoint) and p.GetName() in idx:
        j = UsdPhysics.RevoluteJoint(p)
        lo, hi = j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()
        jlim[p.GetName()] = ((lo if lo is not None else -180.0) * DEG,
                             (hi if hi is not None else 180.0) * DEG)


def fk_probe(qmap, n=10):
    q = base_q.copy()
    for j, v in qmap.items():
        q[idx[j]] = v
    ac.apply_action(ArticulationAction(joint_positions=q))
    for _ in range(n - 1):
        world.step(render=False)
    world.step(render=True)  # sim state reaches USD only on rendered frames
    return world_pos(palm_prim)


def solve_ik(target, iters=30, damp=0.08, dq=0.05, tol=3e-3):
    q = np.array([float(base_q[idx[j]]) for j in IK_JOINTS])
    lo = np.array([jlim[j][0] for j in IK_JOINTS])
    hi = np.array([jlim[j][1] for j in IK_JOINTS])
    err = 1e9
    for it in range(iters):
        p0 = fk_probe(dict(zip(IK_JOINTS, q)))
        e = np.asarray(target) - p0
        err = float(np.linalg.norm(e))
        if err < tol:
            break
        J = np.zeros((3, len(q)))
        for c in range(len(q)):
            qd = q.copy(); qd[c] += dq
            J[:, c] = (fk_probe(dict(zip(IK_JOINTS, qd))) - p0) / dq
        if it == 0 and float(np.abs(J).max()) < 1e-6:
            log("IK diag: JACOBIAN IS ZERO — FK probes are not moving the arm")
            break
        step = J.T @ np.linalg.solve(J @ J.T + damp**2 * np.eye(3), e)
        q = np.clip(q + np.clip(step, -0.2, 0.2), lo, hi)
    ac.apply_action(ArticulationAction(joint_positions=base_q))
    for _ in range(19):
        world.step(render=False)
    world.step(render=True)
    return dict(zip(IK_JOINTS, q)), err


# ---------------------------------------------------------------- shared state
cup_c0, cup_mn0, cup_mx0 = bbox(full_prim)
fluid_total = len(fluid_pts())
fluid_n0 = fluid_in_cup(cup_c0)
palm_p0 = world_pos(palm_prim)
log(f"loaded: cup ctr={cup_c0.round(3)} tilt={cup_tilt_deg():.1f}deg "
    f"fluid={fluid_n0}/{fluid_total} palm={palm_p0.round(3)}")

# ================================================================ PREP
if args.phase == "prep":
    # LATERAL sweep, not a reach-out: pushing along palm->cup (-Y here) extends
    # the arm past its envelope (57-83 mm IK plateaus, the same wall that froze
    # the s4 keyframes). The open fingers already straddle the glass on the -X
    # side, so sweeping the palm toward the cup axis tangentially drives the
    # finger surfaces into the wall — every waypoint stays at constant reach.
    log("PREP: pre-solve LATERAL sweep waypoints on the untouched (kinematic) scene.")
    settle_steps(30)
    palm = world_pos(palm_prim)
    cup_c = bbox(full_prim)[0]
    fing = np.mean([bbox(stage.GetPrimAtPath(f"{ROBOT}/{l}"))[0][:2] for l in
                    ("R_index_intermediate", "R_middle_intermediate",
                     "R_ring_intermediate", "R_pinky_intermediate")], axis=0)
    u = cup_c[:2] - fing
    u = u / np.linalg.norm(u)  # fingers -> cup axis: the direction that makes contact
    z_push = float(palm[2]) - args.push_z_drop
    log(f"station: palm={palm[:2].round(3)} fingers={fing.round(3)} "
        f"sweep dir={u.round(3)} z={z_push:.3f}")

    offsets = [0.02, 0.05, 0.08, 0.11]
    waypoints, errs = [], []
    for s in offsets:
        tgt = np.array([palm[0] + u[0] * s, palm[1] + u[1] * s, z_push])
        qmap, err = solve_ik(tgt)
        waypoints.append({j: float(v) for j, v in qmap.items()})
        errs.append(err)
        for j, v in qmap.items():  # seed the next solve from this waypoint
            base_q[idx[j]] = v
        log(f"  offset {s:.3f}: IK err {err*1000:.1f} mm")

    ok = gate("ik_converged", max(errs) < 0.015,
              f"worst IK residual {max(errs)*1000:.1f} mm over {len(offsets)} waypoints")
    os.makedirs(os.path.dirname(args.traj_json), exist_ok=True)
    with open(args.traj_json, "w") as f:
        json.dump({"joints": IK_JOINTS, "offsets": offsets, "waypoints": waypoints,
                   "ik_err_m": errs, "push_dir_xy": [float(u[0]), float(u[1])],
                   "z_push": z_push, "source_usd": args.usd}, f, indent=2)
    log(f"trajectory -> {args.traj_json}")

# ================================================================ SETTLE
elif args.phase == "settle":
    log("SETTLE: contact live, cups dynamic — hold station 2 s and gate stability.")
    settle_steps(240)
    cup_c = bbox(full_prim)[0]
    drift = float(np.linalg.norm(cup_c - cup_c0))
    tilt = cup_tilt_deg()
    fl = fluid_in_cup(cup_c)
    palm_move = float(np.linalg.norm(world_pos(palm_prim) - palm_p0))
    jv = float(np.abs(np.asarray(robot.get_joint_velocities(), float)).max())
    ok = True
    ok &= gate("cup_settled", drift < 0.015 and tilt < 3.0,
               f"drift {drift*1000:.1f} mm, tilt {tilt:.2f} deg (dynamic, on its foot)")
    ok &= gate("fluid_held", fl >= int(0.90 * fluid_n0),
               f"{fl}/{fluid_n0} particles still in cup")
    ok &= gate("arm_stable", palm_move < 0.020 and jv < 2.0,
               f"palm moved {palm_move*1000:.1f} mm, max joint vel {jv:.2f} rad/s")
    metrics["probes"].update({"drift_mm": drift * 1000, "tilt_deg": tilt,
                              "fluid": fl, "palm_move_mm": palm_move * 1000})

# ================================================================ PUSH
elif args.phase == "push":
    log("PUSH: execute pre-solved waypoints; the cup is on its own physics.")
    with open(args.traj_json) as f:
        traj = json.load(f)
    joints, waypoints = traj["joints"], traj["waypoints"]
    push_dir = np.array(traj["push_dir_xy"])
    settle_steps(60)  # 0.5 s of quiet contact-on hold before moving
    cup_start = bbox(full_prim)[0]

    track = []  # (cap_index, cup_xy, tilt, fluid)

    def probe_cup():
        c = bbox(full_prim)[0]
        track.append({"cap": cap[0], "xy": [float(c[0]), float(c[1])], "z": float(c[2]),
                      "tilt": cup_tilt_deg(), "fluid": fluid_in_cup(c)})

    q_from = {j: float(base_q[idx[j]]) for j in joints}
    for wi, wp in enumerate(waypoints):
        for s in range(args.seg_steps):
            a = 0.5 - 0.5 * math.cos(math.pi * (s + 1.0) / args.seg_steps)
            for j in joints:
                base_q[idx[j]] = (1 - a) * q_from[j] + a * wp[j]
            tick()
            if cap[0] % args.capture_every == 1:  # right after a rendered capture
                probe_cup()
        q_from = wp
        log(f"waypoint {wi+1}/{len(waypoints)} done; cup at "
            f"{np.array(track[-1]['xy']).round(3) if track else '?'}")
    settle_steps(120)  # let the cup coast to rest, keep filming
    _capture()
    probe_cup()

    cup_end = np.array(track[-1]["xy"])
    disp = cup_end - cup_start[:2]
    dmag = float(np.linalg.norm(disp))
    ddir = float(np.dot(disp / max(dmag, 1e-9), push_dir))
    jumps = [float(np.linalg.norm(np.array(b["xy"]) - np.array(a["xy"])))
             for a, b in zip(track, track[1:])]
    fl_end = track[-1]["fluid"]
    tilt_end = track[-1]["tilt"]
    ok = True
    ok &= gate("cup_moved_by_contact", dmag >= 0.025,
               f"cup translated {dmag*1000:.1f} mm (script never wrote its pose)")
    ok &= gate("motion_continuous", max(jumps) < 0.025 if jumps else False,
               f"max inter-frame jump {max(jumps)*1000:.1f} mm — pushed, not teleported")
    ok &= gate("direction_sane", ddir > 0.3,
               f"displacement·push_dir = {ddir:.2f}")
    gate("fluid_report", True, f"{fl_end}/{fluid_n0} particles in cup, tilt {tilt_end:.1f} deg "
         f"(report only — sloshing/tipping is legitimate physics)")
    metrics["probes"].update({"disp_mm": dmag * 1000, "dir_dot": ddir,
                              "tilt_end": tilt_end, "fluid_end": fl_end,
                              "track": track})

# ---------------------------------------------------------------- save + report
if args.phase in ("settle", "push") and args.save_usd:
    author = stage.GetPrimAtPath(ROBOT)
    try:
        from pxr import PhysxSchema
        for p in Usd.PrimRange(author):
            if p.IsA(UsdPhysics.RevoluteJoint) and p.GetName() in idx:
                degv = float(base_q[idx[p.GetName()]] / DEG)
                js = PhysxSchema.JointStateAPI.Apply(p, "angular")
                js.CreatePositionAttr(degv).Set(degv)
                js.CreateVelocityAttr(0.0).Set(0.0)
                drv = UsdPhysics.DriveAPI.Get(p, "angular")
                if drv:
                    drv.CreateTargetPositionAttr(degv).Set(degv)
    except Exception as e:
        log(f"joint-state authoring warn: {e}")
    stage.GetRootLayer().Export(os.path.abspath(args.save_usd))
    log(f"saved {args.save_usd}")

metrics["frames_captured"] = cap[0] // args.capture_every
metrics["result"] = "PASS" if ok else "FAIL"
with open(os.path.join(phase_dir, f"{args.phase}_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
log(f"metrics -> {os.path.join(phase_dir, args.phase + '_metrics.json')}")
log(f"RESULT {args.phase}: {'PASS' if ok else 'FAIL'}")
sim.close()
