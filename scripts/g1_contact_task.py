#!/usr/bin/env python3
"""banner4 FULL pick-lift-pour-replace on REAL contact physics (no kinematic attach).

Sequence (each substage = one headless run: load previous checkpoint, act, gate,
save checkpoint + 4-cam frames + metrics; video assembled after):

  ghost_prep : on a GHOST copy (G1 colliders OFF, cups kinematic) pre-solve the IK
               waypoints a stage needs -> JSON. Real-physics runs never solve IK
               (probing physically disturbs a live world).
  m0_side    : retract the hand from the cup flank and park it at the G1's side.
               This checkpoint is the TASK START.
  m1_approach: side -> staging above the flank -> grasp station. No table strikes,
               cups undisturbed.
  (m2..m6 grasp/lift/move/pour/return are appended as each earlier stage passes.)

The cups are dynamic and the fluid is live in every real run; no code path writes
a cup pose. Env: g1skillgen; World() harness rules apply (physics_dt=1/120, GPU
dynamics + broadphase forced; every measurement probe ends on a rendered frame).
"""
from __future__ import annotations
import argparse
import json
import math
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--phase", required=True,
                choices=["ghost_prep", "side_scan", "grasp_scan", "palm_prep",
                         "claw_prep", "m0_side", "m1_approach", "m1_direct",
                         "m2a_palm", "m2b_claw", "m2c_scan", "m2c_hug", "m2_grasp",
                         "m3_lift", "m3_claw_lift", "pour_prep", "m4_pour"])
ap.add_argument("--usd", required=True)
ap.add_argument("--save_usd", default="")
ap.add_argument("--traj_json", default="output/banner4/pour_seq/contact_task_v01/task_traj.json")
ap.add_argument("--out_dir", default="output/banner4/pour_seq/contact_task_v01")
ap.add_argument("--capture_every", type=int, default=4)
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--seg_steps", type=int, default=90)
ap.add_argument("--close_rate", type=float, default=0.6,
                help="m2c: degrees the finger targets advance per closure "
                "iteration (4 physics ticks). The default 0.6 deg/iter is "
                "18 deg/s, and at that rate the hand drives carry 12-25 deg of "
                "ORDINARY TRACKING LAG in free air -- which is larger than any "
                "contact signal, so neither lag nor effort can tell touching "
                "from moving. Closing 8x slower drops the free-air lag to ~3 "
                "deg and makes contact measurable. Slower is also what the "
                "task wants: this is a glass.")
ap.add_argument("--grip_target", type=float, default=0.12,
                help="m2c: seat a finger once its proximal reaches this "
                "measured effort, N*m. The old 0.30 threshold was NEVER "
                "REACHABLE -- all four proximals top out at 0.11-0.29 -- so "
                "every finger ran to its position cap instead. Friction alone "
                "only needs ~0.03 N*m per finger to hold the 0.3 kg glass "
                "(mu 1.2, 45 mm lever), so 0.12 is a 4x margin.")
ap.add_argument("--max_lag", type=float, default=12.0,
                help="m2c: hard cap on how far a finger's target may run past "
                "its measured joint angle, deg. This is what bounds the "
                "over-press: uncapped, the target ended 10-20 deg INSIDE the "
                "glass, the drive saturated (12 rad/s for 3 s, measured via "
                "--trace_closure) and the pinky was pushed clean through the "
                "5-7 mm wall at 0.39 N*m. Note the drive's ordinary tracking "
                "lag in free air is already 10-20 deg, so lag is a bound, "
                "never a contact TEST.")
ap.add_argument("--trace_closure", default="",
                help="m2c: sample each fingertip's position relative to the cup "
                "axis every 4 ticks and write the trace here as JSON. This is "
                "the probe that shows WHETHER a fingertip enters the glass "
                "gradually (contact never generated) or in a jump (tunnelling).")
ap.add_argument("--engage_standoff", type=float, default=0.0,
                help="ghost_prep: if >0, solve ONLY an 'engage' waypoint at this "
                "standoff (+engage_lift +2cm) and merge into the traj json")
ap.add_argument("--engage_z", type=float, default=0.972,
                help="palm z of the engage waypoint (lower = wrap nearer the "
                "cup's center of mass; 0.972 tipped the glass at the rim)")
ap.add_argument("--claw_roll", type=float, default=-20.0,
                help="claw phases: roll of the hand about the palm normal, deg. "
                "0 = finger stack vertical with the thumb on top, which puts the "
                "thumb's own BASE within 12-35 mm of the rim so its knuckle "
                "arcs over the glass mouth at EVERY thumb joint angle. Rolling "
                "compresses that stack so the thumb drops to wall height. "
                "MEASURED off the meshes (grade_entry.py), mouth intrusion: "
                "roll 0 = 15.3 mm hanging across the mouth, -15 = 8.6, "
                "-20 = 0.6, -25 = 6.0; +20 puts the thumb 10.9 mm INSIDE the "
                "wall. -20 also gives the widest clearances (thumb 13.1, "
                "fingers 16.8, palm 9.6 mm) at 142 deg of wrap.")
ap.add_argument("--claw_thumb", default="-5.7,34,23,34",
                help="claw phases: thumb yaw,pitch,intermediate,distal in deg. "
                "PITCH is what lowers the thumb: extended (-5.7) it stands 83 mm "
                "above the rim and sits 13.1 mm off the wall -- not a grasp. At "
                "pitch 34 it folds down to 3.0 mm off the wall, opposing the "
                "fingers, and the entry is still clear by 3.2 mm (measured, "
                "grade_entry.py). Note the tip CANNOT clear the rim on this hand: "
                "the thumb is ~85 mm long off a base at z 0.974, so at any hand "
                "height where the fingers grip the wall, the tip is above 1.037. "
                "Yaw forward at entry is fatal -- it puts the thumb 8.9 mm INSIDE "
                "the wall and blocks the mouth by 9.4 mm. Yaw is for the hug.")
ap.add_argument("--claw_z", type=float, default=0.940,
                help="claw phases: world z of the hand-base origin at entry. "
                "0.960 is the measured optimum: any higher and the thumb rides "
                "over the rim instead of the wall, any lower and the curled "
                "fingertips reach the counter (they hang to z 0.915).")
ap.add_argument("--claw_open", type=float, default=1.0,
                help="claw phases: scale on the entry curl of all four fingers. "
                "<1 opens the claw wider, which buys the entry clearance needed "
                "to run a DEEPER pocket (--claw_c). Needed because at c=0.050 "
                "all four proximals still seat at the 95 deg CAP rather than on "
                "torque -- they are fully closed and the glass still is not "
                "filling the cage, so only middle/ring ever reach the 0.45 N*m "
                "clamp while index/pinky idle at 0.09-0.21.")
ap.add_argument("--thumb_cap", type=int, default=0,
                help="m2c: after the fence seats, fold the thumb DOWN over the "
                "glass mouth as an axial stop (1). The pour fails without it: "
                "the fence is a RING, so once the cup tilts, gravity pulls it "
                "straight out along its own axis -- in-hand drift went 5.3 -> "
                "19.3 -> 74.1 mm between 0 and 55 deg of tilt and only 4% of the "
                "fluid reached the receiver. The thumb cannot reach the WALL on "
                "this hand (it is above the rim at every yaw), but at the 0.050 "
                "pocket the folded thumb sits 14 mm over the MOUTH, which is a "
                "mechanical stop rather than a friction one. Entry still happens "
                "with the thumb extended, because folded it fouls the rim.")
ap.add_argument("--tilt_deg", type=float, default=110.0,
                help="pour: full tilt angle of the held cup, deg")
ap.add_argument("--finger_cap", type=float, default=95.0,
                help="closure cap on the four finger proximals, deg. The REAL "
                     "joint limit is 97.403; 95 was a self-imposed margin. The "
                     "fingers stop on this CAP, not on torque -- measured holding "
                     "efforts during the lift were 0.198/0.125/0.039/0.229 N*m "
                     "against a 0.45 N*m clamp, i.e. the ring finger is barely "
                     "touching. Raising the cap lets them close onto the GLASS "
                     "and load up, which is what resists the cup rotating inside "
                     "the finger ring during the pour. Never exceed 97.403.")
ap.add_argument("--pour_tilt_keys", type=int, default=4,
                help="tilt keyframes solved/blended through. The stock 4 turns the "
                     "tilt arc into 4 straight chords; the cup is being carried by "
                     "a RING of fingers with no thumb opposing, so every corner "
                     "between chords is an angular-acceleration impulse that walks "
                     "the cup out of the grip. More keys = the hand follows the arc.")
ap.add_argument("--pour_carry_keys", type=int, default=2,
                help="keyframes across the carry to the receiver")
ap.add_argument("--pour_slow", type=float, default=1.0,
                help="multiply every pour blend and settle duration. TOTAL tilt "
                     "time is held independent of --pour_tilt_keys, so keys buy "
                     "smoothness and this buys time; they are separate knobs.")
ap.add_argument("--pour_gap", type=float, default=0.015,
                help="pour: spout height above the receiver rim at full tilt (m)")
ap.add_argument("--entry_freeze", type=float, default=0.002,
                help="m2b: abort the entry push if the glass moves more than "
                "this (m). The guard exists so a claw that is too narrow reports "
                "WHERE it fouled instead of shoving the glass across the counter. "
                "At the shallow 0.0699 pocket 2 mm was never reached; at 0.050 "
                "the fence brushes the wall on the way in and the push froze at "
                "leg 3 step 149/220, leaving the glass 11.5 mm off the pocket "
                "centre -- an asymmetric grip that tilts on lift. 5 mm still "
                "catches a real shove (m2b once moved it 58 mm) while letting "
                "the hand seat around the glass.")
ap.add_argument("--hug_thumb", type=int, default=1,
                help="m2c: run the thumb-yaw hug (1) or leave the thumb parked "
                "out of the way (0). MEASURED on the m2c_hug checkpoint: the "
                "thumb NEVER reaches the wall -- R_thumb_distal is above the "
                "rim at every yaw in a full 80 deg sweep, and as the yaw closed "
                "the index effort collapsed 0.131 -> 0.000, i.e. the thumb was "
                "pushing the glass OUT of the fence while its own pitch drive "
                "saturated against the rim. With a deeper pocket the thumb also "
                "fouls the mouth on entry (-14.0 mm), so 0 is the setting that "
                "matches this hand's geometry.")
ap.add_argument("--claw_c", type=float, default=0.0699,
                help="claw phases: pocket depth along the palm normal, m -- how "
                "far the glass axis sits in FRONT of the hand-base origin. "
                "0.0699 was chosen for a clean, collision-free ENTRY and is "
                "22 mm too deep to GRIP: measured on the m2c_hug checkpoint, "
                "the middle/ring/pinky intermediates bottom out at their 97.4 "
                "deg joint limit still 19.4/21.7/21.2 mm off the wall and the "
                "palm sits 37.5 mm off, so the only contacts are the index tip "
                "(-1.1 mm) and the thumb on the RIM (+0.6 mm) -- a two-point "
                "pinch above the cup's centre of mass, which pivots out (9.2 "
                "deg tilt, 8.9 mm slip on a 12.6 mm lift). Smaller = the glass "
                "sits deeper in the claw and more digits reach the wall, at the "
                "cost of entry clearance (26.9 mm at 0.0699).")
ap.add_argument("--thumb_pitch_deg", type=float, default=0.0,
                help="m2: thumb proximal-pitch target during the yaw sweep "
                "(small values bring the pad down the near wall; bore-guarded)")
args = ap.parse_args()

if not os.path.isfile(args.usd):
    raise SystemExit(f"[task] FATAL: input USD does not exist: {args.usd}")
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
FINGERS = [  # (name, proximal, intermediate, inter/prox ratio from the C0 posture)
    ("index",  "R_index_proximal_joint",  "R_index_intermediate_joint",  28.0 / 45.0),
    ("middle", "R_middle_proximal_joint", "R_middle_intermediate_joint", 30.0 / 48.0),
    ("ring",   "R_ring_proximal_joint",   "R_ring_intermediate_joint",   28.0 / 44.0),
    ("pinky",  "R_pinky_proximal_joint",  "R_pinky_intermediate_joint",  25.0 / 40.0),
]
# design brief 2026-08-14: force-limited grip — the effort CLAMP, not stiffness,
# sets pinch force (kp=400 is a cup-launcher: ~23 N/step growth at 120 Hz).
# view units: kp N*m/rad, kd N*m*s/rad, mef N*m.
# kd tripled vs the brief: contact-driven limit cycles buzzed the free fingers
# at 12.5 rad/s (m0f). Saturation force is set by mef, unchanged.
HAND_GAINS = {
    "R_index_proximal_joint": (20, 0.9, 0.45), "R_index_intermediate_joint": (10, 0.5, 0.35),
    "R_middle_proximal_joint": (20, 0.9, 0.45), "R_middle_intermediate_joint": (10, 0.5, 0.35),
    "R_ring_proximal_joint": (20, 0.9, 0.45), "R_ring_intermediate_joint": (10, 0.5, 0.35),
    "R_pinky_proximal_joint": (20, 0.9, 0.45), "R_pinky_intermediate_joint": (10, 0.5, 0.35),
    "R_thumb_proximal_yaw_joint": (25, 1.5, 1.2), "R_thumb_proximal_pitch_joint": (30, 1.5, 0.70),
    "R_thumb_intermediate_joint": (15, 0.8, 0.40), "R_thumb_distal_joint": (10, 0.5, 0.20),
}
THUMB_YAW = "R_thumb_proximal_yaw_joint"
THUMB_LOCKED = ["R_thumb_proximal_pitch_joint", "R_thumb_intermediate_joint", "R_thumb_distal_joint"]
HAND_LINKS = ["right_hand_base_link", "R_index_intermediate", "R_middle_intermediate",
              "R_ring_intermediate", "R_pinky_intermediate", "R_thumb_distal"]
CONTACT_LINKS = [  # right arm chain + right hand: the links with live colliders
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link",
    "right_wrist_yaw_link", "right_hand_camera_base_link", "right_hand_base_link",
    "R_index_proximal", "R_index_intermediate", "R_middle_proximal", "R_middle_intermediate",
    "R_ring_proximal", "R_ring_intermediate", "R_pinky_proximal", "R_pinky_intermediate",
    "R_thumb_proximal_base", "R_thumb_proximal", "R_thumb_intermediate", "R_thumb_distal",
]
def _pour_fracs(n):
    """(1..n)/n -- with n=4 this is exactly the historical 0.25/0.5/0.75/1.0 and
    with n=2 exactly the historical carry 0.5/1.0, so defaults are a no-op."""
    return [round((i + 1) / float(n), 4) for i in range(n)]


def _pk(f):
    return int(round(f * 100))


POUR_CARRY_F = _pour_fracs(args.pour_carry_keys)
POUR_TILT_F = _pour_fracs(args.pour_tilt_keys)
# hold TOTAL durations fixed against key count; --pour_slow scales them
_S = args.pour_slow
CARRY_SEG = max(20, int(round(180 * 2 / len(POUR_CARRY_F) * _S)))
TILT_SEG = max(20, int(round(150 * 4 / len(POUR_TILT_F) * _S)))
RET_SEG = max(15, int(round(110 * 4 / len(POUR_TILT_F) * _S)))
SETTLE_S = lambda n: max(10, int(round(n * _S)))
# inter-keyframe settle: hold the TOTAL constant as keys increase, otherwise more
# keyframes turn a continuous tilt into a stop-start ratchet -- the opposite of
# the smooth motion more keyframes were added to buy.
KEY_SETTLE_T = max(5, int(round(60 * 4 / len(POUR_TILT_F) * _S)))
KEY_SETTLE_C = max(5, int(round(60 * 2 / len(POUR_CARRY_F) * _S)))

COUNTER_TOP = 0.883
SLAB_X = (-1.613, -1.19)
SLAB_Y = (-6.32, -5.412)
R_WALL = 0.0425  # glass outer wall radius (grasp_scan 2026-08-14)
# Palm plate, probed OFFLINE from the m2 checkpoint mesh (link frame of
# right_hand_base_link): fingers curl toward -X, so the palm FACE normal is -X;
# fingers run along -Y; the thumb hangs off the -Z edge. The flat pressing pad
# is the heel/mid strip (y -70..0 at x -27.4..-25.0); the knuckle half of the
# shell recedes to x -9.6. In the m2 grip the face was ~51 deg off the wall —
# an EDGE contact, i.e. the pivot of the tilt-escape channel (finding 11).
PALM_N_LOC = np.array([-1.0, 0.0, 0.0])       # palm-face normal, link frame
PALM_PAD_LOC = np.array([-0.0250, -0.060, 0.0])  # mid-palm pad point, link frame
PALM_FINGER_LOC = np.array([0.0, -1.0, 0.0])  # wrist->fingers axis, in the face plane
# CLAW ENTRY pose + pocket, MEASURED offline (2026-08-17,
# scratchpad/claw_entry_sweep.py): every right-hand link's real mesh, transformed
# at 288 finger/thumb configurations, and for each the EXACT depth a straight
# push can drive an 85 mm cylinder to (the cup disc sweeps
# {|dy|<=R, alpha >= c-sqrt(R^2-dy^2)}, so c_min = max over hand points of
# alpha + sqrt(R^2-dy^2)).  Winner: all four fingers curled to 45% of range --
# they wrap to the glass EQUATOR (90.6 deg, which is the geometric ceiling for a
# straight push) -- and the thumb parked at BOTH its stops, which leaves it
# behind the palm face where it cannot block the mouth.  That mouth admits a
# 89.0 mm glass; ours is 85.0.  Pocket: the glass axis lands CLAW_C along the
# palm normal and CLAW_S across the palm from the hand-base origin, and at rest
# NOTHING touches (palm 13.2, fingers 6.4, thumb 6.4 mm clear).  The thumb
# reaches the wall after ~16 deg of yaw -- that move is the hug, not this phase.
# Commanded per finger, NOT one number: driven to a common 43.8 deg the four
# fingers jam on each other in a cascade (index reaches it, middle stalls 8 deg
# short, ring 15, pinky 23, each carrying 0.03-0.07 N*m of internal squeeze).
# Tightening the digit contact offset 5 mm -> 2 mm changed it by 0.8 deg, so it
# is the finger HULLS, not offset inflation.  Curling further (68.8, 91.7 deg)
# clears the jam entirely -- 43.8 sits inside an interference band -- but that
# much curl closes the mouth on the glass.  So: command the fan the hand
# actually holds, and nothing fights.  (probe_claw_drive.py / probe_claw_real.py)
CLAW_FING = {k: v * args.claw_open for k, v in
             {"index": 0.765, "middle": 0.624, "ring": 0.499, "pinky": 0.370}.items()}
CLAW_THUMB = dict(zip(["R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint",
                       "R_thumb_intermediate_joint", "R_thumb_distal_joint"],
                      [float(v) * math.pi / 180.0 for v in args.claw_thumb.split(",")]))
CLAW_C, CLAW_S = args.claw_c, -0.100

CAMS = {
    "wide":  ((0.15, -5.05, 1.45), (-1.30, -6.05, 0.95), 24.0),
    "front": ((-1.55, -6.35, 1.10), (-1.40, -5.80, 0.99), 35.0),
    "sideX": ((-1.90, -5.95, 1.05), (-1.40, -5.80, 0.99), 35.0),
    "high":  ((-1.10, -5.45, 1.45), (-1.40, -5.80, 0.99), 35.0),
}
if args.phase == "m4_pour":
    # the action moves to the gap BETWEEN the cups; the grasp-station framing
    # crops the receiver clean out of shot, so the first pour run produced no
    # usable picture of its own 28% transfer.
    _PF = (-1.32, -5.91, 1.02)
    CAMS = {
        "wide":  ((0.15, -5.05, 1.45), (-1.30, -6.05, 0.95), 24.0),
        "front": ((-1.62, -6.48, 1.18), _PF, 32.0),
        "sideX": ((-1.98, -5.86, 1.16), _PF, 32.0),
        "high":  ((-1.02, -5.48, 1.58), _PF, 32.0),
    }


def log(*a):
    print("[task]", *a, flush=True)


metrics = {"phase": args.phase, "usd_in": args.usd, "usd_out": args.save_usd,
           "gates": {}, "probes": {}}


def gate(name, ok, detail):
    metrics["gates"][name] = {"pass": bool(ok), "detail": detail}
    log(f"GATE {name}: {'PASS' if ok else 'FAIL'} — {detail}")
    return bool(ok)


# ------------------------------------------------------------- stage authoring
stage_utils.open_stage(os.path.abspath(args.usd))
stage = omni.usd.get_context().get_stage()

# A USD carrying TIME-SAMPLED drive targets is UNCONTROLLABLE by a World()
# harness: Fabric caches the samples at stage open and re-asserts them every
# frame, pinning the articulation against apply_action (verified 2026-08-14:
# runtime Clear() does NOT purge the cache — the arm stays rigid; only a
# FILE-LEVEL strip works). Fail fast instead of silently freezing.
n_sampled = sum(1 for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT))
                if p.IsA(UsdPhysics.RevoluteJoint)
                for a in p.GetAttributes() if a.GetNumTimeSamples() > 0)
if n_sampled:
    raise SystemExit(f"[task] FATAL: {args.usd} carries {n_sampled} time-sampled joint "
                     "attrs (Play-animation). Control runs need the file-level-stripped "
                     "task base (banner4_task_base_v04.usdc) — see docs.")

GHOST = args.phase in ("ghost_prep", "side_scan", "grasp_scan", "palm_prep",
                       "claw_prep", "pour_prep")
if GHOST:
    # ghost copy: hand may pass through everything; cups pinned
    n = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
            n += 1
    for cup in (FULL_CUP, EMPTY_CUP):
        UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(cup)).GetKinematicEnabledAttr().Set(True)
    log(f"GHOST: {n} G1 colliders OFF, cups kinematic")

if args.phase != "ghost_prep":
    # symmetric high-friction contact + tamed depenetration (design brief),
    # applied in EVERY real phase: default hull contact offsets generate phantom
    # finger-finger contacts that buzz the force-limited drives (12.5 rad/s).
    # Bind the scene's grip material (us 1.4 / ud 1.2) to every hand collider,
    # tighten offsets, cap depenetration velocity, raise cup velocity iterations
    from pxr import UsdShade, PhysxSchema
    grip_mat = UsdShade.Material(stage.GetPrimAtPath("/World/PhysicsMaterials/grip"))
    nb = 0
    for link in CONTACT_LINKS:
        cp = stage.GetPrimAtPath(f"{ROBOT}/{link}/collisions")
        if not cp:
            continue
        UsdShade.MaterialBindingAPI.Apply(cp).Bind(grip_mat, materialPurpose="physics")
        pca = PhysxSchema.PhysxCollisionAPI.Apply(cp)
        pca.CreateContactOffsetAttr(0.005).Set(0.005)
        pca.CreateRestOffsetAttr(0.0).Set(0.0)
        nb += 1
    for cup in (FULL_CUP, EMPTY_CUP):
        cprim = stage.GetPrimAtPath(cup)
        prb = PhysxSchema.PhysxRigidBodyAPI.Apply(cprim)
        prb.CreateMaxDepenetrationVelocityAttr(1.0).Set(1.0)
        prb.CreateSolverVelocityIterationCountAttr(8).Set(8)
    for link in CONTACT_LINKS:
        lp = stage.GetPrimAtPath(f"{ROBOT}/{link}")
        if lp and lp.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(lp).CreateMaxDepenetrationVelocityAttr(1.0).Set(1.0)
    # author the CURRENT hand gains into the drives pre-parse: the input
    # checkpoint carries whatever HAND_GAINS were at ITS save time, and the
    # parsed maxForce beat the runtime set_max_efforts override (m2e ran
    # bit-identical to m2d despite raised clamps)
    for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)):
        if p.IsA(UsdPhysics.RevoluteJoint) and p.GetName() in HAND_GAINS:
            jkp, jkd, jmef = HAND_GAINS[p.GetName()]
            drv = UsdPhysics.DriveAPI.Get(p, "angular")
            if drv:
                drv.CreateStiffnessAttr(0.0).Set(jkp * math.pi / 180.0)
                drv.CreateDampingAttr(0.0).Set(jkd * math.pi / 180.0)
                drv.CreateMaxForceAttr(0.0).Set(jmef)
    log(f"m2 contact prep: grip material + offsets on {nb} hand colliders, "
        "cup depenetration 1.0 m/s, vel iters 8; hand drive gains re-authored")

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
    mef = np.full(robot.num_dof, 1.0e4)  # zeros = drive-OFF (limp), not unclamped
for j in ARM:
    kp[idx[j]] = 4000.0; kd[idx[j]] = 200.0; mef[idx[j]] = 1.0e6
for j, (jkp, jkd, jmef) in HAND_GAINS.items():
    kp[idx[j]] = jkp; kd[idx[j]] = jkd; mef[idx[j]] = jmef
# the torso is the arm's reference frame: weak asset-default waist drives let
# arm swings TWIST it (waist_roll 1.04 rad/s at "rest"; every IK sol drifts)
for j in idx:
    if "waist" in j:
        kp[idx[j]] = 8000.0; kd[idx[j]] = 400.0; mef[idx[j]] = 1.0e6
ac.set_gains(kps=kp, kds=kd)
try:
    av.set_max_efforts(mef[None, :])
except Exception as e:
    log(f"set_max_efforts warn: {e}")

# ------------------------------------------------------------------- helpers
xc = UsdGeom.XformCache(Usd.TimeCode.Default())
palm_prim = stage.GetPrimAtPath(PALM)
full_prim = stage.GetPrimAtPath(FULL_CUP)
empty_prim = stage.GetPrimAtPath(EMPTY_CUP)


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


def link_center(name):
    return bbox(stage.GetPrimAtPath(f"{ROBOT}/{name}"))[0]


def _palm_R():
    """palm rotation, rows = palm basis in world (USD is row-vector)."""
    return np.array(world_T(palm_prim).ExtractRotationMatrix()).reshape(3, 3)


def _cup_in_palm():
    """the cup's body origin in the palm's local frame (m).

    USD is row-vector: world = local * R, so local = world * R.T."""
    T = world_T(palm_prim)
    R = np.array(T.ExtractRotationMatrix()).reshape(3, 3)
    t = np.array(T.ExtractTranslation())
    return (world_pos(full_prim) - t) @ R.T


def _cup_axis_in_palm():
    """the cup's own +z axis, expressed in the palm frame (unit).

    This is the number that says whether the cup is ROTATING inside the
    hand.  The origin-displacement above cannot tell the two apart: the
    cup's body origin sits at its base, so a pure in-hand rotation about
    a contact near mid-height swings that origin tens of mm."""
    cz = np.array(world_T(full_prim).ExtractRotationMatrix().GetRow(2), float)
    cz /= max(np.linalg.norm(cz), 1e-12)
    return cz @ _palm_R().T


def cup_tilt_deg(prim):
    r = world_T(prim).ExtractRotationMatrix()
    up = np.array(r.GetRow(2))
    return float(math.degrees(math.acos(np.clip(abs(up[2]) / np.linalg.norm(up), -1, 1))))


# ---------------------------------------------------- closure tracer
def _cup_frame():
    """(point on the cup axis, unit axis, cavity-floor offset) in world."""
    Tc = world_T(full_prim)
    a = np.array(Tc.ExtractRotationMatrix()).reshape(3, 3)[2]
    a = a / max(np.linalg.norm(a), 1e-12)
    pts = []
    for c in full_prim.GetChildren():
        if c.GetName().startswith("col_wall_"):
            v = UsdGeom.Mesh(c).GetPointsAttr().Get()
            if v:
                pts.append(np.array([Tc.Transform(Gf.Vec3d(q[0], q[1], q[2])) for q in v]))
    W = np.vstack(pts)
    P0 = W.mean(0)
    return P0, a, float(((W - P0) @ a).min())


_TIP_LOCAL = {}


def _tip_setup(links, n=48):
    """cache a subsample of each link's collider verts in its OWN frame."""
    for l in links:
        cp = stage.GetPrimAtPath(f"{ROBOT}/{l}/collisions")
        acc = []
        for q in Usd.PrimRange(cp, Usd.TraverseInstanceProxies()):
            if q.IsA(UsdGeom.Mesh):
                v = UsdGeom.Mesh(q).GetPointsAttr().Get()
                if v:
                    Tq = world_T(q)
                    Tl = world_T(stage.GetPrimAtPath(f"{ROBOT}/{l}")).GetInverse()
                    M = Tq * Tl
                    acc.append(np.array([M.Transform(Gf.Vec3d(a[0], a[1], a[2])) for a in v]))
        if acc:
            A = np.vstack(acc)
            k = np.linspace(0, len(A) - 1, min(n, len(A))).astype(int)
            _TIP_LOCAL[l] = A[k]


def _tip_sample(links):
    """min radius from the cup axis and the height band, per link (mm)."""
    P0, a, floor = _cup_frame()
    out = {}
    for l in links:
        if l not in _TIP_LOCAL:
            continue
        Tl = world_T(stage.GetPrimAtPath(f"{ROBOT}/{l}"))
        Q = np.array([Tl.Transform(Gf.Vec3d(*p)) for p in _TIP_LOCAL[l]])
        d = Q - P0
        s = d @ a
        r = np.linalg.norm(d - np.outer(s, a), axis=1)
        h = s - floor
        inside = (r < 0.0375) & (h > 0) & (h < 0.1175)
        out[l] = [round(float(r.min()) * 1000, 2), round(float(h.min()) * 1000, 1),
                  round(float(h.max()) * 1000, 1), int(inside.sum())]
    return out


fluid_pi = UsdGeom.PointInstancer(stage.GetPrimAtPath(FLUID))


def fluid_pts():
    return np.array([[v[0], v[1], v[2]] for v in fluid_pi.GetPositionsAttr().Get()])


def fluid_in_cup(cup_ctr, r=0.075):
    p = fluid_pts()
    d = np.hypot(p[:, 0] - cup_ctr[0], p[:, 1] - cup_ctr[1])
    return int((d < r).sum())


_timeline = omni.timeline.get_timeline_interface()
cap = [0]
clearance_viol = [0]


def hand_over_counter_low():
    """True if a hand link is in the STRIKE BAND of the tabletop: within 3.5 cm
    of the slab surface while over its footprint. Passes deep below the counter
    (open air under the overhang) are legal — flagging them was a probe bug."""
    for l in HAND_LINKS:
        c = link_center(l)
        if SLAB_X[0] < c[0] < SLAB_X[1] and SLAB_Y[0] < c[1] < SLAB_Y[1] \
                and COUNTER_TOP - 0.035 < c[2] < COUNTER_TOP + 0.005:
            log(f"  CLEARANCE TRIP: {l} at {c.round(3)} (capture {cap[0] // args.capture_every})")
            return True
    return False


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
    if hand_over_counter_low():
        clearance_viol[0] += 1


base_q = np.asarray(robot.get_joint_positions(), float).copy()
motion_rec = []  # (t_seconds, {joint: deg}) recorded each capture for replay baking


def drive():
    ac.apply_action(ArticulationAction(joint_positions=base_q))


def tick():
    drive()
    world.step(render=False)
    if cap[0] % args.capture_every == 0:
        _capture()
        motion_rec.append((cap[0] / 120.0,
                           {j: float(base_q[idx[j]] / DEG) for j in ARM + RHAND}))
    cap[0] += 1


def settle_steps(n):
    for _ in range(n):
        tick()


def blend_to(targets: dict, n: int):
    q0 = {j: float(base_q[idx[j]]) for j in targets}
    for s in range(n):
        a = 0.5 - 0.5 * math.cos(math.pi * (s + 1.0) / n)
        for j, qt in targets.items():
            base_q[idx[j]] = (1 - a) * q0[j] + a * qt
        tick()


jlim = {}
for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)):
    if p.IsA(UsdPhysics.RevoluteJoint) and p.GetName() in idx:
        j = UsdPhysics.RevoluteJoint(p)
        lo, hi = j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()
        jlim[p.GetName()] = ((lo if lo is not None else -180.0) * DEG,
                             (hi if hi is not None else 180.0) * DEG)


def clamp_lim(j, v):
    lo, hi = jlim.get(j, (-math.pi, math.pi))
    return float(np.clip(v, lo, hi))


def fk_probe(qmap, n=10):
    q = base_q.copy()
    for j, v in qmap.items():
        q[idx[j]] = v
    ac.apply_action(ArticulationAction(joint_positions=q))
    for _ in range(n - 1):
        world.step(render=False)
    world.step(render=True)
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


def solve_ik6(target_p, R_t, iters=40, damp=0.10, dq=0.04, w_o=0.10, tol_p=3e-3, tol_o_deg=2.0):
    """Position+orientation IK on ALL 7 arm joints: the palm PAD point goes to
    target_p and the palm link's axes go to R_t (rows = desired world images of
    the link's local axes). GHOST-ONLY: FK probes physically move the arm.
    Returns ({joint: q}, pos_err_m, orient_err_deg)."""
    q = np.array([float(base_q[idx[j]]) for j in ARM])
    lo = np.array([jlim[j][0] for j in ARM])
    hi = np.array([jlim[j][1] for j in ARM])
    R_t = np.asarray(R_t, float)

    def fk(qv):
        qq = base_q.copy()
        for j, v in zip(ARM, qv):
            qq[idx[j]] = v
        ac.apply_action(ArticulationAction(joint_positions=qq))
        for _ in range(9):
            world.step(render=False)
        world.step(render=True)
        T = world_T(palm_prim)
        M = np.array([[T[i][j] for j in range(3)] for i in range(3)])
        return PALM_PAD_LOC @ M + np.array(T.ExtractTranslation()), M

    def err(pad, M):
        e_p = np.asarray(target_p) - pad
        e_o = np.zeros(3)
        for k in range(3):
            e_o += 0.5 * np.cross(M[k], R_t[k])
        return np.concatenate([e_p, w_o * e_o])

    ep = eo = 1e9
    for it in range(iters):
        pad, M = fk(q)
        e = err(pad, M)
        ep = float(np.linalg.norm(e[:3]))
        eo = float(np.linalg.norm(e[3:]) / w_o)
        if ep < tol_p and eo < tol_o_deg * DEG:
            break
        J = np.zeros((6, len(q)))
        for c in range(len(q)):
            qd = q.copy()
            qd[c] += dq
            padd, Md = fk(qd)
            J[:, c] = (err(padd, Md) - e) / dq
        if it == 0 and float(np.abs(J).max()) < 1e-6:
            log("IK6 diag: JACOBIAN IS ZERO — FK probes are not moving the arm")
            break
        step = -J.T @ np.linalg.solve(J @ J.T + damp**2 * np.eye(6), e)
        q = np.clip(q + np.clip(step, -0.15, 0.15), lo, hi)
    ac.apply_action(ArticulationAction(joint_positions=base_q))
    for _ in range(19):
        world.step(render=False)
    world.step(render=True)
    return dict(zip(ARM, [float(v) for v in q])), ep, math.degrees(eo)


def load_traj():
    with open(args.traj_json) as f:
        return json.load(f)


# --------------------------------------------------------------- shared state
cup_c0 = bbox(full_prim)[0]
cup9_c0 = bbox(empty_prim)[0]
fluid_n0 = fluid_in_cup(cup_c0)
palm_p0 = world_pos(palm_prim)
station_arm_q = {j: float(base_q[idx[j]]) for j in ARM}  # v03 pose = grasp station
log(f"loaded: cup4={cup_c0.round(3)} cup9={cup9_c0.round(3)} fluid={fluid_n0} "
    f"palm={palm_p0.round(3)}")


# A rest pose needs no IK (the side point sits outside the frozen-wrist 5-DOF
# envelope anyway: 80 mm residual). Two stages so the forearm never sweeps
# through the cup column: A drops the arm with the shoulder still swung out
# (roll/yaw kept at station values), B then brings it in low, below counter top.
# QA swings OUT at height (dropping in place over the counter struck the slab:
# 22 rad/s spike). QB rests the hand at hip height ABOVE counter level: the
# G1's right flank is angled toward the counter corner, so a fully-hung arm
# (pitch 65) drags the pinky along the counter top (35 trips, 55 rad/s grind).
SIDE_QA = dict(zip(ARM, np.array([18.0, -51.0, 18.7, 10.0, 0.0, 0.0, 0.0]) * DEG))
SIDE_QB = dict(zip(ARM, np.array([40.0, -25.0, 0.0, 30.0, 0.0, 0.0, 0.0]) * DEG))


def cups_still(tol=0.006):
    d4 = float(np.linalg.norm(bbox(full_prim)[0] - cup_c0))
    d9 = float(np.linalg.norm(bbox(empty_prim)[0] - cup9_c0))
    return d4, d9, (d4 < tol and d9 < tol)


def palm_frame():
    """(pad point, face normal, wrist->finger axis) of the palm plate, world.
    Gf matrices are row-vector convention: row i = world image of local axis i."""
    T = world_T(palm_prim)
    M = np.array([[T[i][j] for j in range(3)] for i in range(3)])
    pad = PALM_PAD_LOC @ M + np.array(T.ExtractTranslation())
    n = PALM_N_LOC @ M
    f = PALM_FINGER_LOC @ M
    return pad, n / np.linalg.norm(n), f / np.linalg.norm(f)


def measured_eff(j):
    e = np.asarray(robot.get_measured_joint_efforts(), float).reshape(-1)
    return float(e[idx[j]])


def measured_pos(j):
    q = np.asarray(robot.get_joint_positions(), float).reshape(-1)
    return float(q[idx[j]])


# ============================================================== GHOST PREP
if args.phase == "ghost_prep" and args.engage_standoff > 0:
    # fast path: solve ONLY the engage waypoint (+2cm lift) and MERGE into the
    # existing traj json (the full solve below OVERWRITES it — that silent
    # fallback sent m2b to station_B with no palm jaw)
    log(f"GHOST PREP (engage-only): standoff {args.engage_standoff:.3f} z {args.engage_z:.3f}")
    settle_steps(30)
    cup_c = bbox(full_prim)[0]
    palm = world_pos(palm_prim)
    bearing = (palm[:2] - cup_c[:2])
    bearing = bearing / np.linalg.norm(bearing)
    wp_e = np.array([cup_c[0] + bearing[0] * args.engage_standoff,
                     cup_c[1] + bearing[1] * args.engage_standoff, args.engage_z])
    wp_el = wp_e + np.array([0.0, 0.0, 0.02])
    wp_lift = wp_e + np.array([0.0, 0.0, 0.15])
    traj = load_traj()
    okall = True
    for name, tgt in (("engage", wp_e), ("engage_lift", wp_el), ("lift_high", wp_lift)):
        qmap, err = solve_ik(tgt)
        traj["sols"][name] = {j: float(v) for j, v in qmap.items()}
        traj["waypoints_xyz"][name] = np.asarray(tgt).tolist()
        for j, v in qmap.items():
            base_q[idx[j]] = v
        okall &= err < 0.02
        log(f"  {name}: IK err {err*1000:.1f} mm")
    ok = gate("ik_converged", okall, "engage waypoints")
    with open(args.traj_json, "w") as f:
        json.dump(traj, f, indent=2)

elif args.phase == "ghost_prep":
    log("GHOST PREP: solve m0/m1 waypoints (retreat, lifted, side).")
    settle_steps(30)
    palm = world_pos(palm_prim)
    cup_c = bbox(full_prim)[0]
    fing = np.mean([link_center(l)[:2] for l in
                    ("R_index_intermediate", "R_middle_intermediate",
                     "R_ring_intermediate", "R_pinky_intermediate")], axis=0)
    u = cup_c[:2] - fing
    u = u / np.linalg.norm(u)                      # fingers -> cup axis (lateral)
    torso = bbox(stage.GetPrimAtPath(f"{ROBOT}/torso_link"))[0]
    fwd = cup_c[:2] - torso[:2]
    fwd = fwd / np.linalg.norm(fwd)
    right = np.array([fwd[1], -fwd[0]])
    if np.dot(palm[:2] - torso[:2], right) < 0:    # ensure 'right' is the arm's side
        right = -right

    # With fingers straddling the glass, the ONLY clean escape is straight UP:
    # a lateral retreat arcs the wrist bracket into the rim (118 mm knock) and a
    # backward retreat rakes the far-side fingers through the glass (300
    # particles spilled). Rise until every finger clears the rim (+14 cm), THEN
    # retreat horizontally toward the torso at height.
    # Escape/approach corridor: straight up beside the glass, then OUTWARD along
    # the palm bearing at height (toward-torso retreat drags the fingertips
    # across the cup mouth at rim height — 22 rad/s snag, 117 mm knock). The
    # outward corridor doubles as the approach to the NEW grasp station A
    # (18.7 cm standoff, design brief 2026-08-14).
    bearing = (palm[:2] - cup_c[:2])
    bearing = bearing / np.linalg.norm(bearing)
    wp_up1 = palm + np.array([0.0, 0.0, 0.07])
    wp_up2 = palm + np.array([0.0, 0.0, 0.14])
    wp_out_high = np.array([cup_c[0] + bearing[0] * 0.24, cup_c[1] + bearing[1] * 0.24, wp_up2[2]])
    wp_above_A = np.array([cup_c[0] + bearing[0] * 0.187, cup_c[1] + bearing[1] * 0.187, wp_up2[2]])
    wp_station_A = np.array([cup_c[0] + bearing[0] * 0.187, cup_c[1] + bearing[1] * 0.187, palm[2]])
    wp_station_B = np.array([cup_c[0] + bearing[0] * 0.177, cup_c[1] + bearing[1] * 0.177, palm[2]])
    wp_micro_lift = wp_station_B + np.array([0.0, 0.0, 0.02])
    sols, errs = {}, {}
    for name, tgt in (("up1", wp_up1), ("up2", wp_up2), ("out_high", wp_out_high),
                      ("above_A", wp_above_A), ("station_A", wp_station_A),
                      ("station_B", wp_station_B), ("micro_lift", wp_micro_lift)):
        qmap, err = solve_ik(tgt)
        sols[name] = {j: float(v) for j, v in qmap.items()}
        errs[name] = err
        for j, v in qmap.items():                  # seed continuity
            base_q[idx[j]] = v
        log(f"  {name}: target {np.asarray(tgt).round(3)} IK err {err*1000:.1f} mm")

    # free info for the side-pose choice: drive-settled palm positions of
    # joint-space rest candidates (visual pick happens on m0 frames)
    for nm, cand in (("sideA", [65.0, -20.0, 0.0, 10.0, 0, 0, 0]),
                     ("sideB", [55.0, -25.0, 0.0, 25.0, 0, 0, 0])):
        q = base_q.copy()
        for j, v in zip(ARM, np.array(cand) * DEG):
            q[idx[j]] = v
        ac.apply_action(ArticulationAction(joint_positions=q))
        for _ in range(39):
            world.step(render=False)
        world.step(render=True)
        log(f"  {nm} {cand[:4]}: palm at {world_pos(palm_prim).round(3)}")
    ac.apply_action(ArticulationAction(joint_positions=base_q))
    for _ in range(19):
        world.step(render=False)
    world.step(render=True)

    ok = gate("ik_converged", max(errs.values()) < 0.02,
              f"residuals mm: " + ", ".join(f"{k}={v*1000:.1f}" for k, v in errs.items()))
    os.makedirs(os.path.dirname(args.traj_json), exist_ok=True)
    with open(args.traj_json, "w") as f:
        json.dump({"joints": IK_JOINTS, "sols": sols, "ik_err_m": errs,
                   "station_arm_q": station_arm_q,
                   "station_palm_xyz": palm_p0.tolist(),
                   "waypoints_xyz": {"up1": wp_up1.tolist(), "up2": wp_up2.tolist(),
                                     "out_high": wp_out_high.tolist(),
                                     "above_A": wp_above_A.tolist(),
                                     "station_A": wp_station_A.tolist(),
                                     "station_B": wp_station_B.tolist(),
                                     "micro_lift": wp_micro_lift.tolist()},
                   "source_usd": args.usd}, f, indent=2)
    log(f"trajectory -> {args.traj_json}")

# ============================================================== PALM PREP
elif args.phase == "palm_prep":
    # Ghost-solve the FLAT-PALM corridor (Eli 2026-08-17: "get the palm of the
    # g1 hands flat across the side of the glass"). Target orientation: palm
    # face normal RADIAL into the cup axis (horizontal), fingers tangent,
    # thumb-side UP (thumb-side down jams the thumb into the counter). The
    # corridor honours finding 3: reorient HIGH, descend OUTSIDE the wall.
    log("PALM PREP (ghost): solve flat-palm corridor (up, high, gap, touch).")
    settle_steps(30)
    cup_c = bbox(full_prim)[0]
    palm = world_pos(palm_prim)
    for j in ARM[4:]:
        log(f"  jlim {j}: {jlim[j][0]/DEG:.0f}..{jlim[j][1]/DEG:.0f} deg")
    b = palm[:2] - cup_c[:2]
    b = b / np.linalg.norm(b)                     # cup -> palm, horizontal
    b3 = np.array([b[0], b[1], 0.0])
    n_t = -b3                                     # palm face points INTO the cup
    t_t = np.cross(np.array([0.0, 0.0, 1.0]), b3)  # tangent; this sign = thumb UP
    r_x = -n_t                                    # local +X (back of hand)
    r_y = -t_t                                    # local +Y (toward wrist)
    R_t = np.vstack([r_x, r_y, np.cross(r_x, r_y)])
    z0 = args.engage_z
    wp_high = np.array([cup_c[0] + b[0] * (R_WALL + 0.020),
                        cup_c[1] + b[1] * (R_WALL + 0.020), z0 + 0.14])
    wp_gap = wp_high - np.array([0.0, 0.0, 0.14])
    wp_touch = np.array([cup_c[0] + b[0] * (R_WALL + 0.001),
                         cup_c[1] + b[1] * (R_WALL + 0.001), z0])
    traj = load_traj()
    # leg 1: pure vertical rise at station orientation (position-only, 5J) —
    # reorienting at station height would sweep the fingertips at wall radius
    qmap, err = solve_ik(palm + np.array([0.0, 0.0, 0.14]))
    traj["sols"]["stationA_up"] = {j: float(v) for j, v in qmap.items()}
    for j, v in qmap.items():
        base_q[idx[j]] = v
    log(f"  stationA_up: IK err {err*1000:.1f} mm")
    okall = err < 0.02
    # legs 2-4: full 6D flat-palm waypoints, seeded in sequence
    for name, tgt in (("palm_high", wp_high), ("palm_gap", wp_gap),
                      ("palm_touch", wp_touch)):
        qmap6, ep, eo = solve_ik6(tgt, R_t)
        traj["sols"][name] = qmap6
        traj["waypoints_xyz"][name] = np.asarray(tgt).tolist()
        for j, v in qmap6.items():
            base_q[idx[j]] = v
        okall &= (ep < 0.006 and eo < 6.0)
        log(f"  {name}: pos err {ep*1000:.1f} mm, orient err {eo:.1f} deg")
    # verify the settled touch pose the same way the real phase will measure it
    drive()
    for _ in range(29):
        world.step(render=False)
    world.step(render=True)
    pad, n_w, f_w = palm_frame()
    mis = math.degrees(math.acos(float(np.clip(np.dot(n_w, n_t), -1, 1))))
    spin = math.degrees(math.asin(float(np.clip(f_w[2], -1, 1))))
    thumb_up = bool(-(np.cross(n_w * -1, f_w * -1))[2] > 0)  # local -Z world-z
    log(f"  settled touch pose: pad {pad.round(3)} normal-off-radial {mis:.1f} deg "
        f"finger-axis {spin:+.1f} deg from horizontal thumb_up={thumb_up}")
    ok = gate("palm_ik", okall and mis < 6.0,
              f"corridor solved; settled flatness {mis:.1f} deg")
    traj["palm_bearing"] = b.tolist()
    with open(args.traj_json, "w") as f:
        json.dump(traj, f, indent=2)
    log(f"palm corridor -> {args.traj_json}")

# ============================================================== CLAW PREP
elif args.phase == "claw_prep":
    # Ghost-solve the CLAW ENTRY corridor (Eli 2026-08-17: "simply make a semi
    # circle claw hand pose wide enough for the glass and show the g1 pushing
    # its hand forward so the glass ends up in it").  Orientation is the same
    # flat-palm frame m2a proved; what is new is WHERE the hand stops -- offset
    # across the palm so the glass lands in the finger pocket, not on the heel.
    log("CLAW PREP (ghost): solve the claw-entry corridor (high, p0, p1, p2, in).")
    settle_steps(30)
    cup_c = bbox(full_prim)[0]
    palm = world_pos(palm_prim)
    b = palm[:2] - cup_c[:2]
    b = b / np.linalg.norm(b)
    b3 = np.array([b[0], b[1], 0.0])
    n_w = -b3                                        # palm normal: hand -> cup
    s_w = np.cross(np.array([0.0, 0.0, 1.0]), n_w)   # across the palm, HORIZONTAL
    up = np.array([0.0, 0.0, 1.0])

    def roll_about(v, ax, th):
        return (v * math.cos(th) + np.cross(ax, v) * math.sin(th)
                + ax * float(ax @ v) * (1.0 - math.cos(th)))

    # roll the hand about its own palm normal.  Sign and magnitude settled by
    # GHOST TEST + mesh measurement, never by an AABB proxy: the AABB intrusion
    # check reported 19.6 mm on a pose whose true intrusion was 4.0 mm, and it
    # is what made -20 look wrong for two rounds.  It stays as a report only.
    th = args.claw_roll * DEG
    R_t = np.vstack([-n_w, roll_about(s_w, n_w, th), roll_about(-up, n_w, th)])
    log(f"  roll {args.claw_roll:+.0f} deg about the palm normal")
    org = np.array([cup_c[0], cup_c[1], args.claw_z]) - CLAW_C * n_w - CLAW_S * s_w
    log(f"  cup axis {cup_c[:2].round(3)}  bearing {b.round(3)}  "
        f"hand origin at entry {org.round(3)}")

    def pad_at(o):
        return o + PALM_PAD_LOC @ R_t

    traj = load_traj()
    qmap, err = solve_ik(palm + np.array([0.0, 0.0, 0.14]))
    traj["sols"]["stationA_up"] = {j: float(v) for j, v in qmap.items()}
    for j, v in qmap.items():
        base_q[idx[j]] = v
    okall = err < 0.02
    log(f"  stationA_up: IK err {err*1000:.1f} mm")
    # the push is solved as a STRAIGHT LINE in Cartesian space (4 stations along
    # the palm normal), because a joint-space blend between two IK solutions
    # bows sideways -- and sideways is where the glass is.
    def solve_wp(name, tgt, need=True):
        qmap6, ep, eo = solve_ik6(tgt, R_t)
        traj["sols"][name] = qmap6
        traj["waypoints_xyz"][name] = np.asarray(tgt).tolist()
        for j, v in qmap6.items():
            base_q[idx[j]] = v
        log(f"  {name}: pos err {ep*1000:.1f} mm, orient err {eo:.1f} deg")
        return ep < 0.006 and eo < 6.0

    # p0 first: it is the pose every other station is a small step from, so it
    # seeds them all.  Seeding claw_high off stationA_up instead left it 8 mm /
    # 12.9 deg out -- a tilted arrival that would swing the claw over the rim.
    okall &= solve_wp("claw_p0", pad_at(org - 0.090 * n_w))
    q_p0 = {j: float(base_q[idx[j]]) for j in ARM}
    lift_ok = False
    for lift in (0.14, 0.11, 0.08):
        for j, v in q_p0.items():
            base_q[idx[j]] = v
        if solve_wp("claw_high", pad_at(org - 0.090 * n_w) + np.array([0.0, 0.0, lift])):
            log(f"  claw_high converged at +{lift*100:.0f} cm of lift")
            lift_ok = True
            break
    okall &= lift_ok
    for j, v in q_p0.items():
        base_q[idx[j]] = v
    for name, back in (("claw_p1", 0.050), ("claw_p2", 0.025), ("claw_in", 0.0)):
        okall &= solve_wp(name, pad_at(org - back * n_w))
    # LIFT stations for the hug and the carry.  Both are seeded from claw_in and
    # are a PURE VERTICAL step off it: seeding from stationA_up bowed claw_high
    # 8 mm / 12.9 deg out, and any lateral bow here drags a gripped glass.
    q_in = {j: float(base_q[idx[j]]) for j in ARM}
    for name, ladder in (("claw_lift", (0.020,)),
                         ("claw_carry", (0.150, 0.120, 0.100))):
        got = False
        for up_m in ladder:
            for j, v in q_in.items():
                base_q[idx[j]] = v
            if solve_wp(name, pad_at(org) + np.array([0.0, 0.0, up_m])):
                log(f"  {name} converged at +{up_m*100:.0f} cm of lift")
                traj["waypoints_xyz"][name + "_m"] = float(up_m)
                got = True
                break
        okall &= got
    for j, v in q_in.items():
        base_q[idx[j]] = v
    # put the DIGITS in the claw pose too: the corridor is verified against the
    # shape that will actually enter, not an open hand (straight fingers hang
    # 40 mm lower and read as a counter strike that never happens)
    for nname, prox, inter, ratio in FINGERS:
        base_q[idx[prox]] = clamp_lim(prox, CLAW_FING[nname])
        base_q[idx[inter]] = clamp_lim(inter, 0.765 * args.claw_open * ratio)
    for j, v in CLAW_THUMB.items():
        base_q[idx[j]] = clamp_lim(j, v)
    drive()
    for _ in range(59):
        world.step(render=False)
    world.step(render=True)
    pad, nn, ff = palm_frame()
    o_w = world_pos(palm_prim)
    s_m = np.cross(np.array([0.0, 0.0, 1.0]), nn)
    s_m = s_m / np.linalg.norm(s_m)
    pocket = o_w + CLAW_C * nn + CLAW_S * s_m
    cupn = bbox(full_prim)[0]
    off = float(np.hypot(pocket[0] - cupn[0], pocket[1] - cupn[1]))
    mis = math.degrees(math.acos(float(np.clip(np.dot(nn, n_w), -1, 1))))
    log(f"  settled entry pose: pocket {pocket.round(3)} vs glass axis "
        f"{cupn[:2].round(3)} -> off by {off*1000:.1f} mm, palm {mis:.1f} deg off radial")
    # is the thumb ON THE WALL, or hanging over the mouth?  (AABB corners; the
    # authoritative mesh figure comes from scratchpad/measure_checkpoint.py)
    _, gmn, gmx = bbox(full_prim)
    rim_z, r_rim = float(gmx[2]), float(((gmx[0] - gmn[0]) + (gmx[1] - gmn[1])) / 4)
    intrude, lowest = 0.0, 9.9
    for l in CONTACT_LINKS:
        _, mn, mx = bbox(stage.GetPrimAtPath(f"{ROBOT}/{l}"))
        # only links OVER THE COUNTER can strike it -- the elbow hangs lower but
        # out in free space, and counting it fabricated a 56 mm strike
        if (SLAB_X[0] < (mn[0] + mx[0]) / 2 < SLAB_X[1]
                and SLAB_Y[0] < (mn[1] + mx[1]) / 2 < SLAB_Y[1]):
            lowest = min(lowest, float(mn[2]))
        for cx in (mn[0], mx[0]):
            for cy in (mn[1], mx[1]):
                if mx[2] > rim_z - 0.002:
                    rr = float(np.hypot(cx - cupn[0], cy - cupn[1]))
                    intrude = max(intrude, r_rim - rr)
    for l in ("R_thumb_proximal", "R_thumb_intermediate", "R_thumb_distal"):
        _, mn, mx = bbox(stage.GetPrimAtPath(f"{ROBOT}/{l}"))
        rr = float(np.hypot((mn[0] + mx[0]) / 2 - cupn[0], (mn[1] + mx[1]) / 2 - cupn[1]))
        log(f"    {l:24s} z {mn[2]:.3f}..{mx[2]:.3f}  r {rr*1000:5.1f} mm  "
            f"{'ON THE WALL' if mx[2] < rim_z + 0.002 else 'ABOVE THE RIM'}")
    log(f"  mouth intrusion {intrude*1000:+.1f} mm (want <= 5), lowest hand point "
        f"{lowest:.3f} (counter {COUNTER_TOP:.3f})")
    # intrusion here is an AABB proxy and reads high on diagonal links; the
    # authoritative figure is the mesh measurement in measure_checkpoint.py
    # both figures below are AABB proxies and read pessimistic (this pose's true
    # mesh numbers are 0.6 mm intrusion / 0.895 lowest). Gate loosely here; the
    # real strike test is per-capture in the push phase, and the authoritative
    # clearances come from scratchpad/grade_entry.py on the saved checkpoint.
    ok = gate("claw_ik", okall and off < 0.012 and mis < 6.0
              and lowest > COUNTER_TOP,
              f"corridor solved; pocket {off*1000:.1f} mm from the glass axis, palm "
              f"{mis:.1f} deg off radial, mouth intrusion {intrude*1000:+.1f} mm, "
              f"lowest hand point {lowest:.3f}")
    traj["claw_bearing"] = b.tolist()
    traj["claw_pocket"] = dict(c=CLAW_C, s=CLAW_S, z=args.claw_z)
    with open(args.traj_json, "w") as f:
        json.dump(traj, f, indent=2)
    log(f"claw corridor -> {args.traj_json}")

# ============================================================== SIDE SCAN
elif args.phase == "side_scan":
    # end the blind guessing: FK-settle candidate rest poses, measure EVERY
    # hand link against the counter volume, record the first fully-clear one
    log("SIDE SCAN: measure rest-pose candidates against the counter volume.")
    settle_steps(20)
    CANDS = [
        ("A", [40, -25, 0, 30]), ("B", [40, -25, -30, 30]), ("C", [40, -25, 30, 30]),
        ("D", [55, -15, -20, 15]), ("E", [30, -40, -20, 20]), ("F", [45, -30, -40, 45]),
        ("G", [55, -30, 30, 15]), ("H", [30, -15, 0, 45]),
    ]
    winner = None
    for nm, cand in CANDS:
        q = base_q.copy()
        for j, v in zip(ARM, list(np.array(cand, float) * DEG) + [0.0, 0.0, 0.0]):
            q[idx[j]] = v
        ac.apply_action(ArticulationAction(joint_positions=q))
        for _ in range(49):
            world.step(render=False)
        world.step(render=True)
        stats = []
        for l in HAND_LINKS:
            c = link_center(l)
            over = SLAB_X[0] < c[0] < SLAB_X[1] and SLAB_Y[0] < c[1] < SLAB_Y[1]
            clear = (not over) or c[2] > COUNTER_TOP + 0.012
            stats.append((l, c, clear))
        allclear = all(s[2] for s in stats)
        palm = world_pos(palm_prim)
        worst = min((s[1][2] for s in stats if not s[2]), default=9.9)
        log(f"  {nm} {cand}: palm={palm.round(3)} all_clear={allclear} "
            f"worst_low_z={worst if worst < 9 else '—'}")
        if allclear and winner is None:
            winner = (nm, cand, palm.tolist())
    ok = gate("side_found", winner is not None,
              f"winner: {winner[0]} {winner[1]}" if winner else "no clear candidate")
    if winner:
        traj = load_traj()
        traj["side_q_deg"] = winner[1]
        traj["side_palm_xyz"] = winner[2]
        with open(args.traj_json, "w") as f:
            json.dump(traj, f, indent=2)
        log(f"side pose -> {args.traj_json}")

# ============================================================== M0 — SIDE
elif args.phase == "m0_side":
    log("M0: retract from the cup flank, park the hand at the G1's side.")
    traj = load_traj()
    sols = traj["sols"]
    side_deg = traj.get("side_q_deg")
    side_q = dict(zip(ARM, list(np.array(side_deg, float) * DEG) + [0.0, 0.0, 0.0])) \
        if side_deg else SIDE_QB
    # transit is deterministic: swing east at HELD height (side pose with QA's
    # pitch), only then drop in place — the descent never crosses the counter
    side_high = dict(side_q)
    side_high["right_shoulder_pitch_joint"] = SIDE_QA["right_shoulder_pitch_joint"]
    settle_steps(60)
    blend_to({j: sols["up1"][j] for j in IK_JOINTS}, args.seg_steps)
    blend_to({j: sols["up2"][j] for j in IK_JOINTS}, args.seg_steps)
    blend_to({j: sols["out_high"][j] for j in IK_JOINTS}, args.seg_steps)
    blend_to(SIDE_QA, int(args.seg_steps * 1.5))
    blend_to(side_high, int(args.seg_steps * 1.5))
    blend_to(side_q, int(args.seg_steps * 1.5))
    settle_steps(150)
    palm = world_pos(palm_prim)
    d4, d9, still = cups_still()
    _v = np.abs(np.asarray(robot.get_joint_velocities(), float)).reshape(-1)
    jv = float(_v.max())
    log(f"  max joint vel: {names[int(_v.argmax())]} at {jv:.2f} rad/s")
    ok = True
    ok &= gate("cups_untouched", still, f"cup4 drift {d4*1000:.1f} mm, cup9 {d9*1000:.1f} mm")
    gate("side_pose_report", True, f"palm parked at {palm.round(3)} (visual arbitration)")
    ok &= gate("no_table_strike", clearance_viol[0] == 0,
               f"{clearance_viol[0]} captures with a hand link below counter top over the slab")
    ok &= gate("stable", jv < 2.0, f"max joint vel {jv:.2f} rad/s")
    fl = fluid_in_cup(bbox(full_prim)[0])
    gate("fluid_report", True, f"{fl}/{fluid_n0} particles in cup")

# ============================================================== M1 — APPROACH
elif args.phase == "m1_approach":
    log("M1: side -> staging above flank -> grasp station. Cups must not move.")
    traj = load_traj()
    sols = traj["sols"]
    station = traj["station_arm_q"]
    settle_steps(60)
    # the IK waypoints were solved with the wrist plane at station (B1) angles —
    # restore the wrists first, near the body, before flying those waypoints
    # approach ends at STATION A (18.7 cm standoff, the design-brief grasp
    # station), NOT the old B1 station — descend outside the glass
    side_deg = traj.get("side_q_deg")
    if side_deg:  # rise IN PLACE (east), extend, and only then sweep west high:
        # the westward drift of a low ascent snags fingertips under the slab's
        # invisible 2 cm east face (10.95 rad/s grind, m1i)
        side_q = dict(zip(ARM, list(np.array(side_deg, float) * DEG) + [0.0, 0.0, 0.0]))
        side_high = dict(side_q)
        side_high["right_shoulder_pitch_joint"] = 12.0 * DEG
        blend_to(side_high, args.seg_steps)
        east_high = dict(SIDE_QA)                       # QA pitch/roll/elbow ...
        east_high["right_shoulder_yaw_joint"] = side_q["right_shoulder_yaw_joint"]
        blend_to(east_high, args.seg_steps)             # ... but yaw stays east
    blend_to(SIDE_QA, int(args.seg_steps * 1.5))       # yaw sweeps west LAST, high
    blend_to({j: station[j] for j in ARM[4:]}, 45)     # wrist plane to station
    blend_to({j: sols["out_high"][j] for j in IK_JOINTS}, int(args.seg_steps * 1.5))
    blend_to({j: sols["above_A"][j] for j in IK_JOINTS}, args.seg_steps)
    blend_to({j: sols["station_A"][j] for j in IK_JOINTS}, args.seg_steps)
    settle_steps(120)
    palm = world_pos(palm_prim)
    d4, d9, still = cups_still()
    _v = np.abs(np.asarray(robot.get_joint_velocities(), float)).reshape(-1)
    jv = float(_v.max())
    log(f"  max joint vel: {names[int(_v.argmax())]} at {jv:.2f} rad/s")
    ok = True
    ok &= gate("cups_untouched", still, f"cup4 drift {d4*1000:.1f} mm, cup9 {d9*1000:.1f} mm")
    st = np.array(traj["waypoints_xyz"]["station_A"])
    ok &= gate("at_station_A", float(np.linalg.norm(palm - st)) < 0.02,
               f"palm {palm.round(3)} vs station A {st.round(3)} "
               f"(err {np.linalg.norm(palm - st)*1000:.1f} mm)")
    ok &= gate("no_table_strike", clearance_viol[0] == 0,
               f"{clearance_viol[0]} captures with a hand link below counter top over the slab")
    ok &= gate("stable", jv < 2.0, f"max joint vel {jv:.2f} rad/s")
    fl = fluid_in_cup(bbox(full_prim)[0])
    gate("fluid_report", True, f"{fl}/{fluid_n0} particles in cup")

# ============================================================== GRASP SCAN
elif args.phase == "grasp_scan":
    # The design brief's station numbers were probed against the WEAK-waist
    # torso and are stale. Measure the grasp geometry on THIS config: for each
    # engage standoff, close the fingers kinematically and log where the fence
    # and thumb pad actually sit relative to the glass (R_wall ~ 0.0425 m).
    log("GRASP SCAN: engage standoff + thumb geometry sweep (ghost).")
    traj = load_traj()
    settle_steps(30)
    cup_c = bbox(full_prim)[0]
    palm = world_pos(palm_prim)
    bearing = (palm[:2] - cup_c[:2])
    bearing = bearing / np.linalg.norm(bearing)
    R_WALL = 0.0425
    for j in ("R_index_proximal_joint", "R_thumb_proximal_yaw_joint"):
        log(f"  jlim {j}: {jlim[j][0]/DEG:.0f}..{jlim[j][1]/DEG:.0f} deg")
    results = []
    for standoff in (0.095, 0.105, 0.115):
        tgt = np.array([cup_c[0] + bearing[0] * standoff,
                        cup_c[1] + bearing[1] * standoff, 0.972])
        qmap, err = solve_ik(tgt)
        for j, v in qmap.items():
            base_q[idx[j]] = v
        base_q[idx[THUMB_YAW]] = 66 * DEG
        for close_deg in (50.0, 65.0, 80.0):
            tp_deg = close_deg  # reuse row field for the closure angle
            for nname, prox, inter, ratio in FINGERS:
                base_q[idx[prox]] = clamp_lim(prox, close_deg * DEG)
                base_q[idx[inter]] = clamp_lim(inter, close_deg * DEG * ratio)
            drive()
            for _ in range(59):
                world.step(render=False)
            world.step(render=True)
            cupn = bbox(full_prim)[0]
            fence = []
            for l in ("R_index_intermediate", "R_middle_intermediate",
                      "R_ring_intermediate", "R_pinky_intermediate"):
                c = link_center(l)
                fence.append((float(np.hypot(c[0]-cupn[0], c[1]-cupn[1])), float(c[2])))
            tp = link_center("R_thumb_distal")
            t_r = float(np.hypot(tp[0]-cupn[0], tp[1]-cupn[1]))
            pb = link_center("right_hand_base_link")
            p_r = float(np.hypot(pb[0]-cupn[0], pb[1]-cupn[1]))
            row = {"standoff": standoff, "ik_mm": err*1000, "thumb_pitch": tp_deg,
                   "fence_r_mm": [round(f[0]*1000) for f in fence],
                   "fence_z": [round(f[1], 3) for f in fence],
                   "thumb_r_mm": round(t_r*1000), "thumb_z": round(float(tp[2]), 3),
                   "palm_r_mm": round(p_r*1000), "palm_z": round(float(pb[2]), 3)}
            results.append(row)
            log(f"  so={standoff:.3f} tp={tp_deg:>4.0f}: fence_r={row['fence_r_mm']} "
                f"fence_z={row['fence_z']} thumb_r={row['thumb_r_mm']} "
                f"thumb_z={row['thumb_z']} palm_r={row['palm_r_mm']}")
        # reopen for next standoff
        for nname, prox, inter, ratio in FINGERS:
            base_q[idx[prox]] = 0.0
            base_q[idx[inter]] = 0.0
        base_q[idx[THUMB_YAW]] = 0.0
        base_q[idx["R_thumb_proximal_pitch_joint"]] = 0.0
        drive()
        for _ in range(40):
            world.step(render=False)
    metrics["probes"]["grasp_scan"] = results
    ok = gate("scan_complete", len(results) == 12, f"{len(results)} rows")
    with open(os.path.join(phase_dir, "grasp_scan.json"), "w") as f:
        json.dump(results, f, indent=1)

# ============================================================== M1 DIRECT
elif args.phase == "m1_direct":
    # fallback approach: from the v04 station straight out to station A —
    # a proven lateral blend with no counter interaction
    log("M1-DIRECT: station -> station A (4 cm outward along the bearing).")
    traj = load_traj()
    sols = traj["sols"]
    settle_steps(60)
    blend_to({j: sols["station_A"][j] for j in IK_JOINTS}, args.seg_steps)
    settle_steps(90)
    palm = world_pos(palm_prim)
    st = np.array(traj["waypoints_xyz"]["station_A"])
    d4, d9, still = cups_still()
    _v = np.abs(np.asarray(robot.get_joint_velocities(), float)).reshape(-1)
    jv = float(_v.max())
    ok = True
    ok &= gate("cups_untouched", still, f"cup4 drift {d4*1000:.1f} mm, cup9 {d9*1000:.1f} mm")
    ok &= gate("at_station_A", float(np.linalg.norm(palm - st)) < 0.02,
               f"palm {palm.round(3)} vs station A {st.round(3)} "
               f"(err {np.linalg.norm(palm - st)*1000:.1f} mm)")
    ok &= gate("stable", jv < 2.0, f"max joint vel {jv:.2f} rad/s")
    gate("fluid_report", True, f"{fluid_in_cup(bbox(full_prim)[0])}/{fluid_n0} particles in cup")

# ============================================================== M2a — PALM FLAT
elif args.phase == "m2a_palm":
    # The grasp decomposed further (Eli 2026-08-17): ONE deliverable — the palm
    # plate FLAT against the glass wall. Fingers straight and idle, thumb
    # parked; the plate does all the touching. Corridor: rise at station
    # orientation, reorient high, descend outside the wall, then a slow
    # (~6 mm/s) advance with a live cup-response watch.
    log("M2a: palm plate FLAT on the glass wall — fingers idle, cup undisturbed.")
    traj = load_traj()
    sols = traj["sols"]
    b = np.array(traj["palm_bearing"], float)
    n_t = np.array([-b[0], -b[1], 0.0])
    settle_steps(60)
    # Thumb PARK (ghost scan 2026-08-17, logs/thumb_scan.log): extended at 0/0
    # the thumb pokes the wall 12 mm before the palm (m2a-a: 0.215 N*m on the
    # yaw joint, advance froze at pad gap +11.9 mm). Fully CURLED at yaw-low
    # the moving segments clear (+8/+27 mm); only the rigid thumb mount rides
    # ~2 mm proud of the plate's upper edge and arrives with the pad.
    fing0 = {}
    for nname, prox, inter, ratio in FINGERS:
        fing0[prox] = 0.0
        fing0[inter] = 0.0
    fing0[THUMB_YAW] = clamp_lim(THUMB_YAW, -6.0 * DEG)
    fing0["R_thumb_proximal_pitch_joint"] = clamp_lim("R_thumb_proximal_pitch_joint", 34.0 * DEG)
    fing0["R_thumb_intermediate_joint"] = clamp_lim("R_thumb_intermediate_joint", 46.0 * DEG)
    fing0["R_thumb_distal_joint"] = clamp_lim("R_thumb_distal_joint", 69.0 * DEG)
    blend_to(fing0, 60)
    blend_to({j: sols["stationA_up"][j] for j in IK_JOINTS}, args.seg_steps)
    blend_to({j: sols["palm_high"][j] for j in ARM}, int(args.seg_steps * 1.5))
    blend_to({j: sols["palm_gap"][j] for j in ARM}, args.seg_steps)
    settle_steps(90)
    pad, n_w, f_w = palm_frame()
    mis_gap = math.degrees(math.acos(float(np.clip(np.dot(n_w, n_t), -1, 1))))
    d4g, _, _ = cups_still()
    gate("gap_pose_report", True,
         f"pre-touch: flatness {mis_gap:.1f} deg off radial, pad {pad.round(3)}, "
         f"cup drift {d4g*1000:.1f} mm")
    # slow advance: freeze the arm the moment the cup responds
    q0 = {j: float(base_q[idx[j]]) for j in ARM}
    qt = {j: sols["palm_touch"][j] for j in ARM}
    frozen = -1
    NADV = 360
    for s in range(NADV):
        if frozen < 0:
            a = 0.5 - 0.5 * math.cos(math.pi * (s + 1.0) / NADV)
            for j in ARM:
                base_q[idx[j]] = (1 - a) * q0[j] + a * qt[j]
        tick()
        if frozen < 0 and (cap[0] - 1) % args.capture_every == 0:
            d4 = float(np.linalg.norm(bbox(full_prim)[0] - cup_c0))
            if d4 > 0.0015:
                frozen = s
                log(f"  TOUCH: cup responded {d4*1000:.1f} mm at advance step {s}/{NADV}")
    if frozen < 0:
        log("  advance completed with no cup response (gap gate judges contact)")
    settle_steps(150)
    # ------------------------------------------------ flatness measurements
    pad, n_w, f_w = palm_frame()
    cup_n = bbox(full_prim)[0]
    r_pad = float(np.hypot(pad[0] - cup_n[0], pad[1] - cup_n[1]))
    gap_mm = (r_pad - R_WALL) * 1000.0
    mis = math.degrees(math.acos(float(np.clip(np.dot(n_w, n_t), -1, 1))))
    elev = math.degrees(math.asin(float(np.clip(n_w[2], -1, 1))))
    spin = math.degrees(math.asin(float(np.clip(f_w[2], -1, 1))))
    thumb_up = bool((-np.cross(n_w * -1, f_w * -1))[2] > 0)
    d4, d9, still = cups_still()
    tilt = cup_tilt_deg(full_prim)
    f_eff = {f[0]: abs(measured_eff(f[1])) for f in FINGERS}
    f_eff["thumb"] = abs(measured_eff(THUMB_YAW))
    fmax = max(f_eff.values())
    _v = np.abs(np.asarray(robot.get_joint_velocities(), float)).reshape(-1)
    jv = float(_v.max())
    fl = fluid_in_cup(cup_n)
    ok = True
    ok &= gate("palm_flat", mis < 6.0,
               f"palm normal {mis:.1f} deg off radial (elevation {elev:+.1f} deg)")
    ok &= gate("palm_on_wall", gap_mm < 6.0,
               f"pad->wall gap {gap_mm:+.1f} mm (touch froze at step {frozen})")
    ok &= gate("cup_undisturbed", d4 < 0.006 and tilt < 2.0,
               f"cup drift {d4*1000:.1f} mm, tilt {tilt:.2f} deg")
    ok &= gate("fingers_clear", fmax < 0.05,
               f"max digit effort {fmax:.3f} N*m ({f_eff})")
    ok &= gate("no_table_strike", clearance_viol[0] == 0,
               f"{clearance_viol[0]} captures with a hand link in the strike band")
    ok &= gate("stable", jv < 2.0, f"max joint vel {jv:.2f} rad/s "
               f"({names[int(_v.argmax())]})")
    gate("spin_report", True,
         f"finger axis {spin:+.1f} deg from horizontal, thumb side "
         f"{'UP' if thumb_up else 'DOWN'}")
    gate("fluid_report", True, f"{fl}/{fluid_n0} particles in cup")

# ============================================================== M2b — CLAW
elif args.phase == "m2b_claw":
    # ONE deliverable: form the semicircle claw, push the hand straight forward,
    # and end with the glass sitting INSIDE the claw -- untouched.  Contact is a
    # FAILURE here, not a success: every gate below is about arriving cleanly.
    # The hug (thumb yaw) and the lift are the next stages, not this one.
    log("M2b: semicircle CLAW entry — push forward until the glass is inside it.")
    traj = load_traj()
    sols = traj["sols"]
    b = np.array(traj.get("claw_bearing", traj["palm_bearing"]), float)
    n_t = np.array([-b[0], -b[1], 0.0])
    settle_steps(60)
    # form the claw at the station, far from the glass, so the shape is proven
    # BEFORE it goes anywhere near the cup
    claw = {}
    for nname, prox, inter, ratio in FINGERS:
        claw[prox] = clamp_lim(prox, CLAW_FING[nname])
        claw[inter] = clamp_lim(inter, 0.765 * args.claw_open * ratio)
    for j, v in CLAW_THUMB.items():
        claw[j] = clamp_lim(j, v)
    blend_to(claw, 90)
    settle_steps(60)
    qnow = np.asarray(robot.get_joint_positions(), float).reshape(-1)
    ferr = max(abs(float(qnow[idx[j]]) - claw[j]) for j in claw)
    ok = True
    ok &= gate("claw_formed", ferr < 6.0 * DEG,
               f"claw shape held; worst digit tracking error {ferr/DEG:.1f} deg")
    # travel: rise, reorient high, drop in 90 mm short of the entry pose
    blend_to({j: sols["stationA_up"][j] for j in IK_JOINTS}, args.seg_steps)
    blend_to({j: sols["claw_high"][j] for j in ARM}, int(args.seg_steps * 1.5))
    blend_to({j: sols["claw_p0"][j] for j in ARM}, args.seg_steps)
    settle_steps(90)
    pad, nn, ff = palm_frame()
    mis_stage = math.degrees(math.acos(float(np.clip(np.dot(nn, n_t), -1, 1))))
    d4g, _, _ = cups_still()
    gate("stage_report", True,
         f"staged 90 mm out: palm {mis_stage:.1f} deg off radial, cup drift "
         f"{d4g*1000:.1f} mm")
    # ---- THE PUSH: straight in along the palm normal, three measured legs.
    # The glass moving AT ALL means the claw is not wide enough -> freeze and
    # report where, rather than shove it across the counter.
    frozen = None
    for leg, nm in ((1, "claw_p1"), (2, "claw_p2"), (3, "claw_in")):
        q0 = {j: float(base_q[idx[j]]) for j in ARM}
        qt = {j: sols[nm][j] for j in ARM}
        N = 220
        for s in range(N):
            if frozen is None:
                a = 0.5 - 0.5 * math.cos(math.pi * (s + 1.0) / N)
                for j in ARM:
                    base_q[idx[j]] = (1 - a) * q0[j] + a * qt[j]
            tick()
            if frozen is None and (cap[0] - 1) % args.capture_every == 0:
                d4 = float(np.linalg.norm(bbox(full_prim)[0] - cup_c0))
                if d4 > args.entry_freeze:
                    frozen = (leg, s, d4)
                    log(f"  GLASS MOVED {d4*1000:.1f} mm on leg {leg} step {s}/{N} "
                        f"— freezing the push")
        if frozen is None:
            log(f"  leg {leg} -> {nm}: complete, glass still")
    settle_steps(150)
    # ------------------------------------------------- entry measurements
    pad, nn, ff = palm_frame()
    o_w = world_pos(palm_prim)
    s_m = np.cross(np.array([0.0, 0.0, 1.0]), nn)
    s_m = s_m / np.linalg.norm(s_m)
    pocket = o_w + CLAW_C * nn + CLAW_S * s_m
    cupn = bbox(full_prim)[0]
    off = float(np.hypot(pocket[0] - cupn[0], pocket[1] - cupn[1]))
    mis = math.degrees(math.acos(float(np.clip(np.dot(nn, n_t), -1, 1))))
    # how far round the glass each arm of the claw actually got: alpha is the
    # depth along the palm normal, so alpha > CLAW_C means past the equator
    # WRAP ARC: how far round the glass the claw actually reaches. Measured from
    # each hand link's world AABB corners (link CENTRES understate it by half a
    # phalanx and failed this gate on a claw that was in fact past the equator).
    # 0 deg = straight at the palm; +-90 = the glass equator. The authoritative
    # number comes from the meshes offline -- scratchpad/measure_checkpoint.py.
    pref = link_center("right_hand_base_link")[:2] - cupn[:2]
    pref = pref / np.linalg.norm(pref)
    arc_lo, arc_hi, reach = 999.0, -999.0, {}
    for l in CONTACT_LINKS:
        _, mn, mx = bbox(stage.GetPrimAtPath(f"{ROBOT}/{l}"))
        best_a = None
        for cx in (mn[0], mx[0]):
            for cy in (mn[1], mx[1]):
                for cz in (mn[2], mx[2]):
                    if not (cupn[2] - 0.060 < cz < cupn[2] + 0.060):
                        continue
                    v = np.array([cx, cy]) - cupn[:2]
                    if np.hypot(*v) > R_WALL + 0.035:
                        continue
                    a = math.degrees(math.atan2(v[1] * pref[0] - v[0] * pref[1], float(v @ pref)))
                    arc_lo, arc_hi = min(arc_lo, a), max(arc_hi, a)
                    best_a = a if best_a is None else (a if abs(a) > abs(best_a) else best_a)
        if best_a is not None:
            reach[l] = round(best_a, 1)
    arc = (arc_hi - arc_lo) if arc_hi > arc_lo else 0.0
    # clearance of every hand link to the glass wall, in the glass's height band
    worst_gap, worst_link = 9.9, "—"
    for l in CONTACT_LINKS:
        c = link_center(l)
        if not (COUNTER_TOP < c[2] < cupn[2] + 0.065):
            continue
        g = float(np.hypot(c[0] - cupn[0], c[1] - cupn[1])) - R_WALL
        if g < worst_gap:
            worst_gap, worst_link = g, l
    d4, d9, still = cups_still()
    tilt = cup_tilt_deg(full_prim)
    f_eff = {f[0]: abs(measured_eff(f[1])) for f in FINGERS}
    f_eff["thumb"] = abs(measured_eff(THUMB_YAW))
    fmax = max(f_eff.values())
    _v = np.abs(np.asarray(robot.get_joint_velocities(), float)).reshape(-1)
    jv = float(_v.max())
    fl = fluid_in_cup(cupn)
    ok &= gate("push_completed", frozen is None,
               "hand reached the entry pose" if frozen is None else
               f"frozen on leg {frozen[0]} step {frozen[1]} (glass moved "
               f"{frozen[2]*1000:.1f} mm)")
    ok &= gate("glass_inside_claw", off < 0.012,
               f"glass axis sits {off*1000:.1f} mm from the claw pocket centre")
    ok &= gate("claw_wraps_glass", arc > 120.0,
               f"claw wraps {arc:.1f} deg of the glass ({arc_lo:+.0f} to {arc_hi:+.0f}, "
               f"0 = palm side, +-90 = the equator)")
    ok &= gate("cup_undisturbed", d4 < 0.004 and tilt < 1.5,
               f"cup drift {d4*1000:.1f} mm, tilt {tilt:.2f} deg")
    ok &= gate("nothing_touching", fmax < 0.05,
               f"max digit effort {fmax:.3f} N*m ({f_eff})")
    ok &= gate("no_table_strike", clearance_viol[0] == 0,
               f"{clearance_viol[0]} captures with a hand link in the strike band")
    ok &= gate("stable", jv < 2.0,
               f"max joint vel {jv:.2f} rad/s ({names[int(_v.argmax())]})")
    gate("arc_report", True, f"per-link reach around the glass (deg): {reach}")
    gate("clearance_report", True,
         f"closest hand link to the glass wall: {worst_link} at {worst_gap*1000:+.1f} mm")
    gate("palm_report", True, f"palm {mis:.1f} deg off radial, pocket offset "
         f"{off*1000:.1f} mm")
    gate("fluid_report", True, f"{fl}/{fluid_n0} particles in cup")

# ============================================================== POUR PREP
elif args.phase == "pour_prep":
    # Ghost-solve the carry-and-tilt corridor from the m3 lift checkpoint. The
    # transform is ported from the scripted pour (g1_grasp_stages s4_pour), which
    # reached 70% transfer -- but there the CUP was driven and the palm followed.
    # Here nothing writes a cup pose: only the palm is commanded, along the
    # rigid map the cup would follow if the grip holds. If it does not hold, the
    # cup falls, and that is the honest result.
    log("POUR PREP (ghost): solve carry-to-hover, then the tilt keyframes.")
    settle_steps(30)
    fc, fmn, fmx = bbox(full_prim)
    ec, emn, emx = bbox(empty_prim)
    RIM9 = float(emx[2])
    CUP_H = float(fmx[2] - fmn[2])
    rimoff_z = float(fmx[2] - fc[2])          # rim above the cup's geometry centre
    d = np.array([ec[0] - fc[0], ec[1] - fc[1], 0.0])
    d /= np.linalg.norm(d)
    # sign matters: +theta must DIP the +d rim edge. The scripted pour got this
    # backwards once and poured 14% -- take the cross product, not the intuition.
    tilt_axis = Gf.Vec3d(float(-d[1]), float(d[0]), 0.0)
    T0 = world_T(palm_prim)
    M0 = np.array([[T0[i][j] for j in range(3)] for i in range(3)])
    pad0 = PALM_PAD_LOC @ M0 + np.array(T0.ExtractTranslation())
    M_cup0 = Gf.Matrix4d(world_T(full_prim))
    cup_t0 = np.array(M_cup0.ExtractTranslation())
    # hover: cup upright, base 10 mm over the receiver rim, spout on its axis.
    # FRAME DISCIPLINE: bbox values are GEOMETRY centres, cup_t0 is the PRIM
    # ORIGIN, and the reconstruction baked those ~2 m apart. Everything below is
    # a geometry-frame DELTA applied to both -- never mix the absolute frames.
    hover_geo = np.array([ec[0] - d[0] * R_WALL, ec[1] - d[1] * R_WALL,
                          RIM9 + 0.010 + CUP_H - rimoff_z])
    carry_delta = hover_geo - fc
    if np.linalg.norm(carry_delta) > 0.6:
        raise SystemExit(f"[task] FATAL carry delta {carry_delta} — frame mixing")
    hover_origin = cup_t0 + carry_delta
    spout_hover = np.array([ec[0], ec[1], hover_geo[2] + rimoff_z])
    spout_final = np.array([ec[0], ec[1], RIM9 + args.pour_gap])
    log(f"  receiver {ec[:2].round(3)} rim {RIM9:.3f}; carry delta "
        f"{carry_delta.round(3)} ({np.linalg.norm(carry_delta)*1000:.0f} mm)")

    def smoother(t):
        return 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, t)))

    Mh = Gf.Matrix4d(M_cup0)
    Mh.SetRow(3, Gf.Vec4d(float(hover_origin[0]), float(hover_origin[1]),
                          float(hover_origin[2]), 1.0))

    def rel_at(t):
        """rigid hover->t map, and the tilt angle there. The spout descends on a
        lagged curve that finishes by t=0.55, BEFORE the heavy 60-90 deg part of
        the pour: a spout still 9 cm up while the stream ran splashed 43% away."""
        theta = args.tilt_deg * smoother(t)
        desc = smoother(max(0.0, (t - 0.12) / 0.43))
        spout = spout_hover + (spout_final - spout_hover) * desc
        Tn = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*[-float(v) for v in spout_hover]))
        R = Gf.Matrix4d().SetRotate(Gf.Rotation(tilt_axis, float(theta)))
        Tp = Gf.Matrix4d().SetTranslate(Gf.Vec3d(*[float(v) for v in spout]))
        return Gf.Matrix4d(Mh).GetInverse() * (Mh * Tn * R * Tp), theta

    traj = load_traj()
    okall = True
    for frac in POUR_CARRY_F:
        qm, ep, eo = solve_ik6(pad0 + carry_delta * frac, M0)
        traj["sols"][f"pour_carry{_pk(frac)}"] = qm
        for j, v in qm.items():
            base_q[idx[j]] = v
        log(f"  pour_carry{_pk(frac)}: pos err {ep*1000:.1f} mm, orient {eo:.1f} deg")
        okall &= (ep < 0.008 and eo < 6.0)
    pad_h = pad0 + carry_delta
    for t in POUR_TILT_F:
        rel, theta = rel_at(t)
        p_t = np.array(rel.Transform(Gf.Vec3d(*[float(v) for v in pad_h])))
        rot = Gf.Rotation(tilt_axis, float(theta))
        # rows of R_t are the desired world images of the palm link's local axes,
        # so each row rides the same rotation the cup does
        R_t = np.array([np.array(rot.TransformDir(Gf.Vec3d(*[float(x) for x in M0[k]])))
                        for k in range(3)])
        qm, ep, eo = solve_ik6(p_t, R_t)
        traj["sols"][f"pour_t{_pk(t)}"] = qm
        for j, v in qm.items():
            base_q[idx[j]] = v
        log(f"  pour_t{_pk(t)} (theta {theta:5.1f} deg): pos err {ep*1000:.1f} mm, "
            f"orient {eo:.1f} deg")
        okall &= (ep < 0.012 and eo < 9.0)
    traj["pour"] = {"tilt_deg": args.tilt_deg, "pour_gap": args.pour_gap,
                    "receiver_xy": [float(ec[0]), float(ec[1])],
                    "rim9": RIM9, "carry_delta": carry_delta.tolist()}
    with open(args.traj_json, "w") as f:
        json.dump(traj, f, indent=2)
    ok = gate("pour_ik", okall,
              f"{len(POUR_CARRY_F)} carry + {len(POUR_TILT_F)} tilt keyframes "
              f"solved within tolerance")
    log(f"pour corridor -> {args.traj_json}")

# ================================================================ M4 — POUR
elif args.phase == "m4_pour":
    log("M4: carry over the receiver and pour. Nothing writes a cup pose — the "
        "cup rides the grip or it falls.")
    traj = load_traj()
    sols = traj["sols"]
    settle_steps(60)
    ec, _, emx9 = bbox(empty_prim)
    RIM9 = float(emx9[2])
    palm0 = world_pos(palm_prim)
    cup0 = bbox(full_prim)[0]
    rel0_world = cup0 - palm0                # world-frame offset (rotation-contaminated)
    rel0 = _cup_in_palm()                    # cup origin in the PALM frame
    axis0 = _cup_axis_in_palm()              # cup axis in the PALM frame
    n_start = fluid_in_cup(cup0)

    def in_receiver():
        p = fluid_pts()
        r = np.hypot(p[:, 0] - ec[0], p[:, 1] - ec[1])
        return int(((r < 0.055) & (p[:, 2] < RIM9 + 0.02)).sum())

    def spilled():
        c4 = bbox(full_prim)[0]
        p = fluid_pts()
        r4 = np.hypot(p[:, 0] - c4[0], p[:, 1] - c4[1])
        r9 = np.hypot(p[:, 0] - ec[0], p[:, 1] - ec[1])
        return int(((r4 > 0.075) & (r9 > 0.075) & (p[:, 2] < 0.95)).sum())

    def hold_err():
        """how far the cup has moved WITHIN the hand since the start (m).

        Measured as the cup's body origin expressed in the PALM'S OWN FRAME.
        Both of the obvious shortcuts are wrong once the wrist rotates:
        a world-frame offset rotates with the hand, and bbox() returns a
        world-AXIS-ALIGNED centre, which slides relative to the body as the
        cup tilts.  Either one reports ~90 mm of 'slip' for a cup that is
        welded to the palm."""
        return float(np.linalg.norm(_cup_in_palm() - rel0))

    def twist_deg():
        """how far the cup has ROTATED inside the hand since the start (deg).
        Independent of where the cup's origin sits."""
        return float(math.degrees(math.acos(
            np.clip(float(np.dot(_cup_axis_in_palm(), axis0)), -1.0, 1.0))))

    def hold_err_world():
        """the old world-frame number, kept only so the two can be compared."""
        return float(np.linalg.norm(
            (bbox(full_prim)[0] - world_pos(palm_prim)) - rel0_world))

    ok = True
    gate("start_report", True, f"{n_start} particles held, receiver at "
         f"{ec[:2].round(3)} rim {RIM9:.3f}")
    # ---- A: carry to the hover pose over the receiver ----------------------
    for k in [_pk(f) for f in POUR_CARRY_F]:
        blend_to({j: sols[f"pour_carry{k}"][j] for j in ARM}, CARRY_SEG)
        settle_steps(KEY_SETTLE_C)
        log(f"  carry {k}%: in-hand slip {hold_err()*1000:.1f} mm, TWIST "
            f"{twist_deg():.1f} deg (old world metric {hold_err_world()*1000:.1f} mm), "
            f"cup tilt {cup_tilt_deg(full_prim):.1f} deg")
    ok &= gate("carried", hold_err() < 0.020,
               f"cup still in the hand after the carry ({hold_err()*1000:.1f} mm "
               f"of in-hand drift), tilt {cup_tilt_deg(full_prim):.1f} deg")
    ok &= gate("over_receiver",
               float(np.hypot(bbox(full_prim)[0][0] - ec[0],
                              bbox(full_prim)[0][1] - ec[1])) < 0.090,
               f"cup axis {float(np.hypot(bbox(full_prim)[0][0]-ec[0], bbox(full_prim)[0][1]-ec[1]))*1000:.0f} mm "
               f"from the receiver axis")
    # ---- B: tilt ------------------------------------------------------------
    for k in [_pk(f) for f in POUR_TILT_F]:
        blend_to({j: sols[f"pour_t{k}"][j] for j in ARM}, TILT_SEG)
        settle_steps(KEY_SETTLE_T)
        log(f"  tilt t={k}%: cup tilt {cup_tilt_deg(full_prim):.1f} deg, TWIST "
            f"{twist_deg():.1f} deg, in-hand slip {hold_err()*1000:.1f} mm "
            f"(old world metric {hold_err_world()*1000:.1f} mm), "
            f"receiver {in_receiver()}, spilled {spilled()}")
    settle_steps(SETTLE_S(300))
    n9, sp = in_receiver(), spilled()
    ok &= gate("poured", n9 >= 0.50 * fluid_n0,
               f"{n9}/{fluid_n0} particles in the receiver "
               f"({100.0 * n9 / max(fluid_n0, 1):.0f}%)")
    ok &= gate("not_spilled", sp <= 0.10 * fluid_n0,
               f"{sp}/{fluid_n0} particles spilled on the counter")
    ok &= gate("cup_retained", hold_err() < 0.030 and twist_deg() < 10.0,
               f"the cup rode the grip through the pour ({hold_err()*1000:.1f} mm "
               f"in-hand slip, {twist_deg():.1f} deg of twist)")
    # ---- C: back upright, still holding ------------------------------------
    for k in [_pk(f) for f in POUR_TILT_F[:-1]][::-1]:
        blend_to({j: sols[f"pour_t{k}"][j] for j in ARM}, RET_SEG)
    blend_to({j: sols[f"pour_carry{_pk(POUR_CARRY_F[-1])}"][j] for j in ARM}, RET_SEG)
    settle_steps(SETTLE_S(150))
    gate("return_report", True,
         f"back upright: cup tilt {cup_tilt_deg(full_prim):.1f} deg, twist "
         f"{twist_deg():.1f} deg, in-hand slip "
         f"{hold_err()*1000:.1f} mm, receiver {in_receiver()}, spilled {spilled()}")
    _v = np.abs(np.asarray(robot.get_joint_velocities(), float)).reshape(-1)
    gate("stable_report", True, f"max joint vel {float(_v.max()):.2f} rad/s")

# ========================================================== M2c — SCAN
elif args.phase == "m2c_scan":
    # DIAGNOSTIC, not a stage: m2c_hug closed all four fingers to the 95 deg cap
    # at 0.16-0.31 N*m instead of stopping on torque, and index's effort FELL
    # from 0.101 to 0.058 as it curled -- it rode over the wall.  Guessing a new
    # stop angle would be a third tuning shot in the dark, so sweep the whole
    # closure once and record what every joint actually does.  Both phalanges are
    # logged: the gate only read PROXIMALS, while m2b measured the nearest link
    # to the glass as R_middle_INTERMEDIATE, so the load may sit on joints the
    # gate never looked at.
    log("M2c-scan: sweep the fence, then the thumb, logging every joint.")
    settle_steps(60)
    _, gmn, gmx = bbox(full_prim)
    band = (float(gmn[2]) - 0.002, float(gmx[2]) + 0.002)

    def wall_gap(link, cupn):
        """min radial gap (m) from a link's AABB corners to the glass wall,
        counting only corners inside the glass's own height band."""
        _, mn, mx = bbox(stage.GetPrimAtPath(f"{ROBOT}/{link}"))
        best = 9.9
        for cx in (mn[0], mx[0]):
            for cy in (mn[1], mx[1]):
                for cz in (mn[2], mx[2]):
                    if not (band[0] <= cz <= band[1]):
                        continue
                    best = min(best, float(np.hypot(cx - cupn[0], cy - cupn[1])) - R_WALL)
        return None if best > 9.0 else best

    q0 = {f[0]: (float(base_q[idx[f[1]]]), float(base_q[idx[f[2]]])) for f in FINGERS}

    def fset(nname, prox, inter, ratio, qp):
        p0, i0 = q0[nname]
        base_q[idx[prox]] = clamp_lim(prox, qp)
        base_q[idx[inter]] = clamp_lim(inter, i0 + (base_q[idx[prox]] - p0) * ratio)

    def sample(tag, lead):
        cupn = bbox(full_prim)[0]
        r = {"tag": tag, "lead_deg": round(lead / DEG, 2),
             "cup_drift_mm": round(float(np.linalg.norm(cupn - cup_c0)) * 1000, 2),
             "cup_tilt_deg": round(cup_tilt_deg(full_prim), 2), "d": {}}
        for nname, prox, inter, _ in FINGERS:
            r["d"][nname] = {
                "q_prox": round(float(base_q[idx[prox]]) / DEG, 1),
                "e_prox": round(abs(measured_eff(prox)), 3),
                "e_inter": round(abs(measured_eff(inter)), 3),
                "gap_prox_mm": (lambda g: None if g is None else round(g * 1000, 1))(
                    wall_gap(f"R_{nname}_proximal", cupn)),
                "gap_inter_mm": (lambda g: None if g is None else round(g * 1000, 1))(
                    wall_gap(f"R_{nname}_intermediate", cupn)),
            }
        r["thumb"] = {
            "q_yaw": round(float(base_q[idx[THUMB_YAW]]) / DEG, 1),
            "e_yaw": round(abs(measured_eff(THUMB_YAW)), 3),
            "e_pitch": round(abs(measured_eff("R_thumb_proximal_pitch_joint")), 3),
            "e_inter": round(abs(measured_eff("R_thumb_intermediate_joint")), 3),
            "e_distal": round(abs(measured_eff("R_thumb_distal_joint")), 3),
            "gap_distal_mm": (lambda g: None if g is None else round(g * 1000, 1))(
                wall_gap("R_thumb_distal", cupn)),
            "gap_inter_mm": (lambda g: None if g is None else round(g * 1000, 1))(
                wall_gap("R_thumb_intermediate", cupn)),
        }
        rows.append(r)
        gp = {k: v["gap_inter_mm"] for k, v in r["d"].items()}
        ep = {k: (v["e_prox"], v["e_inter"]) for k, v in r["d"].items()}
        log(f"  [{tag}] lead {r['lead_deg']:5.1f}  drift {r['cup_drift_mm']:5.2f} mm "
            f"tilt {r['cup_tilt_deg']:4.2f}  eff(prox,inter) {ep}  gap_inter {gp}")
        return r

    rows = []
    CAP = min(args.finger_cap, 97.40) * DEG
    lead = max(float(base_q[idx[f[1]]]) for f in FINGERS)
    off = {f[0]: float(base_q[idx[f[1]]]) - lead for f in FINGERS}
    sample("fence", lead)
    while lead < CAP - 1e-9:
        lead = min(lead + 2.0 * DEG, CAP)
        for nname, prox, inter, ratio in FINGERS:
            fset(nname, prox, inter, ratio, lead + off[nname])
        for _ in range(8):
            tick()
        sample("fence", lead)
    # pick the fence angle that loaded the digits hardest, retract to it, and
    # sweep the thumb from THERE -- sweeping with the fence at the cap would
    # characterise a configuration the grasp will never use.
    fr = [r for r in rows if r["tag"] == "fence"]
    best = max(fr, key=lambda r: sum(d["e_prox"] + d["e_inter"] for d in r["d"].values()))
    log(f"  best-loaded fence lead = {best['lead_deg']} deg "
        f"(sum |effort| {sum(d['e_prox'] + d['e_inter'] for d in best['d'].values()):.3f} N*m)")
    lead = best["lead_deg"] * DEG
    tgt = {}
    for nname, prox, inter, ratio in FINGERS:
        p0, i0 = q0[nname]
        tgt[prox] = clamp_lim(prox, lead + off[nname])
        tgt[inter] = clamp_lim(inter, i0 + (tgt[prox] - p0) * ratio)
    blend_to(tgt, 90)
    settle_steps(90)
    sample("fence_best", lead)
    ty = float(base_q[idx[THUMB_YAW]])
    t_cap = jlim[THUMB_YAW][1]
    while ty < t_cap - 1e-9:
        ty = min(ty + 2.0 * DEG, t_cap)
        base_q[idx[THUMB_YAW]] = clamp_lim(THUMB_YAW, ty)
        for _ in range(8):
            tick()
        sample("thumb", lead)
    with open(os.path.join(phase_dir, "m2c_scan.json"), "w") as f:
        json.dump({"rows": rows, "band": band, "R_WALL": R_WALL,
                   "best_fence_deg": best["lead_deg"]}, f, indent=2)
    log(f"scan -> {os.path.join(phase_dir, 'm2c_scan.json')}")
    ok = gate("scan_complete", True, f"{len(rows)} samples recorded")

# ================================================================ M2c — HUG
elif args.phase == "m2c_hug":
    # Close the staged claw onto the glass.
    #
    # WHY THIS IS A NEW STAGE AND NOT A FIX TO m2_grasp: m2_grasp closed from the
    # m1 station, where the fingers RUN OUT OF TRAVEL.  Its loop capped at 85 deg
    # and the four proximals ended at 0.00/0.13/0.14/0.16 N*m against a 0.45 N*m
    # clamp -- index never touched the glass at all.  A +3 deg preload into a
    # joint that is already at its cap does nothing, which is why the grip slipped
    # 5.9 mm on a 12 mm micro-lift.  Measured off the m2b checkpoint (limits are
    # 0..97.4 deg): index 53.6, middle 61.7, ring 68.8, pinky 76.2 deg of travel
    # remain, and the thumb yaw has 80.2 deg -- so here the drives CAN reach their
    # clamps.  Order is fence-then-thumb: a lone digit pressing first shoved the
    # glass 58 mm across the counter and tipped it 47 deg (m2b).
    log("M2c: hug — close the staged claw onto the glass, load the drives.")
    traj = load_traj()
    sols = traj["sols"]
    settle_steps(60)
    ok = True

    TIP_LINKS = ["R_index_intermediate", "R_middle_intermediate",
                 "R_ring_intermediate", "R_pinky_intermediate"]
    _trace = []
    if args.trace_closure:
        _tip_setup(TIP_LINKS)
        log(f"closure trace ON -> {args.trace_closure}")

    def _sample(tag, lead_deg):
        if not args.trace_closure:
            return
        v = np.abs(np.asarray(robot.get_joint_velocities(), float)).reshape(-1)
        _trace.append({"t": cap[0] / 120.0, "tag": tag, "lead": round(lead_deg, 2),
                       "tips": _tip_sample(TIP_LINKS),
                       "eff": {f[0]: round(float(abs(measured_eff(f[1]))), 4) for f in FINGERS},
                       "vmax": round(float(v.max()), 2)})

    SEAT = 0.30        # N*m, 67% of the 0.45 finger clamp — the seat threshold
    TSEAT = 0.55       # N*m, 46% of the 1.2 thumb-yaw clamp
    CAP = min(args.finger_cap, 97.40) * DEG   # hard-clamped to the real joint limit

    # remember the claw's stagger so the wrap SHAPE is preserved while it closes:
    # rewriting every intermediate as prox*ratio would snap middle/ring/pinky
    # back by 5 deg on the first step and un-curl the fence into the glass.
    q0 = {f[0]: (float(base_q[idx[f[1]]]), float(base_q[idx[f[2]]])) for f in FINGERS}

    def fset(nname, prox, inter, ratio, qp):
        p0, i0 = q0[nname]
        base_q[idx[prox]] = clamp_lim(prox, qp)
        base_q[idx[inter]] = clamp_lim(inter, i0 + (base_q[idx[prox]] - p0) * ratio)

    cup_ref = bbox(full_prim)[0]
    fl0 = fluid_in_cup(cup_ref)
    gate("start_report", True,
         f"claw staged, {fl0}/{fluid_n0} particles; travel left (deg) "
         f"{ {f[0]: round((jlim[f[1]][1] - float(base_q[idx[f[1]]])) / DEG, 1) for f in FINGERS} }, "
         f"thumb yaw {(jlim[THUMB_YAW][1] - float(base_q[idx[THUMB_YAW]])) / DEG:.1f}")

    # ---- A: the fence advances AS A UNIT until the first digit meets glass ----
    lead = max(float(base_q[idx[f[1]]]) for f in FINGERS)
    off = {f[0]: float(base_q[idx[f[1]]]) - lead for f in FINGERS}
    contact_at = None
    RATE = args.close_rate * DEG
    NIT = int(260 * 0.6 / max(args.close_rate, 1e-3))
    for it in range(NIT):
        lead = min(lead + RATE, CAP)
        for nname, prox, inter, ratio in FINGERS:
            fset(nname, prox, inter, ratio, lead + off[nname])
        for _ in range(4):
            tick()
        _sample("A", lead / DEG)
        eff = {f[0]: abs(measured_eff(f[1])) for f in FINGERS}
        if max(eff.values()) >= 0.10:
            contact_at = lead
            log(f"  fence first contact, lead {lead / DEG:.1f} deg, efforts "
                f"{ {k: round(v, 3) for k, v in eff.items()} }")
            break
        if lead >= CAP:
            log(f"  fence hit the {CAP / DEG:.0f} deg cap with no contact")
            break
    ok &= gate("fence_contact", contact_at is not None,
               f"fence met the glass with the lead digit at {contact_at / DEG:.1f} deg"
               if contact_at is not None else
               "fence closed to the cap without ever touching the glass")
    d4, _, _ = cups_still()
    gate("fence_contact_drift", True,
         f"cup drift {d4 * 1000:.1f} mm, tilt {cup_tilt_deg(full_prim):.2f} deg at first contact")

    # ---- B: each finger now advances ALONE to its own torque stop -------------
    # the glass is trapped against the palm by phase A, so a single digit can no
    # longer shove it; the others sit 25-40 deg off the wall at first contact.
    # Wind every target back onto its own joint before the per-finger phase.
    # Phase A advances the fence AS A UNIT until the first digit reports
    # contact, and it leaves the other three with a standing command error of
    # 8-23 deg (measured).  Without this snap, phase A's inherited lag is
    # already past any lag bound, so every finger "seats" on its first check no
    # matter how slowly phase B advances -- which is exactly why slowing the
    # closure 8x changed nothing.  Lag is only a contact signal when it is
    # measured from a zero-lag start.
    for nname, prox, inter, ratio in FINGERS:
        fset(nname, prox, inter, ratio, measured_pos(prox))
    for _ in range(30):
        tick()
    log("  targets wound onto the joints; residual lag (deg) "
        f"{ {f[0]: round((float(base_q[idx[f[1]]]) - measured_pos(f[1])) / DEG, 1) for f in FINGERS} }")

    stopped = {f[0]: False for f in FINGERS}
    seat_q = {}
    seat_why = {}
    # A finger that has STOPPED MOVING while its target keeps advancing is
    # seated -- that is what contact with the glass looks like from the joint.
    # Seating on measured effort alone does not work here: the four proximals
    # top out at 0.11-0.29 N*m against a 0.30 threshold, so every finger ran to
    # its POSITION CAP instead, leaving the target 10-20 deg beyond the joint
    # for seconds.  That saturated drive is what pressed the pinky through the
    # 5-7 mm glass wall and what held the whole hand buzzing at 12 rad/s
    # (measured, --trace_closure).  Stall is detected on the tracking error and
    # the target is then wound back onto the joint plus a small preload, so the
    # drive holds the glass instead of chasing a point inside it.
    for it in range(NIT):
        alldone = True
        for nname, prox, inter, ratio in FINGERS:
            if stopped[nname]:
                continue
            alldone = False
            fset(nname, prox, inter, ratio, float(base_q[idx[prox]]) + RATE)
            qm = measured_pos(prox)
            lag = float(base_q[idx[prox]]) - qm
            eff = abs(measured_eff(prox))
            why = ("GRIPPING the glass" if eff >= args.grip_target else
                   "LAG CAP — pressing without load" if lag >= args.max_lag * DEG else
                   "AT CAP" if base_q[idx[prox]] >= CAP else None)
            if why:
                stopped[nname] = True
                # hold at the joint plus the lag it actually needed: never
                # leave the target parked inside the glass.
                fset(nname, prox, inter, ratio,
                     qm + min(lag, args.max_lag * DEG))
                seat_q[nname] = float(base_q[idx[prox]])
                seat_why[nname] = why
                log(f"  {nname:6s} seated at {seat_q[nname] / DEG:5.1f} deg "
                    f"({why}; lag {lag / DEG:.1f} deg, effort {eff:.3f} N*m)")
        for _ in range(4):
            tick()
        _sample("B", float(base_q[idx["R_ring_proximal_joint"]]) / DEG)
        if alldone:
            break
    if args.trace_closure:
        with open(args.trace_closure, "w") as fh:
            json.dump(_trace, fh)
        log(f"closure trace: {len(_trace)} samples -> {args.trace_closure}")
    ok &= gate("fence_seated", sum(stopped.values()) == 4,
               f"{sum(stopped.values())}/4 fingers reached a stop "
               f"({ {k: round(v / DEG, 1) for k, v in seat_q.items()} } deg) "
               f"— {seat_why}")
    _ong = sum(1 for w in seat_why.values() if w.startswith("GRIPPING"))
    ok &= gate("seated_on_glass", _ong >= 3,
               f"{_ong}/4 fingers stopped because they were LOADED on the "
               f"glass, not because they ran out of joint travel")

    # ---- C: the thumb closes the jaw ----------------------------------------
    # the entry pose parks the yaw at its -5.7 deg limit precisely so the thumb
    # does not foul the mouth on the way in; the hug is the yaw coming back.
    t_stop, t_cap = None, jlim[THUMB_YAW][1] - 0.5 * DEG
    for it in range(240 if args.hug_thumb else 0):
        base_q[idx[THUMB_YAW]] = clamp_lim(THUMB_YAW, float(base_q[idx[THUMB_YAW]]) + 0.6 * DEG)
        for _ in range(4):
            tick()
        te = abs(measured_eff(THUMB_YAW))
        if te >= TSEAT or base_q[idx[THUMB_YAW]] >= t_cap:
            t_stop = float(base_q[idx[THUMB_YAW]])
            log(f"  thumb yaw stopped at {t_stop / DEG:.1f} deg (effort {te:.3f} N*m"
                f"{', AT LIMIT' if t_stop >= t_cap else ''})")
            break
    thumb_loaded = abs(measured_eff(THUMB_YAW)) >= 0.30
    if not args.hug_thumb:
        log("  thumb hug DISABLED (--hug_thumb 0): the fence grips alone")
    gate("thumb_hug", True,
         f"thumb yaw at {float(base_q[idx[THUMB_YAW]]) / DEG:.1f} deg, effort "
         f"{abs(measured_eff(THUMB_YAW)):.3f} N*m "
         f"({'loaded' if thumb_loaded else 'NOT loaded — fence-vs-palm closure only'})")

    # ---- C2: thumb CAP over the mouth ---------------------------------------
    # not a grasp contact -- an axial stop. The fence rings the glass and cannot
    # stop it sliding out along its own axis once tilted; the folded thumb can.
    if args.thumb_cap:
        cap_tgt = {"R_thumb_proximal_pitch_joint": 34.0 * DEG,
                   "R_thumb_intermediate_joint": 23.0 * DEG,
                   "R_thumb_distal_joint": 34.0 * DEG}
        blend_to({j: clamp_lim(j, v) for j, v in cap_tgt.items()}, 150)
        settle_steps(120)
        _, _, gmx_c = bbox(full_prim)
        _, tmn, tmx = bbox(stage.GetPrimAtPath(f"{ROBOT}/R_thumb_distal"))
        gate("thumb_cap", True,
             f"thumb folded over the mouth: tip z {tmn[2]:.3f}..{tmx[2]:.3f} vs "
             f"rim {float(gmx_c[2]):.3f}, pitch effort "
             f"{abs(measured_eff('R_thumb_proximal_pitch_joint')):.3f} N*m, cup "
             f"tilt {cup_tilt_deg(full_prim):.2f} deg, drift "
             f"{cups_still(tol=0.05)[0]*1000:.1f} mm")

    # ---- D: preload every seated digit into its clamp ------------------------
    # 4 deg at kp 20 demands 1.4 N*m, so a genuinely blocked joint saturates at
    # 0.45; a joint that still has somewhere to go just follows and reads low.
    pre = {}
    for nname, prox, inter, ratio in FINGERS:
        p0, i0 = q0[nname]
        pre[prox] = clamp_lim(prox, float(base_q[idx[prox]]) + 4.0 * DEG)
        pre[inter] = clamp_lim(inter, i0 + (pre[prox] - p0) * ratio)
    if args.hug_thumb:
        pre[THUMB_YAW] = clamp_lim(THUMB_YAW, float(base_q[idx[THUMB_YAW]]) + 3.0 * DEG)
    blend_to(pre, 45)
    settle_steps(150)

    f_eff = {f[0]: abs(measured_eff(f[1])) for f in FINGERS}
    sat = [n for n, prox, _, _ in FINGERS
           if f_eff[n] >= 0.85 * HAND_GAINS[prox][2]]
    ok &= gate("drives_saturated", len(sat) >= 3,
               f"{len(sat)}/4 finger proximals at >=85% of the 0.45 N*m clamp "
               f"({ {k: round(v, 3) for k, v in f_eff.items()} })")
    d4, _, _ = cups_still(tol=0.05)
    tilt = cup_tilt_deg(full_prim)
    ok &= gate("cup_upright", tilt < 4.0,
               f"cup tilt {tilt:.2f} deg, drift {d4 * 1000:.1f} mm after the hug")
    gate("fluid_after_hug", True, f"{fluid_in_cup(bbox(full_prim)[0])}/{fluid_n0} particles in cup")

    # ---- E: micro-lift — the grip is real or it is not -----------------------
    palm_a = world_pos(palm_prim)
    cup_a = bbox(full_prim)[0]
    ml = sols.get("claw_lift")
    if ml is None:
        gate("micro_lift_grip", False, "no claw_lift waypoint in the traj json — "
             "re-run claw_prep to solve it")
        ok = False
    else:
        blend_to({j: ml[j] for j in ARM}, 120)
        settle_steps(150)
        palm_b = world_pos(palm_prim)
        cup_b = bbox(full_prim)[0]
        dpalm, dcup = palm_b[2] - palm_a[2], cup_b[2] - cup_a[2]
        fl = fluid_in_cup(cup_b)
        ok &= gate("micro_lift_grip",
                   abs(dcup - dpalm) < 0.005 and cup_tilt_deg(full_prim) < 5.0,
                   f"palm +{dpalm * 1000:.1f} mm, cup +{dcup * 1000:.1f} mm "
                   f"(slip {abs(dcup - dpalm) * 1000:.1f} mm), tilt "
                   f"{cup_tilt_deg(full_prim):.2f} deg")
        ok &= gate("fluid_held", fl >= 420, f"{fl}/{fluid_n0} particles in cup")
        # back down to the entry height, STILL GRIPPED — this is the checkpoint
        # m3_claw_lift starts from, so it must end where m2b ended, holding.
        blend_to({j: sols["claw_in"][j] for j in ARM}, 120)
        settle_steps(120)
        held = {f[0]: round(abs(measured_eff(f[1])), 3) for f in FINGERS}
        gate("set_back_report", True,
             f"cup back down, tilt {cup_tilt_deg(full_prim):.2f} deg, "
             f"still gripped ({held} N*m)")
    _v = np.abs(np.asarray(robot.get_joint_velocities(), float)).reshape(-1)
    gate("stable", True, f"max joint vel {float(_v.max()):.2f} rad/s "
         f"({names[int(_v.argmax())]})")

# ========================================================= M3 — LIFT (CLAW)
elif args.phase == "m3_claw_lift":
    log("M3 (claw): friction-carried lift — grip from the m2c hug checkpoint.")
    traj = load_traj()
    sols = traj["sols"]
    settle_steps(60)
    palm_a = world_pos(palm_prim)
    cup_a = bbox(full_prim)[0]
    fl_a = fluid_in_cup(cup_a)
    up_m = float(traj.get("waypoints_xyz", {}).get("claw_carry_m", 0.15))
    blend_to({j: sols["claw_carry"][j] for j in ARM}, 300)   # 2.5 s, slow
    settle_steps(240)                                        # 2 s hold
    palm_b = world_pos(palm_prim)
    cup_b = bbox(full_prim)[0]
    dpalm, dcup = palm_b[2] - palm_a[2], cup_b[2] - cup_a[2]
    fl = fluid_in_cup(cup_b)
    ok = True
    ok &= gate("lifted", dcup >= 0.6 * up_m,
               f"cup rose {dcup * 1000:.1f} mm of a {up_m * 1000:.0f} mm command "
               f"(palm {dpalm * 1000:.1f} mm)")
    ok &= gate("grip_held", abs(dcup - dpalm) < 0.012 and cup_tilt_deg(full_prim) < 12.0,
               f"slip {abs(dcup - dpalm) * 1000:.1f} mm, tilt "
               f"{cup_tilt_deg(full_prim):.2f} deg")
    ok &= gate("fluid_held", fl >= 400, f"{fl}/{fluid_n0} particles in cup ({fl_a} at start)")
    f_eff = {f[0]: round(abs(measured_eff(f[1])), 3) for f in FINGERS}
    f_eff["thumb_yaw"] = round(abs(measured_eff(THUMB_YAW)), 3)
    gate("grip_efforts", True, f"holding efforts {f_eff} N*m")
    _v = np.abs(np.asarray(robot.get_joint_velocities(), float)).reshape(-1)
    gate("stable_report", True, f"max joint vel {float(_v.max()):.2f} rad/s "
         f"({names[int(_v.argmax())]})")

# ============================================================== M2 — GRASP
elif args.phase == "m2_grasp":
    log("M2: antipodal yaw-opposition side pinch (design brief 2026-08-14).")
    traj = load_traj()
    sols = traj["sols"]
    settle_steps(60)

    fence_az = None  # finger-fence azimuth, for the antipodal check

    def thumb_pad():
        c = link_center("R_thumb_distal")
        cup = bbox(full_prim)[0]
        az = c[:2] - cup[:2]
        az = az / max(np.linalg.norm(az), 1e-9)
        return c, az

    def measured(j):
        e = np.asarray(robot.get_measured_joint_efforts(), float).reshape(-1)
        return float(e[idx[j]])

    # POWER WRAP (measured, grasp_scan3): palm face on the NEAR wall at
    # standoff 0.095, fence curled ~80 deg onto the FAR wall. The thumb takes
    # no part — it parks above the rim (true opposition is unreachable at this
    # wrist orientation: the yaw sweep puts the pad over the mouth at every
    # standoff). Force closure is fence-vs-palm.
    ok = True
    # --- 1. engage: palm to the scan-chosen wrap point, fingers OPEN ---
    eng = sols.get("engage", sols["station_B"])
    blend_to({j: eng[j] for j in IK_JOINTS}, 120)
    settle_steps(30)
    d4, _, _ = cups_still()
    gate("engage_report", True, f"cup drift {d4*1000:.1f} mm after engage travel")
    pad, az = thumb_pad()
    cupn = bbox(full_prim)[0]
    pad_r = float(np.hypot(pad[0]-cupn[0], pad[1]-cupn[1]))
    ok &= gate("thumb_clear", pad[2] >= 1.020 or pad_r >= 0.055,
               f"thumb pad z={pad[2]:.3f} r={pad_r*1000:.0f} mm (must clear the rim/wall)")

    # --- 3. finger closure AS A UNIT: advance all four together; the moment
    # ANY finger meets the wall (torque), the whole fence advances a few more
    # degrees together, dragging the cup INTO the palm — a lone finger pressing
    # solo shoved the cup over (58 mm, 47 deg tilt, m2b) ---
    cup_ref = bbox(full_prim)[0]
    contact_at = None
    q_all = float(base_q[idx[FINGERS[0][1]]])
    # phase A: unit advance until the FIRST contact traps the cup on the palm
    for it in range(140):
        q_all = min(q_all + 0.75 * DEG, 85 * DEG)
        for nname, prox, inter, ratio in FINGERS:
            base_q[idx[prox]] = clamp_lim(prox, q_all)
            base_q[idx[inter]] = clamp_lim(inter, q_all * ratio)
        for _ in range(4):
            tick()
        eff = {f[0]: abs(measured(f[1])) for f in FINGERS}
        if max(eff.values()) >= 0.10:
            contact_at = q_all
            log(f"  first contact at {q_all/DEG:.1f} deg "
                f"(efforts {[f'{eff[f[0]]:.2f}' for f in FINGERS]})")
            break
        if q_all >= 85 * DEG:
            log("  closure reached 85 deg cap without contact")
            break
    # phase B: cup now trapped against the palm — each finger advances to its
    # OWN torque stop (the others sit 25-40 deg from the wall at first contact)
    stopped = {f[0]: False for f in FINGERS}
    for it in range(150):
        alldone = True
        for nname, prox, inter, ratio in FINGERS:
            if stopped[nname]:
                continue
            alldone = False
            qp = float(base_q[idx[prox]])
            base_q[idx[prox]] = clamp_lim(prox, qp + 0.75 * DEG)
            base_q[idx[inter]] = clamp_lim(inter, base_q[idx[prox]] * ratio)
            if abs(measured(prox)) >= 0.14 or base_q[idx[prox]] >= 85 * DEG:
                stopped[nname] = True
                log(f"  {nname} seated at {base_q[idx[prox]]/DEG:.1f} deg "
                    f"(effort {measured(prox):.3f} N*m)")
        for _ in range(4):
            tick()
        if alldone:
            break
    # phase C: +3 deg preload into the effort clamps
    pre = {}
    for nname, prox, inter, ratio in FINGERS:
        pre[prox] = clamp_lim(prox, float(base_q[idx[prox]]) + 3.0 * DEG)
        pre[inter] = pre[prox] * ratio
    blend_to(pre, 45)
    settle_steps(120)

    sat = [j for j in [f[1] for f in FINGERS] if abs(measured(j)) >= 0.85 * HAND_GAINS[j][2]]
    ok &= gate("drives_saturated", len(sat) >= 3,
               f"{len(sat)}/4 finger proximals at >=85% clamp "
               f"({[f'{abs(measured(f[1])):.2f}' for f in FINGERS]} N*m)")
    d4, _, _ = cups_still()
    gate("cup_still_held_spot", True, f"cup drift {d4*1000:.1f} mm, tilt {cup_tilt_deg(full_prim):.1f} deg")

    # --- 5. micro-lift: +2 cm, hold 1 s — the grip is real or it isn't ---
    palm_a = world_pos(palm_prim)
    cup_a = bbox(full_prim)[0]
    ml = sols.get("engage_lift", sols["micro_lift"])
    blend_to({j: ml[j] for j in IK_JOINTS}, 90)
    settle_steps(120)
    palm_b = world_pos(palm_prim)
    cup_b = bbox(full_prim)[0]
    dpalm, dcup = palm_b[2] - palm_a[2], cup_b[2] - cup_a[2]
    fl = fluid_in_cup(cup_b)
    ok &= gate("micro_lift_grip", abs(dcup - dpalm) < 0.005 and cup_tilt_deg(full_prim) < 5.0,
               f"palm +{dpalm*1000:.1f} mm, cup +{dcup*1000:.1f} mm "
               f"(slip {abs(dcup-dpalm)*1000:.1f} mm), tilt {cup_tilt_deg(full_prim):.1f} deg")
    ok &= gate("fluid_held", fl >= 420, f"{fl}/{fluid_n0} particles in cup")
    # set back down gently, stay gripped
    blend_to({j: eng[j] for j in IK_JOINTS}, 90)
    settle_steps(90)
    d4f, _, _ = cups_still(tol=0.05)
    gate("set_back_report", True, f"cup back near start (drift {d4f*1000:.1f} mm), still gripped")

# ============================================================== M3 — LIFT
elif args.phase == "m3_lift":
    log("M3: friction-carried lift +15 cm — grip from the m2 checkpoint.")
    traj = load_traj()
    sols = traj["sols"]
    settle_steps(60)
    palm_a = world_pos(palm_prim)
    cup_a = bbox(full_prim)[0]
    fl_a = fluid_in_cup(cup_a)
    blend_to({j: sols["lift_high"][j] for j in IK_JOINTS}, 240)   # 2 s, slow
    settle_steps(240)                                             # 2 s hold
    palm_b = world_pos(palm_prim)
    cup_b = bbox(full_prim)[0]
    dpalm, dcup = palm_b[2] - palm_a[2], cup_b[2] - cup_a[2]
    fl = fluid_in_cup(cup_b)
    ok = True
    ok &= gate("lifted", dcup >= 0.10,
               f"cup rose {dcup*1000:.1f} mm (palm {dpalm*1000:.1f} mm)")
    ok &= gate("grip_held", abs(dcup - dpalm) < 0.012 and cup_tilt_deg(full_prim) < 12.0,
               f"slip {abs(dcup-dpalm)*1000:.1f} mm, tilt {cup_tilt_deg(full_prim):.1f} deg")
    ok &= gate("fluid_held", fl >= 400, f"{fl}/{fluid_n0} particles in cup ({fl_a} at start)")
    _v = np.abs(np.asarray(robot.get_joint_velocities(), float)).reshape(-1)
    gate("stable_report", True, f"max joint vel {float(_v.max()):.2f} rad/s "
         f"({names[int(_v.argmax())]})")

# ------------------------------------------------------------- save + report
if args.phase != "ghost_prep" and args.save_usd:
    try:
        from pxr import PhysxSchema
        for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)):
            if p.IsA(UsdPhysics.RevoluteJoint) and p.GetName() in idx:
                jn = p.GetName()
                degv = float(base_q[idx[jn]] / DEG)
                js = PhysxSchema.JointStateAPI.Apply(p, "angular")
                js.CreatePositionAttr(degv).Set(degv)
                js.CreateVelocityAttr(0.0).Set(0.0)
                drv = UsdPhysics.DriveAPI.Get(p, "angular")
                if drv:
                    drv.CreateTargetPositionAttr(degv).Set(degv)
                    # persist the force-limited hand gains or a GUI Play crushes
                    # the cup (USD stiffness/damping are PER-DEGREE; maxForce is
                    # plain N*m — runtime view gains are never saved by export)
                    if jn in HAND_GAINS:
                        jkp, jkd, jmef = HAND_GAINS[jn]
                        drv.CreateStiffnessAttr(0.0).Set(jkp * math.pi / 180.0)
                        drv.CreateDampingAttr(0.0).Set(jkd * math.pi / 180.0)
                        drv.CreateMaxForceAttr(0.0).Set(jmef)
    except Exception as e:
        log(f"joint-state authoring warn: {e}")
    # bake this substage's executed motion as Play-able replay (GUI: press Play).
    # NOTE the START joint state above is the END pose; a replay checkpoint would
    # jump. So replay bakes are written to a SEPARATE *_replay.usdc whose joint
    # state is the substage's START pose.
    if motion_rec:
        try:
            from pxr import PhysxSchema
            t_end = motion_rec[-1][0] + 2.0
            stage.SetStartTimeCode(0)
            stage.SetEndTimeCode(t_end * 60.0)
            stage.SetTimeCodesPerSecond(60.0)
            first = motion_rec[0][1]
            for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)):
                if p.IsA(UsdPhysics.RevoluteJoint) and p.GetName() in first:
                    jn = p.GetName()
                    drv = UsdPhysics.DriveAPI.Get(p, "angular")
                    if not drv:
                        continue
                    tgt = drv.GetTargetPositionAttr()
                    tgt.Set(first[jn], Usd.TimeCode(0))
                    for ts, qd in motion_rec:
                        tgt.Set(qd[jn], Usd.TimeCode(ts * 60.0))
                    tgt.Set(motion_rec[-1][1][jn], Usd.TimeCode(t_end * 60.0))
                    js = PhysxSchema.JointStateAPI.Apply(p, "angular")
                    js.CreatePositionAttr(first[jn]).Set(first[jn])
            replay_path = os.path.abspath(args.save_usd).replace(".usdc", "_replay.usdc")
            stage.GetRootLayer().Export(replay_path)
            log(f"saved replay {replay_path}")
            # restore end-pose joint state + static targets for the chain checkpoint
            for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)):
                if p.IsA(UsdPhysics.RevoluteJoint) and p.GetName() in idx:
                    jn = p.GetName()
                    degv = float(base_q[idx[jn]] / DEG)
                    drv = UsdPhysics.DriveAPI.Get(p, "angular")
                    if drv:
                        tgt = drv.GetTargetPositionAttr()
                        tgt.Clear()
                        tgt.Set(degv)
                    js = PhysxSchema.JointStateAPI.Apply(p, "angular")
                    js.CreatePositionAttr(degv).Set(degv)
        except Exception as e:
            log(f"replay bake warn: {e}")
    stage.GetRootLayer().Export(os.path.abspath(args.save_usd))
    log(f"saved {args.save_usd}")

metrics["frames_captured"] = cap[0] // args.capture_every
metrics["result"] = "PASS" if ok else "FAIL"
with open(os.path.join(phase_dir, f"{args.phase}_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
log(f"metrics -> {os.path.join(phase_dir, args.phase + '_metrics.json')}")
log(f"RESULT {args.phase}: {'PASS' if ok else 'FAIL'}")
sim.close()
