#!/usr/bin/env python3
"""Does the hand EVER get inside the glass material? Sampled every single step.

Sweeps the right arm through the cup station at several command speeds and, at
every physics step, measures how deep any hand link is inside the cup's glass
solid (annulus wall + base slab, evaluated in the cup's own frame so tilt cannot
corrupt it). Also records how far the cups are pushed.

Penetration that appears only at high command speed is a fast-transient, not a
collider defect: an articulation slammed to a new position target is a very
high-authority actuator and can out-run contact resolution for a frame or two.
Penetration at LOW speed would mean collision detection is genuinely failing.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--joint", default="right_shoulder_yaw_joint")
ap.add_argument("--sweep_deg", type=float, default=30.0)
ap.add_argument("--steps", default="1500,600,240,60",
                help="steps to cover the sweep: fewer steps = faster command")
ap.add_argument("--out", default="output/banner4/phasing/hand_sweep.json")
args = ap.parse_args()

from isaacsim import SimulationApp
sim = SimulationApp({"headless": True})
import omni.usd                                                    # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics                           # noqa: E402
from isaacsim.core.api import World                                # noqa: E402
from isaacsim.core.utils.types import ArticulationAction           # noqa: E402
import isaacsim.core.utils.stage as stage_utils                    # noqa: E402

def log(*a): print("[sweep]", *a, flush=True)

stage_utils.open_stage(os.path.abspath(args.usd))
stage = omni.usd.get_context().get_stage()
ROBOT, CUPS = "/World/G1", ["/World/labg4", "/World/labg9"]
for cup in CUPS:
    UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(cup)).GetKinematicEnabledAttr().Set(False)

world = World(stage_units_in_meters=1.0, physics_dt=1.0/120.0, rendering_dt=1.0/60.0)
pc = world.get_physics_context(); pc.enable_gpu_dynamics(True)
try: pc.set_broadphase_type("GPU")
except Exception: pass
from isaacsim.core.prims import SingleArticulation                 # noqa: E402
robot = SingleArticulation(prim_path=f"{ROBOT}/root_joint", name="g1")
world.scene.add(robot)
world.reset()
for _ in range(120): world.step(render=False)
world.step(render=True)

names = robot.dof_names; idx = {n: i for i, n in enumerate(names)}
ac = robot.get_articulation_controller()
xc = UsdGeom.XformCache(Usd.TimeCode.Default())

LINKS = ["right_hand_base_link", "right_hand_camera_base_link",
         "R_thumb_proximal_base", "R_thumb_proximal", "R_thumb_intermediate", "R_thumb_distal",
         "R_index_proximal", "R_index_intermediate", "R_middle_proximal", "R_middle_intermediate",
         "R_ring_proximal", "R_ring_intermediate", "R_pinky_proximal", "R_pinky_intermediate"]

def cup_frame(cp):
    P = np.array([[v[0],v[1],v[2]] for v in
                  UsdGeom.Mesh(stage.GetPrimAtPath(cp+"/visual")).GetPointsAttr().Get()])
    cx,cy=(P[:,0].min()+P[:,0].max())/2,(P[:,1].min()+P[:,1].max())/2
    r=np.hypot(P[:,0]-cx,P[:,1]-cy); z=P[:,2]; zb,zr=float(z.min()),float(z.max())
    uz=np.unique(np.round(z,6)); mid=[v for v in uz if zb+1e-6<v<zr-1e-6]; zf=float(mid[0])
    rim=r[np.abs(z-zr)<1e-6]; base=r[np.abs(z-zb)<1e-6]
    return dict(cx=cx,cy=cy,zb=zb,zf=zf,zr=zr,r_in=float(np.min(rim[rim>1e-4])),
                r_ob=float(np.max(base)),r_or=float(np.max(rim)))
FR = {c: cup_frame(c) for c in CUPS}

# cache each link's collision points ONCE in link-local space; transform per step
LOCAL = {}
for l in LINKS:
    col = stage.GetPrimAtPath(f"{ROBOT}/{l}/collisions")
    if not col.IsValid(): continue
    acc = []
    Ml = np.array(xc.GetLocalToWorldTransform(stage.GetPrimAtPath(f"{ROBOT}/{l}")), dtype=np.float64)
    Mli = np.linalg.inv(Ml)
    for q in Usd.PrimRange(col, Usd.TraverseInstanceProxies()):
        if q.IsA(UsdGeom.Mesh):
            P = UsdGeom.Mesh(q).GetPointsAttr().Get()
            if not P: continue
            A = np.array([[v[0],v[1],v[2]] for v in P])
            if len(A) > 1500: A = A[::max(1, len(A)//1500)]
            Mw = np.array(xc.GetLocalToWorldTransform(q), dtype=np.float64)
            acc.append((np.c_[A, np.ones(len(A))] @ Mw @ Mli)[:, :3])
    if acc: LOCAL[l] = np.vstack(acc)
log(f"cached {len(LOCAL)} hand links, {sum(len(v) for v in LOCAL.values())} sample points")

def pen_now():
    """max mm any hand link is inside either glass solid, and which link."""
    xc.Clear()
    best, who = 0.0, ""
    Minv = {c: np.linalg.inv(np.array(xc.GetLocalToWorldTransform(stage.GetPrimAtPath(c)), dtype=np.float64))
            for c in CUPS}
    for l, A in LOCAL.items():
        Ml = np.array(xc.GetLocalToWorldTransform(stage.GetPrimAtPath(f"{ROBOT}/{l}")), dtype=np.float64)
        Pw = (np.c_[A, np.ones(len(A))] @ Ml)[:, :3]
        for c in CUPS:
            m = FR[c]
            Ph = np.c_[Pw, np.ones(len(Pw))] @ Minv[c]
            x,y,z = Ph[:,0]-m["cx"], Ph[:,1]-m["cy"], Ph[:,2]
            r = np.hypot(x,y)
            t = np.clip((z-m["zb"])/(m["zr"]-m["zb"]),0,1)
            ro = m["r_ob"] + (m["r_or"]-m["r_ob"])*t
            inw = (r>=m["r_in"])&(r<=ro)&(z>=m["zf"])&(z<=m["zr"])
            if inw.any():
                d = np.minimum.reduce([r-m["r_in"],ro-r,z-m["zf"],m["zr"]-z])[inw].max()
                if d > best: best, who = float(d), f"{l}->{c.split('/')[-1]}"
            inb = (r<=m["r_ob"])&(z>=m["zb"])&(z<=m["zf"])
            if inb.any():
                d = np.minimum.reduce([m["r_ob"]-r,z-m["zb"],m["zf"]-z])[inb].max()
                if d > best: best, who = float(d), f"{l}->{c.split('/')[-1]}(base)"
    return best*1000.0, who

def cpos(c):
    xc.Clear()
    return np.array(xc.GetLocalToWorldTransform(stage.GetPrimAtPath(c)).ExtractTranslation(), float)

q_home = np.asarray(robot.get_joint_positions(), float).copy()
J = args.joint
results = []
for NS in [int(v) for v in args.steps.split(",")]:
    # restore
    ac.apply_action(ArticulationAction(joint_positions=q_home))
    for _ in range(200): world.step(render=False)
    home = {c: cpos(c) for c in CUPS}
    tgt = q_home.copy()
    deg_per_step = args.sweep_deg / NS
    worst, worst_who, worst_s = 0.0, "", -1
    for s in range(NS+1):
        tgt[idx[J]] = q_home[idx[J]] + np.radians(args.sweep_deg)*(s/NS)
        ac.apply_action(ArticulationAction(joint_positions=tgt))
        world.step(render=False)
        p, who = pen_now()
        if p > worst: worst, worst_who, worst_s = p, who, s
    mv = {c: float(np.linalg.norm(cpos(c)-home[c])*1000) for c in CUPS}
    log(f"steps={NS:5d}  ({deg_per_step*120:6.1f} deg/s command)  "
        f"MAX PENETRATION {worst:7.2f} mm  at step {worst_s} [{worst_who or 'none'}]   "
        f"cups moved {mv['/World/labg4']:.1f} / {mv['/World/labg9']:.1f} mm")
    results.append({"steps": NS, "deg_per_s": deg_per_step*120, "max_pen_mm": worst,
                    "where": worst_who, "cup_moved_mm": mv})

os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(results, open(args.out, "w"), indent=2)
log(f"wrote {args.out}")
sim.close()
