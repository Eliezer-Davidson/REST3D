#!/usr/bin/env python3
"""Phasing gate: measure REAL interpenetration between the two cups, and audit
every collider for the "flag on / geometry absent" defect.

Two independent tests, because there are two independent failure modes:

  cup_cup  DYNAMIC. Drive labg9 into labg4 under velocity control and record the
           closest centre-to-centre XY approach. Two cups of outer radii R9 and
           R4 are rigid iff that distance never drops below R9+R4. Anything less
           is interpenetration, measured in mm, with no appeal to contact
           reports -- a contact report cannot detect a contact that was never
           generated, which is precisely the bug.

           --mode press      velocity-driven, slow. Tests CONTACT GENERATION:
                             a slow body that sinks in has no contacts at all.
           --mode ballistic  initial velocity, fast. Tests TUNNELLING (CCD).
           Running both separates the two causes instead of conflating them.

  colliders STATIC, no GPU. Every prim carrying CollisionAPI + collisionEnabled
           is checked for actual geometry in its subtree. The G1 asset ships
           `collisions` Xforms that reference a /Flattened_Prototype_* prim
           absent from the composed stage: the flag reads ON and physics sees
           nothing, so any audit that counts collisionEnabled reports a fully
           collidable robot. That is a silent, total loss of contact.

Exit 1 on any violation. Pure measurement -- this script never fixes anything.
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--test", default="all", choices=["all", "cup_cup", "colliders", "settle"])
ap.add_argument("--mode", default="press", choices=["press", "ballistic", "both", "sweep", "bsweep"])
ap.add_argument("--speeds", default="0.02,0.10,0.30,1.00,2.00",
                help="sweep mode: press speeds in m/s. Penetration that GROWS with "
                     "speed is tunnelling; penetration at the LOWEST speed is a "
                     "contact-generation failure and CCD cannot help it.")
ap.add_argument("--speed", type=float, default=0.05, help="press speed, m/s")
ap.add_argument("--bal_speed", type=float, default=2.0, help="ballistic speed, m/s")
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--tol", type=float, default=0.002, help="allowed penetration, m")
ap.add_argument("--out_dir", default="output/banner4/phasing")
ap.add_argument("--tag", default="probe")
ap.add_argument("--headless", type=int, default=1)
ap.add_argument("--physics_hz", type=float, default=120.0,
                help="World() OVERRIDES the authored physxScene:timeStepsPerSecond, "
                     "so the substep rate must be set here, not in the USD. With "
                     "CCD unavailable under GPU dynamics this is the primary "
                     "anti-tunnelling control: per-step travel = speed / physics_hz.")
args = ap.parse_args()

FULL_CUP, EMPTY_CUP, ROBOT = "/World/labg4", "/World/labg9", "/World/G1"
os.makedirs(args.out_dir, exist_ok=True)
report = {"usd": args.usd, "tests": {}}


def log(*a):
    print("[phase]", *a, flush=True)


# ---------------------------------------------------------------- static audit
def audit_colliders(usd_path):
    """No GPU, no Isaac: pure USD. Returns (n_real, dead_list)."""
    from pxr import Usd, UsdGeom, UsdPhysics
    st = Usd.Stage.Open(os.path.abspath(usd_path))
    real, dead = [], []
    for p in st.Traverse():
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            continue
        a = p.GetAttribute("physics:collisionEnabled")
        if a and a.HasAuthoredValue() and a.Get() is False:
            continue                                   # deliberately off: not a defect
        verts = 0
        # MUST traverse instance proxies: most G1 `collisions` scopes are USD
        # INSTANCES, and a plain PrimRange stops at the proxy and counts zero.
        # Without this the audit reports healthy colliders as dead -- a false
        # negative that sends you chasing a collision bug that does not exist.
        for k in Usd.PrimRange(p, Usd.TraverseInstanceProxies()):
            if k.IsA(UsdGeom.Mesh):
                pts = UsdGeom.Mesh(k).GetPointsAttr().Get()
                verts += len(pts) if pts else 0
            elif k.GetTypeName() in ("Cube", "Sphere", "Cylinder", "Capsule", "Cone", "Plane"):
                verts += 1                             # analytic prim: real geometry
        (real if verts else dead).append((str(p.GetPath()), verts))
    return real, dead


# ------------------------------------------------------------ geometry helpers
def cup_radius(stage, path):
    """Outer radius from the collision mesh itself, about its own vertical axis."""
    from pxr import Usd, UsdGeom
    m = None
    for k in Usd.PrimRange(stage.GetPrimAtPath(path)):
        if k.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(k); break
    P = np.array([[v[0], v[1], v[2]] for v in m.GetPointsAttr().Get()])
    cx, cy = (P[:, 0].min() + P[:, 0].max()) / 2, (P[:, 1].min() + P[:, 1].max()) / 2
    return float(np.hypot(P[:, 0] - cx, P[:, 1] - cy).max())


def run_static():
    real, dead = audit_colliders(args.usd)
    log(f"colliders with real geometry : {len(real)}")
    log(f"colliders DEAD (flag on, no geometry): {len(dead)}")
    for p, _ in dead:
        log(f"    DEAD  {p}")
    report["tests"]["colliders"] = {
        "n_real": len(real), "n_dead": len(dead),
        "dead": [p for p, _ in dead],
        "pass": len(dead) == 0,
    }
    return len(dead) == 0


if args.test == "colliders":
    ok = run_static()
    json.dump(report, open(f"{args.out_dir}/{args.tag}_report.json", "w"), indent=2)
    log(f"RESULT {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)

# ------------------------------------------------------------------ dynamic run
static_ok = run_static() if args.test == "all" else True

from isaacsim import SimulationApp
sim = SimulationApp({"headless": bool(args.headless)})

import omni.usd                                                    # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf          # noqa: E402
from isaacsim.core.api import World                                # noqa: E402
import isaacsim.core.utils.stage as stage_utils                    # noqa: E402
from isaacsim.core.prims import SingleRigidPrim                    # noqa: E402

stage_utils.open_stage(os.path.abspath(args.usd))
stage = omni.usd.get_context().get_stage()

# Isolate the cups: the robot is not under test here, and a stray arm collider
# touching a cup would confound the measurement (one variable at a time).
n_off = 0
rp = stage.GetPrimAtPath(ROBOT)
if rp.IsValid():
    for prim in Usd.PrimRange(rp):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
            n_off += 1
log(f"robot colliders disabled for isolation: {n_off}")

for cup in (FULL_CUP, EMPTY_CUP):
    UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(cup)).GetKinematicEnabledAttr().Set(False)

R4, R9 = cup_radius(stage, FULL_CUP), cup_radius(stage, EMPTY_CUP)
CONTACT_D = R4 + R9
log(f"outer radii: labg4 {R4*1000:.2f} mm, labg9 {R9*1000:.2f} mm "
    f"-> contact at {CONTACT_D*1000:.2f} mm centre-to-centre")

world = World(stage_units_in_meters=1.0, physics_dt=1.0 / args.physics_hz,
              rendering_dt=1.0 / 60.0)
log(f"physics at {args.physics_hz:.0f} Hz -> a cup at 2 m/s advances "
    f"{2.0/args.physics_hz*1000:.2f} mm per step")
pc = world.get_physics_context()
pc.enable_gpu_dynamics(True)
try:
    pc.set_broadphase_type("GPU")
except Exception:
    pass

bcache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])


def bbox_centre(path):
    bcache.Clear()
    b = bcache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedBox()
    mn, mx = b.GetMin(), b.GetMax()
    return np.array([(mn[i] + mx[i]) / 2.0 for i in range(3)])


def quat_to_R(q):
    """Isaac returns (w, x, y, z)."""
    w, x, y, z = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


world.reset()
for _ in range(60):                       # let the scene settle before pushing
    world.step(render=False)

rb4 = SingleRigidPrim(prim_path=FULL_CUP, name="cup4");  rb4.initialize()
rb9 = SingleRigidPrim(prim_path=EMPTY_CUP, name="cup9"); rb9.initialize()

# Calibrate body-origin -> geometric-axis offset ONCE, in each body's local frame.
# The rigid-body origin is not the geometry centre in this scene, and a bbox read
# between physics steps is stale unless a render is forced -- so every in-loop
# measurement comes from the physics pose, transformed by this constant offset.
world.step(render=True)                   # one render so the bbox read is current
CAL = {}
for tag, rb, path in (("4", rb4, FULL_CUP), ("9", rb9, EMPTY_CUP)):
    p0, q0 = rb.get_world_pose()
    c0 = bbox_centre(path)
    CAL[tag] = quat_to_R(q0).T @ (c0 - np.asarray(p0, dtype=float))
    log(f"cup{tag}: body origin -> geom centre offset (local, mm) "
        f"{np.round(CAL[tag]*1000, 2)}")


def centre(tag, rb):
    p, q = rb.get_world_pose()
    return np.asarray(p, dtype=float) + quat_to_R(q) @ CAL[tag]


def sweep(mode, speed):
    c4, c9 = centre("4", rb4), centre("9", rb9)
    d0 = c4[:2] - c9[:2]
    u = d0 / np.linalg.norm(d0)
    log(f"--- {mode} @ {speed} m/s : start separation {np.linalg.norm(d0)*1000:.2f} mm, "
        f"heading ({u[0]:+.3f},{u[1]:+.3f})")
    trace, dmin, dmin_step = [], 1e9, -1
    if mode == "ballistic":
        rb9.set_linear_velocity(np.array([u[0] * speed, u[1] * speed, 0.0]))
    for s in range(args.steps):
        if mode == "press":               # hold the push speed: tests contact GEN
            # do NOT read the velocity back here -- on the GPU tensor pipeline the
            # read/write interleave silently drops the write and the cup never moves.
            rb9.set_linear_velocity(np.array([u[0] * speed, u[1] * speed, 0.0]))
        world.step(render=False)
        c4, c9 = centre("4", rb4), centre("9", rb9)
        d = float(np.linalg.norm(c4[:2] - c9[:2]))
        trace.append(d)
        if d < dmin:
            dmin, dmin_step = d, s
        if s > 30 and d > CONTACT_D * 1.6 and dmin < CONTACT_D:
            break                          # passed clean through and out the far side
    pen = max(0.0, CONTACT_D - dmin)
    # A run that never closed to the contact distance proves NOTHING about
    # penetration -- counter friction stalls the push at low speed. Reporting it
    # as a pass would manufacture a clean row out of a test that never ran.
    reached = dmin <= CONTACT_D * 1.02
    passed = reached and pen <= args.tol
    log(f"    closest approach {dmin*1000:.2f} mm at step {dmin_step} "
        f"(contact at {CONTACT_D*1000:.2f} mm)")
    verdict = "PASS" if passed else ("NO CONTACT REACHED (inconclusive)" if not reached else "FAIL")
    log(f"    PENETRATION = {pen*1000:.2f} mm   -> {verdict} (tol {args.tol*1000:.1f} mm)")
    return {"mode": mode, "speed": speed, "contact_distance_m": CONTACT_D,
            "closest_m": dmin, "penetration_m": pen, "pass": bool(passed),
            "reached_contact": bool(reached), "verdict": verdict,
            "trace_mm": [round(t * 1000, 3) for t in trace[::5]]}


if args.mode == "sweep":
    modes = [("press", float(v)) for v in args.speeds.split(",")]
elif args.mode == "bsweep":
    # BALLISTIC sweep: the cup carries momentum and contact is ALLOWED to stop it.
    # This is the physically meaningful gate -- a velocity-forced pusher is an
    # infinitely strong actuator that no contact can ever halt, so its residual
    # penetration measures solver stiffness, not phasing.
    modes = [("ballistic", float(v)) for v in args.speeds.split(",")]
elif args.mode == "both":
    modes = [("press", args.speed), ("ballistic", args.bal_speed)]
else:
    modes = [(args.mode, args.speed if args.mode == "press" else args.bal_speed)]


def settle_test(n=720):
    """Did the collider swap break the scene? A cup must sit still and keep its
    fluid. Measures drift, tilt and particle retention over 6 s of pure rest."""
    import omni.usd as _ou
    # the fluid is a PointInstancer with `positions` (NOT `points`) -- same prim
    # and same accessor production uses, so the counts are directly comparable
    fpi = UsdGeom.PointInstancer(stage.GetPrimAtPath("/World/fluid"))

    def fluid_in(c, r=0.075):
        if not fpi:
            return -1
        pos = fpi.GetPositionsAttr().Get()
        if pos is None:
            return -1
        P = np.array([[v[0], v[1], v[2]] for v in pos])
        return int((np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]) < r).sum())

    def tilt(rb):
        _, q = rb.get_world_pose()
        up = quat_to_R(q) @ np.array([0.0, 0.0, 1.0])
        return float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))

    c40, c90 = centre("4", rb4), centre("9", rb9)
    f0 = fluid_in(c40)
    for _ in range(n):
        world.step(render=False)
    c41, c91 = centre("4", rb4), centre("9", rb9)
    f1 = fluid_in(c41)
    d4, d9 = np.linalg.norm(c41 - c40) * 1000, np.linalg.norm(c91 - c90) * 1000
    t4, t9 = tilt(rb4), tilt(rb9)
    log(f"    labg4 drift {d4:.2f} mm, tilt {t4:.2f} deg")
    log(f"    labg9 drift {d9:.2f} mm, tilt {t9:.2f} deg")
    log(f"    fluid particles in labg4: {f0} -> {f1}" + ("  (no particle system found)" if f0 < 0 else ""))
    ok = d4 < 5.0 and d9 < 5.0 and t4 < 3.0 and t9 < 3.0 and (f0 < 0 or f1 >= 0.95 * f0)
    log(f"    SETTLE {'PASS' if ok else 'FAIL'}")
    return {"drift_mm": [d4, d9], "tilt_deg": [t4, t9],
            "fluid": [f0, f1], "pass": bool(ok)}

if args.test == "settle":
    report["tests"]["settle"] = settle_test()
    report["pass"] = report["tests"]["settle"]["pass"]
    json.dump(report, open(f"{args.out_dir}/{args.tag}_report.json", "w"), indent=2)
    log(f"RESULT {'PASS' if report['pass'] else 'FAIL'}")
    sim.close()
    sys.exit(0 if report["pass"] else 1)

results = []
for i, (mode, sp) in enumerate(modes):
    if i > 0:
        world.reset()
        for _ in range(60):
            world.step(render=False)
    results.append(sweep(mode, sp))

real = [r for r in results if r["reached_contact"]]
report["tests"]["cup_cup"] = {"radii_mm": [R4 * 1000, R9 * 1000],
                              "contact_distance_mm": CONTACT_D * 1000,
                              "runs": results,
                              "n_conclusive": len(real),
                              "pass": bool(real) and all(r["pass"] for r in real)}
report["pass"] = static_ok and report["tests"]["cup_cup"]["pass"]
json.dump(report, open(f"{args.out_dir}/{args.tag}_report.json", "w"), indent=2)
log(f"wrote {args.out_dir}/{args.tag}_report.json")
log(f"RESULT {'PASS' if report['pass'] else 'FAIL'}")
sim.close()
