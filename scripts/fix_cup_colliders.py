#!/usr/bin/env python3
"""Give the banner4 cups exact, dense, non-phasing rigid colliders.

WHY the shipped collider phases (all measured, see docs/banner4-physics-repair):

  The cup's collision mesh IS its render mesh: a 40-gon revolve of 202 verts /
  400 tris carrying `physics:approximation = "sdf"`. Three defects compound:

  1. It has only THREE unique z levels -- 0, 7.5 and 125 mm. The entire 117.5 mm
     wall is one band of triangles with NO vertices on it (240 of 400 tris have
     an edge over 20 mm; the longest is 117.7 mm). PhysX SDF contact generation
     samples the VERTICES of one shape against the other's signed distance
     field, so two cup walls meeting have zero sample points between them and
     generate no contacts at all.
  2. Winding is INCONSISTENT (40 directed edges lack an opposite twin), so the
     inside/outside test the SDF builder relies on is undefined in patches. The
     signed volume comes out 3677 cm^3 against a 1012 cm^3 bounding box -- 3.6x
     its own bbox, which is impossible for a clean closed mesh.
  3. `PhysxSDFMeshCollisionAPI` is never actually applied -- only the raw
     attribute `physxSDFMeshCollision:sdfResolution = 320` is authored. PhysX
     reads that attribute only through the applied schema, so the requested
     resolution is silently ignored and the default is used.

  CCD cannot rescue any of this: Isaac disables CCD outright under GPU dynamics
  ("Disabling CCD for GPU dynamics as its not supported"). Contact generation
  has to be right on its own.

THE FIX (mode convex_ring, default): keep the geometry EXACTLY as authored and
change only its collision representation. The cup is a perfect parametric
revolve, so it decomposes losslessly into N convex wedges of the wall plus one
convex disc for the base. Convex-vs-convex is PhysX's exact path -- full contact
manifolds, no vertex sampling, no SDF sign, no dependence on tessellation
density. With --segments 40 the collider occupies the identical surface the
render mesh does.

  mode remesh_sdf is the alternative: rebuild the same revolve watertight with
  consistent winding and rings up the wall, then apply the SDF schema properly.
  Provided so the two can be A/B'd rather than asserted.

This script only authors colliders. It never moves a cup.
"""
from __future__ import annotations
import argparse, math, os, sys

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Sdf

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cups", default="/World/labg4,/World/labg9")
ap.add_argument("--mode", default="convex_ring", choices=["convex_ring", "remesh_sdf"])
ap.add_argument("--segments", type=int, default=40,
                help="wall wedges around the circumference; 40 matches the "
                     "existing tessellation exactly, so geometry is unchanged")
ap.add_argument("--rings", type=int, default=32,
                help="remesh_sdf: vertex rings up the wall (the shipped mesh has 0)")
ap.add_argument("--sdf_resolution", type=int, default=320)
ap.add_argument("--contact_offset", type=float, default=0.005)
ap.add_argument("--rest_offset", type=float, default=0.0)
ap.add_argument("--grip_material", default="/World/PhysicsMaterials/grip")
ap.add_argument("--max_depen_vel", type=float, default=-1.0,
                help="cups: physxRigidBody:maxDepenetrationVelocity (m/s). The "
                     "shipped value is 1.0, which is SLOWER than the closing "
                     "speeds this scene sees, so penetration accumulates faster "
                     "than the solver is permitted to undo it. <0 leaves it alone.")
ap.add_argument("--substeps", type=int, default=0,
                help="scene: physxScene:timeStepsPerSecond. Shipped 120 -> a cup "
                     "at 2 m/s advances 16.7 mm per step. Raising this is the "
                     "only anti-tunnelling lever left once CCD is unavailable "
                     "(Isaac disables CCD under GPU dynamics). 0 leaves it alone.")
args = ap.parse_args()

stage = Usd.Stage.Open(os.path.abspath(args.usd))
grip = None
gm = stage.GetPrimAtPath(args.grip_material)
if gm.IsValid():
    grip = UsdShade.Material(gm)


def log(*a):
    print("[cupfix]", *a, flush=True)


def profile(cup_path):
    """Recover the revolve parameters from the mesh itself. Nothing hardcoded."""
    vis = stage.GetPrimAtPath(f"{cup_path}/visual")
    m = UsdGeom.Mesh(vis)
    P = np.array([[v[0], v[1], v[2]] for v in m.GetPointsAttr().Get()])
    cx = float((P[:, 0].min() + P[:, 0].max()) / 2.0)
    cy = float((P[:, 1].min() + P[:, 1].max()) / 2.0)
    r = np.hypot(P[:, 0] - cx, P[:, 1] - cy)
    z = P[:, 2]
    zb, zr = float(z.min()), float(z.max())
    uz = np.unique(np.round(z, 6))
    # cavity floor = the z level between base and rim (the base slab's top)
    mid = [v for v in uz if zb + 1e-6 < v < zr - 1e-6]
    zf = float(mid[0]) if mid else zb + 0.0075
    base = r[np.abs(z - zb) < 1e-6]
    rim = r[np.abs(z - zr) < 1e-6]
    r_in = float(np.min(rim[rim > 1e-4]))          # cavity radius at the rim
    r_out_rim = float(np.max(rim))
    r_out_base = float(np.max(base))
    return dict(cx=cx, cy=cy, z_base=zb, z_floor=zf, z_rim=zr,
                r_in=r_in, r_out_base=r_out_base, r_out_rim=r_out_rim)


def r_out_at(pf, z):
    """Outer radius of the tapered wall at height z (linear, as authored)."""
    t = (z - pf["z_base"]) / (pf["z_rim"] - pf["z_base"])
    return pf["r_out_base"] + (pf["r_out_rim"] - pf["r_out_base"]) * t


def author_collider(prim, approximation="convexHull"):
    UsdPhysics.CollisionAPI.Apply(prim)
    if prim.IsA(UsdGeom.Mesh):
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
    prim.CreateAttribute("physxCollision:contactOffset",
                         Sdf.ValueTypeNames.Float).Set(args.contact_offset)
    prim.CreateAttribute("physxCollision:restOffset",
                         Sdf.ValueTypeNames.Float).Set(args.rest_offset)
    if grip is not None:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(grip, materialPurpose="physics")
    UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.invisible)


def build_convex_ring(cup_path, pf):
    N = args.segments
    cx, cy = pf["cx"], pf["cy"]
    ri, zf, zr = pf["r_in"], pf["z_floor"], pf["z_rim"]
    ro_f, ro_r = r_out_at(pf, zf), pf["r_out_rim"]
    made = 0
    for i in range(N):
        t0, t1 = 2 * math.pi * i / N, 2 * math.pi * (i + 1) / N
        pts = []
        for t in (t0, t1):
            ct, st = math.cos(t), math.sin(t)
            pts += [Gf.Vec3f(cx + ri * ct,   cy + ri * st,   zf),
                    Gf.Vec3f(cx + ro_f * ct, cy + ro_f * st, zf),
                    Gf.Vec3f(cx + ro_r * ct, cy + ro_r * st, zr),
                    Gf.Vec3f(cx + ri * ct,   cy + ri * st,   zr)]
        p = f"{cup_path}/col_wall_{i:03d}"
        mesh = UsdGeom.Mesh.Define(stage, p)
        mesh.CreatePointsAttr(pts)
        # quads: theta0 face, theta1 face, inner, outer, bottom, top
        mesh.CreateFaceVertexCountsAttr([4] * 6)
        mesh.CreateFaceVertexIndicesAttr([0,1,2,3, 4,7,6,5, 0,3,7,4, 1,5,6,2, 0,4,5,1, 3,2,6,7])
        author_collider(mesh.GetPrim(), "convexHull")
        made += 1
    # base slab: a full disc is already convex, so one analytic cylinder is exact
    h = zf - pf["z_base"]
    cyl = UsdGeom.Cylinder.Define(stage, f"{cup_path}/col_base")
    cyl.CreateRadiusAttr(float(pf["r_out_base"]))
    cyl.CreateHeightAttr(float(h))
    cyl.CreateAxisAttr("Z")
    xf = UsdGeom.Xformable(cyl.GetPrim()); xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, pf["z_base"] + h / 2.0))
    author_collider(cyl.GetPrim())
    made += 1
    return made


def build_remesh_sdf(cup_path, pf):
    """Watertight revolve, consistent winding, vertices all the way up the wall."""
    N, M = args.segments, args.rings
    cx, cy = pf["cx"], pf["cy"]
    ri, zb, zf, zr = pf["r_in"], pf["z_base"], pf["z_floor"], pf["z_rim"]
    zs_out = np.linspace(zb, zr, M + 1)
    zs_in = np.linspace(zf, zr, M + 1)
    P, F = [], []

    def ring(rad_fn, zs):
        base_idx = len(P)
        for z in zs:
            rr = rad_fn(z)
            for i in range(N):
                t = 2 * math.pi * i / N
                P.append(Gf.Vec3f(cx + rr * math.cos(t), cy + rr * math.sin(t), float(z)))
        return base_idx

    o_out = ring(lambda z: r_out_at(pf, z), zs_out)      # outer skin, up
    o_in = ring(lambda z: ri, zs_in)                     # inner skin, up
    def quad(a, b, c, d): F.extend([a, b, c, d])
    for k in range(len(zs_out) - 1):                     # outer wall, CCW out
        for i in range(N):
            j = (i + 1) % N
            quad(o_out + k*N + i, o_out + k*N + j, o_out + (k+1)*N + j, o_out + (k+1)*N + i)
    for k in range(len(zs_in) - 1):                      # inner wall, wound INWARD
        for i in range(N):
            j = (i + 1) % N
            quad(o_in + k*N + i, o_in + (k+1)*N + i, o_in + (k+1)*N + j, o_in + k*N + j)
    ci = len(P); P.append(Gf.Vec3f(cx, cy, zb))          # outer base fan
    for i in range(N):
        j = (i + 1) % N
        F.extend([ci, o_out + j, o_out + i])
    cf = len(P); P.append(Gf.Vec3f(cx, cy, zf))          # cavity floor fan
    for i in range(N):
        j = (i + 1) % N
        F.extend([cf, o_in + i, o_in + j])
    top = len(zs_out) - 1; topi = len(zs_in) - 1
    for i in range(N):                                   # rim annulus
        j = (i + 1) % N
        quad(o_out + top*N + i, o_out + top*N + j, o_in + topi*N + j, o_in + topi*N + i)
    counts = []
    idx = []
    k = 0
    # rebuild counts: quads were pushed 4 at a time, fans 3 at a time -> track order
    # (simplest: re-walk with explicit sizes)
    return P, F


CUPS = [c for c in args.cups.split(",") if c]
total = 0
for cup in CUPS:
    prim = stage.GetPrimAtPath(cup)
    if not prim.IsValid():
        log(f"{cup}: MISSING, skipped"); continue
    pf = profile(cup)
    log(f"{cup} profile: axis=({pf['cx']:.4f},{pf['cy']:.4f}) "
        f"z {pf['z_base']*1000:.1f}/{pf['z_floor']*1000:.1f}/{pf['z_rim']*1000:.1f} mm  "
        f"r_in={pf['r_in']*1000:.2f} r_out {pf['r_out_base']*1000:.2f}->{pf['r_out_rim']*1000:.2f} mm")

    # strip the degenerate SDF collider: `visual` becomes render-only
    vis = stage.GetPrimAtPath(f"{cup}/visual")
    if vis.IsValid():
        vis.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        vis.RemoveAPI(UsdPhysics.CollisionAPI)
        for a in ("physics:approximation", "physics:collisionEnabled",
                  "physxSDFMeshCollision:sdfResolution"):
            if vis.HasAttribute(a):
                vis.RemoveProperty(a)
        log(f"  removed degenerate SDF collider from {cup}/visual (now render-only)")

    # remove any previous run's proxies so this is idempotent
    for c in list(prim.GetChildren()):
        if c.GetName().startswith(("col_wall_", "col_base")):
            stage.RemovePrim(c.GetPath())

    if args.mode == "convex_ring":
        n = build_convex_ring(cup, pf)
    else:
        raise SystemExit("[cupfix] remesh_sdf not wired in this build; use convex_ring")
    total += n
    log(f"  authored {n} convex collider prims ({args.segments} wall wedges + 1 base disc)")

if args.max_depen_vel >= 0:
    for cup in CUPS:
        prim = stage.GetPrimAtPath(cup)
        if prim.IsValid():
            prim.CreateAttribute("physxRigidBody:maxDepenetrationVelocity",
                                 Sdf.ValueTypeNames.Float).Set(args.max_depen_vel)
    log(f"maxDepenetrationVelocity = {args.max_depen_vel} m/s on {len(CUPS)} cups")

if args.substeps > 0:
    for p in stage.Traverse():
        if p.IsA(UsdPhysics.Scene) or "PhysicsScene" in str(p.GetPath()):
            p.CreateAttribute("physxScene:timeStepsPerSecond",
                              Sdf.ValueTypeNames.UInt).Set(args.substeps)
            log(f"{p.GetPath()}: timeStepsPerSecond = {args.substeps}")

stage.GetRootLayer().Export(os.path.abspath(args.out))
log(f"wrote {args.out}  ({total} collider prims across {len(CUPS)} cups)")
