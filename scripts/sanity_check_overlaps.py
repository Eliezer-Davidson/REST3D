#!/usr/bin/env python3
"""Scene sanity gate: NO body may interpenetrate another at spawn. Ever.

Checks every DYNAMIC rigid body (and robot articulation links) against all other
collider-bearing prims, visible or not — invisible colliders are exactly the ones
that bite. Static-vs-static scan furniture is skipped (reconstructed meshes
interpenetrate each other by nature and never move). Authored FilteredPairs are
respected: a filtered pair is non-colliding by design.

Method: AABB broad pass with --tol slack, then a vertex-in-box refinement for
mesh colliders (a room-sized floor mesh AABB must not flag a foot standing on it).

Collision filters (FilteredPairsAPI) are deliberately IGNORED: a filter that
legalizes an interpenetration is exactly the failure this gate exists to catch.
Filters are for contact quality, never for tolerating overlap.

Pure pxr (rest3d env), no GPU. Exit 1 on any violation.
"""
from __future__ import annotations
import argparse, math, sys
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--tol", type=float, default=0.003, help="allowed penetration (m): resting-contact slack")
ap.add_argument("--robot", default="", help="optionally reference a robot USD before checking")
ap.add_argument("--robot_path", default="/World/G1")
ap.add_argument("--robot_pos", default="")
ap.add_argument("--robot_yaw_deg", type=float, default=0.0)
args = ap.parse_args()

stage = Usd.Stage.Open(args.usd)
if args.robot:
    prim = stage.DefinePrim(args.robot_path, "Xform")
    prim.GetReferences().AddReference(args.robot)
    xf = UsdGeom.Xformable(prim); xf.ClearXformOpOrder()
    if args.robot_pos:
        x, y, z = [float(v) for v in args.robot_pos.split(",")]
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
    if args.robot_yaw_deg:
        h = math.radians(args.robot_yaw_deg) / 2.0
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(math.cos(h), 0, 0, math.sin(h)))

xc = UsdGeom.XformCache(Usd.TimeCode.Default())


def colliders(root):
    """[(owner_body_path|None, collider_path, verts|None, mn, mx)] — ignores visibility."""
    out = []
    def prim_verts(p):
        M = np.array(xc.GetLocalToWorldTransform(p), dtype=np.float64)
        if p.IsA(UsdGeom.Mesh):
            pts = p.GetAttribute("points").Get()
            if pts:
                return (np.array([[q[0], q[1], q[2], 1.0] for q in pts]) @ M)[:, :3]
        elif p.IsA(UsdGeom.Cube):
            s = (UsdGeom.Cube(p).GetSizeAttr().Get() or 2.0) / 2.0
            return (np.array([[a*s, b*s, c*s, 1.0] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) @ M)[:, :3]
        elif p.IsA(UsdGeom.Cylinder):
            cyl = UsdGeom.Cylinder(p)
            r = cyl.GetRadiusAttr().Get() or 1.0
            hh = (cyl.GetHeightAttr().Get() or 2.0) / 2.0
            return (np.array([[a*r, b*r, c*hh, 1.0] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) @ M)[:, :3]
        elif p.IsA(UsdGeom.Sphere):
            r = UsdGeom.Sphere(p).GetRadiusAttr().Get() or 1.0
            return (np.array([[a*r, b*r, c*r, 1.0] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) @ M)[:, :3]
        return None

    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            continue
        # geometry may be ON the prim or (robot convention) on child prims under
        # an Xform "collisions" scope — gather all of it
        parts = [v for v in (prim_verts(q) for q in Usd.PrimRange(p, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))) if v is not None]
        if not parts:
            continue
        V = np.vstack(parts)
        owner = None
        q = p
        while q and str(q.GetPath()) not in ("/", ""):
            if q.HasAPI(UsdPhysics.RigidBodyAPI):
                en = q.GetAttribute("physics:rigidBodyEnabled")
                if not en or en.Get() is not False:
                    owner = str(q.GetPath())
                break
            q = q.GetParent()
        out.append((owner, str(p.GetPath()), V, V.min(0), V.max(0)))
    return out


def top_of(path):
    seg = path.split("/")
    return "/World/" + seg[2] if len(seg) > 2 else path


def filtered_targets(path):
    t = set()
    p = stage.GetPrimAtPath(path)
    while p and str(p.GetPath()) not in ("/", ""):
        if p.HasAPI(UsdPhysics.FilteredPairsAPI):
            for tgt in UsdPhysics.FilteredPairsAPI(p).GetFilteredPairsRel().GetTargets():
                t.add(str(tgt))
        p = p.GetParent()
    return t


all_c = colliders(stage.GetPrimAtPath("/World"))
dyn = [c for c in all_c if c[0] is not None]
print(f"[sanity] {len(all_c)} colliders, {len(dyn)} on enabled rigid bodies")

# Refinement = what PhysX actually collides: point-inside-CONVEX-HULL depth.
# (Nearly all colliders here are convexHull-approximated or primitives; a hull test
# also catches vertex-sparse decomposition pieces whose faces cross a body with no
# vertex nearby. Sparse-vertex FACES through a hull are covered by the symmetric test.)
from scipy.spatial import ConvexHull

_hulls = {}


def hull_eqs(cp, V):
    if cp not in _hulls:
        try:
            _hulls[cp] = ConvexHull(V).equations
        except Exception:
            _hulls[cp] = None
    return _hulls[cp]


def max_depth(P, eqs):
    """deepest penetration of points P into the hull (A.x + b <= 0 inside)."""
    if eqs is None or P is None or len(P) == 0:
        return 0.0
    S = P @ eqs[:, :3].T + eqs[:, 3]
    return float((-S.max(axis=1)).max())


violations = []
for o1, cp1, V1, mn1, mx1 in dyn:
    for o2, cp2, V2, mn2, mx2 in all_c:
        if o2 is not None and cp2 <= cp1:
            continue                                  # dedupe dynamic-dynamic
        if o1 == o2 or (o2 and top_of(o1) == top_of(o2)):
            continue                                  # same body / same articulation
        pen = np.minimum(mx1, mx2) - np.maximum(mn1, mn2)
        if not (pen > args.tol).all():
            continue
        depth = max(max_depth(V1, hull_eqs(cp2, V2)), max_depth(V2, hull_eqs(cp1, V1)))
        if depth <= args.tol:
            continue
        violations.append((cp1, cp2, depth))

if violations:
    print(f"[sanity] FAIL — {len(violations)} interpenetrating pairs (deeper than {args.tol*1000:.0f} mm):")
    for a, b, d in sorted(violations, key=lambda v: -v[2])[:40]:
        print(f"  depth~{d*1000:6.1f} mm  {a}  <->  {b}")
    sys.exit(1)
print("[sanity] PASS — no dynamic body interpenetrates anything at spawn")
