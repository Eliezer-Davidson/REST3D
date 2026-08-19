#!/usr/bin/env python3
"""Set the cups' centre of mass to the glass shell's true centroid.

banner4 authored CoM at base+20 mm on a 125 mm glass, deliberately: the comment
in integrate_fluid_banner4.py reads "LOW center of mass -> bottom-heavy,
self-righting, won't topple". That made the cup sit still on the counter -- and
it makes the cup a WEEBLE, which actively fights being tilted. During a pour the
grip must overcome m*g*d*sin(theta) for the whole rotation, where d is the
distance from the grip down to that artificially low CoM. Measured symptom: the
cup lags the hand, torque builds, then it breaks loose and lurches.

The rigid body here represents the GLASS ONLY -- the water is separate PBD
particles carrying their own mass -- so the physical CoM is the shell's
centroid, computed from the mesh by area-weighting its triangles.
"""
import argparse, os
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--cups", default="/World/labg4,/World/labg9")
ap.add_argument("--com_frac", type=float, default=-1.0,
                help="override: CoM height as a fraction of glass height. "
                     "<0 = compute the shell centroid from the mesh (default)")
args = ap.parse_args()
stage = Usd.Stage.Open(os.path.abspath(args.usd))

def shell_centroid_z(prim):
    m = UsdGeom.Mesh(prim)
    P = np.array([[v[0], v[1], v[2]] for v in m.GetPointsAttr().Get()])
    cnt = np.array(m.GetFaceVertexCountsAttr().Get())
    idx = np.array(m.GetFaceVertexIndicesAttr().Get())
    tris, o = [], 0
    for c in cnt:
        f = idx[o:o+c]; o += c
        for k in range(1, c-1):
            tris.append([f[0], f[k], f[k+1]])
    T = np.array(tris)
    a, b, c = P[T[:,0]], P[T[:,1]], P[T[:,2]]
    area = 0.5*np.linalg.norm(np.cross(b-a, c-a), axis=1)      # area-weight the shell
    cz = (a[:,2]+b[:,2]+c[:,2])/3.0
    return float((area*cz).sum()/area.sum()), float(P[:,2].min()), float(P[:,2].max())

for cp in [c for c in args.cups.split(",") if c]:
    prim = stage.GetPrimAtPath(cp)
    vis = stage.GetPrimAtPath(cp + "/visual")
    if not prim.IsValid() or not vis.IsValid():
        print(f"[com] {cp}: MISSING"); continue
    zc, zb, zr = shell_centroid_z(vis)
    if args.com_frac >= 0:
        zc = zb + args.com_frac*(zr-zb)
    mass = UsdPhysics.MassAPI.Apply(prim)
    old = mass.GetCenterOfMassAttr().Get()
    new = Gf.Vec3f(float(old[0]), float(old[1]), float(zc))
    mass.CreateCenterOfMassAttr().Set(new)
    print(f"[com] {cp}: CoM z {old[2]:.4f} -> {zc:.4f}   "
          f"({(old[2]-zb)*1000:.1f} mm -> {(zc-zb)*1000:.1f} mm above the base, "
          f"glass {(zr-zb)*1000:.0f} mm tall)")

stage.GetRootLayer().Export(os.path.abspath(args.out))
print(f"[com] wrote {args.out}")
