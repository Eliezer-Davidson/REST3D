#!/usr/bin/env python3
"""Uniformly rescale a banner4-style USD scene by baking a factor into geometry.

Structure (verified): bodies carry only xformOp:translate; meshes are raw points
with no per-mesh transform; a few Cubes/Cylinders use scale/radius/height. So a
clean bake = multiply, everywhere: mesh points, extents, cylinder radius/height,
translate ops, scale ops. /World stays identity -> no double-scaling, so fluid can
be added afterwards in true world coordinates.
"""
import argparse
from pxr import Usd, UsdGeom, Gf, Vt

ap = argparse.ArgumentParser()
ap.add_argument("--in_usd", required=True)
ap.add_argument("--out_usd", required=True)
ap.add_argument("--scale", type=float, default=2.5)
args = ap.parse_args()
S = args.scale

stage = Usd.Stage.Open(args.in_usd)
stage.Export(args.out_usd)
stage = Usd.Stage.Open(args.out_usd)

n_pts = n_cyl = n_t = n_s = 0
for prim in stage.Traverse():
    # geometry: mesh points
    pa = prim.GetAttribute("points")
    if pa and pa.IsValid() and pa.Get():
        pa.Set(Vt.Vec3fArray([p * S for p in pa.Get()])); n_pts += 1
    # procedural cylinders (fake-liquid): scale local size (NOT via xform)
    if prim.IsA(UsdGeom.Cylinder):
        c = UsdGeom.Cylinder(prim)
        c.GetRadiusAttr().Set(float(c.GetRadiusAttr().Get()) * S)
        c.GetHeightAttr().Set(float(c.GetHeightAttr().Get()) * S)
        n_cyl += 1
    # transforms: translate + scale ops
    if prim.IsA(UsdGeom.Xformable):
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            t = op.GetOpType()
            if t == UsdGeom.XformOp.TypeTranslate:
                op.Set(op.Get() * S); n_t += 1
            elif t == UsdGeom.XformOp.TypeScale:
                op.Set(op.Get() * S); n_s += 1

# recompute extents from the scaled geometry (do NOT hand-scale them -> avoids
# double-scaling on Cubes whose size comes from a scale op)
n_ext = 0
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Boundable):
        b = UsdGeom.Boundable(prim)
        ext = UsdGeom.Boundable.ComputeExtentFromPlugins(b, Usd.TimeCode.Default())
        if ext:
            b.CreateExtentAttr().Set(ext); n_ext += 1

stage.GetRootLayer().Save()
print(f"[rescale] x{S}: points={n_pts} cylinders={n_cyl} translates={n_t} "
      f"scaleOps={n_s} extents_recomputed={n_ext}")
print(f"[rescale] wrote {args.out_usd}")
