# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Build an Isaac Sim USD from the assets exported by export_sim_scene.py.

Needs the `pxr` (usd-core) package — on this machine that lives in the genrecon env,
not rest3d:
  /home/user/miniconda3/envs/genrecon/bin/python scripts/export_usd_scene.py \
      --assets output/office_lounge/sim/assets_zup \
      --out    output/office_lounge/sim/usd/office_lounge.usdc

Layout per object (Z-up, metersPerUnit=1):
  /World/<name>            Xform  + RigidBodyAPI + MassAPI(density)
  /World/<name>/visual     Mesh with per-vertex displayColor, no collision
  /World/<name>/col_XX     invisible Mesh, CollisionAPI, convexHull approximation
plus a static /World/floor slab so the scene is self-supporting when the sim's own
ground plane is disabled (as unitree_sim_isaaclab does).
"""
import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, Vt


def load_obj(path):
    V, C, F = [], [], []
    for line in open(path):
        if line.startswith("v "):
            p = line.split()
            V.append((float(p[1]), float(p[2]), float(p[3])))
            C.append((float(p[4]), float(p[5]), float(p[6])) if len(p) >= 7 else (0.7, 0.7, 0.7))
        elif line.startswith("f "):
            idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            for k in range(1, len(idx) - 1):
                F.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(V, np.float32), np.asarray(C, np.float32), np.asarray(F, np.int32)


def add_mesh(stage, path, V, F, color=None, vertex_colors=None):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray((3,) * len(F)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr("none")
    if vertex_colors is not None:
        pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec3fArray.FromNumpy(np.clip(vertex_colors, 0, 1)))
    elif color is not None:
        mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    return mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--density", type=float, default=200.0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    stage = Usd.Stage.CreateNew(args.out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    # group collision parts by object
    parts = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(args.assets, "*_col_*.obj"))):
        name = re.sub(r"_col_\d+$", "", os.path.basename(p)[:-4])
        parts[name].append(p)

    for vis_path in sorted(glob.glob(os.path.join(args.assets, "*_visual.obj"))):
        name = os.path.basename(vis_path)[:-len("_visual.obj")]
        safe = re.sub(r"[^A-Za-z0-9_]", "_", name)
        xf = UsdGeom.Xform.Define(stage, f"/World/{safe}")
        prim = xf.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateDensityAttr(args.density)

        V, C, F = load_obj(vis_path)
        add_mesh(stage, f"/World/{safe}/visual", V, F, vertex_colors=C)

        for k, cp in enumerate(parts.get(name, [])):
            Vc, _, Fc = load_obj(cp)
            m = add_mesh(stage, f"/World/{safe}/col_{k:02d}", Vc, Fc, color=(0.8, 0.5, 0.2))
            UsdPhysics.CollisionAPI.Apply(m.GetPrim())
            mc = UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim())
            mc.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
            UsdGeom.Imageable(m.GetPrim()).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        print(f"[usd] {name}: visual {len(V)}v + {len(parts.get(name, []))} colliders")

    # static floor slab: collider without RigidBodyAPI
    floor = UsdGeom.Cube.Define(stage, "/World/floor")
    floor.CreateSizeAttr(1.0)
    fx = UsdGeom.Xformable(floor.GetPrim())
    fx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
    fx.AddScaleOp().Set(Gf.Vec3f(20.0, 20.0, 0.1))
    floor.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.55, 0.55, 0.53)]))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())

    stage.GetRootLayer().Save()
    print(f"[usd] wrote {args.out}")


if __name__ == "__main__":
    main()
