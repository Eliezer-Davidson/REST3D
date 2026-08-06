#!/usr/bin/env python3
"""Force lab glassware to sit on table with rigid-body + collision in USD."""

from __future__ import annotations

import argparse
from collections.abc import Iterable

from pxr import Gf, Usd, UsdGeom, UsdPhysics


LABG_NAMES = ("labg0", "labg1", "labg4", "labg5", "labg7", "labg8")


def _iter_prims(stage: Usd.Stage, names: Iterable[str]):
    for n in names:
        p = stage.GetPrimAtPath(f"/World/{n}")
        if p.IsValid() and p.IsActive():
            yield n, p


def _bbox_min(prim, cache: UsdGeom.BBoxCache) -> Gf.Vec3d:
    return cache.ComputeWorldBound(prim).ComputeAlignedBox().GetMin()


def _ensure_translate_op(prim):
    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    op = xform.AddTranslateOp()
    op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    return op


def _set_world_min_z(prim, target_min_z: float, cache: UsdGeom.BBoxCache) -> float:
    mn = _bbox_min(prim, cache)
    dz = target_min_z - float(mn[2])
    op = _ensure_translate_op(prim)
    cur = op.Get()
    op.Set(Gf.Vec3d(float(cur[0]), float(cur[1]), float(cur[2]) + dz))
    return dz


def _ensure_dynamic_rigid(prim) -> None:
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateRigidBodyEnabledAttr(True)
    rb.CreateKinematicEnabledAttr(False)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateDensityAttr(650.0)


def _ensure_mesh_collision(prim) -> None:
    UsdPhysics.CollisionAPI.Apply(prim)
    mc = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mc.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)


def _ensure_physics_scene(stage: Usd.Stage) -> None:
    scene = stage.GetPrimAtPath("/World/PhysicsScene")
    if not scene.IsValid():
        ps = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        ps.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        ps.CreateGravityMagnitudeAttr(9.81)


def _add_table_support(stage: Usd.Stage, top_z: float) -> None:
    # Invisible static collision slab under the moved glassware cluster.
    support = UsdGeom.Cube.Define(stage, "/World/labg_table_support")
    support.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(support.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(-0.45, -2.72, top_z - 0.005))
    xf.AddScaleOp().Set(Gf.Vec3f(0.24, 0.16, 0.005))
    UsdPhysics.CollisionAPI.Apply(support.GetPrim())
    UsdGeom.Imageable(support.GetPrim()).CreateVisibilityAttr(UsdGeom.Tokens.invisible)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_usd", required=True)
    ap.add_argument("--out_usd", required=True)
    ap.add_argument("--target_min_z", type=float, default=0.3708, help="Desired world min-z for all moved labg objects")
    ap.add_argument("--support_top_z", type=float, default=0.3706, help="Top surface height of helper collision slab")
    args = ap.parse_args()

    stage = Usd.Stage.Open(args.in_usd)
    if stage is None:
        raise RuntimeError(f"failed to open {args.in_usd}")
    stage.Export(args.out_usd)
    stage = Usd.Stage.Open(args.out_usd)

    _ensure_physics_scene(stage)
    _add_table_support(stage, args.support_top_z)

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    moved = 0
    for name, prim in _iter_prims(stage, LABG_NAMES):
        _ensure_dynamic_rigid(prim)
        col = stage.GetPrimAtPath(f"/World/{name}/col_00")
        if col.IsValid():
            _ensure_mesh_collision(col)
        dz = _set_world_min_z(prim, args.target_min_z, cache)
        moved += 1
        print(f"[fix] {name}: applied dz={dz:.5f} rigid+collision enabled")

    if moved == 0:
        raise RuntimeError("no active labg objects found")

    stage.GetRootLayer().Save()
    print(f"[fix] wrote {args.out_usd} ({moved} objects grounded)")


if __name__ == "__main__":
    main()
