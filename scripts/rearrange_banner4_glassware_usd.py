#!/usr/bin/env python3
"""Move selected banner4 glassware to side counter and tune transparency."""

from __future__ import annotations

import argparse
from collections.abc import Iterable

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt


def _bbox_center(prim, bbox_cache: UsdGeom.BBoxCache) -> Gf.Vec3d:
    box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    return (box.GetMin() + box.GetMax()) * 0.5


def _bbox_min_z(prim, bbox_cache: UsdGeom.BBoxCache) -> float:
    box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    return float(box.GetMin()[2])


def _set_parent_translate(prim, delta: Gf.Vec3d) -> None:
    xf = UsdGeom.Xformable(prim)
    op = None
    for candidate in xf.GetOrderedXformOps():
        if candidate.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op = candidate
            break
    if op is None:
        op = xf.AddTranslateOp()
        op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    cur = op.Get()
    op.Set(Gf.Vec3d(cur[0] + delta[0], cur[1] + delta[1], cur[2] + delta[2]))


def _set_mesh_opacity(mesh_prim, opacity: float) -> None:
    mesh = UsdGeom.Mesh(mesh_prim)
    mesh.CreateDisplayOpacityAttr(Vt.FloatArray([opacity]))


def _targets() -> dict[str, tuple[float, float]]:
    # Two rows on the side counter (right-side long bench region in banner4),
    # placed in a confirmed countertop zone (same support z as right-counter items).
    return {
        "labg0": (-0.62, -2.86),  # beaker
        "labg1": (-0.50, -2.84),  # vial
        "labg4": (-0.38, -2.82),  # beaker
        "labg5": (-0.56, -2.64),  # vial
        "labg7": (-0.42, -2.62),  # beaker
        "labg8": (-0.28, -2.60),  # vial
    }


def _iter_labgs(stage: Usd.Stage, names: Iterable[str]):
    for name in names:
        prim = stage.GetPrimAtPath(f"/World/{name}")
        if prim.IsValid():
            yield name, prim


def _bind_preview_material(stage: Usd.Stage, prim, mat_name: str, rgb: tuple[float, float, float], opacity: float) -> None:
    looks = UsdGeom.Scope.Define(stage, "/World/Looks")
    _ = looks  # keep scope defined
    mat_path = Sdf.Path(f"/World/Looks/{mat_name}")
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path.AppendChild("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(max(0.02, min(1.0, opacity))))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.08)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.45)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(prim).Bind(mat)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_usd", required=True)
    ap.add_argument("--out_usd", required=True)
    ap.add_argument("--glass_opacity", type=float, default=0.12, help="Display opacity for moved beaker/vial meshes")
    ap.add_argument("--liquid_opacity", type=float, default=0.42, help="Display opacity for liquid sub-mesh/cylinder")
    ap.add_argument("--support_z", type=float, default=0.373, help="Expected side-table support height (world Z)")
    ap.add_argument("--z_clearance", type=float, default=0.0015, help="Small lift above support to prevent z-fighting")
    args = ap.parse_args()

    stage = Usd.Stage.Open(args.in_usd)
    if stage is None:
        raise RuntimeError(f"failed to open {args.in_usd}")
    stage.Export(args.out_usd)
    stage = Usd.Stage.Open(args.out_usd)

    targets = _targets()
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    moved = 0
    for name, prim in _iter_labgs(stage, targets.keys()):
        ctr = _bbox_center(prim, bbox_cache)
        min_z = _bbox_min_z(prim, bbox_cache)
        tx, ty = targets[name]
        target_min_z = args.support_z + args.z_clearance
        dz = target_min_z - min_z
        delta = Gf.Vec3d(tx - ctr[0], ty - ctr[1], dz)
        _set_parent_translate(prim, delta)
        visual = stage.GetPrimAtPath(f"/World/{name}/visual")
        if visual.IsValid():
            # Also set explicit translucent material because some Isaac render paths
            # ignore/under-emphasize displayOpacity-only edits.
            _set_mesh_opacity(visual, max(0.05, min(1.0, args.glass_opacity)))
            UsdGeom.Gprim(visual).CreateDoubleSidedAttr(True)
            _bind_preview_material(stage, visual, f"{name}_glass", (0.78, 0.89, 0.96), args.glass_opacity)
        liquid = stage.GetPrimAtPath(f"/World/{name}/liquid")
        if liquid.IsValid():
            _set_mesh_opacity(liquid, max(0.05, min(1.0, args.liquid_opacity)))
            _bind_preview_material(stage, liquid, f"{name}_liquid", (0.20, 0.50, 0.88), args.liquid_opacity)
        moved += 1
        print(
            f"[move] {name}: from=({ctr[0]:.3f},{ctr[1]:.3f},{ctr[2]:.3f}) "
            f"to=({tx:.3f},{ty:.3f},{ctr[2] + dz:.3f}) dz={dz:.4f}"
        )

    if moved == 0:
        raise RuntimeError("no target glassware found to move")

    stage.GetRootLayer().Save()
    print(f"[move] wrote {args.out_usd} with {moved} moved glass objects")


if __name__ == "__main__":
    main()
