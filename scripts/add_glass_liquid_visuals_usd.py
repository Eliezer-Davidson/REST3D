#!/usr/bin/env python3
"""Add lightweight liquid visuals inside lab glassware in a USD stage.

This is intentionally low-risk: it adds simple visual cylinders only
(no particle/fluid simulation), which keeps GPU load minimal.
"""

from __future__ import annotations

import argparse
import re

from pxr import Gf, Sdf, Usd, UsdGeom, Vt


def _bbox_from_mesh_points(mesh_prim) -> tuple[Gf.Vec3d, Gf.Vec3d]:
    mesh = UsdGeom.Mesh(mesh_prim)
    pts = mesh.GetPointsAttr().Get()
    if not pts:
        raise ValueError(f"mesh has no points: {mesh_prim.GetPath()}")
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    mn = Gf.Vec3d(min(xs), min(ys), min(zs))
    mx = Gf.Vec3d(max(xs), max(ys), max(zs))
    return mn, mx


def _liquid_color_for(name: str) -> tuple[float, float, float]:
    palette = [
        (0.15, 0.42, 0.85),
        (0.22, 0.70, 0.45),
        (0.82, 0.20, 0.18),
        (0.90, 0.65, 0.15),
        (0.55, 0.30, 0.70),
    ]
    m = re.search(r"(\d+)$", name)
    idx = int(m.group(1)) if m else 0
    return palette[idx % len(palette)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_usd", required=True, help="Input USD/USDC path")
    ap.add_argument("--out_usd", required=True, help="Output USD/USDC path")
    ap.add_argument("--fill_ratio", type=float, default=0.45, help="Fraction of vessel height to fill [0,1]")
    args = ap.parse_args()

    fill_ratio = max(0.05, min(0.95, args.fill_ratio))
    stage = Usd.Stage.Open(args.in_usd)
    if stage is None:
        raise RuntimeError(f"failed to open USD: {args.in_usd}")

    stage.Export(args.out_usd)
    stage = Usd.Stage.Open(args.out_usd)
    world = stage.GetPrimAtPath("/World")
    if not world.IsValid():
        raise RuntimeError("missing /World prim")

    added = 0
    for child in world.GetChildren():
        name = child.GetName()
        if not name.startswith("labg"):
            continue
        visual = stage.GetPrimAtPath(child.GetPath().AppendChild("visual"))
        if not visual.IsValid():
            print(f"[liquid] skip {name}: no /visual mesh")
            continue

        mn, mx = _bbox_from_mesh_points(visual)
        ex = max(1e-6, mx[0] - mn[0])
        ey = max(1e-6, mx[1] - mn[1])
        ez = max(1e-6, mx[2] - mn[2])

        radius = 0.35 * min(ex, ey)
        height = fill_ratio * ez
        zc = mn[2] + 0.5 * height
        xc = 0.5 * (mn[0] + mx[0])
        yc = 0.5 * (mn[1] + mx[1])

        liq_path = child.GetPath().AppendChild("liquid")
        cyl = UsdGeom.Cylinder.Define(stage, liq_path)
        cyl.CreateRadiusAttr(float(radius))
        cyl.CreateHeightAttr(float(height))
        cyl.CreateAxisAttr("z")
        xform = UsdGeom.Xformable(cyl.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(xc, yc, zc))

        rgb = _liquid_color_for(name)
        cyl.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*rgb)]))
        cyl.CreateDisplayOpacityAttr(Vt.FloatArray([0.55]))

        added += 1
        print(
            f"[liquid] {name}: center=({xc:.3f},{yc:.3f},{zc:.3f}) "
            f"r={radius:.4f} h={height:.4f}"
        )

    if added == 0:
        raise RuntimeError("no labg* prims found; nothing added")

    stage.GetRootLayer().Save()
    print(f"[liquid] wrote {args.out_usd} with {added} liquid visuals")


if __name__ == "__main__":
    main()
