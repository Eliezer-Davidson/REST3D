#!/usr/bin/env python3
"""Project MuJoCo-authored lab glassware into assets_zup for USD export.

This reads lab glass bodies (labg*) already authored in an MJCF file and writes
matching *_visual.obj and *_col_00.obj files into an assets_zup directory so
scripts/export_usd_scene.py will include them in the Isaac Sim USD.
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET


def _parse_floats(text: str | None, n: int, default: float) -> list[float]:
    if not text:
        return [default] * n
    vals = [float(v) for v in text.split()]
    if len(vals) != n:
        raise ValueError(f"expected {n} floats, got {len(vals)} from {text!r}")
    return vals


def _load_obj(path: str) -> tuple[list[list[float]], list[tuple[int, int, int]]]:
    verts: list[list[float]] = []
    tris: list[tuple[int, int, int]] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    tris.append((idx[0], idx[k], idx[k + 1]))
    if not verts or not tris:
        raise ValueError(f"invalid/empty obj: {path}")
    return verts, tris


def _write_obj(path: str, verts: list[list[float]], tris: list[tuple[int, int, int]], rgb: list[float] | None) -> None:
    with open(path, "w") as fh:
        if rgb is None:
            for v in verts:
                fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        else:
            for v in verts:
                fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {rgb[0]:.4f} {rgb[1]:.4f} {rgb[2]:.4f}\n")
        for a, b, c in tris:
            fh.write(f"f {a+1} {b+1} {c+1}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", required=True, help="MuJoCo XML that already contains labg* meshes/bodies")
    ap.add_argument("--mujoco_assets", required=True, help="MuJoCo assets dir containing lab_*.obj")
    ap.add_argument("--assets_zup", required=True, help="assets_zup dir where *_visual/_col_*.obj are written")
    args = ap.parse_args()

    os.makedirs(args.assets_zup, exist_ok=True)
    root = ET.parse(args.mjcf).getroot()

    asset = root.find("asset")
    world = root.find("worldbody")
    if asset is None or world is None:
        raise ValueError("MJCF missing <asset> or <worldbody>")

    mesh_defs: dict[str, tuple[str, list[float]]] = {}
    for m in asset.findall("mesh"):
        name = m.get("name", "")
        if not name.startswith("labg"):
            continue
        mesh_file = m.get("file")
        if not mesh_file:
            continue
        scale = _parse_floats(m.get("scale"), 3, 1.0)
        mesh_defs[name] = (mesh_file, scale)

    n_written = 0
    for b in world.findall("body"):
        body_name = b.get("name", "")
        if not body_name.startswith("labg"):
            continue
        pos = _parse_floats(b.get("pos"), 3, 0.0)
        visual = None
        for g in b.findall("geom"):
            if g.get("class") == "visual":
                visual = g
                break
        if visual is None:
            continue

        mesh_name = visual.get("mesh")
        if not mesh_name or mesh_name not in mesh_defs:
            continue
        mesh_file, scale = mesh_defs[mesh_name]
        src_obj = os.path.join(args.mujoco_assets, mesh_file)
        verts, tris = _load_obj(src_obj)
        for v in verts:
            v[0] = v[0] * scale[0] + pos[0]
            v[1] = v[1] * scale[1] + pos[1]
            v[2] = v[2] * scale[2] + pos[2]

        rgba = _parse_floats(visual.get("rgba"), 4, 1.0)
        rgb = rgba[:3]
        vis_out = os.path.join(args.assets_zup, f"{body_name}_visual.obj")
        col_out = os.path.join(args.assets_zup, f"{body_name}_col_00.obj")
        _write_obj(vis_out, verts, tris, rgb=rgb)
        _write_obj(col_out, verts, tris, rgb=None)
        n_written += 1
        print(f"[inject] wrote {body_name} -> {os.path.basename(vis_out)}, {os.path.basename(col_out)}")

    if n_written == 0:
        raise RuntimeError("no labg* bodies found to inject")
    print(f"[inject] done: {n_written} glassware bodies exported to {args.assets_zup}")


if __name__ == "__main__":
    main()
