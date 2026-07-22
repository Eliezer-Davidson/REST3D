# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Export a stage3 scene to MuJoCo (MJCF) and prepare assets for Isaac Sim (USD).

Input : a stage3 variant dir (global_scene or global_scene_w_walls) whose obj_files/
        hold world-frame, Y-up, metric OBJs with vertex colors.
Output: <out>/mujoco/<scene>.xml + <out>/mujoco/assets/*.obj
        <out>/assets_zup/*_visual.obj + *_col_*.obj   (shared by the USD builder)

Per object:
  - vertices are rotated Y-up -> Z-up (x,y,z) -> (x,-z,y), the convention MuJoCo and
    Isaac both use, with gravity on -Z;
  - a decimated visual mesh keeps the look (MuJoCo OBJs can't carry vertex colors, so
    the MJCF visual geom gets the mesh's mean color; the USD builder re-reads the
    per-vertex colors from the *_visual.obj written here);
  - CoACD decomposes the shape into convex parts for collision (a raw convex hull
    would fill the space under the table top and between the sofa cushions).

Usage:
  python scripts/export_sim_scene.py \
      --scene_dir output/office_lounge/stage3/global_scene \
      --out output/office_lounge/sim
"""
import argparse
import os
import glob

import open3d  # noqa: F401  -- MUST import before coacd: the reverse order segfaults
                # (both vendor native libs; coacd-first poisons open3d's loader).
import numpy as np


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
    return np.asarray(V, np.float64), np.asarray(C, np.float64), np.asarray(F, np.int32)


def yup_to_zup(V):
    return np.stack([V[:, 0], -V[:, 2], V[:, 1]], axis=1)


def save_obj(path, V, F, C=None):
    with open(path, "w") as fh:
        if C is None:
            for v in V:
                fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        else:
            for v, c in zip(V, C):
                fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
        for f in F:
            fh.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")


def decimate(V, C, F, target_faces):
    import open3d as o3d
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    m.vertex_colors = o3d.utility.Vector3dVector(np.clip(C, 0, 1))
    if len(F) > target_faces:
        m = m.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    V2 = np.asarray(m.vertices)
    F2 = np.asarray(m.triangles, np.int32)
    C2 = np.asarray(m.vertex_colors)
    if len(C2) != len(V2):
        C2 = np.full_like(V2, 0.7)
    return V2, C2, F2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--visual_faces", type=int, default=50_000)
    ap.add_argument("--coacd_faces", type=int, default=20_000,
                    help="decimation target before CoACD (it scales badly with input size)")
    ap.add_argument("--coacd_threshold", type=float, default=0.05)
    ap.add_argument("--max_hulls", type=int, default=32)
    ap.add_argument("--density", type=float, default=200.0,
                    help="kg/m^3 for collision geoms (foam-furniture-ish default)")
    args = ap.parse_args()

    scene = os.path.basename(os.path.dirname(os.path.dirname(args.scene_dir.rstrip("/"))))
    zup_dir = os.path.join(args.out, "assets_zup")
    mj_dir = os.path.join(args.out, "mujoco")
    mj_assets = os.path.join(mj_dir, "assets")
    os.makedirs(zup_dir, exist_ok=True)
    os.makedirs(mj_assets, exist_ok=True)

    import coacd
    coacd.set_log_level("error")

    bodies = []
    for path in sorted(glob.glob(os.path.join(args.scene_dir, "obj_files", "*.obj"))):
        name = os.path.basename(path)[:-4]
        V, C, F = load_obj(path)
        V = yup_to_zup(V)

        vis_obj = os.path.join(zup_dir, f"{name}_visual.obj")
        if not os.path.exists(vis_obj):
            Vd, Cd, Fd = decimate(V, C, F, args.visual_faces)
            save_obj(vis_obj, Vd, Fd, Cd)
        mean_rgb = C.mean(axis=0)

        col_paths = sorted(glob.glob(os.path.join(zup_dir, f"{name}_col_*.obj")))
        if not col_paths:
            Vc, _, Fc = decimate(V, C, F, args.coacd_faces)
            mesh = coacd.Mesh(Vc, Fc)
            parts = coacd.run_coacd(
                mesh, threshold=args.coacd_threshold,
                max_convex_hull=args.max_hulls, seed=0)
            for k, (pv, pf) in enumerate(parts):
                p = os.path.join(zup_dir, f"{name}_col_{k:02d}.obj")
                save_obj(p, np.asarray(pv), np.asarray(pf, np.int32))
                col_paths.append(p)
            print(f"[coacd] {name}: {len(parts)} convex parts", flush=True)
        else:
            print(f"[coacd] {name}: cached ({len(col_paths)} parts)", flush=True)
        bodies.append((name, mean_rgb, col_paths))

    # ---- MJCF -------------------------------------------------------------
    # Assets are symlinked into mujoco/assets so the XML stays relocatable with
    # relative meshdir; copy if the filesystem refuses symlinks.
    for f in glob.glob(os.path.join(zup_dir, "*.obj")):
        dst = os.path.join(mj_assets, os.path.basename(f))
        if not os.path.exists(dst):
            try:
                os.symlink(os.path.relpath(f, mj_assets), dst)
            except OSError:
                import shutil
                shutil.copy(f, dst)

    L = []
    L.append(f'<mujoco model="{scene}">')
    L.append('  <compiler meshdir="assets" angle="radian"/>')
    L.append('  <option timestep="0.002"/>')
    L.append('  <visual><global offwidth="1920" offheight="1080"/></visual>')
    L.append('  <default>')
    L.append('    <default class="visual">')
    L.append('      <geom type="mesh" contype="0" conaffinity="0" group="2" mass="0"/>')
    L.append('    </default>')
    L.append('    <default class="collision">')
    L.append(f'      <geom type="mesh" group="3" density="{args.density}" '
             'friction="0.9 0.005 0.0001" rgba="0.8 0.5 0.2 0.35"/>')
    L.append('    </default>')
    L.append('  </default>')
    L.append('  <asset>')
    L.append('    <texture type="skybox" builtin="gradient" rgb1="0.9 0.9 0.95" '
             'rgb2="0.6 0.65 0.75" width="64" height="64"/>')
    L.append('    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.62 0.62 0.60" '
             'rgb2="0.55 0.55 0.53" width="128" height="128"/>')
    L.append('    <material name="floor_mat" texture="floor_tex" texrepeat="8 8"/>')
    for name, _, col_paths in bodies:
        L.append(f'    <mesh name="{name}_visual" file="{name}_visual.obj"/>')
        for p in col_paths:
            b = os.path.basename(p)[:-4]
            L.append(f'    <mesh name="{b}" file="{os.path.basename(p)}"/>')
    L.append('  </asset>')
    L.append('  <worldbody>')
    L.append('    <light pos="0 -1 3" dir="0 0.3 -1" directional="true"/>')
    L.append('    <geom name="floor" type="plane" size="10 10 0.1" material="floor_mat" '
             'friction="0.9 0.005 0.0001"/>')
    for name, rgb, col_paths in bodies:
        L.append(f'    <body name="{name}">')
        L.append('      <freejoint/>')
        L.append(f'      <geom class="visual" mesh="{name}_visual" '
                 f'rgba="{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} 1"/>')
        for p in col_paths:
            b = os.path.basename(p)[:-4]
            L.append(f'      <geom class="collision" mesh="{b}"/>')
        L.append('    </body>')
    L.append('  </worldbody>')
    L.append('</mujoco>')

    xml_path = os.path.join(mj_dir, f"{scene}.xml")
    with open(xml_path, "w") as fh:
        fh.write("\n".join(L))
    n_col = sum(len(b[2]) for b in bodies)
    print(f"[mjcf] {xml_path}  ({len(bodies)} bodies, {n_col} collision parts)")


if __name__ == "__main__":
    main()
