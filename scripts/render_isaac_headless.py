#!/usr/bin/env python3
"""Headless Isaac Sim RTX render of a USD scene from several fixed cameras -> PNG.

Sidesteps the Wayland/GLFW GUI-visibility problem on this box by rendering
headless (no window) via replicator render products + rgb annotators.

Run in an Isaac env (isaacsim 5.1.0.0), e.g.:
  ~/miniconda3/envs/g1skillgen/bin/python scripts/render_isaac_headless.py \
      --in_usd output/banner4/sim/usd/banner4_with_liquids_clear_table_v8.usdc \
      --out_dir output/banner4/isaac_render
"""
from __future__ import annotations

import argparse
import os

# ---- args before SimulationApp (it eats argv) -------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--in_usd", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--width", type=int, default=1600)
ap.add_argument("--height", type=int, default=900)
ap.add_argument("--warmup", type=int, default=140, help="RTX convergence frames before first capture")
ap.add_argument("--pathtrace", action="store_true", help="Use PathTracing (better glass, slower)")
args = ap.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

from isaacsim import SimulationApp  # noqa: E402

sim = SimulationApp({"headless": True, "width": args.width, "height": args.height})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import carb  # noqa: E402
from pxr import UsdLux, UsdGeom, Sdf, Gf  # noqa: E402

# renderer mode
settings = carb.settings.get_settings()
if args.pathtrace:
    settings.set("/rtx/rendermode", "PathTracing")
    settings.set("/rtx/pathtracing/spp", 32)
    settings.set("/rtx/pathtracing/totalSpp", 256)
else:
    settings.set("/rtx/rendermode", "RaytracedLighting")

ctx = omni.usd.get_context()
ctx.open_stage(args.in_usd)
stage = ctx.get_stage()
print(f"[render] opened {args.in_usd}", flush=True)

# --- lighting: dome + key so the scene isn't lit only by displayColor -------
if not stage.GetPrimAtPath("/World/ViewDome").IsValid():
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/ViewDome"))
    dome.CreateIntensityAttr(1000.0)
if not stage.GetPrimAtPath("/World/ViewKey").IsValid():
    key = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/ViewKey"))
    key.CreateIntensityAttr(2000.0)
    kx = UsdGeom.Xformable(key.GetPrim())
    kx.ClearXformOpOrder()
    kx.AddRotateXYZOp().Set(Gf.Vec3f(-40.0, 15.0, 0.0))

import omni.replicator.core as rep  # noqa: E402

# (name, camera_position, look_at)  -- world is Z-up, metres.
# Moved glassware cluster centroid ~ (-0.35, -2.72, 0.38); island vessels ~ (-0.15, -1.0, 0.42)
CAMERAS = [
    ("cluster",     (-0.35, -1.45, 0.86),  (-0.34, -2.72, 0.40)),
    ("graze_left",  ( 0.85, -2.66, 0.435), (-0.60, -2.76, 0.385)),
    ("overview",    ( 2.30,  1.25, 1.85),  (-0.15, -1.65, 0.30)),
]

products = []
for name, pos, look in CAMERAS:
    cam = rep.create.camera(position=pos, look_at=look)
    rp = rep.create.render_product(cam, (args.width, args.height))
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    annot.attach([rp])
    products.append((name, annot))

mode = "PathTracing" if args.pathtrace else "RaytracedLighting"
n_steps = max(12, args.warmup // 8)
print(f"[render] converging: {n_steps} orchestrator steps ({mode})...", flush=True)
# orchestrator.step() (not sim.update) is what actually populates rgb annotators
# and accumulates RTX subframes for a clean, denoised frame.
for i in range(n_steps):
    rep.orchestrator.step(rt_subframes=16, delta_time=0.0, pause_timeline=True)
    if i % 4 == 0:
        print(f"[render]   step {i}/{n_steps}", flush=True)

try:
    from PIL import Image
    def save(rgb, path):
        Image.fromarray(rgb).save(path)
except Exception:
    import imageio.v2 as imageio
    def save(rgb, path):
        imageio.imwrite(path, rgb)


def to_rgb(data, w, h):
    if isinstance(data, dict):
        data = data.get("data", data)
    arr = np.asarray(data)
    if arr.ndim == 1:  # flat buffer -> reshape by resolution
        ch = arr.size // (w * h) if (w * h) else 0
        if ch in (3, 4):
            arr = arr.reshape((h, w, ch))
        else:
            raise ValueError(f"unexpected flat buffer size {arr.size} for {w}x{h}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr[:, :, :3]


for name, annot in products:
    arr = to_rgb(annot.get_data(), args.width, args.height)
    out = os.path.join(args.out_dir, f"v8_{name}.png")
    save(arr, out)
    print(f"[render] wrote {out}  shape={arr.shape}", flush=True)

sim.close()
print("[render] done", flush=True)
