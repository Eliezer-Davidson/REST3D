#!/usr/bin/env python3
"""Load banner4_fluid_v12, run physics to settle (no robot, no tilt), and render:
  - an overview (confirm the lab did NOT collapse)
  - a closeup of the source/target cups (confirm fluid stays in the cup)
"""
from __future__ import annotations
import argparse, os

ap = argparse.ArgumentParser()
ap.add_argument("--usd", default="output/banner4/sim/usd/banner4_fluid_v12.usdc")
ap.add_argument("--out_dir", default="output/banner4/fluid_check")
ap.add_argument("--steps", type=int, default=120)
ap.add_argument("--every", type=int, default=4)
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

from isaacsim import SimulationApp
sim = SimulationApp({"headless": True, "width": args.width, "height": args.height})

import numpy as np, carb, omni.usd
from pxr import UsdGeom, Gf, UsdLux, Sdf
from isaacsim.core.api import World
import omni.replicator.core as rep

carb.settings.get_settings().set("/rtx/rendermode", "RaytracedLighting")
omni.usd.get_context().open_stage(os.path.abspath(args.usd))
stage = omni.usd.get_context().get_stage()
# banner4 ships no lights -> add them or the render is black
UsdLux.DomeLight.Define(stage, "/World/ValDome").CreateIntensityAttr(800.0)
_k = UsdLux.DistantLight.Define(stage, "/World/ValKey"); _k.CreateIntensityAttr(2200.0)
_kx = UsdGeom.Xformable(_k.GetPrim()); _kx.ClearXformOpOrder(); _kx.AddRotateXYZOp().Set(Gf.Vec3f(-45, 20, 0))
world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
# SDF collision + particles REQUIRE GPU dynamics; World() doesn't apply the authored
# flag when loading an existing stage, so force it on or the SDF cups misbehave.
_pc = world.get_physics_context()
_pc.enable_gpu_dynamics(True)
try:
    _pc.set_broadphase_type("GPU")
except Exception:
    pass

for _cp in stage.Traverse():
    if _cp.IsA(UsdGeom.Camera):
        UsdGeom.Camera(_cp).CreateClippingRangeAttr().Set(Gf.Vec2f(0.005, 100000.0))

CAMERAS = [
    ("overview", (5.75, 3.10, 4.60), (-0.40, -4.10, 0.75)),
    ("cups",     (-0.05, -4.85, 1.42), (-1.33, -6.53, 0.98)),
]
prods = []
for nm, pos, look in CAMERAS:
    cam = rep.create.camera(position=pos, look_at=look)
    for _cp in stage.Traverse():
        if _cp.IsA(UsdGeom.Camera):
            UsdGeom.Camera(_cp).CreateClippingRangeAttr().Set(Gf.Vec2f(0.005, 100000.0))
    rpd = rep.create.render_product(cam, (args.width, args.height))
    an = rep.AnnotatorRegistry.get_annotator("rgb")
    an.attach([rpd])
    prods.append((nm, an))

try:
    from PIL import Image
    def save(a, p): Image.fromarray(a).save(p)
except Exception:
    import imageio.v2 as imageio
    def save(a, p): imageio.imwrite(p, a)


def to_rgb(d, w, h):
    a = np.asarray(d)
    if a.ndim == 1:
        a = a.reshape((h, w, a.size // (w * h)))
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a[:, :, :3]


world.reset()
cap = 0
for f in range(args.steps):
    world.step(render=True)
    if f % args.every == 0:
        for nm, an in prods:
            save(to_rgb(an.get_data(), args.width, args.height),
                 os.path.join(args.out_dir, f"{nm}_{cap:03d}.png"))
        cap += 1
    if f % 30 == 0:
        print(f"[val] step {f}/{args.steps} captured={cap}", flush=True)
print(f"[val] done: {cap} frames/camera in {args.out_dir}", flush=True)
sim.close()
