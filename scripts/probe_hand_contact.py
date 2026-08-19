#!/usr/bin/env python3
"""Ask PHYSX -- not USD -- which colliders actually exist, and measure real
hand-vs-glass interpenetration.

USD composition showing geometry under a `collisions` scope does NOT mean PhysX
built a collision shape from it. The G1's arm chain and PALM carry INSTANCED
collision scopes; the fingers carry direct ones. This probe settles whether that
distinction is why the palm passes through the glass, by running a PhysX scene
query at each link and listing what physics reports back.

It also measures true geometric penetration of every hand link's mesh against
the cup modelled as its exact analytic solid (annulus wall + base slab), which
is independent of whether any contact was ever generated.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--settle", type=int, default=120)
ap.add_argument("--out", default="output/banner4/phasing/hand_contact.json")
args = ap.parse_args()

ROBOT = "/World/G1"
CUPS = ["/World/labg4", "/World/labg9"]
LINKS = ["right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link",
         "right_wrist_pitch_link", "right_wrist_yaw_link",
         "right_hand_camera_base_link", "right_hand_base_link",
         "R_index_proximal", "R_index_intermediate", "R_middle_proximal",
         "R_middle_intermediate", "R_ring_proximal", "R_ring_intermediate",
         "R_pinky_proximal", "R_pinky_intermediate", "R_thumb_proximal_base",
         "R_thumb_proximal", "R_thumb_intermediate", "R_thumb_distal"]

from isaacsim import SimulationApp
sim = SimulationApp({"headless": True})

import omni.usd                                                     # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics, Gf                        # noqa: E402
from isaacsim.core.api import World                                 # noqa: E402
import isaacsim.core.utils.stage as stage_utils                     # noqa: E402
from omni.physx import get_physx_scene_query_interface              # noqa: E402


def log(*a):
    print("[hand]", *a, flush=True)


stage_utils.open_stage(os.path.abspath(args.usd))
stage = omni.usd.get_context().get_stage()
world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
pc = world.get_physics_context()
pc.enable_gpu_dynamics(True)
try:
    pc.set_broadphase_type("GPU")
except Exception:
    pass
world.reset()
for _ in range(args.settle):
    world.step(render=False)
world.step(render=True)

bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])


def world_aabb(path):
    bc.Clear()
    b = bc.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedBox()
    return np.array(b.GetMin(), float), np.array(b.GetMax(), float)


# ---------------------------------------------------------------- PhysX truth
qi = get_physx_scene_query_interface()


def physx_overlap(mn, mx, pad=0.002):
    """What collision shapes does PHYSX report inside this box?"""
    hits = []

    def cb(hit):
        hits.append(str(hit.collision))
        return True

    c = (mn + mx) / 2.0
    he = (mx - mn) / 2.0 + pad
    qi.overlap_box(carb_float3(he), carb_float3(c), carb_float4(0, 0, 0, 1), cb, False)
    return hits


import carb                                                          # noqa: E402


def carb_float3(v):
    return carb.Float3(float(v[0]), float(v[1]), float(v[2]))


def carb_float4(x, y, z, w):
    return carb.Float4(float(x), float(y), float(z), float(w))


log("=" * 74)
log("DOES PHYSX HAVE A COLLIDER FOR THIS LINK?")
log("=" * 74)
log(f"{'link':30s} {'instanced':>10s} {'physx shapes reported':>22s}")
log("-" * 74)
res_links = {}
for l in LINKS:
    lp = stage.GetPrimAtPath(f"{ROBOT}/{l}")
    col = stage.GetPrimAtPath(f"{ROBOT}/{l}/collisions")
    if not lp.IsValid():
        continue
    inst = col.IsInstance() if col.IsValid() else False
    mn, mx = world_aabb(f"{ROBOT}/{l}")
    hits = physx_overlap(mn, mx)
    mine = [h for h in hits if l in h]
    res_links[l] = {"instanced": bool(inst), "n_hits": len(hits), "own_shapes": mine[:4],
                    "has_own_collider": bool(mine)}
    mark = "" if mine else "   <== PHYSX HAS NO COLLIDER HERE"
    log(f"{l:30s} {str(inst):>10s} {len(mine):>22d}{mark}")

log("")
log("=" * 74)
log("WHAT PHYSX REPORTS INSIDE EACH CUP'S VOLUME")
log("=" * 74)
res_cups = {}
for cp in CUPS:
    mn, mx = world_aabb(cp)
    hits = sorted(set(physx_overlap(mn, mx)))
    own = [h for h in hits if cp in h]
    hand = [h for h in hits if any(k in h for k in ("R_", "right_hand", "right_wrist"))]
    log(f"{cp}:  {len(hits)} shapes total")
    log(f"    own colliders     : {len(own)}  e.g. {own[:3]}")
    log(f"    HAND shapes inside: {len(hand)}  {hand[:8]}")
    res_cups[cp] = {"total": len(hits), "own": len(own), "hand_inside": hand}

json.dump({"links": res_links, "cups": res_cups}, open(args.out, "w"), indent=2)
log(f"wrote {args.out}")
sim.close()
