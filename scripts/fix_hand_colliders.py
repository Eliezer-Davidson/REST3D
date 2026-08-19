#!/usr/bin/env python3
"""Repair the G1 right-arm colliders: populate /collisions scopes that carry the
CollisionAPI but contain NO geometry.

The g1_29dof_with_inspire asset has 65 rigid bodies and collision geometry on only
24 of them -- the finger/thumb bones. Every arm link, both palms, the torso and the
feet ship a `collisions` Xform that has UsdPhysics.CollisionAPI + MeshCollisionAPI
(convexHull) applied and `collisionEnabled = True`, but whose geometry is an
internal reference to a /Flattened_Prototype_* prim that does not exist in the
composed stage. The flag reads ON everywhere, so any audit that counts
collisionEnabled reports a fully-collidable robot; physics sees nothing.

Consequence in banner4: the palm, wrist, forearm and elbow pass straight through
the glass and the counter, so the hand can be driven deeper than geometry allows
and the fingers/thumb end up inside the cup outline.

Fix: give each empty /collisions the same geometry its sibling /visuals already
has, via an internal reference (no data duplication). Scope is limited to the
right arm chain -- the links whose collisionEnabled is ALREADY True, i.e. the ones
the scene intends to collide. Legs/torso are deliberately left alone: they are
switched off in this scene on purpose and giving them colliders would drop the
posed robot onto the counter.
"""
import argparse
import os

from pxr import Usd, UsdGeom, UsdPhysics

CHAIN = ["right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
         "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link",
         "right_wrist_yaw_link", "right_hand_camera_base_link", "right_hand_base_link"]

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--robot", default="/World/G1")
ap.add_argument("--links", default=",".join(CHAIN))
args = ap.parse_args()

stage = Usd.Stage.Open(os.path.abspath(args.usd))
if stage is None:
    raise SystemExit(f"cannot open {args.usd}")


def geom_points(prim):
    n = 0
    for q in Usd.PrimRange(prim):
        if q.GetTypeName() == "Mesh":
            pts = UsdGeom.Mesh(q).GetPointsAttr().Get()
            n += 0 if pts is None else len(pts)
    return n


fixed, skipped, deinstanced = [], [], []
for link in [l for l in args.links.split(",") if l]:
    lp = stage.GetPrimAtPath(f"{args.robot}/{link}")
    if not lp.IsValid():
        skipped.append((link, "link missing"))
        continue
    col = stage.GetPrimAtPath(f"{args.robot}/{link}/collisions")
    vis = stage.GetPrimAtPath(f"{args.robot}/{link}/visuals")
    if not col.IsValid():
        col = stage.DefinePrim(f"{args.robot}/{link}/collisions", "Xform")
    if not vis.IsValid():
        skipped.append((link, "no /visuals to copy from"))
        continue
    # These 9 scopes are INSTANCES (instanceable=True) whose prototype is empty --
    # that is why the flag reads ON while physics sees nothing, and why authoring
    # into them fails with "authoring to an instance proxy is not allowed". The
    # healthy finger scopes are not instanced. De-instance before repairing.
    if col.IsInstanceable():
        col.SetInstanceable(False)
        deinstanced.append(link)

    have = geom_points(col)
    if have > 0:
        skipped.append((link, f"already has {have} pts"))
        continue
    added = 0
    for child in vis.GetChildren():
        dst = stage.DefinePrim(col.GetPath().AppendChild(child.GetName()), "Xform")
        dst.GetReferences().AddInternalReference(child.GetPath())
        added += 1
    # the APIs are already authored on these prims; assert them anyway so the fix
    # also works on a stage where only the geometry convention is present
    UsdPhysics.CollisionAPI.Apply(col).CreateCollisionEnabledAttr(True).Set(True)
    mc = UsdPhysics.MeshCollisionAPI.Apply(col)
    if not mc.GetApproximationAttr().HasAuthoredValue():
        mc.CreateApproximationAttr("convexHull").Set("convexHull")
    now = geom_points(col)
    fixed.append((link, added, now, geom_points(vis)))

print(f"{'link':30s} {'refs':>5s} {'coll pts':>9s} {'vis pts':>9s}")
for link, added, now, vn in fixed:
    ok = "OK" if now > 0 else "STILL EMPTY"
    print(f"{link:30s} {added:5d} {now:9d} {vn:9d}  {ok}")
for link, why in skipped:
    print(f"{link:30s} skipped: {why}")
if deinstanced:
    print(f"\nde-instanced {len(deinstanced)} empty-prototype collision scopes: "
          + ", ".join(deinstanced))

stage.GetRootLayer().Export(os.path.abspath(args.out))
print(f"\nwrote {args.out}  ({len(fixed)} links repaired, {len(skipped)} skipped)")
