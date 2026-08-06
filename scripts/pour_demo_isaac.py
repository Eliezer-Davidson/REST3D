#!/usr/bin/env python3
"""Standalone Isaac Sim demo of REAL liquid being poured between two beakers.

- Two beakers use the watertight beaker.obj with PhysX SDF-mesh collision, so the
  hollow interior actually contains fluid.
- Source beaker: kinematic rigid body, scripted tilt -> pours.
- Receiver beaker: static collider on the ground.
- Fluid: PhysX position-based-dynamics particle set (real pourable liquid).
- Runs headless, captures RGB frames from a fixed camera, encodes an MP4.

Run:  ~/miniconda3/envs/g1skillgen/bin/python scripts/pour_demo_isaac.py --out_dir output/pour_demo
"""
from __future__ import annotations
import argparse, os, math

ap = argparse.ArgumentParser()
ap.add_argument("--out_dir", default="output/pour_demo")
ap.add_argument("--beaker_obj", default="lab_demo/props/beaker.obj")
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--settle", type=int, default=70, help="physics steps upright before pouring")
ap.add_argument("--pour", type=int, default=150, help="steps over which the tilt ramps")
ap.add_argument("--hold", type=int, default=50, help="steps held at full tilt")
ap.add_argument("--tilt_deg", type=float, default=115.0)
ap.add_argument("--capture_every", type=int, default=3)
ap.add_argument("--glass_op", type=float, default=0.14, help="glass opacity (raise to debug visibility)")
ap.add_argument("--cam_pos", default="0.30,-0.30,0.19")
ap.add_argument("--cam_look", default="0.02,0.0,0.095")
args = ap.parse_args()
GLASS_OP = args.glass_op
CAM_POS = tuple(float(x) for x in args.cam_pos.split(","))
CAM_LOOK = tuple(float(x) for x in args.cam_look.split(","))
os.makedirs(args.out_dir, exist_ok=True)
frames_dir = os.path.join(args.out_dir, "frames")
os.makedirs(frames_dir, exist_ok=True)

from isaacsim import SimulationApp
sim = SimulationApp({"headless": True, "width": args.width, "height": args.height})

import numpy as np
import carb, omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, UsdLux, UsdShade, Sdf, Vt
from omni.physx.scripts import particleUtils
from isaacsim.core.api import World
import omni.replicator.core as rep

carb.settings.get_settings().set("/rtx/rendermode", "RaytracedLighting")

world = World(stage_units_in_meters=1.0, physics_dt=1.0/120.0, rendering_dt=1.0/60.0)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()

# physics scene path (simulation owner for the particle system)
scene_path = None
for p in stage.Traverse():
    if p.IsA(UsdPhysics.Scene):
        scene_path = p.GetPath().pathString
        break
print("[pour] physics scene:", scene_path, flush=True)

# particles REQUIRE GPU dynamics
scene_api = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(scene_path))
scene_api.CreateEnableGPUDynamicsAttr(True)
scene_api.CreateBroadphaseTypeAttr("GPU")
print("[pour] GPU dynamics enabled", flush=True)

# lights
UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(700.0)
key = UsdLux.DistantLight.Define(stage, "/World/Key"); key.CreateIntensityAttr(2200.0)
kx = UsdGeom.Xformable(key.GetPrim()); kx.ClearXformOpOrder(); kx.AddRotateXYZOp().Set(Gf.Vec3f(-45, 20, 0))


def load_obj(path):
    V, F = [], []
    for line in open(path):
        s = line.split()
        if not s:
            continue
        if s[0] == "v":
            V.append((float(s[1]), float(s[2]), float(s[3])))
        elif s[0] == "f":
            idx = [int(t.split("/")[0]) - 1 for t in s[1:]]
            for k in range(1, len(idx) - 1):
                F.append((idx[0], idx[k], idx[k + 1]))
    return V, F


def glass_material(path, rgb):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(GLASS_OP)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.06)
    sh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.45)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


V, F = load_obj(args.beaker_obj)
pts = [Gf.Vec3f(*v) for v in V]
counts = [3] * len(F)
indices = [i for f in F for i in f]


def make_beaker(path, translate, kinematic):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateDoubleSidedAttr(True)
    prim = mesh.GetPrim()
    xf = UsdGeom.Xformable(prim); xf.ClearXformOpOrder()
    top = xf.AddTranslateOp(); top.Set(Gf.Vec3d(*translate))
    rop = xf.AddOrientOp(); rop.Set(Gf.Quatf(1, 0, 0, 0))
    UsdPhysics.CollisionAPI.Apply(prim)
    mc = UsdPhysics.MeshCollisionAPI.Apply(prim); mc.CreateApproximationAttr("sdf")
    sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
    sdf.CreateSdfResolutionAttr(320)
    if kinematic:
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        rb.CreateRigidBodyEnabledAttr(True)
        rb.CreateKinematicEnabledAttr(True)
    mat = glass_material(path + "_glass", (0.80, 0.90, 0.97))
    UsdShade.MaterialBindingAPI(prim).Bind(mat)
    return prim, top, rop


SRC_T = (-0.045, 0.0, 0.120)
RCV_T = (0.075, 0.0, 0.0)
src_prim, src_translate, src_orient = make_beaker("/World/beaker_src", SRC_T, kinematic=True)
rcv_prim, _, _ = make_beaker("/World/beaker_rcv", RCV_T, kinematic=False)

# ---- particle system + fluid ----
PCO = 0.006          # particle contact offset
FRO = 0.0035         # fluid rest offset
particleUtils.add_physx_particle_system(
    stage, "/World/particleSystem", simulation_owner=scene_path,
    contact_offset=PCO, rest_offset=0.005, particle_contact_offset=PCO,
    solid_rest_offset=0.0035, fluid_rest_offset=FRO,
    enable_ccd=True, solver_position_iterations=16, max_velocity=8.0,
)
# fluid grid inside the source beaker cavity (beaker local z 0..0.10, inner r~0.028)
spacing = 2.0 * FRO
cx, cy, cz = SRC_T
lower = Gf.Vec3f(cx - 0.018, cy - 0.018, cz + 0.020)
_pos, _vel = particleUtils.create_particles_grid(lower, spacing, 6, 6, 8)  # returns (positions, velocities)
positions = Vt.Vec3fArray(_pos)
vels = Vt.Vec3fArray(_vel)
fluid = particleUtils.add_physx_particleset_pointinstancer(
    stage, Sdf.Path("/World/fluid"), positions, vels, Sdf.Path("/World/particleSystem"),
    self_collision=True, fluid=True, particle_group=0,
    particle_mass=0.0, density=1000.0, num_prototypes=1,
)
print(f"[pour] fluid particles: {len(positions)}", flush=True)

# color + size the particle prototype (rendered as blue spheres)
inst = UsdGeom.PointInstancer(stage.GetPrimAtPath("/World/fluid"))
proto_targets = inst.GetPrototypesRel().GetTargets()
print(f"[dbg] fluid instances={len(inst.GetPositionsAttr().Get() or [])} proto_targets={list(proto_targets)}", flush=True)
if proto_targets:
    for pt in proto_targets:
        sp = stage.GetPrimAtPath(pt)
        s = UsdGeom.Sphere(sp)
        if s:
            s.CreateRadiusAttr(float(FRO * 1.6))
            s.GetPrim().CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set(
                Vt.Vec3fArray([Gf.Vec3f(0.10, 0.45, 0.95)]))
            print(f"[dbg] set prototype {pt} radius={FRO*1.6:.4f}", flush=True)

# report actual geometry positions so we can frame the camera correctly
_bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
for nm in ["/World/beaker_src", "/World/beaker_rcv", "/World/fluid"]:
    b = _bc.ComputeWorldBound(stage.GetPrimAtPath(nm)).ComputeAlignedBox()
    mn, mx = b.GetMin(), b.GetMax()
    print(f"[dbg] {nm}: min=({mn[0]:.3f},{mn[1]:.3f},{mn[2]:.3f}) max=({mx[0]:.3f},{mx[1]:.3f},{mx[2]:.3f})", flush=True)

# ---- camera + capture ----
cam = rep.create.camera(position=CAM_POS, look_at=CAM_LOOK)
# default near-clip (~1 m) culls these ~0.4 m-away beakers -> set it small
for _cp in stage.Traverse():
    if _cp.IsA(UsdGeom.Camera):
        UsdGeom.Camera(_cp).CreateClippingRangeAttr().Set(Gf.Vec2f(0.005, 100000.0))
rp = rep.create.render_product(cam, (args.width, args.height))
annot = rep.AnnotatorRegistry.get_annotator("rgb")
annot.attach([rp])


def to_rgb(data, w, h):
    arr = np.asarray(data)
    if arr.ndim == 1:
        ch = arr.size // (w * h)
        arr = arr.reshape((h, w, ch))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr[:, :, :3]


try:
    from PIL import Image
    def save(rgb, p): Image.fromarray(rgb).save(p)
except Exception:
    import imageio.v2 as imageio
    def save(rgb, p): imageio.imwrite(p, rgb)


def tilt_quat(deg):
    q = Gf.Rotation(Gf.Vec3d(0, 1, 0), deg).GetQuat()  # about +Y
    return Gf.Quatf(q.GetReal(), *[float(x) for x in q.GetImaginary()])


world.reset()
total = args.settle + args.pour + args.hold
cap = 0
for f in range(total):
    if f < args.settle:
        deg = 0.0
    elif f < args.settle + args.pour:
        t = (f - args.settle) / max(1, args.pour)
        deg = args.tilt_deg * (0.5 - 0.5 * math.cos(math.pi * t))  # smootherstep-ish
    else:
        deg = args.tilt_deg
    src_orient.Set(tilt_quat(deg))
    world.step(render=True)
    if f % args.capture_every == 0:
        arr = to_rgb(annot.get_data(), args.width, args.height)
        save(arr, os.path.join(frames_dir, f"f{cap:04d}.png"))
        cap += 1
    if f % 30 == 0:
        print(f"[pour] step {f}/{total} tilt={deg:.0f}deg captured={cap}", flush=True)

print(f"[pour] done: {cap} frames in {frames_dir}", flush=True)
sim.close()
