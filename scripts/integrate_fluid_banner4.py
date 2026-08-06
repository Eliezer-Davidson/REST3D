#!/usr/bin/env python3
"""Make banner4 physics-ready for robot manipulation + add REAL fluid to one cup.

- All dynamic furniture -> static (rigidBodyEnabled=False) so the lab doesn't collapse.
- SOURCE + TARGET cups -> dynamic with SDF-mesh collision (graspable + can contain/pour fluid).
- SOURCE cup: fake liquid cylinder removed, real PhysX PBD fluid particles added.
- GPU dynamics enabled on the physics scene (required for particles).

Run in an Isaac env:
  ~/miniconda3/envs/g1skillgen/bin/python scripts/integrate_fluid_banner4.py \
     --in_usd  output/banner4/sim/usd/banner4_with_liquids_v11.usdc \
     --out_usd output/banner4/sim/usd/banner4_fluid_v12.usdc
"""
from __future__ import annotations
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--in_usd", required=True)
ap.add_argument("--out_usd", required=True)
ap.add_argument("--source", default="labg4")
ap.add_argument("--target", default="labg9")
ap.add_argument("--fro", type=float, default=0.005, help="fluid rest offset (real-scale ~0.005)")
ap.add_argument("--pco", type=float, default=0.009, help="particle contact offset")
args = ap.parse_args()

FOOT_PROTRUSION = 0.006  # how far the invisible ground-contact foot sticks below the glass base

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf, Vt, UsdShade
from omni.physx.scripts import particleUtils

stage = Usd.Stage.Open(args.in_usd)
stage.Export(args.out_usd)
stage = Usd.Stage.Open(args.out_usd)
world = stage.GetPrimAtPath("/World")
bcache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

# high-friction physics material to keep small cups from sliding/tipping
grip = UsdShade.Material.Define(stage, "/World/PhysicsMaterials/grip")
gm = UsdPhysics.MaterialAPI.Apply(grip.GetPrim())
gm.CreateStaticFrictionAttr(1.4)
gm.CreateDynamicFrictionAttr(1.2)
gm.CreateRestitutionAttr(0.0)

KEEP_DYNAMIC = {args.source, args.target}

# 1) freeze everything dynamic except the two target cups -> static environment
frozen = 0
for c in world.GetChildren():
    n = c.GetName()
    if n in KEEP_DYNAMIC:
        continue
    if c.HasAPI(UsdPhysics.RigidBodyAPI):
        rb = UsdPhysics.RigidBodyAPI(c)
        rb.GetRigidBodyEnabledAttr().Set(False) if rb.GetRigidBodyEnabledAttr() else rb.CreateRigidBodyEnabledAttr(False)
        frozen += 1
print(f"[fluid] froze {frozen} dynamic bodies -> static", flush=True)


def ensure_support(name, prim):
    """Flat, invisible, high-friction static slab flush under the cup base for stable contact."""
    box = bcache.ComputeWorldBound(prim).ComputeAlignedBox()
    mn, mx = box.GetMin(), box.GetMax()
    cx, cy, base = 0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1]), float(mn[2])
    sp = f"/World/{name}_base_support"
    if not stage.GetPrimAtPath(sp).IsValid():
        cube = UsdGeom.Cube.Define(stage, sp)
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cube.GetPrim()); xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, base - 0.003))
        xf.AddScaleOp().Set(Gf.Vec3f(0.06, 0.06, 0.004))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(grip, materialPurpose="physics")
        UsdGeom.Imageable(cube.GetPrim()).CreateVisibilityAttr(UsdGeom.Tokens.invisible)


def make_sdf_cup(name):
    """Give a cup accurate SDF-mesh collision on its hollow visual mesh; drop convex col."""
    prim = stage.GetPrimAtPath(f"/World/{name}")
    vis = stage.GetPrimAtPath(f"/World/{name}/visual")
    if not vis.IsValid():
        raise RuntimeError(f"{name}: no /visual")
    UsdPhysics.CollisionAPI.Apply(vis)
    mc = UsdPhysics.MeshCollisionAPI.Apply(vis)
    mc.CreateApproximationAttr("sdf")
    sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(vis)
    sdf.CreateSdfResolutionAttr(320)
    UsdShade.MaterialBindingAPI(vis).Bind(grip, materialPurpose="physics")
    # remove the convex-hull collider (a solid blob that can't hold fluid)
    col = stage.GetPrimAtPath(f"/World/{name}/col_00")
    if col.IsValid():
        stage.RemovePrim(col.GetPath())
    # ensure dynamic + sane mass
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    (rb.GetRigidBodyEnabledAttr() or rb.CreateRigidBodyEnabledAttr(True)).Set(True)
    (rb.GetKinematicEnabledAttr() or rb.CreateKinematicEnabledAttr(False)).Set(False)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(0.3)  # ~300 g cup
    # LOW center of mass -> bottom-heavy, self-righting, won't topple. COM is in the
    # body's LOCAL frame, and this cup's geometry is offset (local z ~0.9), so derive
    # it from the actual mesh points rather than assuming a base at 0.
    _vp = np.array([[v[0], v[1], v[2]] for v in UsdGeom.Mesh(vis).GetPointsAttr().Get()])
    _lcx = float((_vp[:, 0].min() + _vp[:, 0].max()) * 0.5)
    _lcy = float((_vp[:, 1].min() + _vp[:, 1].max()) * 0.5)
    _lbz = float(_vp[:, 2].min())
    mass.CreateCenterOfMassAttr(Gf.Vec3f(_lcx, _lcy, _lbz + 0.02))
    # SDF contains fluid but gives a domed/ring bottom contact (cup rocks to ~11deg).
    # Add a flat convex cylinder FOOT at the base for a clean, stable ground contact.
    foot_r = 0.42 * min(_vp[:, 0].max() - _vp[:, 0].min(), _vp[:, 1].max() - _vp[:, 1].min())
    foot_h = FOOT_PROTRUSION
    foot = UsdGeom.Cylinder.Define(stage, f"/World/{name}/foot")
    foot.CreateRadiusAttr(float(foot_r)); foot.CreateHeightAttr(foot_h); foot.CreateAxisAttr("Z")
    fxf = UsdGeom.Xformable(foot.GetPrim()); fxf.ClearXformOpOrder()
    # top flush with the glass base (does NOT poke into the cavity -> won't eject fluid);
    # protrudes 6mm below so IT, not the domed SDF base, is the ground contact.
    fxf.AddTranslateOp().Set(Gf.Vec3d(_lcx, _lcy, _lbz - foot_h * 0.5))
    UsdPhysics.CollisionAPI.Apply(foot.GetPrim())
    UsdShade.MaterialBindingAPI(foot.GetPrim()).Bind(grip, materialPurpose="physics")
    UsdGeom.Imageable(foot.GetPrim()).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    # stabilize small dynamic cup: damping + more solver iterations
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    prb.CreateAngularDampingAttr(6.0)
    prb.CreateLinearDampingAttr(1.0)
    prb.CreateSolverPositionIterationCountAttr(32)
    prb.CreateSolverVelocityIterationCountAttr(4)
    return prim


# 1b) reseat the two cups cleanly ON TOP of the counter. The reconstruction left
# their bases ~a few cm inside the counter; at real scale that interpenetrates the
# counter collider and PhysX ejects the cup (the tipping we saw). Ray-cast the
# counter surface under each cup and lift the cup to rest on it.
_xc = UsdGeom.XformCache(Usd.TimeCode.Default())
_A, _B, _C = [], [], []
for _p in stage.Traverse():
    if not _p.IsA(UsdGeom.Mesh):
        continue
    _pp = str(_p.GetPath()); _top = _pp.split("/")[2] if len(_pp.split("/")) > 2 else ""
    if _top.startswith("labg") or "support" in _pp or _top in ("Looks", "fluid"):
        continue
    _m = UsdGeom.Mesh(_p)
    _pts = _m.GetPointsAttr().Get(); _fvc = _m.GetFaceVertexCountsAttr().Get(); _fvi = _m.GetFaceVertexIndicesAttr().Get()
    if not _pts or not _fvc or not _fvi:
        continue
    _M = np.array(_xc.GetLocalToWorldTransform(_p), dtype=np.float64)
    _V = (np.array([[q[0], q[1], q[2], 1.0] for q in _pts]) @ _M)[:, :3]
    _i = 0
    for _cc in _fvc:
        for _k in range(1, _cc - 1):
            _A.append(_V[_fvi[_i]]); _B.append(_V[_fvi[_i + _k]]); _C.append(_V[_fvi[_i + _k + 1]])
        _i += _cc
_A = np.array(_A); _B = np.array(_B); _C = np.array(_C)
_tmn = np.minimum(np.minimum(_A, _B), _C); _tmx = np.maximum(np.maximum(_A, _B), _C)


def _surface_under(px, py, zceil):
    m = (_tmn[:, 0] <= px) & (_tmx[:, 0] >= px) & (_tmn[:, 1] <= py) & (_tmx[:, 1] >= py) & (_tmn[:, 2] <= zceil)
    if not m.any():
        return None
    a, b, c = _A[m], _B[m], _C[m]
    v0 = b[:, :2] - a[:, :2]; v1 = c[:, :2] - a[:, :2]; v2 = np.array([px, py]) - a[:, :2]
    d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01; den[den == 0] = 1e-12
    vv = (d11 * d20 - d01 * d21) / den; ww = (d00 * d21 - d01 * d20) / den; uu = 1 - vv - ww
    ins = (uu >= -0.02) & (vv >= -0.02) & (ww >= -0.02)
    z = (a[:, 2] + vv * (b[:, 2] - a[:, 2]) + ww * (c[:, 2] - a[:, 2]))[ins]
    z = z[z <= zceil]
    return float(z.max()) if z.size else None


def reseat_on_counter(name):
    prim = stage.GetPrimAtPath(f"/World/{name}")
    box = bcache.ComputeWorldBound(prim).ComputeAlignedBox(); mn, mx = box.GetMin(), box.GetMax()
    cx, cy, base, top = 0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1]), float(mn[2]), float(mx[2])
    surf = _surface_under(cx, cy, top)
    if surf is None:
        print(f"[fluid] reseat {name}: no counter found under it, left as-is", flush=True); return
    # the invisible foot is excluded from the bbox, so the cup's TRUE lowest point is
    # (glass base - foot protrusion). Land THAT on the counter so the foot doesn't
    # interpenetrate (which was launching the empty cup and sloshing fluid out).
    delta = (surf + 0.002) - (base - FOOT_PROTRUSION)
    xf = UsdGeom.Xformable(prim)
    op = next((o for o in xf.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    if op is None:
        op = xf.AddTranslateOp(); op.Set(Gf.Vec3d(0, 0, 0))
    cur = op.Get(); op.Set(Gf.Vec3d(cur[0], cur[1], cur[2] + delta))
    print(f"[fluid] reseat {name}: base {base:.3f}->{surf+0.003:.3f} (counter z={surf:.3f}, delta {delta:+.3f})", flush=True)


src_prim = make_sdf_cup(args.source)
tgt_prim = make_sdf_cup(args.target)
print(f"[fluid] SDF + foot on {args.source} (source) + {args.target} (target)", flush=True)

# reseat AFTER the foot exists -> the foot bottom (cup's lowest point) rests on the counter
bcache.Clear()
reseat_on_counter(args.source)
reseat_on_counter(args.target)

# flat high-friction static slab flush under each cup for a reliable contact surface
bcache.Clear()
ensure_support(args.source, src_prim)
ensure_support(args.target, tgt_prim)

# 2) remove the SOURCE's fake liquid cylinder
liq = stage.GetPrimAtPath(f"/World/{args.source}/liquid")
if liq.IsValid():
    stage.RemovePrim(liq.GetPath())
    print(f"[fluid] removed fake liquid cylinder from {args.source}", flush=True)

# 3) GPU dynamics on the physics scene
scene = stage.GetPrimAtPath("/World/PhysicsScene")
sapi = PhysxSchema.PhysxSceneAPI.Apply(scene)
sapi.CreateEnableGPUDynamicsAttr(True)
sapi.CreateBroadphaseTypeAttr("GPU")
print("[fluid] GPU dynamics enabled", flush=True)

# 4) real PBD fluid inside the SOURCE cup cavity (cup has since been reseated)
bcache.Clear()
box = bcache.ComputeWorldBound(src_prim).ComputeAlignedBox()
mn, mx = box.GetMin(), box.GetMax()
cx, cy = 0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1])
base, top = float(mn[2]), float(mx[2])
half = 0.18 * min(mx[0] - mn[0], mx[1] - mn[1])   # inset so grid CORNERS clear the side walls
FRO = args.fro
PCO = args.pco
particleUtils.add_physx_particle_system(
    stage, "/World/fluidSystem", simulation_owner="/World/PhysicsScene",
    contact_offset=PCO, rest_offset=PCO, particle_contact_offset=PCO,
    solid_rest_offset=FRO, fluid_rest_offset=FRO,
    enable_ccd=True, solver_position_iterations=16, max_velocity=8.0,
)
spacing = 2.0 * FRO
# Start WELL above the glass bottom (base = cup min-z = foot bottom, ~6mm below the
# glass base, whose interior floor is ~2cm higher still). Spawning in free space up
# in the cavity lets particles FALL and settle, instead of starting embedded in the
# solid glass base and phasing straight through it.
lower = Gf.Vec3f(cx - half, cy - half, base + 0.050)
nx = ny = max(3, int((2 * half) / spacing))
nz = 6
_pos, _vel = particleUtils.create_particles_grid(lower, spacing, nx, ny, nz)
positions = Vt.Vec3fArray(_pos)
vels = Vt.Vec3fArray(_vel)
particleUtils.add_physx_particleset_pointinstancer(
    stage, Sdf.Path("/World/fluid"), positions, vels, Sdf.Path("/World/fluidSystem"),
    self_collision=True, fluid=True, particle_group=0,
    particle_mass=0.0, density=1000.0, num_prototypes=1,
)
# color/size the fluid spheres
inst = UsdGeom.PointInstancer(stage.GetPrimAtPath("/World/fluid"))
for pt in inst.GetPrototypesRel().GetTargets():
    s = UsdGeom.Sphere(stage.GetPrimAtPath(pt))
    if s:
        s.CreateRadiusAttr(float(FRO * 1.5))
        s.GetPrim().CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set(
            Vt.Vec3fArray([Gf.Vec3f(0.10, 0.45, 0.95)]))
print(f"[fluid] added {len(positions)} fluid particles in {args.source} "
      f"(cavity center {cx:.3f},{cy:.3f} base z={base:.3f})", flush=True)

stage.GetRootLayer().Save()
print(f"[fluid] wrote {args.out_usd}", flush=True)
app.close()
