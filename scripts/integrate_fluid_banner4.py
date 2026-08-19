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
ap.add_argument("--pellets", type=int, default=450, help="target fluid particle count")
ap.add_argument("--iso", type=int, default=1, help="1: skin a smooth isosurface over the fluid (visual only)")
ap.add_argument("--src_xy", default="", help="move source cup center to this world 'x,y' (e.g. counter front edge)")
ap.add_argument("--tgt_xy", default="", help="move target cup center to this world 'x,y'")
ap.add_argument("--counter_front_y", type=float, default=None,
                help="world y of the counter's front edge; the contact slab is clamped behind it")
args = ap.parse_args()

# How far the invisible ground-contact foot sticks below the glass base. Keep SMALL:
# this is literally how high the glass visually floats above the counter. Any value
# works geometrically (the foot's flat bottom is always below the domed base tip);
# 6 mm read as an obvious hover in the GUI.
FOOT_PROTRUSION = 0.002

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

# 1a-bis) optionally move the cups (world XY of the GLASS CENTER). Added so the
# cups can sit at the counter's FRONT edge, where a robot standing on open floor
# can reach them — the original reconstruction placement is at the counter's back,
# reachable only by standing inside the kneehole (= interpenetrating the counter,
# which the overlap sanity gate forbids).
for _nm, _spec in ((args.source, args.src_xy), (args.target, args.tgt_xy)):
    if not _spec:
        continue
    _tx, _ty = [float(v) for v in _spec.split(",")]
    _prim = stage.GetPrimAtPath(f"/World/{_nm}")
    _b = bcache.ComputeWorldBound(_prim).ComputeAlignedBox()
    _cx, _cy = 0.5 * (_b.GetMin()[0] + _b.GetMax()[0]), 0.5 * (_b.GetMin()[1] + _b.GetMax()[1])
    _xf = UsdGeom.Xformable(_prim)
    _op = next(o for o in _xf.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeTranslate)
    _v = _op.Get()
    _op.Set(Gf.Vec3d(_v[0] + (_tx - _cx), _v[1] + (_ty - _cy), _v[2]))
    print(f"[fluid] moved {_nm}: center ({_cx:.3f},{_cy:.3f}) -> ({_tx:.3f},{_ty:.3f})", flush=True)
    bcache.Clear()


def build_counter_slab(surfs, centers):
    """ONE invisible static box whose top is flush with the measured counter surface,
    spanning the counter work area around the cups. The cups rest on THIS, never on
    the reconstruction's decomposed convex hulls: contact-report tracing showed the
    hull is a noisy contact partner (faceted edges, ±0.4 mm height noise) that kept
    kicking the empty cup into a permanent skitter. Pairs cup<->cabinetry are
    broadphase-filtered (see filter_cup_vs_counter), so this slab IS the counter for
    the cups — make it big enough to cover anywhere the robot may set a cup down."""
    top = float(np.mean(surfs)) + 0.0005
    cx = float(np.mean([c[0] for c in centers]))
    cy = float(np.mean([c[1] for c in centers]))
    half_x = 0.30
    # the slab must NEVER extend past the counter's front edge: invisible support
    # hanging in open air is a scene lie (a cup set there would float; the robot's
    # torso can graze it). Clamp the +y side at the counter edge.
    y_lo = cy - 0.30
    y_hi = min(cy + 0.30, args.counter_front_y - 0.005) if args.counter_front_y is not None else cy + 0.30
    cy_s, half_y = 0.5 * (y_lo + y_hi), 0.5 * (y_hi - y_lo)
    sp = "/World/counter_contact_slab"
    cube = UsdGeom.Cube.Define(stage, sp)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube.GetPrim()); xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy_s, top - 0.01))     # half-height 0.01 -> top flush
    xf.AddScaleOp().Set(Gf.Vec3f(2 * half_x, 2 * half_y, 0.02))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(grip, materialPurpose="physics")
    UsdGeom.Imageable(cube.GetPrim()).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    print(f"[fluid] counter contact slab: top z={top:.4f}, {2*half_x:.2f} x {2*half_y:.2f} m at ({cx:.3f},{cy_s:.3f})", flush=True)
    return top


def filter_cup_vs_counter(name):
    """Make the cup ignore the cabinetry's decomposed hull colliders entirely.
    The flush counter_contact_slab stands in for the counter surface."""
    cup = stage.GetPrimAtPath(f"/World/{name}")
    box = bcache.ComputeWorldBound(cup).ComputeAlignedBox()
    mn, mx = box.GetMin(), box.GetMax()
    cx, cy, base = 0.5 * (mn[0] + mx[0]), 0.5 * (mn[1] + mx[1]), float(mn[2])
    targets = set()
    for p in stage.Traverse():
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            continue
        pp = str(p.GetPath())
        top_owner = "/".join(pp.split("/")[:3])
        if top_owner.startswith(f"/World/{name}") or "labg" in top_owner or "slab" in pp:
            continue
        m = UsdGeom.Mesh(p)
        if not m or not m.GetPointsAttr().Get():
            continue
        M = np.array(_xc.GetLocalToWorldTransform(p), dtype=np.float64)
        V = (np.array([[q[0], q[1], q[2], 1.0] for q in m.GetPointsAttr().Get()]) @ M)[:, :3]
        if (V[:, 0].min() < cx < V[:, 0].max() and V[:, 1].min() < cy < V[:, 1].max()
                and V[:, 2].min() < base and V[:, 2].max() > base - 0.30):
            targets.add(top_owner)
    api = UsdPhysics.FilteredPairsAPI.Apply(cup)
    rel = api.CreateFilteredPairsRel()
    for t in sorted(targets):
        rel.AddTarget(Sdf.Path(t))
    print(f"[fluid] {name}: collision-filtered against {sorted(targets)}", flush=True)


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
        print(f"[fluid] reseat {name}: no counter found under it, left as-is", flush=True); return None
    # the invisible foot is excluded from the bbox, so the cup's TRUE lowest point is
    # (glass base - foot protrusion). Land THAT just above the counter so the foot
    # neither interpenetrates (depenetration launch) nor hovers visibly.
    delta = (surf + 0.0005) - (base - FOOT_PROTRUSION)
    xf = UsdGeom.Xformable(prim)
    op = next((o for o in xf.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    if op is None:
        op = xf.AddTranslateOp(); op.Set(Gf.Vec3d(0, 0, 0))
    cur = op.Get(); op.Set(Gf.Vec3d(cur[0], cur[1], cur[2] + delta))
    print(f"[fluid] reseat {name}: glass base -> {surf + 0.0005 + FOOT_PROTRUSION:.4f} "
          f"(counter z={surf:.4f}, delta {delta:+.4f})", flush=True)
    return surf


src_prim = make_sdf_cup(args.source)
tgt_prim = make_sdf_cup(args.target)
print(f"[fluid] SDF + foot on {args.source} (source) + {args.target} (target)", flush=True)

# reseat AFTER the foot exists -> the foot bottom (cup's lowest point) rests on the counter
bcache.Clear()
src_surf = reseat_on_counter(args.source)
tgt_surf = reseat_on_counter(args.target)

# one flush counter slab for both cups + broadphase-filter the noisy cabinetry hulls
bcache.Clear()
if src_surf is not None and tgt_surf is not None:
    _cs = []
    for _n in (args.source, args.target):
        _b = bcache.ComputeWorldBound(stage.GetPrimAtPath(f"/World/{_n}")).ComputeAlignedBox()
        _cs.append((0.5 * (_b.GetMin()[0] + _b.GetMax()[0]), 0.5 * (_b.GetMin()[1] + _b.GetMax()[1])))
    _slab_top = build_counter_slab([src_surf, tgt_surf], _cs)
    # re-land each cup on the SLAB top (the mean-height slab can sit above a cup's
    # locally-measured counter surface -> spawn interpenetration -> kick)
    bcache.Clear()
    for _n in (args.source, args.target):
        _p = stage.GetPrimAtPath(f"/World/{_n}")
        _b = bcache.ComputeWorldBound(_p).ComputeAlignedBox()
        _foot_bottom = float(_b.GetMin()[2]) - FOOT_PROTRUSION
        _dz = (_slab_top + 0.0005) - _foot_bottom
        if abs(_dz) > 1e-6:
            _xf = UsdGeom.Xformable(_p)
            _op = next(o for o in _xf.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeTranslate)
            _v = _op.Get(); _op.Set(Gf.Vec3d(_v[0], _v[1], _v[2] + _dz))
            print(f"[fluid] {_n}: re-landed on slab ({_dz:+.4f})", flush=True)
    filter_cup_vs_counter(args.source)
    filter_cup_vs_counter(args.target)

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
# author the physics rate INTO the asset: GUI Play (and any loader that honors the
# stage) defaults to 60 Hz when unauthored, and at 60 Hz the cup/counter contact
# solve is unstable — the cup gets launched off the counter at t=0, flinging the
# fluid. At 120 Hz the same scene is stable (A/B'd 2026-08-07).
sapi.CreateTimeStepsPerSecondAttr(120)
print("[fluid] GPU dynamics enabled, physics rate authored at 120 Hz", flush=True)

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
# Circle-packed spawn column: a square grid's CORNERS cap the count (they hit the SDF
# side walls first), a disc doesn't. Radius matches the proven square-corner envelope
# (0.26*width ~ the old half*sqrt(2)). Spawn in free air above the interior floor so
# nothing starts embedded in the solid glass base (finding #8).
r_spawn = 0.26 * min(mx[0] - mn[0], mx[1] - mn[1])
z0 = base + 0.030
per_layer = sum(1 for ix in range(-8, 9) for iy in range(-8, 9)
                if (ix * spacing) ** 2 + (iy * spacing) ** 2 <= r_spawn ** 2)
zmax_spawn = top - 0.008  # keep the column top under the rim
nz = min(max(1, -(-args.pellets // per_layer)), int((zmax_spawn - z0) / spacing))
pts = []
for iz in range(nz):
    z = z0 + iz * spacing
    for ix in range(-8, 9):
        for iy in range(-8, 9):
            x, y = ix * spacing, iy * spacing
            if x * x + y * y <= r_spawn ** 2:
                pts.append(Gf.Vec3f(cx + x, cy + y, z))
positions = Vt.Vec3fArray(pts)
vels = Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 0.0)] * len(pts))
print(f"[fluid] spawn: {per_layer}/layer x {nz} layers, r={r_spawn*1000:.1f}mm, "
      f"z {z0:.3f}..{z0 + nz * spacing:.3f} (rim {top:.3f})", flush=True)
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

# 5) optional liquid look: PhysX isosurface — a smooth mesh skinned over the particles
# every frame ON THE GPU. Purely a render-side layer: the same 444 pellets simulate
# underneath, so containment/stability results carry over unchanged.
if args.iso:
    ps_prim = stage.GetPrimAtPath("/World/fluidSystem")
    iso = PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(ps_prim)
    iso.CreateIsosurfaceEnabledAttr(True)
    # gridSpacing / surfaceDistance / gridSmoothingRadius left at schema default
    # (-inf = auto-derived from the particle system's offsets)
    water = UsdShade.Material.Define(stage, "/World/Looks/waterIso")
    wsh = UsdShade.Shader.Define(stage, "/World/Looks/waterIso/Shader")
    wsh.CreateIdAttr("UsdPreviewSurface")
    wsh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.08, 0.38, 0.85))
    wsh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.55)
    wsh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.02)
    wsh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.33)
    water.CreateSurfaceOutput().ConnectToSource(wsh.ConnectableAPI(), "surface")
    # the runtime-generated isosurface mesh lives under the particle system and
    # inherits its material binding; the raw pellet spheres get hidden
    UsdShade.MaterialBindingAPI(ps_prim).Bind(water)
    UsdGeom.Imageable(stage.GetPrimAtPath("/World/fluid")).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    print("[fluid] isosurface enabled + water material bound (pellet spheres hidden)", flush=True)

stage.GetRootLayer().Save()
print(f"[fluid] wrote {args.out_usd}", flush=True)
app.close()
