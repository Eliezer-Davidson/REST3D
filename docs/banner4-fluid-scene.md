# banner4 — real-scale lab scene with real simulated fluid

**Status (2026-08-06): DONE in headless validation, awaiting one interactive Play test.**

A lab reconstructed by REST3D from the single photo `banner4.jpg`, turned into a
physics-ready Isaac Sim scene: real metric scale, static furniture, and **real PhysX
PBD fluid inside a graspable cup**. The intent is that a robot (e.g. G1) can pick up
the full cup and pour it into the empty one — the grasp/pour is the *robot's* job; this
scene only provides a correct, stable, fluid-holding world for it to act in.

## The deliverable

| Thing | Value |
|---|---|
| Current scene | `output/banner4/sim/usd/banner4_fluid_v13.usdc` |
| Scale | uniform ×2.5 over the raw reconstruction (real metric; bench ≈91 cm, beaker ≈12.5 cm) |
| Source cup (has fluid) | `labg4` — dynamic, SDF collision, ~125 blue PBD particles |
| Target cup (empty) | `labg9` — dynamic, SDF collision |
| Furniture | ~40 bodies frozen static (`rigidBodyEnabled=False`) |
| Both cups at rest | **0.0° tilt, zero drift, fluid contained** (headless, 260 steps) |

> `output/` is git-ignored, so the `.usdc` scenes and render frames live only on disk.
> The committed, reproducible artifact is the **script pipeline** below.

## Pipeline (reproduce from the raw glassware scene)

All Isaac scripts run in the `g1skillgen` env; `rescale_scene.py` is pxr-only and runs in `rest3d`.

```bash
# 1. rescale raw reconstruction to real metric (rest3d env — pxr only)
~/miniconda3/envs/rest3d/bin/python scripts/rescale_scene.py \
    --in_usd  output/banner4/sim/usd/banner4_with_liquids_v11.usdc \
    --out_usd output/banner4/sim/usd/banner4_scaled_v13.usdc --scale 2.5

# 2. make physics-ready + add real fluid to labg4 (g1skillgen env — Isaac)
~/miniconda3/envs/g1skillgen/bin/python scripts/integrate_fluid_banner4.py \
    --in_usd  output/banner4/sim/usd/banner4_scaled_v13.usdc \
    --out_usd output/banner4/sim/usd/banner4_fluid_v13.usdc --fro 0.0035

# 3. validate headless: settle physics, render overview + cup closeup
~/miniconda3/envs/g1skillgen/bin/python scripts/validate_v12.py \
    --usd output/banner4/sim/usd/banner4_fluid_v13.usdc --steps 260 --every 15
# frames -> output/banner4/fluid_check/{overview,cups}_NNN.png

# 4. open interactively (GUI) to press Play
~/miniconda3/envs/g1skillgen/bin/python output/_pipeline/view_isaac.py \
    --scene output/banner4/sim/usd/banner4_fluid_v13.usdc
```

Scripts committed for this arc:
- `integrate_fluid_banner4.py` — **the deliverable**: static furniture, SDF cups + foot + low COM, PBD fluid, GPU dynamics.
- `rescale_scene.py` — uniform metric rebake (multiply geometry, then *recompute* extents).
- `validate_v12.py` — headless settle + 2-camera render check.
- `pour_demo_isaac.py` — standalone proven 2-beaker pour demo (fluid mechanism reference).
- `render_isaac_headless.py` — multi-camera RTX render with the near-clip fix.
- `add_glass_liquid_visuals_usd.py`, `fix_labg_physics_contact_usd.py`,
  `inject_mujoco_glassware_to_assets.py`, `rearrange_banner4_glassware_usd.py` —
  the earlier (Aug 5) glassware-editing steps that produced `..._v11.usdc`.

## Key findings today (the hard-won physics traps)

Full writeup in the vault memory note `isaac-sdf-fluid-cup-stability.md`. In the order they bite:

1. **PhysX SDF-mesh collision REQUIRES GPU dynamics.** With it off, the SDF collider is
   silently broken — the cup drifts/tips as if it has no floor. Same flag gates particles.
2. **`World()` loading an existing stage does NOT apply the authored `enableGPUDynamics`.**
   Force it at runtime: `world.get_physics_context().enable_gpu_dynamics(True)`. This masqueraded
   as a scene bug for many iterations — the USD had the flag set, but the harness ran with it off.
3. **Instrument, don't eyeball.** A pose trace (log world pos + tilt each step, `render=False`)
   showed the cups settling at a *stable* ~11° — impossible for a flat base → the SDF gives a
   **domed/ring base contact**.
4. **Fix the domed base with a thin flat "foot"** cylinder collider at the base (top flush with the
   glass base — must not poke into the cavity or it ejects fluid), protruding ~6 mm below so IT is
   the ground contact. Keep SDF for walls/fluid.
5. **The invisible foot is excluded from `BBoxCache`** → a bbox-based reseat buries the foot in the
   counter → depenetration kick launches the light empty cup. Reseat the TRUE lowest point
   (`glass_base − FOOT_PROTRUSION`).
6. **Fluid must spawn clear of the glass**, not referenced to the cup's min-z (which is the foot,
   below the glass base). v13 spawns the grid at `base + 0.050`, inset `0.18·width`, 6 Z-layers,
   SDF res 320 — otherwise the bottom layer starts inside the solid base and gets ejected ("phases
   through" on Play).
7. **Monocular reconstruction has no metric scale** → banner4 came out ~40% real; ×2.5 fixes it.
   Rebake by multiplying geometry then *recomputing* extents (hand-scaling extents double-scales Cubes).

## Where to proceed tomorrow

1. **Confirm the Play test.** The GUI was left open on `banner4_fluid_v13.usdc`. Press Play and watch
   ~10–20 s: cups should stay at 0° and the fluid stay in `labg4`. Headless validation already passes;
   this only checks the interactive path.
2. **If it misbehaves on Play:** the GUI's Play isn't running GPU dynamics. Patch
   `output/_pipeline/view_isaac.py` to force `get_physics_context().enable_gpu_dynamics(True)` +
   `set_broadphase_type("GPU")` after world creation, mirroring `validate_v12.py`. (This is the single
   most likely remaining issue — it is finding #2 reappearing in a different harness.)
3. **Robot pick-and-pour (the actual goal, robot's job):** drop a G1 into this scene. The loader
   **must** force GPU dynamics on at load — the authored flag is not reliably applied. Without it the
   cups have no collision and the fluid won't simulate.
4. **Optional polish** (not requested): more particles for a fuller pour; brighter liquid color;
   a second full/empty pair if multiple pour targets are wanted.

## Caveat to carry forward

The one thing that will bite anyone reusing this scene: **whatever loads the USD must explicitly
enable GPU dynamics at runtime.** The flag is authored in the file, but `isaacsim.core.api.World`
(and likely other loaders) does not apply it when opening an existing stage. Every harness here
that works (`validate_v12.py`, `pose_trace.py`) forces it on; `view_isaac.py` does not yet.
