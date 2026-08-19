# banner4 — real-scale lab scene with real simulated fluid

**Status (2026-08-14 night): REAL-CONTACT TASK CHAIN day 1 — m0 (side rest) and m1
(approach) PASS with full gates; m2 grasp reaches a real partial grip (cup rises 6 mm,
tilt-escape blocker documented). Full findings + 11 engine laws:
`docs/contact-task-2026-08-14.md`. Earlier same day: CONTACT SANDBOX built — the
kinematic era is over.** Eli rejected
the kinematic-follow mechanism (hand phased through the cup, cup tethered). Audit found
ALL 58 G1 colliders authored `collisionEnabled=False` and both cups `kinematic=True`.
`scripts/g1_contact_sandbox.py` (prep/settle/push) enables the 21 right-arm+hand
colliders, flips the cups dynamic, and proved it: 2 s settle (cup 0.00° on its foot,
444/444 fluid, arm stable), then a lateral hand sweep pushed the cup **109.6 mm by
contact alone** (max inter-frame jump 3.1 mm, direction·push=0.92; cup tips over at the
end — real physics; the script never writes a cup pose). Reach lesson: pushing along
palm→cup (−Y) hits the arm envelope (57–83 mm IK plateaus, same wall as the s4
keyframes); LATERAL waypoints at constant reach all solve at ≤1.4 mm. Scene:
`banner4_contact_sandbox_v01.usdc` (pre-push, upright) / `banner4_contact_push_v02.usdc`
(post-push); evidence `pour_seq/contact_sandbox_v01/`. Next: re-script grasp-lift-pour
against contact, starting with finger closure that actually grips.

**v02/v03 (2026-08-14 afternoon, Eli's GUI review):** v02 bakes the push sweep as
TIME-SAMPLED drive targets (5 arm joints, t=1–5 s, 60 tps) so GUI Play runs the knock on
the asset's own stiffness-70 drives — verified on the pure timeline path (no `World()`):
palm 117 mm, cup knocked 105 mm. Two visible-vs-invisible-geometry defects found and
fixed in **v03** (current): (a) labg9's spawn straddled first the slab edge (fell to the
floor — kinematic cups had masked the bad spawn forever) and then the VISIBLE counter's
diagonal corner (scanned top ends ~x=−1.20 in the cups' y-band); now at (−1.31,−5.955),
probed stable 15 s. (b) Eli's "liquid pours out and FREEZES" was neither a physics nor
an iso bug — probes show particles never sleep (0 bit-identical over 15 s) and the iso
mesh tracks them to ~5 mm; the frozen-looking tongue was fluid at rest ON THE INVISIBLE
SLAB, which overhung the visible corner by ~19 cm. The slab is now shaped to the
measured visible counter (x[−1.613,−1.19], y[−6.32,−5.412]); acceptance probe: knock
106 mm, labg9 exact, 0 particles resting past the visible corner. LAW (extends finding
#10): every invisible proxy collider must match the VISIBLE geometry's footprint — any
overhang eventually renders as "objects/fluid floating in air", any shortfall as
"falling through the table"; kinematic bodies mask bad spawns until the day they go
dynamic. No PBD material is authored on the fluid (all defaults) — cohesion/viscosity
are NOT available as an explanation for odd fluid rest shapes in this scene.

**Status (2026-08-13 evening, SUPERSEDED — kinematic-follow): PICK-LIFT-POUR ACHIEVED, scripted + gated.
`g1_grasp_stages.py` runs four checkpointed stages (s1_approach → s2_close → s3_lift →
s4_pour); the lift retains 423/444 PBD particles (95.3%) and the 115° pour transfers
295/423 (70%) into labg9 with 17% loose. Checkpoints `banner4_g1_s{1..4}_*_v0{6..9}.usdc`;
evidence under `pour_seq/scripted_v06/`; full findings in
`docs/scripted-grasp-pour-2026-08-13.md`. Remaining polish: set-back-down stage,
hand-sync past 104°, splash reduction.**

**Harness findings 2026-08-13 (bite anyone scripting against this scene):**
1. **Sim state reaches USD only on RENDERED frames.** `world.step(render=False)` loops
   read FROZEN authored xforms through XformCache/BBoxCache — an IK built on such probes
   gets a exactly-zero Jacobian and silently returns its seed (this also afflicted
   `g1_pour_sequence.py`'s fk(); its baseline IK likely never converged). Any measurement
   probe must end on `world.step(render=True)`, or read the physics view directly.
2. **A kinematic fluid-holding cup must move a hair every physics step.** Driving it from
   render-gated palm reads teleports it in ~3.3 mm jumps every 4th step; the SDF wall
   slams the PBD fluid and ejects 2/3 of it even with the cup perfectly level. The fix
   that passes: pre-solve all IK waypoints, then script the cup's carry trajectory
   per-step on the joint interpolation schedule (endpoint-matched to the palm, ≤1.4 mm).
3. **Carry follows must not compose with palm rotation.** `M_off * palm_T` swings the cup
   like a lever under mid-interpolation palm yaw (measured 4.5 cm sideways) and tilts it
   under pitch (74% spill). Use translation-delta + level-locked rotation, and hold palm
   orientation with 6-DOF IK (drift gated < 8°, achieved 2.0°).
4. **`SimulationApp.close()` hangs routinely after work completes** — wrap every headless
   run in `timeout -k 10 <cap>` and kill-and-verify by PID before any relaunch (match
   `ps -eo pid,comm,args | awk '$2 ~ /^python/ && /<script>/'`, never a bare cmdline grep).

A lab reconstructed by REST3D from the single photo `banner4.jpg`, turned into a
physics-ready Isaac Sim scene: real metric scale, static furniture, and **real PhysX
PBD fluid inside a graspable cup**. The intent is that a robot (e.g. G1) can pick up
the full cup and pour it into the empty one — the grasp/pour is the *robot's* job; this
scene only provides a correct, stable, fluid-holding world for it to act in.

## The deliverable

| Thing | Value |
|---|---|
| Current scene | `output/banner4/sim/usd/banner4_fluid_v18.usdc` |
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
    --out_usd output/banner4/sim/usd/banner4_fluid_v14.usdc --fro 0.0035

# 3. validate headless: settle physics, render overview + cup closeup
~/miniconda3/envs/g1skillgen/bin/python scripts/validate_v12.py \
    --usd output/banner4/sim/usd/banner4_fluid_v14.usdc \
    --out_dir output/banner4/fluid_check_v14 --steps 260 --every 15
# frames -> output/banner4/fluid_check_v14/{overview,cups}_NNN.png

# 4. open interactively (GUI) to press Play — scene path is POSITIONAL (no --scene flag;
#    view_isaac.py reads sys.argv[1], and a flag there is silently "opened" as a file)
~/miniconda3/envs/g1skillgen/bin/python output/_pipeline/view_isaac.py \
    output/banner4/sim/usd/banner4_fluid_v14.usdc
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
8. **Author `timeStepsPerSecond` INTO the asset (v13→v14, 2026-08-07).** v13 authored no physics
   rate, so GUI Play ran the kit default **60 Hz** — and at 60 Hz the cup/counter contact solve
   explodes: the cup is launched ~1 m off the counter at t=0 (before the fluid even lands), flinging
   the particles. That was the "water jumps out when I press Play." The same scene at **120 Hz** is
   rock stable. Not a fluid problem at all: spawn height, spacing (already exactly 2×fluidRestOffset
   = rest distance), and drop-vs-emit were all red herrings. A/B'd headless both ways AND through a
   timeline-Play probe (`probe_gui_play.py` pattern: `timeline.play()` + `sim.update()`, no `World()`,
   no forced dt — the honest reproduction of pressing Play). v14 fix:
   `PhysxSchema.PhysxSceneAPI.CreateTimeStepsPerSecondAttr(120)`. Result 96/96 particles contained,
   cup drift 0.000 m.
9. **The GUI honors authored physics-scene attrs; python `World()` overrides them.** The timeline-Play
   probe confirmed authored `enableGPUDynamics`/`broadphaseType` ARE picked up by the GUI path (the
   particles simulate without any runtime forcing) — the earlier "must force GPU dynamics" caveat
   applies to `World()`-based harnesses, which also override the authored timestep with their
   `physics_dt` argument. So: assets carry the truth for the GUI; every `World()` harness must
   replicate it (`physics_dt=1/120`, `enable_gpu_dynamics(True)`, broadphase GPU).
10. **The reconstruction's decomposed convex-hull colliders are BAD contact partners for small
   dynamic bodies.** The empty cup skittered violently (bounced meters) whenever its foot could
   interact with the cabinetry hull — faceted edges + ±0.4 mm height noise; its twin 25 cm away
   sat still on a 0.1 mm coin-flip. Shim-level fixes (thicker foot, prouder slab, no slab, no SDF,
   moved cup) ALL failed; contact-report tracing (`PhysxContactReportAPI`) found the flickering
   foot↔hull pair. Real fix (v16): `UsdPhysics.FilteredPairsAPI` on each cup vs the cabinetry prim
   + ONE invisible slab (60 cm square, top flush with the measured counter surface) as the cups'
   only floor. Both cups: 0.00 mm oscillation. Corollary: anything the robot must set down on
   scanned furniture wants a flush proxy slab + filter, not the raw hulls.
11. **Isosurface renders in the GUI viewport but NOT in headless replicator captures.** v18 applies
   `PhysxParticleIsosurfaceAPI` (defaults, auto spacing) + UsdPreviewSurface water bound to
   `/World/fluidSystem`, pellets hidden. The runtime mesh (`/World/fluidSystem/Isosurface`,
   ~1k triangles, correct world placement, material bound) is fully formed in USD, yet headless
   RTX captures draw nothing — even displayColor-red with material stripped. GUI viewport renders
   it fine (Eli confirmed: liquid falls and pools). So: judge iso VISUALS only in the GUI; keep
   using particle-position probes for physics (positions still sync to USD with the iso on).

## Where to proceed next

1. ~~Confirm the Play test~~ — **done 2026-08-07 via the timeline-Play probe on v14** (96/96 contained,
   cup still). Optionally eyeball it in the GUI on `banner4_fluid_v14.usdc` for the satisfying visual.
2. **Robot pick-and-pour (the actual goal, robot's job):** drop a G1 into this scene. If the loader
   uses `World()`/SimulationContext it **must** pass `physics_dt=1/120` and force GPU dynamics —
   `World()` overrides both authored values. A plain GUI/timeline load needs nothing: v14 carries
   the correct authored physics.
3. **Optional polish** (not requested): more particles for a fuller pour; brighter liquid color;
   a second full/empty pair if multiple pour targets are wanted.

## Caveat to carry forward

`isaacsim.core.api.World()` **overrides authored physics-scene state** — both GPU dynamics
(finding #2) and the timestep (finding #8). Any `World()`-based harness must set
`physics_dt=1/120.0` and `enable_gpu_dynamics(True)` + GPU broadphase itself. GUI/timeline
loading needs nothing: v14 authors all three.
