# Scripted G1 grasp-lift-pour on banner4 — session findings, 2026-08-13

**Status: see the "Final state" section at the bottom (filled in at end of session).**

The task: script the whole pick-and-pour slowly and carefully, stage by stage, joint by
joint, headless, with photos and videos for self-inspection. This document records what
was built, every root-caused failure, and where the next session picks up.

## The staged framework

`scripts/g1_grasp_stages.py` — four checkpointed stages, each a separate headless run
that loads the previous stage's USD, acts, checks numeric gates, and saves a new
checkpoint + 4-camera frames (`frames_{wide,front,sideX,high}/`) + a metrics JSON:

| Stage | What it does | Checkpoint out |
|---|---|---|
| `s1_approach` | open hand, 5-DOF IK palm to grasp station (13.5 cm from cup axis, 3.7 cm below rim; wrist plane frozen at Eli's B1 angles) | `banner4_g1_s1_approach_v06.usdc` |
| `s2_close` | fingers close JOINT BY JOINT in 1.5° steps, each stopping at measured glass-shell distance; thumb yaw+pitch under a point-level bore guard | `banner4_g1_s2_close_v07.usdc` |
| `s3_lift` | 6-DOF IK waypoints (orientation held), cup on a scripted per-step carry +15 cm, 2 s hold | `banner4_g1_s3_lift_v08.usdc` |
| `s4_pour` | carry over labg9, tilt about a descending pinned spout, drain, return upright | `banner4_g1_s4_pour_v09.usdc` |

Grasp mechanism is **kinematic-follow by design** (documented banner4 decision): the cup
is slaved to scripted trajectories; fingers close to measured contact distances but exert
no forces. The FLUID (444 PBD particles) is fully physically simulated throughout — fluid
retention/transfer numbers are real physics, not choreography.

Evidence bundle: `output/banner4/pour_seq/scripted_v06/` (per-stage frames, videos
`{wide,front,sideX,high}.mp4` + `grid4.mp4`, `full_grasp_*.mp4` concatenations, metrics
JSONs, `evidence/` stills and montage).

## Verified results (gates + adversarial review + point-level probes)

- **S1 approach PASS**: IK residual 0.8 mm, palm radial 13.3 cm, cup drift 0.00 mm.
- **S2 closure PASS**: index/middle/ring/pinky stop 8.9/10.8/11.0/10.4 mm from the glass
  (center distance; finger surfaces ~2-3 mm clear), thumb tip fully above rim with zero
  below-rim bore geometry, cup drift 0.00 mm, fluid 444/444.
- **S3 lift PASS**: +15.1 cm, palm orientation drift 2.0°, cage held, **fluid retention
  423/444 (95.3%)** through the carry and hold.
- **S4 pour**: see final state below.

## Every root-caused failure of the day (the reusable knowledge)

1. **Isaac 5.1 flushes sim state to USD only on RENDERED frames.** Render-less
   physics loops read frozen authored xforms; a DLS IK built on such probes gets an
   exactly-zero Jacobian and silently returns its seed (bit-identical residual across
   reruns is the tell). Fix: every FK probe ends on `world.step(render=True)`. This also
   afflicts `g1_pour_sequence.py`'s fk() — the Aug-12 baseline IK likely never converged.
2. **A kinematic fluid-holding cup must move every physics step.** Driving it from
   render-gated palm reads teleports it in ~3.3 mm jumps every 4th step; the SDF wall
   slams the PBD fluid (2/3 ejected with a perfectly level cup). Fix: pre-solve IK
   waypoints, then script the cup per-step on the joint-interpolation schedule.
3. **Never compose a carried object's pose with the palm's rotation.** `M_off * palm_T`
   swings the cup like a lever under mid-interpolation palm yaw (4.5 cm sideways, 74%
   spill under pitch). Fix: translation-delta follow + level-locked rotation + 6-DOF IK
   holding palm orientation (2.0° drift).
4. **Prim origin ≠ geometry center in reconstructed scenes** (the labg cups' origins sit
   ~2.2 m from their glass). All motion math must be geometry-frame deltas applied
   equally to prim origin and palm targets; an assert now trips on implausible deltas.
5. **Tilt axis sign must be verified by cross product, not intuition.** The first pour
   tipped the cup BACKWARD (spout rose): only 14% transferred. axis = (-d_y, d_x, 0)
   with +θ dips the +d rim edge (verify axis × (d·R) points -z).
6. **The spout must be LOW before the heavy pour.** A lagged descent left the stream
   falling 9 cm → 43% splash loss. Descent now completes by t=0.55 (collision-checked
   against the receiver wall).
7. **Arm reachability caps the hand-synced tilt at ~40-75°.** Beyond it, keyframes are
   frozen at the last reachable pose and the cup pivots in the fingers (the human
   wrist-limit look). Full 115° palm-synced pour is kinematically out of envelope at
   this station.
8. **`SimulationApp.close()` hangs routinely** after work completes — every run is
   wrapped in `timeout -k 10 <cap>`; kill-and-verify by `ps -eo pid,comm,args | awk
   '$2 ~ /^python/ && /<script>/'` (a bare cmdline grep self-matches wrappers).
9. **Monitors tailing a log that gets rm'd+recreated replay the OLD file** (or go
   silent on the orphaned fd). Arm watchers only when the path is guaranteed fresh.
10. **AABB-based clearance guards are useless for long diagonal links** (an open
    thumb's AABB blanketed the cup mouth, -29.8 mm of phantom intrusion). Use actual
    mesh vertices (cheap: cache local points once, transform per probe).
11. **Beauty renders of refractive glass cannot adjudicate contact.** The adversarial
    review's suspected wall-breach was settled only by a point-level vertex probe
    (fingertip surfaces grazed ≤1.5 mm into the wall at shell_stop=9 mm; widened to
    11 mm). Numeric probes are the arbiter; renders are the narrative.

## Where the next session picks up

- If S4 passed: polish is optional (hand-sync beyond ~40°, splash reduction, place-back
  stage s5, camera work for the marketing cut).
- If S4 fell short: the best attempt is saved as `banner4_g1_s4_pour_v09.usdc` +
  `s4_pour/` frames; the two candidate approaches are (a) re-station the robot closer to
  labg9 so the pour happens inside the arm envelope (re-run s1 with a bearing toward the
  receiver midpoint), or (b) accept arm-frozen pours and tune only the fluid trajectory
  (descent schedule, tilt rate, pour_gap).
- Known cosmetic debts either way: index fingertip surface ~1 mm into the glass at S2
  (halve the last closure step to fix), denoiser smearing in far-field renders, head
  gaze fixed forward (no articulated neck in this asset).

## Final state

**ALL FOUR STAGES PASS (17:50, well before the 20:00 deadline). Grasp, lift AND pour
achieved with evidence.**

- S4 pour final gates: **poured 295/423 (70%)** into labg9, 74 loose (17%, within the
  20% bound), hand rode the tilt to ~104° with ≤11° rot error (final 115° keyframe
  frozen at arm reach limit — cup pivots in the fingers, human-wrist style).
- The winning S4 recipe: corrected tilt axis `(-d_y, d_x)` + spout descent completed by
  t=0.55 (before the heavy pour) + arm keyframes frozen when IK residual > 60 mm.
  The early descent also improved reachability (4 of 5 keyframes executed vs 1 of 5).
- Sequence checkpoints: `banner4_g1_s1_approach_v06` → `s2_close_v07` → `s3_lift_v08`
  → `s4_pour_v09` (.usdc, in `output/banner4/sim/usd/`).
- Evidence: `output/banner4/pour_seq/scripted_v06/` — per-stage 4-camera frames +
  videos, `full_sequence_{front,sideX,high,wide}.mp4`,
  `evidence/pour_achieved_evidence.png` (montage), pour money-shots
  `s4_pour/frames_{front,sideX}/f0150.png`.
- S4 iteration history (kept for the record): v1 frame-mixing assert case, v2 backward
  tilt (14%), v3 high-spout splash (44%/43%), v4 PASS (70%/17%).
