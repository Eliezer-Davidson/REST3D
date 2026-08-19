# Real-contact pick-lift-pour on banner4 — session findings, 2026-08-14

**The mandate (Eli):** replace the kinematic-follow demo with REAL rigid+fluid contact
physics end to end — hand from the G1's side, to the cup flank without hitting the
table, grasp, lift, move, tilt-pour into the second glass, un-tilt, replace, return.
Slow and meticulous; every substage gated, rendered, and delivered. Hard stop 24:00.

**Status: see "Final state" at the bottom (filled at wrap-up).**

## The two scenes

- `banner4_contact_sandbox_v03.usdc` — Eli's interactive sandbox: cups dynamic, right
  arm+hand colliders live, baked Play-animation (arm sweeps and knocks the cup on real
  contact). Confirmed by Eli in the GUI.
- `banner4_task_base_v04.usdc` — the SAME scene with the animation stripped at FILE
  level: the only controllable base for scripted work (see finding 1).

## Substage framework

`scripts/g1_contact_task.py` — checkpointed phases, each a headless run: load previous
checkpoint, act, gate, save checkpoint + 4-cam frames + metrics + a Play-able
`*_replay.usdc`. Ghost phases (colliders off, cups kinematic) pre-solve IK waypoints
and scan geometry; real phases never solve IK (probing physically disturbs the world).

| Phase | What | Status at wrap |
|---|---|---|
| m0_side | vertical escape from the cup, park hand at side rest | **PASS** (cups 0.0 mm, 444/444 fluid, 0 strikes, 0.12 rad/s) |
| m1_approach | side → grasp station A | **PASS** (1.7 mm, 0.00 rad/s, cups 0.0 mm) |
| m2_grasp | power wrap: palm jaw + torque-seated fence | see final state |
| m3_lift | +15 cm friction carry | see final state |
| m4–m6 | carry / pour / return | not reached — see final state |

## Root-caused findings (each cost a failed run; all verified by A/B)

1. **A USD carrying time-sampled drive targets is uncontrollable by a `World()`
   harness.** Fabric caches the samples at stage open and re-asserts them every frame;
   `apply_action` is silently overridden (zero-Jacobian IK, arm rigid even at zero
   gains). Runtime `attr.Clear()` does NOT purge the cache — only a FILE-level strip
   works. Timeline-play scenes and scripted-control scenes must be SEPARATE FILES.
   The task script now fails fast on any sampled joint attr.
2. **The torso is the arm's reference frame — stiffen the waist.** Asset-default waist
   drives (~70 stiffness) let every arm swing twist the torso (waist_roll 1.04 rad/s at
   "rest"); every IK solution drifts by centimeters. kp 8000/kd 400 on waist_* turned
   7–80 mm residuals into ≤2.2 mm and rest velocity into 0.09 rad/s.
3. **With fingers straddling a glass, the only clean escape/approach is vertical.**
   Lateral retreat arcs the wrist camera-bracket into the rim (118 mm knock); backward
   retreat rakes the far-side fingers through the glass (300 particles spilled);
   toward-torso retreat at rim height snags fingertips on the mouth (22 rad/s).
   Rise +14 cm beside the glass, then translate outward ALONG THE PALM BEARING.
4. **Force-limited fingers + default hull contact offsets = limit-cycle buzz.** Phantom
   finger-finger contacts rattle weak drives at 12–55 rad/s. Tighten contactOffset to
   0.005/restOffset 0 on all hand colliders and triple the brief's damping.
5. **Invisible proxy geometry must match the visible geometry — in EVERY direction.**
   The contact slab's east face (a 2 cm invisible wall at the visible corner) snagged
   ascending fingertips; its north strip overhangs the visible counter's back edge and
   tripped a too-crude clearance probe on legal under-counter passes. Probe the STRIKE
   BAND (surface ±3.5 cm), not "below top".
6. **Measure grasp geometry on the CURRENT config; briefs go stale.** The design
   brief's antipodal thumb station was probed against the weak-waist torso; with the
   waist stiff, thumb yaw sweeps put the pad OVER THE MOUTH at every standoff/pitch
   (r 4–37 mm, z above rim). True thumb opposition is unreachable at this wrist
   orientation. The measured alternative: **power wrap** — palm face on the near wall
   (standoff 0.095, z 0.945) + fence curled to the far wall (contact ~36° after the
   palm nudges the cup; scan predicted 72–78° for an unmoved cup).
7. **Grasp force flows through the contact link's OWN joint clamp.** The brief's
   lever-matched max_efforts assumed fingertip contact; the wrap contacts the
   intermediate link near its joint, so the intermediate clamp (0.12 N·m) capped the
   squeeze at ~2.4 N/finger and the cup crept (5.9 mm micro-lift slip). Raise the
   clamp on the joint that actually bears the contact.
8. **Close a fence as a unit only until first contact; then seat per-finger by
   torque.** A lone finger advancing solo shoves the cup over (58 mm, 47° tilt);
   stopping ALL fingers at first contact leaves three of them in the air (only
   pinky+palm carried: 7.8 mm slip). Trap first, then seat each finger to its own
   torque stop, then preload into the clamps.
9. **Chained runs must cap each phase tightly** (`timeout -k 10 ~900–1500 s`):
   `SimulationApp.close()` hangs after success and a generous cap wastes the whole
   margin between phases. Kill-verify by PID between links.
10. **`ast.parse` catches syntax, not NameError.** A missing constant killed a full
    Isaac boot 6 s in. Preflight = ast + name-presence sweep for module-level refs.
11. **A grasp equilibrium can be GEOMETRIC, not force-limited — raising clamps then
    changes nothing.** Three runs (m2d/e/f) were bit-identical at efforts
    0.13–0.16 N·m despite clamps of 0.30 → 0.45 (runtime AND pre-parse authored):
    the cup relieves the fence-vs-palm squeeze by TILTING (~11°) about its
    palm-contact edge — a compliant escape channel whose stiffness (gravity
    restoring moment), not the drives, sets the contact force. Bit-identical
    physics across a parameter change is the tell that the parameter is not the
    binding constraint. Fix directions (untested): contact geometry that resists
    the tilt — palm contact HIGHER + fence LOWER so the couple opposes rotation;
    or a second fence contact row (curl the intermediates to touch at a second
    height); or tilt the palm plane a few degrees so the cup wedges instead of
    pivoting.

## Where the next session picks up

- Chain state and checkpoints: `output/banner4/sim/usd/banner4_task_*.usdc`;
  trajectory/waypoint JSON: `output/banner4/pour_seq/contact_task_v01/task_traj.json`;
  per-phase evidence under `output/banner4/pour_seq/contact_task_v01/<phase>/`.
- The design brief (4-lens workflow + synthesis) is preserved in the workflow journal;
  its pour recipe (waist-yaw carry +26°, wrist-roll-only tilt, 0.18 m lift) is still
  the plan for m4–m6 — re-validate its numbers against the stiff-waist config first
  (finding 6).
- Phone evidence hub (artifact, updated per substage):
  https://claude.ai/code/artifact/1f3281c7-de63-48a2-8acb-a865ab778252

## Final state (22:20, wrap-up per the deadline protocol)

**Banked with full gates + 4-cam video + Play-able replays:**
- **m0_side PASS** — vertical escape from the cup, side rest. Cups 0.0 mm, 444/444
  fluid, 0 strikes, 0.12 rad/s. `banner4_task_m0_side_v01.usdc` (+`_replay`).
- **m1_approach PASS** — side → grasp station A at 1.7 mm, 0.00 rad/s, cups 0.0 mm.
  `banner4_task_m1_station_v01.usdc` (+`_replay`).

**m2_grasp best attempt (FAIL by gates, real partial grip):** power wrap at
standoff 0.095/z 0.945 — engage travel clean, palm jaw traps the cup (24.5 mm slide,
0.0° tilt), fence seats by torque (pinky 44°/0.14 N·m; middle+ring 85°), preload holds
3 fingers + palm at 0.13–0.16 N·m. Micro-lift: **the cup rises 6.0 mm in the grip**
(palm +11.8), then slips 5.9 mm at 10.9° tilt — the tilt-escape channel (finding 11).
All 444 particles retained; cup set back upright, still gripped.
`banner4_task_m2_grasp_v01.usdc` + `m2_grasp/grid4.mp4`. m3–m6 not reached.

**Tomorrow's shortest path:** fix the tilt channel (finding 11's three candidate
geometries), re-run m2→m3 with the proven chain commands (logs `m2f`/`m3c` have the
exact invocations), then the design brief's pour plan (waist-yaw carry, wrist-roll
tilt) re-validated per finding 6.
