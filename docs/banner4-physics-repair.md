# banner4 — cup rigid-body phasing: root cause and repair (2026-08-19)

Reported symptom: *"the hands are still phasing through the cups at some points
and the cups phase through each other."*

Both symptoms have **one** root cause: the cups' collision representation.

## What the collider actually was

The cup's collider **is its render mesh** — `/World/labg4/visual`, a 40-gon
revolve of **202 verts / 400 triangles**, tagged `physics:approximation = "sdf"`.
Measured off the shipped USD:

| property | measured | why it matters |
|---|---|---|
| unique z levels | **3** — 0, 7.5, 125 mm | the whole 117.5 mm wall is ONE band of triangles |
| verts on the wall | **0** | 122 at the base, 80 at the rim, nothing between |
| longest triangle edge | **117.7 mm** | 240 of 400 tris have an edge > 20 mm |
| winding | **INCONSISTENT** | 40 directed edges have no opposite twin |
| signed volume | **3677 cm³** vs a **1012 cm³** bbox | 3.6x its own bounding box — impossible for a clean closed mesh |
| `PhysxSDFMeshCollisionAPI` | **never applied** | only the bare attr `sdfResolution=320` is authored, so PhysX ignores it |

Three failure mechanisms follow, and they compound:

1. **No sample points on the wall.** PhysX SDF collision generates contacts by
   sampling one shape's *vertices* against the other's signed distance field.
   Two cup walls meeting have zero vertices between them, so no contacts are
   generated at all — the walls pass through each other.
2. **Undefined SDF sign.** Inconsistent winding leaves the inside/outside test
   undefined in patches, so a shape can read as outside while geometrically
   inside. This is what makes the hand phase *at some points* and not others.
3. **Ignored resolution.** `sdfResolution = 320` has never taken effect.

## The constraint that removes the usual escape hatch

```
[Warning] [isaacsim.core.simulation_manager] Disabling CCD for GPU dynamics as its not supported
```

Isaac disables continuous collision detection outright under GPU dynamics. CCD
is not available here, so **contact generation has to be correct on its own** —
there is no swept-volume safety net behind it.

## The repair

`scripts/fix_cup_colliders.py --mode convex_ring`

The cup is a perfect parametric revolve (cavity r 37.5 mm; outer r 42.5 -> 45.0
mm tapered; base slab 0 -> 7.5 mm; height 125 mm), so it decomposes
**losslessly** into 40 convex wall wedges plus one convex base disc. At
`--segments 40` that is the existing tessellation, so the collider occupies the
identical surface — **the geometry does not change, only its collision
representation.** Every wedge is verified convex (8 of 8 hull vertices).

Convex-vs-convex is PhysX's exact path: full contact manifolds, no vertex
sampling, no SDF sign, no dependence on tessellation density.

The degenerate SDF collider is stripped from `visual`, which becomes
render-only. All parameters are recovered from the mesh itself; nothing is
hardcoded, so the script works on any cup of this family.

## Measured result — `scripts/probe_phasing.py`

Two cups of outer radius 45.0 mm are rigid iff their centre-to-centre distance
never drops below **90.0 mm**. Anything less is interpenetration, measured
directly from physics poses. Contact reports cannot be used for this: a contact
report cannot detect a contact that was never generated, which is the bug.

Push mode = sustained velocity, i.e. a high-authority actuator like a robot arm.

| push speed | shipped SDF | convex ring | |
|---|---|---|---|
| 0.30 m/s | 2.95 mm | 2.63 mm | |
| 1.00 m/s | 12.56 mm | 3.18 mm | -75% |
| 2.00 m/s | **75.63 mm** | **5.85 mm** | **-92%** |

At 2 m/s the shipped collider let one cup end up 14.4 mm from the other's centre
when they can never be closer than 90.0 — it went essentially all the way
through.

## No regression — verified on the real task

Re-ran the winning grasp chain (`--claw_c 0.050 --claw_thumb=-5.7,-5.7,0,0
--entry_freeze 0.005 --hug_thumb 0`) with only the collider changed:

| stage | original (SDF) | convex ring |
|---|---|---|
| claw wrap | 182.9 deg | **182.9 deg** (identical) |
| claw entry effort | 0.000 N·m | 0.000 N·m |
| hug grip, index/middle/ring/pinky | 0.239/0.217/**0.144**/0.233 N·m | 0.248/0.229/**0.240**/0.258 N·m |
| micro-lift slip | 2.0 mm | **1.7 mm** |
| micro-lift tilt | 3.87 deg | **3.39 deg** |
| fluid retained | 444/444 | 444/444 |
| settle drift / tilt | 0.00 mm / 0.00 deg | 0.00 mm / 0.00 deg |

Grip improves on every finger; the ring finger gains 67%. The index now stops at
87 deg against the **glass** rather than bottoming out at its 95.4 deg joint
limit — a real contact stop instead of a limit stop.

`push_completed` and `cup_undisturbed` fail in BOTH runs, identically
(`frozen on leg 3 step 173 (glass moved 5.2 mm)`; drift 6.1 vs 6.2 mm). Those
are pre-existing and unrelated to the collider.

## Knobs tested and rejected (each as a single variable)

| change | result |
|---|---|
| contact offset 5 -> 20 mm | **no effect at all** (2.63/3.23/5.85 mm — identical). Late contact generation is not the limiter. |
| physics 120 -> 480 Hz | **worse** at low speed (4.38 vs 2.63 mm). The press test re-asserts its velocity 4x as often, so it does not isolate the timestep. |
| CCD | **unavailable** — Isaac disables it under GPU dynamics. |

## Two measurement traps that produced wrong answers

1. **`Usd.PrimRange` does not descend into instance proxies.** Most G1
   `collisions` scopes are USD *instances*. A plain PrimRange stops at the proxy
   and counts zero geometry, which reported 9 healthy colliders — including the
   palm's 88,338-vert mesh — as "flag on, no geometry". The corrected audit
   (`Usd.TraverseInstanceProxies()`) finds **965 real colliders, 0 dead**. There
   was never a missing-collider bug. `scripts/probe_phasing.py` now traverses
   proxies and says so in a comment.
2. **Never run two Isaac processes at once on this box.** A concurrently-run
   baseline silently dropped its `set_linear_velocity` commands and reported the
   cup as never moving. Re-run alone it moved normally. Every A/B here was run
   sequentially.

A third, in the probe itself: a run that never closes to the contact distance
proves nothing about penetration — counter friction stalls a slow push. Those
are reported `NO CONTACT REACHED (inconclusive)`, never as a pass.

---

# Part 2 — the hand-through-glass report was REAL (2026-08-19, later)

Part 1 fixed cup-vs-cup. It did **not** fix hand-vs-cup, and the earlier verdict
here — *"could not reproduce, 12.4 mm clearance"* — was measured at a pose the
user had parked in the sandbox, **not at the grasp**. It is withdrawn.

## Measured at the actual grasp

Counting collider vertices lying inside the cup cavity (r < 37.5 mm,
0 < h < 117.5 mm) at each saved checkpoint of the chain:

| stage | index tip | middle tip | ring tip | pinky tip |
|---|---|---|---|---|
| station, before anything | 0 | 0 | 0 | 0 |
| after claw **staging** | 0 | 0 | 0 | 0 |
| after the **hug closes** | **7391** | 0 | **7236** | **7364** |
| after the **lift** | 6526 | 7994 | 9249 | 9892 |

34,232 fingertip collider vertices end up inside the glass. It happens during
the **closure**, and there are two different mechanisms.

## Mechanism A — the index finger is staged ABOVE THE RIM

At staging the index intermediate spans **121.7–135.8 mm** above the cavity
floor. The rim is at 117.5. It is over the mouth before the closure starts, so
closing simply lowers it into the cavity. Nothing is violated — it is a POSE
defect, fixed by lowering `--claw_z`, not a physics one.

Ring and pinky, by contrast, never rise above 76.1 and 57.6 mm. They are below
the rim the whole time, so they crossed the **wall**.

## Mechanism B — the closure never stops on contact, so it presses through

`--trace_closure` samples each fingertip's distance from the cup axis every 4
ticks. It shows:

- middle, ring and pinky approach and **stop cleanly at 44–45 mm — the outer
  wall.** Contact generation works. The largest movement in any one sample is
  **1.3 mm**, so nothing is tunnelling (which had been the leading hypothesis,
  since CCD is unavailable under GPU dynamics).
- all four fingers then report `AT CAP` at 95.2–95.6° with efforts of
  **0.108 / 0.147 / 0.149 / 0.285 N·m**. The seat threshold was **0.30 N·m** —
  **it is not reachable by any finger, so it never fires**, and every finger
  runs to its position cap instead.
- middle and ring physically stop against the glass at t≈1.2–1.8 s, but the
  loop keeps advancing their targets until t≈4.1 s. The commanded position ends
  **10–20° inside the glass**.
- max joint velocity **pins at 12 rad/s from t≈0.9 s and never comes down** — a
  saturated drive fighting a contact for three seconds. This is the same
  12.5 rad/s "buzz" an earlier comment blamed on phantom finger-finger contacts.
- at the very end the pinky's effort spikes to 0.39 N·m and it is pushed from
  44 mm to 34 mm — **through the wall**.

**The wall penetration is caused by the closure controller, not the collision
geometry.**

### The repair

Seat on a **reachable force target** (`--grip_target`, default 0.12 N·m — four
times the ~0.03 N·m per finger that friction actually needs to hold a 0.3 kg
glass at µ 1.2 on a 45 mm lever) with a **hard cap on tracking lag**
(`--max_lag`, default 12°) so the target can never park inside the glass. Gate
`seated_on_glass` fails if fewer than 3 of 4 fingers stopped under load rather
than by running out of travel.

**A rejected intermediate:** seating on tracking lag alone (target ≥ 3° past the
measured joint). It fires immediately — the drive's ordinary tracking lag in
free air is already 10–20°, so lag cannot distinguish "moving" from "stopped".
Grip collapsed to 0.009–0.027 N·m and the micro-lift slipped 12.1 mm (from
1.7 mm). **Lag is a bound, never a contact test.**

## Why this wrecked the pour

The grasp that results is not a wrap — it is four fingers hooked near the cup's
axis, **~70 mm above its centre of mass**. The cup hangs off it like a pendulum,
so rotating the hand barely rotates the cup: commanded **72°** of tilt, the cup
reached **46.8°**.

## The pour angle, computed correctly

Earlier this file's reasoning used `atan((H−h)/R)` = 65.9°. That closed form
requires the liquid surface to still meet the wall on **both** sides, i.e.
tilt ≤ `atan(h/R)` = **41.9°**, so it was outside its validity range. Solving
the tilted-cylinder volume numerically against the measured asset geometry
(cavity R **37.5 mm**, H **117.5 mm**, fill **30.2 mm**):

| cup tilt | most that can pour, quasi-statically |
|---|---|
| 46.8° (what the cup actually reached) | **0 %** |
| 69.8° | first drop clears the rim |
| 72° (what was commanded) | 15 % |
| 80° | 64 % |
| 85° | 87 % |
| 88° | 97 % |

So the 62 % that reached the receiver was **thrown there by the slip lurch**
(the cup snaps 33.1° → 50.3° in one keyframe), not poured. A real pour needs the
cup itself — not the wrist — to reach ~85°.

## A measurement bug that was masking all of this

`cup_retained` measured in-hand drift as a world-frame offset between the palm
position and the cup's **axis-aligned bbox centre**. Both rotate with the wrist,
so a cup welded to the palm scores ~90 mm of "slip" at 72° of tilt — which is
almost exactly the 94.6 mm that was being reported as grip failure. It now
measures the cup's origin in the **palm's own frame**, plus a separate **twist**
angle (the cup's axis in the palm frame vs. at the start), which is the number
that actually says whether the cup is rotating in the hand.
