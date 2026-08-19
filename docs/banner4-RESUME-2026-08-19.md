# banner4 grasp-and-pour — pause state, 2026-08-19

Paused mid-task for a physical machine move. Nothing was running at pause time;
GPU clear, no partial writes, all checkpoints and metrics verified intact.

## Where the work stands

REAL-contact grasp + lift ACHIEVED and reproduced bit-for-bit across two
independent chains. No kinematic attach; no code path writes a cup pose.

| stage | measured |
|---|---|
| m2b_claw entry | wrap **182.9 deg**, max digit effort **0.000 N.m** (arrives clean) |
| m2c_hug | 4/4 fingers seat; micro-lift slip **2.0 mm**, tilt 3.87 deg |
| m3_claw_lift | cup rose **146.1 mm** of a 150 mm command, slip **2.7 mm**, tilt 12.91 deg, **444/444 fluid** |
| carry | 128 mm across the counter, 5.3 mm in-hand drift |
| m4_pour (45 deg) | **16-28% transferred**, 0-1 particles spilled; cup slides 66-88 mm in-hand |

THE fix was one number: claw pocket depth `CLAW_C` 0.0699 -> 0.050.

## Exact command to resume

    conda activate g1skillgen
    python scripts/g1_contact_task.py --phase <phase> \
      --claw_c 0.050 --claw_thumb=-5.7,-5.7,0,0 \
      --entry_freeze 0.005 --hug_thumb 0

Phase order: m2a_palm -> m2b_claw -> m2c_hug -> m3_claw_lift -> pour_prep (ghost) -> m4_pour
Trajectory in play: `output/banner4/pour_seq/contact_task_v01/task_traj_c050.json`
Checkpoints: `output/banner4/sim/usd/final/f_m2b|f_m2c|f_m3.usdc`
Evidence: `output/banner4/pour_seq/final_v01/evidence/`

Note the `--claw_thumb=-5.7,...` `=` form is required; argparse reads the leading
minus as a flag otherwise.

## The one open problem

The four fingers form a RING. On tilt, gravity pulls the cup straight out along
its own axis with nothing blocking the mouth end. In-hand drift runs
5.3 -> 19.3 -> 74.1 mm between 0 and 55 deg of tilt.

The Inspire thumb cannot help: `R_thumb_distal` sits above the rim at EVERY yaw
across a full 80 deg sweep, and as yaw closes the index effort collapses
0.131 -> 0.000 N.m (the thumb pushes the glass OUT of the fence). Folding it as
an axial cap also fails - tip at z 1.037 vs rim 1.013 presses the cup down,
slip 2.0 -> 7.8 mm.

Rejected: raising the finger force clamp. 0.45 N.m is the real Inspire spec and
two of four fingers already reach it; raising it fakes the hardware.

UNTESTED LEAD: the glass is 85 mm across and this hand's jaw closes to ~24 mm
with the thumb level. A smaller-diameter vessel is the one change that would let
the thumb oppose at all. This changes the SCENE, not the controller - confirm
before pursuing.

## Decisions waiting on Eli

1. Commit the session's scripts? Nothing is committed. `scripts/g1_contact_task.py`
   and siblings are still UNTRACKED.
2. Fold these real results into the "Glass, Water, Hands" showcase artifact, or
   keep the beauty version separate? Not touched without a go-ahead in case it
   has already been shared.
3. Both NVIDIA MCP servers (`isaac-sim`, `isaac-docs`) are still Pending approval
   and need a Claude Code restart.
