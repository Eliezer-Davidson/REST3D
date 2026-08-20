# EDSRest3d — Usage guide

*Single photo in → physically stable 3D scene out → exported into a robotics simulator.*
Install first: [INSTALL-EDS.md](INSTALL-EDS.md). All commands assume
`source rest3d_blackwell_env.sh` in the current shell.

## The pipeline (three stages, same as upstream)

```bash
# Stage 1+2 — infer the 3D scene from one image
conda activate rest3d
export GEMINI_API_KEY=<your key>
bash 1_infer_scenecanon.sh          # set INPUT=<your image> inside, or use the demo
# → output/<image_name>/stage2/scene_canon/

# Stage 3 — physically stabilize the scene (CPU PhysX on Blackwell)
conda activate gym
bash 2_stable_scene.sh
# → output/<image_name>/stage3/global_scene/      ← the canonical result
#   (never use global_scene_v2 anywhere — that name is a hand-edit, produced by no code here)

# Interactive replay in the browser
bash 3_replay_in_simulator.sh       # add --gpu_physics only on pre-Blackwell GPUs
```

## Running stage 2 on a 12 GB GPU — the four-flag recipe

```bash
REST3D_HALF_WEIGHTS=1 REST3D_CPU_CONSTRUCT=1 REST3D_OFFLOAD_EMBEDDER=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash 1_infer_scenecanon.sh
```

- fp16 weights (decoders stay fp32 — they run outside autocast)
- models constructed on CPU (kills a transient fp32-on-GPU spike)
- idle generators/embedder paged to RAM during mesh decode (~5.4 GB freed)
- **layout post-optimization runs automatically, deferred**: its inputs are captured to a
  bundle during the loop and replayed after the model stack is released, so the full
  published method runs with identical math — no flag needed.
  `REST3D_SKIP_LAYOUT_POSTOPT=1` skips post-opt entirely;
  `REST3D_KEEP_POSTOPT_BUNDLES=1` keeps the bundles for determinism probes.

Useful env overrides: `REST3D_GEMINI_MODEL` (the default preview model 503s under load),
`REST3D_VLM_MAX_RETRIES` (exponential backoff is built in).

## Exporting a scene to a simulator (the EDS addition)

```bash
# stage3 scene → MuJoCo MJCF (with CoACD convex collision decomposition)
python scripts/export_sim_scene.py --scene output/<image_name>/stage3/global_scene \
    --out export/<image_name>/

# exported assets → Isaac Sim USD (RigidBodyAPI + convexHull colliders + displayColor)
python scripts/export_usd_scene.py --assets export/<image_name>/ \
    --out export/<image_name>/scene.usd

# headless Isaac render / scripted demos
python scripts/render_isaac_headless.py --usd export/<image_name>/scene.usd
```

Worked examples of downstream use live in `docs/`: `banner4-fluid-scene.md`
(real-scale lab scene with PBD fluid in a graspable cup), `banner4-physics-repair.md`,
`contact-task-2026-08-14.md`, `scripted-grasp-pour-2026-08-13.md` — a Unitree G1 with
Inspire hands grasping, lifting and pouring from an exported scene, on real contact
physics. Supporting tools are in `scripts/` (`g1_*.py`, `probe_*.py`, `fix_*_colliders.py`).

## Known behaviours (read before trusting a result)

- **Metastable stage-3 poses** (e.g. an object balanced on a sofa backrest) can survive
  Isaac Gym's settle yet fall in MuJoCo — always validate exported scenes in the target
  engine.
- Monocular scale: large objects land near-metric; small/occluded objects can drift.
- `import open3d` before `import coacd`, always (segfault otherwise).
- On multi-GPU or pre-Blackwell machines the upstream defaults may be faster — this
  port's value is that it runs at all on 12 GB.

## Credits

Method and original code: [REST3D](https://github.com/ShirleyMaxx/REST3D), Xiaoxuan Ma
et al., Carnegie Mellon University — CC BY-NC 4.0. Adaptation, port and exporters:
**Eliezer Davidson** (eliezerdavidson6666@gmail.com), for non-commercial research.
