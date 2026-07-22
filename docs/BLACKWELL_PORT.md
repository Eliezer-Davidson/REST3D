# Running REST3D on a 12 GB Blackwell GPU (RTX 5070 Ti Laptop, sm_120)

This branch makes the full three-stage pipeline run end-to-end on hardware well below
the documented requirements (upstream asks ≥32 GB VRAM for Stage 2; Isaac Gym ships no
GPU-PhysX kernels for Blackwell at all). Everything is opt-in: with the env flags unset
and `use_gpu_physics=True`, stock upstream behaviour is preserved.

Validated 2026-07-21/22 on a real capture: 7 objects, all stages exit 0, final scene
physically stable (CEM penetration 0), exported to MuJoCo + Isaac Sim and settle-tested
in both.

## Environment

The pinned cu121 stack cannot launch kernels on sm_120. This checkout runs:
torch 2.8.0+cu128, kaolin 0.18 (cu128 wheel), pytorch3d 0.7.8 built from source
(TORCH_CUDA_ARCH_LIST=12.0), spconv-cu126, numpy pinned 1.26. `rest3d_blackwell_env.sh`
wires CUDA_HOME to a CUDA 12.8 toolchain for gsplat's runtime JIT. Stage 3 uses a
separate py3.8 env with Isaac Gym Preview 4.

## Changes

| file | what |
|---|---|
| `rest3d/utils/vlm.py` | `REST3D_GEMINI_MODEL` env override (the default preview model 503s under load); exponential backoff on 5xx/connection errors; `REST3D_VLM_MAX_RETRIES` |
| `scripts/infer_scene3d.py` | per-object prune (extract mesh/bbox inside the loop, free GPU tensors) — fixes cross-object VRAM accumulation |
| `rest3d/models/build_sam3dobj.py` | `CUDA_HOME` setdefault (external toolchain wins); `REST3D_SKIP_LAYOUT_POSTOPT` gate |
| `rest3d/models/cem_opt.py` | `use_gpu_physics` parameter; CPU PhysX pipeline (GPU PhysX is impossible on sm_120: `libPhysXGpu_64.so` carries SASS for sm_50–80 only, no PTX) |
| `rest3d/config/stable_scene_cfg.py` | `use_gpu_physics: bool = False` config flag |
| `scripts/replay_in_simulator.py` | `--gpu_physics` flag (CPU default) |
| `scripts/export_sim_scene.py` | NEW: stage3 scene → MuJoCo MJCF with CoACD convex collision decomposition |
| `scripts/export_usd_scene.py` | NEW: exported assets → Isaac Sim USD (RigidBodyAPI, convexHull colliders, vertex displayColor) |
| `docs/patches/*.patch` | the low-VRAM edits to the vendored `sam-3d-objects/` and `sam3/` checkouts (those dirs are separate repos, so the diffs ship here as patches) |

## Stage 2 on 12 GB — the five-flag recipe

```
REST3D_HALF_WEIGHTS=1 REST3D_CPU_CONSTRUCT=1 REST3D_OFFLOAD_EMBEDDER=1 \
REST3D_SKIP_LAYOUT_POSTOPT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

1. fp16 weights (decoders stay fp32 — they run outside autocast);
2. construct models on CPU (kills a transient fp32-on-GPU spike);
3. page the idle generators/embedder to RAM during mesh decode (~5.4 GB freed for the
   dense grid);
4. skip the gaussian-splat layout post-optimization (the single biggest memory peak;
   Stage 3 re-optimizes poses anyway — cost: image-space placement is looser);
5. the per-object prune above.

Do NOT combine with `REST3D_CPU_OFFLOAD` on a ~30 GB-RAM machine — it stacks every
model in system RAM at construction and gets OOM-killed.

## Stage 3 on CPU PhysX

`use_gpu_physics=False` runs the identical CEM optimization on CPU PhysX. Measured on a
6-object scene, pop 2048: local groups + global ≈ 13 min, penetration 0. Env isolation
via per-env collision groups keeps CPU scaling linear.

## Known behaviours

- Metastable Stage-3 poses (e.g. an object balanced on a sofa backrest) can survive
  Isaac Gym's settle yet fall in MuJoCo — validate exported scenes in the target engine.
- CoACD note: `import open3d` must precede `import coacd` (reverse order segfaults).
- Monocular scale: large objects land near-metric; small/occluded objects can drift
  (one tumbler reconstructed ~40 % undersized). Verify against a known dimension.
