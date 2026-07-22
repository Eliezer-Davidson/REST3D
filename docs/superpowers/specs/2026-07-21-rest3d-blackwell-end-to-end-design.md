# REST3D end-to-end on a 12GB Blackwell laptop — design

**Date:** 2026-07-21
**Machine:** RTX 5070 Ti Laptop GPU (Blackwell, sm_120, 12GB / ~11.5GB usable), Intel Core Ultra 9 275HX (24 threads), 30GB RAM.
**Goal:** Run all three REST3D stages to completion on this machine, from the single demo image to a physically stabilized scene.

## Background

REST3D has three stages (the shell-script names are off by one — `2_stable_scene.sh` produces `stage3/`):

| Stage | Entry | Env | Status before this work |
|---|---|---|---|
| 1 — scene tree (VLM + SAM3) | `infer_scenetree.py` | `rest3d` (py3.11, torch 2.8+cu128) | **Working** |
| 2 — mesh reconstruction (SAM-3D-Objects) | `infer_scene3d.py` | `rest3d` | **OOM** — upstream wants ≥32GB VRAM |
| 3 — CEM physics stabilization (Isaac Gym) | `stable_scene.py` → `cem_opt.py` | `gym` (py3.8, torch 2.2.2+cu121) | **Assumed impossible** on Blackwell |
| (3b — viser replay) | `replay_in_simulator.py` | `gym` | tied to Stage 3 |

Two blockers were believed fatal on this GPU. Both were investigated empirically and **both are solvable on this machine.**

## Finding 1 — Isaac Gym CPU PhysX works; only GPU PhysX is impossible

- `IsaacGym_Preview_4_Package/.../libPhysXGpu_64.so` contains compiled SASS for **sm_50/60/70/75/80 only, and zero PTX** (verified with `cuobjdump`). No JIT fallback exists for sm_120, so GPU PhysX dies with "no kernel image is available." This is a hard wall; Isaac Gym Preview 4 is frozen/deprecated.
- **CPU PhysX runs correctly here.** Verified: boxes drop and settle to the exact expected height; root-state tensors return on `cpu`; the tensor API the CEM optimizer uses is fully supported by the CPU pipeline (it is strictly more permissive than the GPU pipeline).
- The Stage 3 code is already device-agnostic: `cem_opt.py:545` sets `self._device = self.root_states.device`; every `device=` in `cem_opt.py`, `stable_scene.py`, `coll_det.py` derives from it. **No hardcoded `.cuda()` anywhere.** `torch.cuda.empty_cache()` (`stable_scene.py:380`) is a verified safe no-op when CUDA is never initialized.
- **Migrating to Isaac Sim / Isaac Lab is unnecessary** — it would be a ~20GB install and a multi-day USD/articulation rewrite of `cem_opt.py` for no benefit over the two-line CPU switch.

### Measured wall-clock (this box, CPU PhysX, V-HACD assets)

At the default `cem_pop_size = 2048`: one 60-step settle ≈ 1.1s (primitives) to ≈ 5.2s (7 V-HACD mesh actors); asset load ≈ 0.29s each. Demo scene ≈ 3,660 sim steps total → **order 3–10 minutes.**

Scaling is **linear** in env count because actors use collision group = `env_idx` (`cem_opt.py:158,165`) and envs are grid-spaced (`env_half`), so there is no cross-env broadphase. A benchmark that instead stacks all envs at one origin shows false O(N²·⁴) (~2.8s per *step* at 2048 → hours); that does **not** reflect the real code. **Keep the default 2048.**

## Finding 2 — Stage 2 OOM is cross-object tensor retention, fixable losslessly

`infer_scene3d.py:119-123` runs SAM-3D-Objects per mask and does `outputs.append(output.copy())` — a **shallow** copy that keeps every object's full decoded mesh + gaussians resident in VRAM until the loop ends (`make_scene_untextured_mesh` / `get_object_bboxes_from_outputs` are called on the whole list *after* the loop). This is the measured "10.38GB by object 3" accumulation.

Downstream (Stage 3) consumes **only OBJ geometry + URDFs** (`scene_io.py` `load_trimesh_any`); gaussians/textures are never read. Within Stage 2, each object's output is needed only to produce: the `glb` (trimesh, already CPU), the pose (`rotation`/`scale`/`translation`, tiny), and the bbox — which needs `gaussian[0].get_xyz` (`mesh.py:144`). So the gaussian is needed **per object**, not retained across objects.

## Design

### Part A — Stage 3: config-flag CPU physics (chosen: config flag, default CPU here)

1. `rest3d/config/stable_scene_cfg.py`: add `use_gpu_physics: bool = False`.
2. `rest3d/models/cem_opt.py` `create_sim_and_viewer(...)`: add param `use_gpu_physics: bool = False`; set `params.use_gpu_pipeline = use_gpu_physics` and `params.physx.use_gpu = use_gpu_physics` (replacing the hardcoded `True`).
3. Stage 3 call site (`cem_opt.py:501`): pass `use_gpu_physics=getattr(args, "use_gpu_physics", False)`.
4. Replay call site (`replay_in_simulator.py:312`): add a `--gpu_physics` CLI flag (default `False`) and pass it through; use `--viser` (headless, no Vulkan) for visualization since the Isaac Gym GPU viewer/camera can't run here.

Default-`False` means both entry points get CPU physics with no extra flags on this machine, while a ≥32GB GPU can re-enable GPU physics by setting the field / flag. Upstream numerics are otherwise unchanged. (Note: on CPU, the immediate `refresh_actor_root_state_tensor` after a setter returns *correct* data rather than the GPU pipeline's documented stale read, so results may differ slightly from — and be more correct than — a GPU reference; this is expected, not a bug.)

### Part B — Stage 2: prune + CPU-move retained tensors (chosen: prune, lossless)

Refactor the `infer_scene3d.py` inference loop so that, per object, it: (a) runs inference, (b) extracts the CPU trimesh via `make_scene_untextured_mesh(output)`, (c) computes bbox/axes via `get_object_bboxes_from_outputs(output)` **before** freeing, (d) retains only mesh + bbox + axes, (e) `del output` and `torch.cuda.empty_cache()`. Both helper functions already iterate over `*outputs`, so calling them with a single output is valid and produces identical results.

- **Lossless:** no precision reduction; only where tensors live changes.
- Try **without** the earlier `REST3D_HALF_WEIGHTS` fp16 hack first (offload + prune may fit in fp32); fall back to enabling it only if a real run still OOMs.
- Keep the existing `REST3D_CPU_OFFLOAD` per-phase paging (it handles the *within-object* peak); prune handles the *across-object* accumulation. They are complementary.

## Execution plan (chosen: full end-to-end, report back)

1. Implement Part A + Part B; smoke-test imports/parse.
2. Stage 2: run `infer_scene3d.py` on the demo image → verify **7/7** OBJs + URDFs written.
3. Stage 3: `bash 2_stable_scene.sh` (CPU physics) → verify a stabilized scene + poses JSON, capture wall-clock.
4. Replay: `replay_in_simulator.py --viser` → confirm it serves.
5. Report OBJ count, settle time, stability, and any surprises.

## Non-goals / risks

- **Not** porting to Isaac Sim/Lab, MuJoCo, or PyBullet — CPU PhysX makes that unnecessary; a backend swap would change physics semantics vs. the paper.
- Stage 2 quality: prune is lossless, but the ≥32GB reference remains the gold standard; a real run must show clean geometry.
- Stage 3 CPU-vs-GPU numeric drift is expected (see Part A note).
- Wall-clock at 2048 envs is a proxy estimate until the real `stable_scene.py` run measures it; `cem_pop_size` stays configurable as a lever if needed.
