# EDSRest3d — Installation (12 GB Blackwell port)

*By Eliezer Davidson. This adaptation of [REST3D](https://github.com/ShirleyMaxx/REST3D)
(Xiaoxuan Ma et al., CMU) runs the full published method on a single 12 GB consumer GPU
(tested: RTX 5070 Ti Laptop, sm_120 "Blackwell") and adds robotics-simulator export.
Non-commercial research only, per the upstream [CC BY-NC 4.0 license](LICENSE).*

For the original (24 GB+, cu121) setup, follow the upstream [INSTALL.md](INSTALL.md).
This guide covers what is **different** here.

## 0. Prerequisites

- NVIDIA driver ≥ 580.x and a CUDA **12.8** toolkit installed on the system
  (the port JIT-compiles gsplat kernels at runtime — `rest3d_blackwell_env.sh` wires
  `CUDA_HOME` to it).
- Conda (two envs are used: `rest3d` for stages 1–2, `gym` (Python 3.8 + Isaac Gym
  Preview 4) for stage 3). Isaac Gym Preview 4 must be obtained from NVIDIA directly —
  its license does not allow redistribution, so it is NOT included here.
- A Google **Gemini API key** in `GEMINI_API_KEY` (the VLM stage; see upstream INSTALL.md).

## 1. Main environment (`rest3d`) — the Blackwell stack

The upstream cu121 pins cannot launch kernels on sm_120. Replace them with:

```bash
conda create -n rest3d python=3.10 -y && conda activate rest3d
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
pip install spconv-cu126 "numpy==1.26.*"
# pytorch3d 0.7.8 must be BUILT FROM SOURCE for sm_120:
TORCH_CUDA_ARCH_LIST="12.0" pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.8"
pip install -r requirements.txt   # remaining upstream deps
```

Then, in every shell that runs the pipeline:

```bash
source rest3d_blackwell_env.sh    # wires CUDA_HOME for gsplat's runtime JIT
```

## 2. Vendored model checkouts + the EDS patches

`sam3/` and `sam-3d-objects/` are separate upstream repos (not part of this tree). Clone
them per upstream INSTALL.md, then apply the low-VRAM patches shipped here:

```bash
cd sam-3d-objects
git apply ../docs/patches/sam3d-objects-blackwell-12gb.patch
git apply ../docs/patches/sam3d-objects-deferred-postopt.patch
cd ../sam3
git apply ../docs/patches/sam3-agent-fixes.patch
```

## 3. Stage-3 environment (`gym`)

Per upstream: Python 3.8 env with Isaac Gym Preview 4. One EDS difference: **GPU PhysX is
impossible on sm_120** (`libPhysXGpu_64.so` ships SASS for sm_50–80 only, no PTX). The
port defaults to **CPU PhysX** (`use_gpu_physics=False`) — same CEM optimization,
measured ≈13 min for a 6-object scene at population 2048, penetration 0.

## 4. Simulator-export extras (the EDS addition)

```bash
pip install coacd open3d mujoco   # exporters: MJCF + convex decomposition
# Isaac Sim (optional, for USD replay): install per NVIDIA docs
```

**Import-order landmine:** `import open3d` must come **before** `import coacd`
(the reverse order segfaults — both vendor native libs). The shipped scripts already
import in the safe order.

## 5. Smoke test

```bash
conda activate rest3d && source rest3d_blackwell_env.sh
export GEMINI_API_KEY=<your key>
bash 1_infer_scenecanon.sh        # demo image → output/<name>/stage2/scene_canon/
```

If stage 2 OOMs on your GPU, see the four-flag recipe in [USAGE-EDS.md](USAGE-EDS.md) —
and do **not** set `REST3D_CPU_OFFLOAD` on a ≤32 GB-RAM machine (it stacks every model
in system RAM at construction and gets OOM-killed).
