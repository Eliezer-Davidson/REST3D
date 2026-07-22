# Source this before running Stages 1-2 on the RTX 5070 Ti (Blackwell, sm_120).
#   conda activate rest3d && source rest3d_blackwell_env.sh && bash 1_infer_scenecanon.sh
#
# Why this exists: the repo's pinned cu121 stack cannot launch kernels on sm_120, so this
# checkout runs on torch 2.8.0+cu128 with a source-built pytorch3d. gsplat still JIT-compiles
# its CUDA kernels at runtime, and the rest3d env's own nvcc is 12.1 (max sm_90) -- so we point
# CUDA_HOME at a separate CUDA 12.8 toolkit env that can emit sm_120.

export SAM3D_OBJECTS_ROOT="/home/user/work/REST3D/sam-3d-objects"

# CUDA 12.8 toolchain (only for compiling extensions; runtime CUDA libs come from torch's wheels)
_CU128="/home/user/miniconda3/envs/cu128build"
_CU128_TARGETS="$_CU128/targets/x86_64-linux"
export CUDA_HOME="$_CU128"
export PATH="$_CU128/bin:$PATH"
# conda-forge puts CUDA headers/libs under targets/<triple>/, not $PREFIX/include, and torch's
# headers pull in cusparse.h -- without these the JIT compile fails to find it.
export CPLUS_INCLUDE_PATH="$_CU128_TARGETS/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export C_INCLUDE_PATH="$_CU128_TARGETS/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export LIBRARY_PATH="$_CU128_TARGETS/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST="12.0"

# GEMINI_API_KEY (kept outside the repo, mode 600)
if [ -f "$HOME/.rest3d_env" ]; then
    . "$HOME/.rest3d_env"
else
    echo "WARNING: ~/.rest3d_env not found -- GEMINI_API_KEY unset, Stage 1 will fail" >&2
fi
