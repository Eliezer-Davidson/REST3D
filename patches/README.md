# Local modifications to the vendored Meta packages

The `sam3/` and `sam-3d-objects/` directories are gitignored clones of Meta's
releases, and they carry local modifications that ARE this fork's port — the
Gemini agent backend and the 12 GB deferred-postopt path. Captured here as
patches on 2026-08-20 (pre-wipe backup) because the nested repos have no
pushable remote.

To reconstruct exactly:

    git clone https://github.com/facebookresearch/sam3.git
    git -C sam3 checkout 46957e4
    git -C sam3 apply ../patches/vendored-sam3-local-mods.patch

    git clone https://github.com/facebookresearch/sam-3d-objects.git
    git -C sam-3d-objects checkout f91db41
    git -C sam-3d-objects apply ../patches/vendored-sam-3d-objects-local-mods.patch

sam3 modified: sam3/agent/agent_core.py, client_sam3.py, inference.py
sam-3d-objects modified: sam3d_objects/__init__.py, model/io.py,
pipeline/inference_pipeline.py, pipeline/inference_pipeline_pointmap.py;
NEW: pipeline/deferred_postopt.py
