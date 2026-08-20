> ## 🔀 Research fork notice
>
> This is a personal **research fork** of [REST3D](https://github.com/ShirleyMaxx/REST3D)
> by Xiaoxuan Ma et al. (Carnegie Mellon University), maintained by
> **Eliezer Davidson**.
>
> Changes vs upstream, both for non-commercial research only:
> 1. **`blackwell-12gb` port** — runs the full published method on a single 12 GB
>    consumer GPU (RTX 5070 Ti / Blackwell, CUDA 12.8), via deferred two-stage layout
>    post-optimization.
> 2. **Simulator export** — exporters that carry REST3D reconstructions into robotics
>    simulators (MuJoCo / Isaac) as physics scenes.
>
> All credit for the method and code belongs to the original authors. This fork is used
> strictly under the upstream [CC BY-NC 4.0 license](LICENSE): **non-commercial research
> only**, with attribution; no affiliation with or endorsement by the original authors is
> implied.

# EDSRest3d

**Eliezer Davidson's research adaptation of REST3D — runs the full method on a 12 GB
consumer GPU and exports reconstructed scenes into robotics simulators (MuJoCo / Isaac).**
📦 Install: [INSTALL-EDS.md](INSTALL-EDS.md) · 📖 Use: [USAGE-EDS.md](USAGE-EDS.md) · 🔧 Port details: [docs/BLACKWELL_PORT.md](docs/BLACKWELL_PORT.md)

---

## Upstream: REST3D — Reconstructing Physically Stable 3D Scenes from a Single Image

<p align="center">
  <a href="https://shirleymaxx.github.io/">Xiaoxuan Ma</a>&emsp;
  <a href="https://jiashunwang.github.io/">Jiashun Wang</a>&emsp;
  <a href="https://nicolasugrinovic.github.io/">Nicolás Ugrinovic</a>&emsp;
  <a href="https://yehonathanlitman.github.io/">Yehonathan Litman</a>&emsp;
  <a href="https://kriskitani.github.io/">Kris Kitani</a>
</p>

<p align="center">
  Carnegie Mellon University
</p>
  

<p align="center"><a href="https://arxiv.org/abs/2605.30338"><img src="https://img.shields.io/badge/arXiv-REST3D-d55c5c?logo=arxiv&style=flat" alt="arXiv"></a> &nbsp;<a href="https://shirleymaxx.github.io/REST3D/"><img src="https://img.shields.io/badge/Project-Page-6bbf59?logo=googlechrome&style=flat" alt="Project Page"></a> &nbsp;<a href="https://shirleymaxx.github.io/REST3D/#interactive"><img src="https://img.shields.io/badge/Interactive-3D-f2c14e?logo=unity&style=flat" alt="Interactive 3D"></a></p>

<h3 align="center">
  ⚡️ TL;DR: From a single casual image to a visually consistent and physically stable interactive 3D scene.
</h3>
<p align="center">
  <img src="assets/teaser.gif" alt="REST3D teaser" width="100%">
</p>


## News

🚩 **2026.06**: Released the code.

**2026.05**: Released the arXiv paper and project page.


## 🛠️ Installation

Please follow [INSTALL.md](INSTALL.md) for detailed installation instructions.

## 🚀 Quick start
Follow the steps below to reconstruct a physically stable 3D scene from a single image and interactively inspect object stability in physics simulator.

> Remember to set your API key before running (`GEMINI_API_KEY` — see [INSTALL.md](INSTALL.md)).

**Step 1. Infer 3D scene from a single image**


```bash
conda activate rest3d
bash 1_infer_scenecanon.sh. #Change `INPUT` to your image path (defaults to the demo image)
```
Outputs are saved to `output/<image_name>/stage2/scene_canon/`.

**Step 2. Stabilize the scene**

```bash
conda activate gym
bash 2_stable_scene.sh
```
Outputs are saved to `output/<image_name>/stage3/`:
- `global_scene/` — physically stable scene
- `global_scene_w_walls/` — (optional) additionally fits walls to the scene and adjusts wall-attached object positions accordingly

**🤗 Visualize and interact with the physically stable scene**

```bash
conda activate gym
bash 3_replay_in_simulator.sh
```
Follow the printed URL to open the scene in a browser and interactively inspect the physics simulation settling process.


## Citation
If you find this work useful, please cite:
```bibtex
@article{ma2026rest3d,
  title     = {REST3D: Reconstructing Physically Stable 3D Scenes from a Single Image},
  author    = {Ma, Xiaoxuan and Wang, Jiashun and Ugrinovic, Nicol\'{a}s and Litman, Yehonathan and Kitani, Kris},
  booktitle = {arXiv preprint arXiv:2605.30338},
  year      = {2026}
}
```

## Acknowledgements

This repository builds upon the following excellent open-source projects: [SAM 3](https://github.com/facebookresearch/sam3), [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects), and [Isaac Gym](https://developer.nvidia.com/isaac-gym).

## License

This project is released under the [CC&nbsp;BY-NC&nbsp;4.0](https://creativecommons.org/licenses/by-nc/4.0/).
See [`LICENSE`](LICENSE) for details.

**Non-commercial use only.** For commercial licensing, please contact the [author](mailto:xiaoxuam@andrew.cmu.edu). Please note that it also relies on external libraries, which may be subject to their own licenses and terms of use.
