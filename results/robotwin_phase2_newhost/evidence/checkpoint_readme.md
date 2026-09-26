---
library_name: pytorch
license: mit
pipeline_tag: robotics
tags:
- robotics
- vla
- lerobot
- robotwin
---

# LAWAM RoboTwin SFT Release

This repository contains the LAWAM SFT checkpoint for RoboTwin experiments, presented in the paper [LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies](https://huggingface.co/papers/2606.15768).

* **Project Page:** [LaWAM Project Page](https://rlinf.github.io/LaWAM/)
* **Code:** [GitHub Repository](https://github.com/RLinf/LaWAM)

## Dataset Notes

The RoboTwin training data used for this checkpoint uses EEF actions and is derived from lingbot-va / [robbyant/robotwin-clean-and-aug-lerobot](https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot/tree/main/lerobot_robotwin_eef_aug_500/beat_block_hammer-aloha-agilex_randomized_500-1000).

Compared with the referenced public RoboTwin data, our processed version has the following differences:

- We use the EEF-action variant.
- We convert the data into LeRobot 3.0 format.
- The accompanying `dataset_statistics.json` and `config.yaml` correspond to this processed LeRobot 3.0 version, not the raw public release.

## Files

- `final_model/pytorch_model.pt`: released LAWAM checkpoint.
- `config.yaml`: training/configuration metadata for this release.
- `dataset_statistics.json`: dataset statistics for the processed training data.

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@misc{chen2026lawam,
  title = {LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies},
  author = {Chen, Jialei and Wang, Kai and Chen, Kang and Chen, Shuaihang and Gao, Feng and Tang, Wenhao Rect, Zhiyuan and Liu, Weilin and Yao, Zhuyu and Li, Boxun and Xu, Yuanbo and Yu, Chao},
  journal = {arXiv preprint arXiv:2606.15768},
  year = {2026},
  archiveprefix = {arXiv},
  primaryclass = {cs.RO},
}
```