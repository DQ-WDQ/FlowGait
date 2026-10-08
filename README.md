# FlowGait

This repository contains the dataset and training code accompanying our **CHI 2026** paper:

> **FlowGait: Enabling Robust Long-Term Gait Recognition Across Real-World Covariates with mmWave Radar**  
> Dequan Wang, Chenming He, Lingyu Wang, Chengzhen Meng, Xiaoran Fan, Yanyong Zhang

- **Paper:** https://dl.acm.org/doi/10.1145/3772318.3790623

FlowGait is a mmWave-based gait recognition framework for robust, long-term deployment in real-world environments. It addresses changes in gait caused by clothing, carried items, walking routes, and natural variation over time. After a one-time enrollment with labeled walking data, FlowGait adapts to each user's evolving gait through self-training and continual learning on unlabeled daily walks. The framework combines a Transformer tailored to radar range-Doppler heatmaps, two-stage step-traversal pseudo-labeling, and a fixed-size core set for replay to reduce catastrophic forgetting and keep model updates efficient.

## Primary workflow

- Model: `model_img/flowgait_net.py` (`FlowgaitNet`)
- Image-sequence datasets: `data/rds.py`
- Shared training loop: `training/engine.py`
- Main training entry point: `train_flowgaitnet.py`
- Action-subset training entry point: `train_flowgaitnet_action_subset.py`

The maintained workflow imports datasets from `data.rds`. Retained semi-supervised experiments share a configurable reader in `data/rds_experiments.py`.

## Requirements

Python 3.9 or newer is required. Install PyTorch and TorchVision builds that match your CUDA environment, then install FlowGait.

## Dataset

- **Download:** https://pan.ustc.edu.cn/share/index/949b5ac3317d4214ba7d?p=1
- **Password:** `lUEo`

### Input directory layout

The image dataset reader expects one folder per sequence. Identity and action are the two parent folders:

```text
dataset/
  <identity>/
    <action>/
      <sequence>/
        0.png
        1.png
        ...
        19.png
```

Each sequence contains 20 grayscale frames. The current model configuration expects each cropped frame to have shape `11 × 220`.


## Citation

If you use this code or dataset, please cite our paper:

```bibtex
@inproceedings{wang2026flowgait,
  title     = {FlowGait: Enabling Robust Long-Term Gait Recognition Across Real-World Covariates with mmWave Radar},
  author    = {Wang, Dequan and He, Chenming and Wang, Lingyu and Meng, Chengzhen and Fan, Xiaoran and Zhang, Yanyong},
  booktitle = {Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems},
  year      = {2026},
  doi       = {10.1145/3772318.3790623}
}
```

## Related Work: RDGait

Also check out our earlier work, [RDGait](https://github.com/DQ-WDQ/RDGait), on mmWave gait recognition in complex indoor environments. Its publicly available dataset includes 125 participants across two indoor scenarios and five walking behaviors, with multiple radar representations: range-Doppler stacks, micro-Doppler spectrograms, range-time spectrograms, and point clouds.

[Dataset and documentation](https://github.com/DQ-WDQ/RDGait)
