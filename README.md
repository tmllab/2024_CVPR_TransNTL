# TransNTL
This repository contains a Pytorch implementation of CVPR 2024 Highlight paper "Your Transferability Barrier is Fragile: Free-Lunch for Transferring the Non-Transferable Learning." (still under updating...)

## Usage

### Preparing Data
Download and save datasets to `./data`, then run the following command:
```
python data_split.py
```
We provide pre-split datasets in [Google Drive](https://drive.google.com/drive/folders/1j5K3nimlyP2Bzw_T32Fy3PqbOPlBdnLQ?usp=sharing). You could download them and save to `./data_presplit/`.

### Pretraining NTL Models
We also provide saved model files in [Google Drive](https://drive.google.com/drive/folders/1j5K3nimlyP2Bzw_T32Fy3PqbOPlBdnLQ?usp=sharing) which were pretrained on our pre-split datasets. You could save them to `./saved_models/`.

Alternatively, you could pre-train NTL models by yourself. In this way, please use parameters in `./config/*/pretrain.yml`.

### Training TransNTL
Please run the following command to training TransNTL for attacking NTL models.
```
python run_attack.py
```

## Citation
```
@inproceedings{hong2024your,
  title={Your Transferability Barrier is Fragile: Free-Lunch for Transferring the Non-Transferable Learning},
  author={Hong, Ziming and Shen, Li and Liu, Tongliang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={28805--28815},
  year={2024}
}
```

## Acknowledgements
Parts of our codes are based on following projects: 
- https://github.com/conditionWang/NTL
- https://github.com/LyWang12/CUTI-Domain
- https://github.com/zhuohuangai/SharpDRO


