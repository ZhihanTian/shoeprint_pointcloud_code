# Point Cloud Shoeprint Classification

This repository provides the implementation of a deep learning architecture for 3D shoeprint point cloud classification.
It combines **Multi-Scale Set Abstraction (MSA)** and **Channel & Spatial Attention(CAS)** for enhanced feature extraction.

## 🧠 Model Overview

- Multi-branch MSA layers
- CAS attention mechanism at shallow layer
- Global feature pooling and classification

## 🗃 Dataset Format

Your dataset should follow this structure:
```
data_root/
├── 1/
│ ├── 1_1.xyz
│ ├── 1_2.xyz
├── 2/
│ ├── 2_1.xyz
...
```
- Each `.xyz` file contains 3D points in text format.

## 🚀 Getting Started

Install dependencies:
```bash
pip install -r requirements.txt
```

Set your dataset path in `Config.data_root` in `train.py` and run training:
```
python train.py
```
