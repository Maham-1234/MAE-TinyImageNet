# Masked Autoencoder (MAE) — TinyImageNet

Self-supervised image representation learning using a Masked Autoencoder built from scratch in PyTorch.

## Overview

This project implements the MAE architecture from [He et al., 2022](https://arxiv.org/abs/2111.06377) — a ViT-Base encoder paired with a lightweight ViT-Small decoder that learns visual representations by reconstructing images with 75% of patches masked. No labels, no pretrained weights.

## Architecture

| Component | Model | Dim | Depth | Heads | Params |
|-----------|-------|-----|-------|-------|--------|
| Encoder | ViT-Base/16 | 768 | 12 | 12 | ~86M |
| Decoder | ViT-Small/16 | 384 | 12 | 6 | ~22M |

- **196 total patches** (14×14 grid from 224×224 image)
- **49 visible patches** fed to encoder (25%)
- **147 masked patches** reconstructed by decoder (75%)

## Project Structure

```
mae-tinyimagenet/
├── mae_assignment.ipynb   # Full training notebook
├── requirements.txt       # Dependencies
└── README.md
```

## Training Configuration

```python
NUM_EPOCHS   = 30
BATCH_SIZE   = 128
BASE_LR      = 2e-4        
WEIGHT_DECAY = 0.1
MASK_RATIO   = 0.75
```

**Techniques used:** Mixed precision (AMP) · AdamW · Cosine LR decay · Gradient clipping · Early stopping (patience=5) · Dual GPU (DataParallel)

## Results

| Metric | Score |
|--------|-------|
| PSNR | ~22–24 dB |
| SSIM | ~0.75–0.82 |

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/Maham-1234/MAE-TinyImageNet.git
cd mae-tinyimagenet
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add the dataset**

On Kaggle, add [TinyImageNet](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet) to your notebook. Dataset will be available at:
```
/kaggle/input/tiny-imagenet/tiny-imagenet-200/
```

**4. Run the notebook**

Open `mae_assignment.ipynb` on Kaggle with **GPU T4×2** accelerator enabled and run all cells.

## Demo

The final cell launches a **Gradio app** where you can:
- Upload any image
- Adjust the masking ratio (10–95%) with a slider
- See the masked input and reconstruction side by side
- Get PSNR and SSIM scores in real time



## References

- He, K., et al. *Masked Autoencoders Are Scalable Vision Learners*. CVPR 2022. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
- Dosovitskiy, A., et al. *An Image is Worth 16x16 Words*. ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
