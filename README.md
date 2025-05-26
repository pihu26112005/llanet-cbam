# DFD — Deepfake Video & Image Detector

- CS671 Course Project · Group 37  
- Final Evaluation PPT: [View on Canva](https://www.canva.com/design/DAGoAiMCg9k/r0Z1bWFoJVnz2YbR27ZESg/edit?utm_content=DAGoAiMCg9k&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

This project implements and enhances LLANet, a multi-branch network architecture designed for deepfake detection. The system is extended by integrating Convolutional Block Attention Modules (CBAM) into the Enhanced Feature Pyramid Network (E-FPN) backbone, and further improved with Restormer-based preprocessing for robust noise handling and artifact suppression in both video frames and images.

---

## Features

-  LLANet implementation with multi-task branches
-  CBAM integration into the E-FPN backbone for better local attention
-  Restormer-based preprocessing to denoise inputs and enhance deepfake artifacts
-  Supports major deepfake datasets: Celeb-DF, DFDC, DFD, FaceForensics++
-  Modular config-based training and evaluation pipeline

---

## Model Architecture

- LLANet (Local-to-Global Attention Aggregation Network) as backbone
- CBAM: Convolutional Block Attention Module integrated at multiple scales in E-FPN
- Restormer: Transformer-based denoiser applied as preprocessing on input frames
- Multi-task supervision: heatmaps, binary mask, and classification outputs

---

## Tech Stack

- Python 3.8+
- PyTorch
- OpenCV, NumPy, Matplotlib
- CBAM (custom module)
- Restormer (image restoration transformer)
- Hydra + YAML config support

---

## Project Structure

- models/: LLANet, E-FPN, CBAM modules
- losses/: Heatmap, SBI, and binary mask losses
- datasets/: CelebDF, DFDC, DFD loaders
- package_utils/: Restormer preprocessing, landmark extraction, cropping
- scripts/: Train/test pipelines (train.sh, test.sh)

---


## How to Run

1. Clone this repository

```bash
git clone https://github.com/yourusername/dfd-llanet.git
cd dfd-llanet
```

2. Setup environment
``` bash
conda create -n dfd python=3.8
conda activate dfd
pip install -r requirements.txt
```

3. Train model

``` bash
bash scripts/train.sh
```

4.Evaluate on test set
``` bash
bash scripts/test.sh

```


