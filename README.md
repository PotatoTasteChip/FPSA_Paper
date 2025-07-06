# YOLO-FPSA: YOLO + Frequency Phase Self-Attention

##  Overview
This repo modifies Ultralytics YOLO to add FFT-based frequency and phase attention for object detection in distorted or blurry or 물방울 images.

##  How to Use

```bash
conda env create -f environment.yml
conda activate FPSA_Paper

python ./scripts/train.py