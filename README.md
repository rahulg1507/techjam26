# techjam26 — Robust AI-Generated Image Detection

TikTok TechJam 2026 — Track 5: Robust Detection of AI-Generated Images Under Real-World Transformations

## Project Overview
A model that classifies images as AI-generated vs authentic, robust to real-world
post-processing (JPEG compression, blur, resize, noise, color jitter, cropping).
It fuses CLIP embeddings with frequency-domain (FFT) features and uses an adaptive
blur-detection gate to route each image to either a CLIP-only or fused classifier,
since frequency fusion improves clean accuracy but degrades under heavy blur or
downsampling.

## Team
| Name | Role | Contribution |
|------|------|---------------|
| TBD  | Data pipeline | |
| TBD  | Model training | |
| TBD  | Evaluation harness | |
| TBD  | Error analysis / report | |
| TBD  | Demo video / writeup | |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Project Structure
```
techjam26/
├── data/               # datasets (gitignored — see data/README.md)
├── src/
│   ├── features.py           # CLIP feature extraction
│   ├── transforms.py         # robustness augmentations (JPEG, blur, noise, etc.)
│   ├── frequency_features.py # FFT frequency-domain feature extraction
│   ├── blur_detector.py      # Laplacian blur-score heuristic
│   ├── adaptive_predict.py   # blur-gated CLIP-only/fused prediction
│   ├── train.py              # classifier training
│   ├── evaluate.py           # clean vs transformed robustness comparison
│   ├── error_analysis.py     # adaptive misclassification analysis
│   └── infer.py              # inference script -> outputs JSON {image_path, pred}
├── outputs/            # predictions, checkpoints (gitignored)
├── reports/            # error analysis, robustness table, tech report
└── requirements.txt
```

## Reproducing Results
```bash
python src/train.py --data_dir data/cifake/train --output outputs/classifier.pkl
python src/evaluate.py --data_dir data/cifake/test --mode adaptive --clip_only_classifier <path> --fused_classifier <path> --blur_threshold 1197.3215
```

## Datasets Used
- CIFAKE (Kaggle) — https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
- SID_Set (Hugging Face) — https://huggingface.co/datasets/saberzl/SID_Set
- WildFake (ModelScope) — https://modelscope.cn/datasets/hy2628982280/WildFake/summary
- Validation only (not for training): COCO val2017 (real) + DALL·E Advanced (AIGC)

## Limitations & Future Work
_(fill in before submission)_

## Demo Video
_(YouTube link here)_
