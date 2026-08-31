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
| Name                         | Contribution                                                                                                                                        |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Rahul Ganesh                 | Data pipeline, model training, adaptive gating system, robustness and cross-dataset evaluation, error analysis, inference script, and documentation |
| Gokul Sai Ramesh             | Demo video production                                                                                                                               |
| Kumaresan Kugapryan          | Track selection and early project discussion                                                                                                        |
| Ryan Priyank Sunjai Dhas     | Testing and feedback on early prototypes                                                                                                            |
| Gowrinath Jayamani Kanagaraj | Project planning and coordination                                                                                                                   |

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

## Results

**In-distribution (CIFAKE test set):**

| Mode                     | Clean Accuracy |
| ------------------------ | -------------- |
| CLIP-only                | 88.4%          |
| Fused (CLIP + frequency) | 90.7%          |
| Adaptive (blur-gated)    | 90.6%          |

Adaptive mode matches or exceeds both baselines across most robustness transforms
(JPEG compression, blur, resize, noise, color jitter, center crop), and specifically
recovers CLIP-only's robustness on heavy blur/downsampling where naive fusion
degrades sharply. Full results in reports/robustness_table_adaptive.md.

**Cross-dataset generalization (zero-shot on SID_Set):**

Accuracy dropped to ~57% when evaluating our CIFAKE-trained adaptive model on
SID_Set (a higher-resolution, differently-sourced dataset), with the model
skewing toward predicting "fake" (recall ~99%, precision ~54%). This indicates
our classifier learned CIFAKE-specific patterns rather than fully generalizable
AI-generation signatures — see Limitations for discussion. Full results in
reports/robustness_table_sidset.md.

## Limitations & Future Work

- **Distribution shift over transform robustness:** Our adaptive model handles
  post-processing transforms (JPEG, blur, resize, noise) well within-distribution,
  but zero-shot cross-dataset evaluation on SID_Set revealed a sharp accuracy drop
  (~90% to ~57%), indicating the classifier learned CIFAKE-specific patterns rather
  than fully generalizable AI-generation signatures. Robustness to transforms and
  robustness to distribution shift are distinct challenges.
- **Resolution mismatch:** Our blur-detection threshold was tuned on CIFAKE's
  native 32x32 images, which have inherently compressed blur-score ranges compared
  to full-resolution datasets like SID_Set. Retuning per-dataset may be necessary.
- **Small evaluation samples:** Due to compute/time constraints, cross-dataset
  evaluation used a 1000-image-per-class sample rather than the full 210K-image
  SID_Set training split.
- **Future work:** domain adaptation or fine-tuning on a mix of datasets, testing
  additional frequency-domain features less tied to resolution, and expanding
  cross-dataset evaluation to WildFake.

## Demo Video
[https://youtu.be/lQfBDpBt76M](https://youtu.be/lQfBDpBt76M)
