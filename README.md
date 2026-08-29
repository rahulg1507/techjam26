# techjam26 — Robust AI-Generated Image Detection

TikTok TechJam 2026 — Track 5: Robust Detection of AI-Generated Images Under Real-World Transformations

## Project Overview
A model that classifies images as AI-generated vs authentic, robust to real-world
post-processing (JPEG compression, blur, resize, noise, color jitter, cropping).

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
│   ├── datasets.py     # dataset loading + labeling
│   ├── transforms.py   # robustness augmentations (JPEG, blur, noise, etc.)
│   ├── features.py     # CLIP feature extraction
│   ├── train.py         # train classifier on top of features
│   ├── infer.py         # inference script -> outputs JSON {image_path, pred}
│   └── evaluate.py      # clean vs transformed robustness comparison
├── notebooks/          # exploration / EDA
├── outputs/            # predictions, checkpoints (gitignored)
├── reports/            # error analysis, robustness table, tech report
└── requirements.txt
```

## Reproducing Results
1. Download datasets per `data/README.md`
2. `python src/train.py --config configs/baseline.yaml`
3. `python src/infer.py --input_dir <path_to_images> --output outputs/preds.json`
4. `python src/evaluate.py --preds outputs/preds.json --labels <ground_truth.csv>`

## Datasets Used
- CIFAKE (Kaggle) — https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
- SID_Set (Hugging Face) — https://huggingface.co/datasets/saberzl/SID_Set
- WildFake (ModelScope) — https://modelscope.cn/datasets/hy2628982280/WildFake/summary
- Validation only (not for training): COCO val2017 (real) + DALL·E Advanced (AIGC)

## Limitations & Future Work
_(fill in before submission)_

## Demo Video
_(YouTube link here)_
