"""
Inference script — required deliverable per problem statement Section 5.5:
"A script that takes an image directory as input and outputs a confidence
score for each image... Output should be a JSON file containing
image_path and pred for each image."
"""
import argparse
import json
import pickle
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from features import ClipFeatureExtractor

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_classifier(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory of images to score")
    parser.add_argument("--output", required=True, help="Path to write predictions JSON")
    parser.add_argument("--classifier", default="outputs/classifier.pkl")
    args = parser.parse_args()

    extractor = ClipFeatureExtractor()
    clf = load_classifier(args.classifier)

    image_paths = [
        p for p in Path(args.input_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTS
    ]

    results = []
    for path in tqdm(image_paths, desc="Scoring images"):
        try:
            img = Image.open(path)
            feat = extractor.extract(img).numpy().reshape(1, -1)
            pred = float(clf.predict_proba(feat)[0, 1])  # P(AIGC)
        except Exception as e:
            print(f"Warning: failed on {path}: {e}")
            pred = None
        results.append({"image_path": str(path), "pred": pred})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
