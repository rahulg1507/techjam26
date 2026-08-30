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

from adaptive_predict import adaptive_predict
from features import ClipFeatureExtractor

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_classifier(path: str):
    """Load a serialized classifier from disk."""
    with Path(path).open("rb") as classifier_file:
        return pickle.load(classifier_file)


def main() -> None:
    """Score images with the blur-gated adaptive prediction pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory of images to score")
    parser.add_argument("--output", required=True, help="Path to write predictions JSON")
    parser.add_argument(
        "--clip_only_classifier",
        required=True,
        help="Path to the CLIP-only pickled classifier",
    )
    parser.add_argument(
        "--fused_classifier",
        required=True,
        help="Path to the fused-feature pickled classifier",
    )
    parser.add_argument(
        "--blur_threshold",
        type=float,
        required=True,
        help="Use CLIP-only prediction below this blur score",
    )
    args = parser.parse_args()

    extractor = ClipFeatureExtractor()
    clip_only_classifier = load_classifier(args.clip_only_classifier)
    fused_classifier = load_classifier(args.fused_classifier)

    image_paths = [
        p for p in Path(args.input_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTS
    ]

    results = []
    for path in tqdm(image_paths, desc="Scoring images"):
        try:
            with Image.open(path) as image:
                pred = adaptive_predict(
                    image,
                    clip_only_classifier,
                    fused_classifier,
                    extractor,
                    args.blur_threshold,
                )
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
