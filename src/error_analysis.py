"""Record misclassified images from adaptive robustness evaluation."""

import argparse
import csv
import pickle
import random
from collections import Counter
from pathlib import Path

from PIL import Image

from adaptive_predict import adaptive_predict
from blur_detector import estimate_blur_score
from features import ClipFeatureExtractor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REAL_DIRECTORY_NAME = "REAL"
FAKE_DIRECTORY_NAME = "FAKE"
REAL_LABEL = 0
FAKE_LABEL = 1
PREDICTION_THRESHOLD = 0.5
DEFAULT_BLUR_THRESHOLD = 1197.3215
DEFAULT_MAXIMUM_IMAGES_PER_CLASS = 300
DEFAULT_OUTPUT_PATH = "reports/misclassified.csv"
DEFAULT_RANDOM_SEED = 42
TRANSFORM_MARKERS = (
    "jpeg_90", "jpeg_70", "jpeg_50", "jpeg_30", "blur_0.5", "blur_1.0",
    "blur_2.0", "resize_0.5", "resize_0.25", "noise_0.02", "noise_0.05",
    "noise_0.10", "color_jitter", "center_crop",
)


def load_classifier(classifier_path: str):
    """Load a serialized classifier for adaptive prediction."""
    with Path(classifier_path).open("rb") as classifier_file:
        return pickle.load(classifier_file)


def load_labeled_paths(
    data_directory: str, max_per_class: int | None
) -> tuple[list[Path], list[int]]:
    """Collect sampled image paths and labels from REAL and FAKE folders."""
    if max_per_class is not None and max_per_class < 1:
        raise ValueError("max_per_class must be at least 1")

    root = Path(data_directory)
    image_paths = []
    labels = []
    sampling_generator = random.Random(DEFAULT_RANDOM_SEED)
    for directory_name, label in (
        (REAL_DIRECTORY_NAME, REAL_LABEL),
        (FAKE_DIRECTORY_NAME, FAKE_LABEL),
    ):
        class_directory = root / directory_name
        if not class_directory.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_directory}")
        class_image_paths = [
            image_path
            for image_path in sorted(class_directory.rglob("*"))
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if max_per_class is not None and len(class_image_paths) > max_per_class:
            class_image_paths = sorted(
                sampling_generator.sample(class_image_paths, max_per_class)
            )
        image_paths.extend(class_image_paths)
        labels.extend([label] * len(class_image_paths))

    if not image_paths:
        raise ValueError(f"No supported images found in {root}")
    return image_paths, labels


def classify_condition(image_path: Path) -> str:
    """Return a transform marker from a path, or clean when none is present."""
    path_text = str(image_path).lower()
    for transform_marker in TRANSFORM_MARKERS:
        if transform_marker.lower() in path_text:
            return transform_marker
    return "clean"


def analyze_misclassifications(
    image_paths: list[Path],
    labels: list[int],
    clip_only_classifier,
    fused_classifier,
    feature_extractor: ClipFeatureExtractor,
    blur_threshold: float,
) -> list[dict[str, object]]:
    """Run adaptive inference and return records only for incorrect predictions."""
    misclassified = []
    for image_path, true_label in zip(image_paths, labels):
        with Image.open(image_path) as image:
            image_copy = image.convert("RGB").copy()
        predicted_probability = adaptive_predict(
            image_copy,
            clip_only_classifier,
            fused_classifier,
            feature_extractor,
            blur_threshold,
        )
        predicted_label = int(predicted_probability >= PREDICTION_THRESHOLD)
        if predicted_label != true_label:
            misclassified.append(
                {
                    "image_path": str(image_path),
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "predicted_probability": predicted_probability,
                    "blur_score": estimate_blur_score(image_copy),
                    "condition": classify_condition(image_path),
                }
            )
    return misclassified


def save_misclassifications(
    misclassified: list[dict[str, object]], output_path: str
) -> None:
    """Write misclassification details to a CSV for follow-up analysis."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "image_path", "true_label", "predicted_label",
        "predicted_probability", "blur_score",
    ]
    with destination.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=columns)
        writer.writeheader()
        for record in misclassified:
            writer.writerow({column: record[column] for column in columns})


def print_summary(misclassified: list[dict[str, object]]) -> None:
    """Print total errors and their clean/transform path breakdown."""
    condition_counts = Counter(record["condition"] for record in misclassified)
    print(f"Total misclassified: {len(misclassified)}")
    print("Breakdown by condition:")
    if not condition_counts:
        print("  clean: 0")
    else:
        for condition, count in sorted(condition_counts.items()):
            print(f"  {condition}: {count}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for adaptive error analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, help="Directory containing REAL and FAKE")
    parser.add_argument(
        "--clip_only_classifier", required=True, help="Path to the CLIP-only classifier"
    )
    parser.add_argument(
        "--fused_classifier", required=True, help="Path to the fused classifier"
    )
    parser.add_argument(
        "--blur_threshold", type=float, default=DEFAULT_BLUR_THRESHOLD,
        help="Blur threshold used to choose the classifier",
    )
    parser.add_argument(
        "--max_per_class", type=int, default=DEFAULT_MAXIMUM_IMAGES_PER_CLASS,
        help="Maximum number of images to evaluate from each class",
    )
    return parser.parse_args()


def main() -> None:
    """Run adaptive inference and save misclassified image details."""
    arguments = parse_arguments()
    image_paths, labels = load_labeled_paths(
        arguments.data_dir, arguments.max_per_class
    )
    feature_extractor = ClipFeatureExtractor()
    misclassified = analyze_misclassifications(
        image_paths,
        labels,
        load_classifier(arguments.clip_only_classifier),
        load_classifier(arguments.fused_classifier),
        feature_extractor,
        arguments.blur_threshold,
    )
    save_misclassifications(misclassified, DEFAULT_OUTPUT_PATH)
    print_summary(misclassified)
    print(f"Wrote misclassifications to {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
