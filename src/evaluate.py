"""Evaluate classifier robustness to real-world image transformations."""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from features import ClipFeatureExtractor
from transforms import TRANSFORMS


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REAL_DIRECTORY_NAME = "REAL"
FAKE_DIRECTORY_NAME = "FAKE"
REAL_LABEL = 0
FAKE_LABEL = 1
DEFAULT_CLASSIFIER_PATH = "outputs/classifier.pkl"
DEFAULT_REPORT_PATH = "reports/robustness_table.md"
DEFAULT_BATCH_SIZE = 32
METRIC_ZERO_DIVISION_VALUE = 0


def load_labeled_images(data_directory: str) -> tuple[list[Image.Image], list[int]]:
    """Load RGB images and labels from REAL and FAKE subdirectories."""
    root = Path(data_directory)
    images = []
    labels = []

    for directory_name, label in (
        (REAL_DIRECTORY_NAME, REAL_LABEL),
        (FAKE_DIRECTORY_NAME, FAKE_LABEL),
    ):
        class_directory = root / directory_name
        if not class_directory.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_directory}")

        image_paths = sorted(class_directory.rglob("*"))
        for image_path in image_paths:
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB").copy())
                labels.append(label)

    if not images:
        raise ValueError(f"No supported images found in {root}")

    return images, labels


def load_classifier(classifier_path: str):
    """Load the classifier produced by the training workflow."""
    with Path(classifier_path).open("rb") as classifier_file:
        return pickle.load(classifier_file)


def extract_embeddings(
    images: list[Image.Image],
    feature_extractor: ClipFeatureExtractor,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Extract embeddings in batches so each evaluation condition stays efficient."""
    embedding_batches = []
    for start_index in range(0, len(images), batch_size):
        image_batch = images[start_index : start_index + batch_size]
        embedding_batches.append(feature_extractor.extract_batch(image_batch))
    return torch.cat(embedding_batches, dim=0).numpy()


def calculate_metrics(expected_labels, predicted_labels) -> dict[str, float]:
    """Calculate the classification metrics used in the robustness summary."""
    return {
        "accuracy": accuracy_score(expected_labels, predicted_labels),
        "precision": precision_score(
            expected_labels,
            predicted_labels,
            zero_division=METRIC_ZERO_DIVISION_VALUE,
        ),
        "recall": recall_score(
            expected_labels,
            predicted_labels,
            zero_division=METRIC_ZERO_DIVISION_VALUE,
        ),
        "f1": f1_score(
            expected_labels,
            predicted_labels,
            zero_division=METRIC_ZERO_DIVISION_VALUE,
        ),
    }


def evaluate_condition(
    condition_images: list[Image.Image],
    labels: list[int],
    classifier,
    feature_extractor: ClipFeatureExtractor,
) -> dict[str, float]:
    """Extract features and score one clean or transformed image condition."""
    embeddings = extract_embeddings(condition_images, feature_extractor)
    predictions = classifier.predict(embeddings)
    return calculate_metrics(labels, predictions)


def print_metrics(condition_name: str, metrics: dict[str, float]) -> None:
    """Print metrics for one condition while evaluation is running."""
    print(
        f"{condition_name} — "
        f"accuracy: {metrics['accuracy']:.4f}, "
        f"precision: {metrics['precision']:.4f}, "
        f"recall: {metrics['recall']:.4f}, "
        f"F1: {metrics['f1']:.4f}"
    )


def build_report(metrics_by_condition: dict[str, dict[str, float]]) -> str:
    """Build the markdown table required for the robustness evaluation summary."""
    report_lines = [
        "# Robustness Evaluation Summary",
        "",
        "| Condition | Accuracy | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition_name, metrics in metrics_by_condition.items():
        report_lines.append(
            f"| {condition_name} | {metrics['accuracy']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} |"
        )
    return "\n".join(report_lines) + "\n"


def save_report(report: str, report_path: str) -> None:
    """Create the report directory and write the robustness table."""
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")


def evaluate_dataset(
    data_directory: str,
    classifier_path: str = DEFAULT_CLASSIFIER_PATH,
    report_path: str = DEFAULT_REPORT_PATH,
) -> dict[str, dict[str, float]]:
    """Evaluate clean images and every registered transform, then save the report."""
    images, labels = load_labeled_images(data_directory)
    classifier = load_classifier(classifier_path)
    feature_extractor = ClipFeatureExtractor()
    metrics_by_condition = {}

    clean_metrics = evaluate_condition(images, labels, classifier, feature_extractor)
    metrics_by_condition["Clean"] = clean_metrics
    print_metrics("Clean", clean_metrics)

    for transform_name, transform in TRANSFORMS.items():
        transformed_images = [transform(image) for image in images]
        transform_metrics = evaluate_condition(
            transformed_images, labels, classifier, feature_extractor
        )
        metrics_by_condition[transform_name] = transform_metrics
        print_metrics(transform_name, transform_metrics)

    save_report(build_report(metrics_by_condition), report_path)
    print(f"Wrote robustness report to {report_path}")
    return metrics_by_condition


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for robustness evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, help="Directory containing REAL and FAKE")
    parser.add_argument(
        "--classifier",
        default=DEFAULT_CLASSIFIER_PATH,
        help="Path to the trained classifier",
    )
    parser.add_argument(
        "--report_path",
        default=DEFAULT_REPORT_PATH,
        help="Path to the markdown robustness table",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line robustness evaluation workflow."""
    arguments = parse_arguments()
    evaluate_dataset(
        data_directory=arguments.data_dir,
        classifier_path=arguments.classifier,
        report_path=arguments.report_path,
    )


if __name__ == "__main__":
    main()
