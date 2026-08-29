"""Evaluate classifier robustness to real-world image transformations."""

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from features import ClipFeatureExtractor
from frequency_features import extract_frequency_features
from transforms import TRANSFORMS


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REAL_DIRECTORY_NAME = "REAL"
FAKE_DIRECTORY_NAME = "FAKE"
REAL_LABEL = 0
FAKE_LABEL = 1
DEFAULT_CLASSIFIER_PATH = "outputs/classifier.pkl"
DEFAULT_REPORT_PATH = "reports/robustness_table.md"
DEFAULT_BATCH_SIZE = 32
DEFAULT_RANDOM_SEED = 42
DEFAULT_USE_FREQUENCY_FEATURES = True
METRIC_ZERO_DIVISION_VALUE = 0


def load_labeled_images(
    data_directory: str, max_per_class: int | None = None
) -> tuple[list[Image.Image], list[int]]:
    """Load RGB images and labels from REAL and FAKE subdirectories."""
    if max_per_class is not None and max_per_class < 1:
        raise ValueError("max_per_class must be at least 1")

    root = Path(data_directory)
    images = []
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

        for image_path in tqdm(
            class_image_paths,
            desc=f"Loading {directory_name} images",
            unit="image",
        ):
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
    use_frequency_features: bool = DEFAULT_USE_FREQUENCY_FEATURES,
) -> np.ndarray:
    """Extract CLIP embeddings, optionally fused with frequency features."""
    embedding_batches = []
    for start_index in range(0, len(images), batch_size):
        image_batch = images[start_index : start_index + batch_size]
        clip_embeddings = feature_extractor.extract_batch(image_batch)
        if use_frequency_features:
            frequency_feature_arrays = [
                extract_frequency_features(image) for image in image_batch
            ]
            frequency_embeddings = torch.from_numpy(
                np.stack(frequency_feature_arrays)
            ).to(dtype=clip_embeddings.dtype)
            clip_embeddings = torch.cat(
                (clip_embeddings, frequency_embeddings), dim=1
            )
        embedding_batches.append(clip_embeddings)
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
    use_frequency_features: bool = DEFAULT_USE_FREQUENCY_FEATURES,
) -> dict[str, float]:
    """Extract features and score one clean or transformed image condition."""
    embeddings = extract_embeddings(
        condition_images,
        feature_extractor,
        use_frequency_features=use_frequency_features,
    )
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
    max_per_class: int | None = None,
    use_frequency_features: bool = DEFAULT_USE_FREQUENCY_FEATURES,
) -> dict[str, dict[str, float]]:
    """Evaluate clean images and every registered transform, then save the report."""
    images, labels = load_labeled_images(data_directory, max_per_class)
    classifier = load_classifier(classifier_path)
    feature_extractor = ClipFeatureExtractor()
    metrics_by_condition = {}

    clean_metrics = evaluate_condition(
        images,
        labels,
        classifier,
        feature_extractor,
        use_frequency_features=use_frequency_features,
    )
    metrics_by_condition["Clean"] = clean_metrics
    print_metrics("Clean", clean_metrics)

    for transform_name, transform in tqdm(
        TRANSFORMS.items(), desc="Evaluating transforms", unit="transform"
    ):
        transformed_images = [transform(image) for image in images]
        transform_metrics = evaluate_condition(
            transformed_images,
            labels,
            classifier,
            feature_extractor,
            use_frequency_features=use_frequency_features,
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
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=None,
        help="Maximum number of images to sample from each class",
    )
    parser.add_argument(
        "--use_frequency_features",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_FREQUENCY_FEATURES,
        help="Include frequency-domain features with CLIP embeddings",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line robustness evaluation workflow."""
    arguments = parse_arguments()
    evaluate_dataset(
        data_directory=arguments.data_dir,
        classifier_path=arguments.classifier,
        report_path=arguments.report_path,
        max_per_class=arguments.max_per_class,
        use_frequency_features=arguments.use_frequency_features,
    )


if __name__ == "__main__":
    main()
