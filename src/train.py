"""Train a lightweight classifier on fused CLIP and frequency features."""

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from features import ClipFeatureExtractor
from frequency_features import extract_frequency_features
from transforms import random_augment


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REAL_DIRECTORY_NAME = "REAL"
FAKE_DIRECTORY_NAME = "FAKE"
REAL_LABEL = 0
FAKE_LABEL = 1
DEFAULT_OUTPUT_PATH = "outputs/classifier.pkl"
DEFAULT_AUGMENTATION_PROBABILITY = 0.5
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_BATCH_SIZE = 32
DEFAULT_RANDOM_SEED = 42
CLASSIFIER_MAXIMUM_ITERATIONS = 1000
METRIC_ZERO_DIVISION_VALUE = 0
PROBABILITY_MINIMUM = 0.0
PROBABILITY_MAXIMUM = 1.0


def collect_image_paths(
    data_directory: str, max_per_class: int | None = None
) -> tuple[list[Path], list[int]]:
    """Collect labeled paths from REAL and FAKE directories."""
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

        for image_path in class_image_paths:
            image_paths.append(image_path)
            labels.append(label)

    if not image_paths:
        raise ValueError(f"No supported images found in {root}")

    return image_paths, labels


def load_images(image_paths: list[Path]) -> list[Image.Image]:
    """Load images into memory so files are closed before feature extraction."""
    images = []
    for image_path in tqdm(image_paths, desc="Loading images", unit="image"):
        with Image.open(image_path) as image:
            images.append(image.convert("RGB").copy())
    return images


def augment_training_images(
    images: list[Image.Image], augmentation_probability: float
) -> list[Image.Image]:
    """Augment a fraction of images to expose the classifier to real-world distortions."""
    augmented_images = []
    for image in images:
        if random.random() < augmentation_probability:
            augmented_images.append(random_augment(image))
        else:
            augmented_images.append(image)
    return augmented_images


def extract_embeddings(
    images: list[Image.Image],
    feature_extractor: ClipFeatureExtractor,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> object:
    """Extract fused CLIP and frequency embeddings in efficient batches.

    CLIP inference remains batched for speed while each image receives a
    frequency-domain descriptor that captures artifacts absent from semantic
    features.
    """
    embedding_batches = []
    for start_index in tqdm(
        range(0, len(images), batch_size), desc="Extracting embeddings", unit="batch"
    ):
        image_batch = images[start_index : start_index + batch_size]
        clip_embeddings = feature_extractor.extract_batch(image_batch)
        frequency_feature_arrays = [
            extract_frequency_features(image) for image in image_batch
        ]
        frequency_embeddings = torch.from_numpy(
            np.stack(frequency_feature_arrays)
        ).to(dtype=clip_embeddings.dtype)
        embedding_batches.append(
            torch.cat((clip_embeddings, frequency_embeddings), dim=1)
        )

    return torch.cat(embedding_batches, dim=0).numpy()


def calculate_metrics(expected_labels, predicted_labels) -> dict[str, float]:
    """Calculate the metrics used to assess classification and robustness."""
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


def report_metrics(split_name: str, metrics: dict[str, float]) -> None:
    """Print a consistent summary for one dataset split."""
    print(
        f"{split_name} — "
        f"accuracy: {metrics['accuracy']:.4f}, "
        f"precision: {metrics['precision']:.4f}, "
        f"recall: {metrics['recall']:.4f}, "
        f"F1: {metrics['f1']:.4f}"
    )


def save_classifier(classifier, output_path: str) -> None:
    """Persist the fitted classifier for use by the inference script."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output_file:
        pickle.dump(classifier, output_file)


def train_classifier(
    data_directory: str,
    output_path: str = DEFAULT_OUTPUT_PATH,
    augmentation_probability: float = DEFAULT_AUGMENTATION_PROBABILITY,
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
    max_per_class: int | None = None,
) -> LogisticRegression:
    """Train, evaluate, and save a logistic regression image classifier."""
    if not PROBABILITY_MINIMUM <= augmentation_probability <= PROBABILITY_MAXIMUM:
        raise ValueError("augmentation probability must be between 0 and 1")
    if not PROBABILITY_MINIMUM < validation_split < PROBABILITY_MAXIMUM:
        raise ValueError("validation split must be between 0 and 1")

    image_paths, labels = collect_image_paths(data_directory, max_per_class)
    training_paths, validation_paths, training_labels, validation_labels = train_test_split(
        image_paths,
        labels,
        test_size=validation_split,
        random_state=DEFAULT_RANDOM_SEED,
        stratify=labels,
    )

    feature_extractor = ClipFeatureExtractor()
    training_images = augment_training_images(
        load_images(training_paths), augmentation_probability
    )
    validation_images = load_images(validation_paths)
    training_embeddings = extract_embeddings(training_images, feature_extractor)
    validation_embeddings = extract_embeddings(validation_images, feature_extractor)

    classifier = LogisticRegression(max_iter=CLASSIFIER_MAXIMUM_ITERATIONS)
    classifier.fit(training_embeddings, training_labels)
    report_metrics(
        "Train",
        calculate_metrics(training_labels, classifier.predict(training_embeddings)),
    )
    report_metrics(
        "Validation",
        calculate_metrics(validation_labels, classifier.predict(validation_embeddings)),
    )
    save_classifier(classifier, output_path)
    print(f"Saved classifier to {output_path}")
    return classifier


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for the training workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, help="Directory containing REAL and FAKE")
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_PATH, help="Path to save the classifier"
    )
    parser.add_argument(
        "--augment_prob",
        type=float,
        default=DEFAULT_AUGMENTATION_PROBABILITY,
        help="Fraction of training images to augment",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=DEFAULT_VALIDATION_SPLIT,
        help="Fraction of images reserved for validation",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=None,
        help="Maximum number of images to sample from each class",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line training workflow."""
    arguments = parse_arguments()
    train_classifier(
        data_directory=arguments.data_dir,
        output_path=arguments.output,
        augmentation_probability=arguments.augment_prob,
        validation_split=arguments.val_split,
        max_per_class=arguments.max_per_class,
    )


if __name__ == "__main__":
    main()
