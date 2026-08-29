"""Choose CLIP-only or fused inference based on an image's blur level."""

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

from blur_detector import estimate_blur_score
from features import ClipFeatureExtractor
from frequency_features import extract_frequency_features


DEFAULT_MAXIMUM_IMAGES_PER_CLASS = 200
DEFAULT_PERCENTILE_VALUES = (10, 25, 50, 75)
FAKE_CLASS_INDEX = 1
CLIP_ONLY_FEATURE_DIMENSION = 1
INITIAL_BEST_ACCURACY = -1.0
PREDICTION_THRESHOLD = 0.5


def extract_adaptive_features(
    image,
    feature_extractor: ClipFeatureExtractor,
    use_frequency_features: bool,
) -> np.ndarray:
    """Extract one CLIP embedding, optionally concatenated with frequency features."""
    clip_features = feature_extractor.extract(image).numpy()
    if not use_frequency_features:
        return clip_features.reshape(CLIP_ONLY_FEATURE_DIMENSION, -1)

    frequency_features = extract_frequency_features(image)
    fused_features = np.concatenate((clip_features, frequency_features))
    return fused_features.reshape(CLIP_ONLY_FEATURE_DIMENSION, -1)


def adaptive_predict(
    image,
    clip_only_classifier,
    fused_classifier,
    feature_extractor: ClipFeatureExtractor,
    blur_threshold: float,
) -> float:
    """Predict FAKE probability using the classifier suited to image detail."""
    blur_score = estimate_blur_score(image)
    use_clip_only_features = blur_score < blur_threshold
    classifier = (
        clip_only_classifier if use_clip_only_features else fused_classifier
    )
    features = extract_adaptive_features(
        image,
        feature_extractor,
        use_frequency_features=not use_clip_only_features,
    )
    return float(classifier.predict_proba(features)[0, FAKE_CLASS_INDEX])


def find_best_threshold(
    images,
    labels,
    clip_only_classifier,
    fused_classifier,
    feature_extractor: ClipFeatureExtractor,
    candidate_thresholds,
) -> float:
    """Select the blur threshold with the highest adaptive validation accuracy."""
    best_threshold = None
    best_accuracy = INITIAL_BEST_ACCURACY
    for threshold in candidate_thresholds:
        predictions = [
            adaptive_predict(
                image,
                clip_only_classifier,
                fused_classifier,
                feature_extractor,
                float(threshold),
            )
            for image in images
        ]
        predicted_labels = [
            int(prediction >= PREDICTION_THRESHOLD) for prediction in predictions
        ]
        accuracy = accuracy_score(labels, predicted_labels)
        print(f"Blur threshold {float(threshold):.4f}: accuracy {accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    if best_threshold is None:
        raise ValueError("candidate_thresholds must contain at least one threshold")
    return best_threshold


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for adaptive threshold tuning."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, help="Directory containing REAL and FAKE")
    parser.add_argument(
        "--clip_only_classifier", required=True, help="Path to the CLIP-only classifier"
    )
    parser.add_argument(
        "--fused_classifier", required=True, help="Path to the fused-feature classifier"
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=DEFAULT_MAXIMUM_IMAGES_PER_CLASS,
        help="Maximum images to load from each class for tuning",
    )
    return parser.parse_args()


def main() -> None:
    """Load a tuning sample and print the best adaptive blur threshold."""
    from evaluate import load_labeled_images

    arguments = parse_arguments()
    if arguments.max_per_class < 1:
        raise ValueError("max_per_class must be at least 1")

    images, labels = load_labeled_images(
        arguments.data_dir, max_per_class=arguments.max_per_class
    )
    clip_only_classifier = _load_classifier(arguments.clip_only_classifier)
    fused_classifier = _load_classifier(arguments.fused_classifier)
    feature_extractor = ClipFeatureExtractor()
    blur_scores = np.array([estimate_blur_score(image) for image in images])
    candidate_thresholds = np.percentile(
        blur_scores, DEFAULT_PERCENTILE_VALUES
    )
    best_threshold = find_best_threshold(
        images,
        labels,
        clip_only_classifier,
        fused_classifier,
        feature_extractor,
        candidate_thresholds,
    )
    print(f"Best blur threshold: {best_threshold:.4f}")


def _load_classifier(classifier_path: str):
    """Load a serialized classifier for adaptive inference."""
    with Path(classifier_path).open("rb") as classifier_file:
        return pickle.load(classifier_file)


if __name__ == "__main__":
    main()
