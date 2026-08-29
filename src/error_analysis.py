"""Summarize representative classifier errors for the evaluation report."""

import argparse
import json
from collections import Counter
from pathlib import Path


REAL_DIRECTORY_NAME = "REAL"
FAKE_DIRECTORY_NAME = "FAKE"
REAL_LABEL = 0
FAKE_LABEL = 1
CONFIDENCE_THRESHOLD = 0.5
DEFAULT_PREDICTIONS_PATH = "outputs/preds.json"
DEFAULT_REPORT_PATH = "reports/error_analysis.md"
DEFAULT_EXAMPLE_COUNT = 5
TRANSFORM_NAMES = (
    "jpeg_90",
    "jpeg_70",
    "jpeg_50",
    "jpeg_30",
    "blur_0.5",
    "blur_1.0",
    "blur_2.0",
    "resize_0.5",
    "resize_0.25",
    "noise_0.02",
    "noise_0.05",
    "noise_0.10",
    "color_jitter",
    "center_crop",
)


def load_predictions(predictions_path: str) -> list[dict]:
    """Load image paths and fake-confidence scores from inference JSON."""
    with Path(predictions_path).open("r", encoding="utf-8") as predictions_file:
        predictions = json.load(predictions_file)
    if not isinstance(predictions, list):
        raise ValueError("Prediction file must contain a JSON list")
    return predictions


def find_ground_truth_label(image_path: str, data_directory: str) -> int:
    """Infer the label from the nearest REAL or FAKE directory in the dataset."""
    path_parts = [part.upper() for part in Path(image_path).parts]
    label_by_directory_name = {
        REAL_DIRECTORY_NAME: REAL_LABEL,
        FAKE_DIRECTORY_NAME: FAKE_LABEL,
    }
    for path_part in reversed(path_parts):
        if path_part in label_by_directory_name:
            return label_by_directory_name[path_part]

    relative_path = Path(image_path)
    if not relative_path.is_absolute():
        relative_path = Path(data_directory) / relative_path
    path_parts = [part.upper() for part in relative_path.parts]
    for path_part in reversed(path_parts):
        if path_part in label_by_directory_name:
            return label_by_directory_name[path_part]

    raise ValueError(f"Could not determine REAL/FAKE label for {image_path}")


def classify_errors(
    predictions: list[dict], data_directory: str
) -> tuple[list[dict], list[dict]]:
    """Separate valid predictions into false-positive and false-negative records."""
    false_positives = []
    false_negatives = []
    for prediction in predictions:
        image_path = prediction.get("image_path")
        confidence = prediction.get("pred")
        if image_path is None or confidence is None:
            continue
        confidence = float(confidence)
        ground_truth = find_ground_truth_label(image_path, data_directory)
        if ground_truth == REAL_LABEL and confidence >= CONFIDENCE_THRESHOLD:
            false_positives.append(
                {"image_path": image_path, "confidence": confidence}
            )
        elif ground_truth == FAKE_LABEL and confidence < CONFIDENCE_THRESHOLD:
            false_negatives.append(
                {"image_path": image_path, "confidence": confidence}
            )

    false_positives.sort(key=lambda error: error["confidence"], reverse=True)
    false_negatives.sort(key=lambda error: error["confidence"])
    return false_positives, false_negatives


def select_examples(errors: list[dict], example_count: int) -> list[dict]:
    """Keep the most confidently incorrect examples for a concise report."""
    return errors[:example_count]


def identify_patterns(errors: list[dict]) -> list[str]:
    """Identify recurring transform names in error paths when metadata is available."""
    transform_counts = Counter()
    for error in errors:
        path_text = str(error["image_path"]).lower()
        for transform_name in TRANSFORM_NAMES:
            if transform_name.lower() in path_text:
                transform_counts[transform_name] += 1

    if not transform_counts:
        return [
            "No clear transformation pattern was available from the image paths. "
            "Review the examples visually alongside the robustness table."
        ]

    most_common_count = transform_counts.most_common()
    highest_count = most_common_count[0][1]
    recurring_transforms = [
        transform_name
        for transform_name, count in most_common_count
        if count == highest_count
    ]
    transform_list = ", ".join(f"`{name}`" for name in recurring_transforms)
    return [
        f"The most frequent transform marker in error paths was {transform_list} "
        f"({highest_count} example(s)); inspect these conditions first."
    ]


def format_examples(errors: list[dict]) -> list[str]:
    """Format error records as markdown table rows."""
    if not errors:
        return ["| _None_ | — |"]
    return [
        f"| `{error['image_path']}` | {error['confidence']:.4f} |"
        for error in errors
    ]


def build_report(false_positives: list[dict], false_negatives: list[dict]) -> str:
    """Build the concise Error Analysis Note markdown document."""
    all_errors = false_positives + false_negatives
    report_lines = [
        "# Error Analysis Note",
        "",
        "Confidence is the classifier's predicted probability that an image is FAKE "
        f"(threshold: {CONFIDENCE_THRESHOLD:.1f}).",
        "",
        "## False Positives (REAL predicted as FAKE)",
        "",
        "| Image path | FAKE confidence |",
        "|---|---:|",
        *format_examples(false_positives),
        "",
        "## False Negatives (FAKE predicted as REAL)",
        "",
        "| Image path | FAKE confidence |",
        "|---|---:|",
        *format_examples(false_negatives),
        "",
        "## Patterns Noticed",
        "",
    ]
    report_lines.extend(f"- {pattern}" for pattern in identify_patterns(all_errors))
    report_lines.append("")
    return "\n".join(report_lines)


def save_report(report: str, report_path: str) -> None:
    """Create the report directory and write the analysis note."""
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")


def analyze_errors(
    predictions_path: str = DEFAULT_PREDICTIONS_PATH,
    data_directory: str = "data",
    report_path: str = DEFAULT_REPORT_PATH,
    example_count: int = DEFAULT_EXAMPLE_COUNT,
) -> tuple[list[dict], list[dict]]:
    """Load predictions, summarize errors, and save the markdown note."""
    if example_count < 1:
        raise ValueError("example count must be at least 1")
    predictions = load_predictions(predictions_path)
    false_positives, false_negatives = classify_errors(predictions, data_directory)
    selected_false_positives = select_examples(false_positives, example_count)
    selected_false_negatives = select_examples(false_negatives, example_count)
    save_report(
        build_report(selected_false_positives, selected_false_negatives), report_path
    )
    print(
        f"Found {len(false_positives)} false positives and "
        f"{len(false_negatives)} false negatives"
    )
    print(f"Wrote error analysis report to {report_path}")
    return selected_false_positives, selected_false_negatives


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for error analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        "--preds",
        default=DEFAULT_PREDICTIONS_PATH,
        help="Inference JSON containing image_path and pred fields",
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Dataset directory containing REAL and FAKE folders",
    )
    parser.add_argument(
        "--report_path",
        default=DEFAULT_REPORT_PATH,
        help="Path to the markdown error-analysis note",
    )
    parser.add_argument(
        "--example_count",
        type=int,
        default=DEFAULT_EXAMPLE_COUNT,
        help="Number of examples to include per error category",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line error-analysis workflow."""
    arguments = parse_arguments()
    analyze_errors(
        predictions_path=arguments.predictions,
        data_directory=arguments.data_dir,
        report_path=arguments.report_path,
        example_count=arguments.example_count,
    )


if __name__ == "__main__":
    main()
