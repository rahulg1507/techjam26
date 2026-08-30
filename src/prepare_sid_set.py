"""Create a reproducible REAL/FAKE image sample from the SID-Set parquet shards."""

import argparse
import io
import random
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from tqdm import tqdm


REAL_LABEL = 0
FAKE_LABEL = 1
TAMPERED_LABEL = 2
REAL_DIRECTORY_NAME = "REAL"
FAKE_DIRECTORY_NAME = "FAKE"
DEFAULT_PARQUET_DIRECTORY = "data/sid_set/data"
DEFAULT_OUTPUT_DIRECTORY = "data/sid_set_sample"
DEFAULT_MAX_PER_CLASS = 1000
DEFAULT_NUMBER_OF_FILES = 4
DEFAULT_RANDOM_SEED = 42
PARQUET_COLUMNS = ["img_id", "image", "label"]
PARQUET_FILE_TEMPLATE = "train-{file_index:05d}-of-00249.parquet"


def reservoir_add(
    sample: list[tuple[Any, bytes]],
    candidate: tuple[Any, bytes],
    candidate_count: int,
    maximum_size: int,
    random_generator: random.Random,
) -> None:
    """Keep a uniform fixed-size sample without retaining every image in memory."""
    if len(sample) < maximum_size:
        sample.append(candidate)
        return

    replacement_index = random_generator.randrange(candidate_count)
    if replacement_index < maximum_size:
        sample[replacement_index] = candidate


def extract_image_bytes(image_data: Any) -> bytes:
    """Return the encoded image bytes from the parquet image column."""
    if not isinstance(image_data, dict) or "bytes" not in image_data:
        raise ValueError("The image column must contain dictionaries with a 'bytes' key")

    image_bytes = image_data["bytes"]
    if not isinstance(image_bytes, (bytes, bytearray)):
        raise ValueError("The image 'bytes' value must be bytes")
    return bytes(image_bytes)


def sample_images(
    parquet_directory: Path, number_of_files: int, maximum_per_class: int
) -> dict[int, list[tuple[Any, bytes]]]:
    """Read parquet shards and return uniform samples for real and fake images."""
    samples = {REAL_LABEL: [], FAKE_LABEL: []}
    candidate_counts = {REAL_LABEL: 0, FAKE_LABEL: 0}
    random_generator = random.Random(DEFAULT_RANDOM_SEED)

    parquet_paths = [
        parquet_directory / PARQUET_FILE_TEMPLATE.format(file_index=file_index)
        for file_index in range(number_of_files)
    ]
    missing_paths = [path for path in parquet_paths if not path.is_file()]
    if missing_paths:
        missing_path_list = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing parquet file(s): {missing_path_list}")

    for parquet_path in tqdm(parquet_paths, desc="Reading parquet files", unit="file"):
        dataframe = pd.read_parquet(parquet_path, columns=PARQUET_COLUMNS)
        for row in dataframe.itertuples(index=False):
            if row.label not in samples:
                continue

            candidate_counts[row.label] += 1
            reservoir_add(
                samples[row.label],
                (row.img_id, extract_image_bytes(row.image)),
                candidate_counts[row.label],
                maximum_per_class,
                random_generator,
            )

    return samples


def save_images(samples: dict[int, list[tuple[Any, bytes]]], output_directory: Path) -> None:
    """Decode and save sampled images in the directory layout used by training."""
    output_directories = {
        REAL_LABEL: output_directory / REAL_DIRECTORY_NAME,
        FAKE_LABEL: output_directory / FAKE_DIRECTORY_NAME,
    }
    for directory in output_directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    total_images = sum(len(sample) for sample in samples.values())
    progress_bar = tqdm(total=total_images, desc="Saving images", unit="image")
    for label, sample in samples.items():
        for image_id, image_bytes in sample:
            output_path = output_directories[label] / f"{image_id}.jpg"
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.convert("RGB").save(output_path, format="JPEG")
            progress_bar.update(1)
    progress_bar.close()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for selecting and exporting the sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet_dir",
        default=DEFAULT_PARQUET_DIRECTORY,
        help=f"Directory containing SID-Set parquet files (default: {DEFAULT_PARQUET_DIRECTORY})",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=f"Directory for REAL and FAKE images (default: {DEFAULT_OUTPUT_DIRECTORY})",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=DEFAULT_MAX_PER_CLASS,
        help=f"Maximum images to export for each class (default: {DEFAULT_MAX_PER_CLASS})",
    )
    parser.add_argument(
        "--num_files",
        type=int,
        default=DEFAULT_NUMBER_OF_FILES,
        help=f"Number of sequential shards to read (default: {DEFAULT_NUMBER_OF_FILES})",
    )
    return parser.parse_args()


def main() -> None:
    """Build the requested REAL/FAKE SID-Set sample."""
    arguments = parse_arguments()
    if arguments.max_per_class < 1:
        raise ValueError("--max_per_class must be at least 1")
    if arguments.num_files < 1:
        raise ValueError("--num_files must be at least 1")

    samples = sample_images(
        Path(arguments.parquet_dir), arguments.num_files, arguments.max_per_class
    )
    save_images(samples, Path(arguments.output_dir))

    print(
        f"Saved {len(samples[REAL_LABEL])} real and {len(samples[FAKE_LABEL])} "
        f"fake images to {arguments.output_dir}"
    )


if __name__ == "__main__":
    main()
