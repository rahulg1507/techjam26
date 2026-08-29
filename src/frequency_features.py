"""Frequency features expose AIGC artifacts that semantic features can miss.

Compression and blur alter frequency-domain artifacts differently from semantic
CLIP features, so combining both feature types can improve robustness when
generated or authentic images undergo real-world transformations.
"""

import numpy as np
from PIL import Image


FREQUENCY_BAND_COUNT = 8
LOG_MAGNITUDE_OFFSET = 1.0
EMPTY_BAND_FEATURE_VALUE = 0.0
GRAYSCALE_CHANNEL_COUNT = 1
HALF_DIMENSION_DIVISOR = 2.0
ORIGIN_COORDINATE = 0.0
SMOKE_TEST_IMAGE_SIZE = 224
SMOKE_TEST_COLOR_VALUE = 128


def extract_frequency_features(image: Image.Image) -> np.ndarray:
    """Return mean log-magnitude values for concentric radial FFT bands."""
    grayscale_pixels = np.asarray(image.convert("L"), dtype=np.float32)
    frequency_spectrum = np.fft.fftshift(np.fft.fft2(grayscale_pixels))
    magnitude_spectrum = np.abs(frequency_spectrum)
    log_magnitude_spectrum = np.log(
        magnitude_spectrum + LOG_MAGNITUDE_OFFSET
    )

    image_height, image_width = grayscale_pixels.shape
    vertical_coordinates, horizontal_coordinates = np.indices(
        (image_height, image_width), dtype=np.float32
    )
    center_y = (image_height - GRAYSCALE_CHANNEL_COUNT) / HALF_DIMENSION_DIVISOR
    center_x = (image_width - GRAYSCALE_CHANNEL_COUNT) / HALF_DIMENSION_DIVISOR
    radial_distance = np.sqrt(
        (vertical_coordinates - center_y) ** 2
        + (horizontal_coordinates - center_x) ** 2
    )
    maximum_radius = float(radial_distance.max())
    band_edges = np.linspace(
        ORIGIN_COORDINATE,
        maximum_radius,
        FREQUENCY_BAND_COUNT + GRAYSCALE_CHANNEL_COUNT,
    )
    band_indices = np.digitize(radial_distance, band_edges, right=False)
    band_indices = np.clip(
        band_indices - GRAYSCALE_CHANNEL_COUNT,
        ORIGIN_COORDINATE,
        FREQUENCY_BAND_COUNT - GRAYSCALE_CHANNEL_COUNT,
    )

    features = np.full(
        FREQUENCY_BAND_COUNT, EMPTY_BAND_FEATURE_VALUE, dtype=np.float32
    )
    for band_index in range(FREQUENCY_BAND_COUNT):
        band_pixels = log_magnitude_spectrum[band_indices == band_index]
        if band_pixels.size:
            features[band_index] = float(band_pixels.mean())
    return features


def main() -> None:
    """Run a smoke test to verify the fixed-length frequency descriptor."""
    dummy_image = Image.new(
        "RGB",
        (SMOKE_TEST_IMAGE_SIZE, SMOKE_TEST_IMAGE_SIZE),
        color=(SMOKE_TEST_COLOR_VALUE,) * 3,
    )
    feature_vector = extract_frequency_features(dummy_image)
    print(f"Frequency feature vector shape: {feature_vector.shape}")


if __name__ == "__main__":
    main()
