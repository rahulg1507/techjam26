"""Estimate image blur with a fast Laplacian-variance heuristic."""

import cv2
import numpy as np
from PIL import Image, ImageFilter


LAPLACIAN_DEPTH = cv2.CV_64F
SMOKE_TEST_IMAGE_SIZE = 128
SMOKE_TEST_CHANNEL_COUNT = 3
SMOKE_TEST_RANDOM_SEED = 42
SMOKE_TEST_BLUR_RADIUS = 8.0
PIXEL_MINIMUM = 0
PIXEL_MAXIMUM = 255


def estimate_blur_score(image: Image.Image) -> float:
    """Return Laplacian-response variance, a fast, well-known blur heuristic.

    No machine-learning model is needed: lower variance indicates a blurrier,
    lower-detail image, while higher variance indicates sharper details.
    """
    grayscale_pixels = np.asarray(image.convert("L"))
    laplacian_response = cv2.Laplacian(grayscale_pixels, LAPLACIAN_DEPTH)
    return float(laplacian_response.var())


def main() -> None:
    """Compare blur scores for deterministic noisy and blurred dummy images."""
    random_generator = np.random.default_rng(SMOKE_TEST_RANDOM_SEED)
    noisy_pixels = random_generator.integers(
        PIXEL_MINIMUM,
        PIXEL_MAXIMUM + 1,
        size=(
            SMOKE_TEST_IMAGE_SIZE,
            SMOKE_TEST_IMAGE_SIZE,
            SMOKE_TEST_CHANNEL_COUNT,
        ),
        dtype=np.uint8,
    )
    sharp_image = Image.fromarray(noisy_pixels, mode="RGB")
    blurred_image = sharp_image.filter(
        ImageFilter.GaussianBlur(radius=SMOKE_TEST_BLUR_RADIUS)
    )

    sharp_score = estimate_blur_score(sharp_image)
    blurred_score = estimate_blur_score(blurred_image)
    print(f"Sharp image blur score: {sharp_score:.4f}")
    print(f"Blurred image blur score: {blurred_score:.4f}")
    print(f"Blurred scores lower: {blurred_score < sharp_score}")


if __name__ == "__main__":
    main()
