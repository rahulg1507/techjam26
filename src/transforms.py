"""
Robustness transforms matching the challenge's transformation table (Section 5.2).
Used both as training-time augmentation and for building the clean-vs-transformed
robustness evaluation table.
"""
import io
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

# Transform parameters are named so the challenge settings are easy to audit and
# adjust without scattering unexplained values through the implementation.
JPEG_QUALITY_HIGH = 90
JPEG_QUALITY_MEDIUM_HIGH = 70
JPEG_QUALITY_MEDIUM = 50
JPEG_QUALITY_LOW = 30

BLUR_SIGMA_LIGHT = 0.5
BLUR_SIGMA_MEDIUM = 1.0
BLUR_SIGMA_HEAVY = 2.0

RESIZE_SCALE_HALF = 0.5
RESIZE_SCALE_QUARTER = 0.25

NOISE_SIGMA_LIGHT = 0.02
NOISE_SIGMA_MEDIUM = 0.05
NOISE_SIGMA_HEAVY = 0.10

COLOR_JITTER_FACTOR = 0.2
CENTER_CROP_PERCENTAGE = 0.8

PIXEL_MAX_VALUE = 255.0
PIXEL_MIN_VALUE = 0.0
NORMALIZED_PIXEL_MAXIMUM = 1.0
MINIMUM_IMAGE_DIMENSION = 1
DEFAULT_AUGMENTATION_PROBABILITY = 0.5


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    """Simulate social-media re-encoding that introduces JPEG artifacts."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    """Simulate camera focus loss or platform processing that softens details."""
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def resize_then_upscale(img: Image.Image, scale: float) -> Image.Image:
    """Simulate thumbnail generation and subsequent upscaling by a platform."""
    w, h = img.size
    small = img.resize(
        (
            max(MINIMUM_IMAGE_DIMENSION, int(w * scale)),
            max(MINIMUM_IMAGE_DIMENSION, int(h * scale)),
        ),
        Image.BILINEAR,
    )
    return small.resize((w, h), Image.BILINEAR)


def gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    """Simulate sensor and transmission noise that can obscure generation cues."""
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / PIXEL_MAX_VALUE
    noise = np.random.normal(PIXEL_MIN_VALUE, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, PIXEL_MIN_VALUE, NORMALIZED_PIXEL_MAXIMUM)
    return Image.fromarray((noisy * PIXEL_MAX_VALUE).astype(np.uint8))


def color_jitter(
    img: Image.Image, factor_range: float = COLOR_JITTER_FACTOR
) -> Image.Image:
    """Simulate varied display, lighting, and platform color processing."""
    out = img.convert("RGB")
    for enhancer_cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = NORMALIZED_PIXEL_MAXIMUM + random.uniform(-factor_range, factor_range)
        out = enhancer_cls(out).enhance(factor)
    return out


def center_crop(
    img: Image.Image, crop_pct: float = CENTER_CROP_PERCENTAGE
) -> Image.Image:
    """Simulate user or platform cropping that removes contextual image regions."""
    w, h = img.size
    new_w, new_h = int(w * crop_pct), int(h * crop_pct)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.BILINEAR)


# Registry used by both training-time augmentation and eval harness
TRANSFORMS = {
    "jpeg_90": lambda img: jpeg_compress(img, JPEG_QUALITY_HIGH),
    "jpeg_70": lambda img: jpeg_compress(img, JPEG_QUALITY_MEDIUM_HIGH),
    "jpeg_50": lambda img: jpeg_compress(img, JPEG_QUALITY_MEDIUM),
    "jpeg_30": lambda img: jpeg_compress(img, JPEG_QUALITY_LOW),
    "blur_0.5": lambda img: gaussian_blur(img, BLUR_SIGMA_LIGHT),
    "blur_1.0": lambda img: gaussian_blur(img, BLUR_SIGMA_MEDIUM),
    "blur_2.0": lambda img: gaussian_blur(img, BLUR_SIGMA_HEAVY),
    "resize_0.5": lambda img: resize_then_upscale(img, RESIZE_SCALE_HALF),
    "resize_0.25": lambda img: resize_then_upscale(img, RESIZE_SCALE_QUARTER),
    "noise_0.02": lambda img: gaussian_noise(img, NOISE_SIGMA_LIGHT),
    "noise_0.05": lambda img: gaussian_noise(img, NOISE_SIGMA_MEDIUM),
    "noise_0.10": lambda img: gaussian_noise(img, NOISE_SIGMA_HEAVY),
    "color_jitter": lambda img: color_jitter(img, COLOR_JITTER_FACTOR),
    "center_crop": lambda img: center_crop(img, CENTER_CROP_PERCENTAGE),
}


def random_augment(
    img: Image.Image, p: float = DEFAULT_AUGMENTATION_PROBABILITY
) -> Image.Image:
    """Apply a random subset of transforms for training-time robustness augmentation."""
    for fn in TRANSFORMS.values():
        if random.random() < p:
            img = fn(img)
    return img
