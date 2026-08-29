"""
Robustness transforms matching the challenge's transformation table (Section 5.2).
Used both as training-time augmentation and for building the clean-vs-transformed
robustness evaluation table.
"""
import io
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    """quality in {90, 70, 50, 30}"""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    """sigma in {0.5, 1.0, 2.0}"""
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def resize_then_upscale(img: Image.Image, scale: float) -> Image.Image:
    """scale in {0.5, 0.25} — downscale then upscale back (thumbnail simulation)"""
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    """sigma in {0.02, 0.05, 0.10}, applied in [0,1] pixel space"""
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 1)
    return Image.fromarray((noisy * 255).astype(np.uint8))


def color_jitter(img: Image.Image, factor_range: float = 0.2) -> Image.Image:
    """brightness/contrast/saturation +/- factor_range (default 20%)"""
    out = img.convert("RGB")
    for enhancer_cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = 1.0 + random.uniform(-factor_range, factor_range)
        out = enhancer_cls(out).enhance(factor)
    return out


def center_crop(img: Image.Image, crop_pct: float = 0.8) -> Image.Image:
    """crop 80% from center"""
    w, h = img.size
    new_w, new_h = int(w * crop_pct), int(h * crop_pct)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.BILINEAR)


# Registry used by both training-time augmentation and eval harness
TRANSFORMS = {
    "jpeg_90": lambda img: jpeg_compress(img, 90),
    "jpeg_70": lambda img: jpeg_compress(img, 70),
    "jpeg_50": lambda img: jpeg_compress(img, 50),
    "jpeg_30": lambda img: jpeg_compress(img, 30),
    "blur_0.5": lambda img: gaussian_blur(img, 0.5),
    "blur_1.0": lambda img: gaussian_blur(img, 1.0),
    "blur_2.0": lambda img: gaussian_blur(img, 2.0),
    "resize_0.5": lambda img: resize_then_upscale(img, 0.5),
    "resize_0.25": lambda img: resize_then_upscale(img, 0.25),
    "noise_0.02": lambda img: gaussian_noise(img, 0.02),
    "noise_0.05": lambda img: gaussian_noise(img, 0.05),
    "noise_0.10": lambda img: gaussian_noise(img, 0.10),
    "color_jitter": lambda img: color_jitter(img, 0.2),
    "center_crop": lambda img: center_crop(img, 0.8),
}


def random_augment(img: Image.Image, p: float = 0.5) -> Image.Image:
    """Apply a random subset of transforms for training-time robustness augmentation."""
    for fn in TRANSFORMS.values():
        if random.random() < p:
            img = fn(img)
    return img
