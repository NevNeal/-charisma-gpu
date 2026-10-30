"""Getting foreground pixels out of images the way R charisma does."""

import numpy as np
from PIL import Image


def png_pixels(path):
    """Foreground pixels of a cut-out image with transparency, as [P, 3] uint8.
    Foreground is R charisma's round(alpha) >= 1, i.e. alpha >= 128 of 255.
    An image without an alpha channel is all foreground, as in R."""
    a = np.asarray(Image.open(path).convert("RGBA"))
    return a[a[..., 3] >= 128][:, :3]


def mask_pixels(image, mask):
    """Pixels of an RGB image (path, PIL image or H x W x 3 array) under a
    boolean H x W mask, as [P, 3] uint8."""
    if not isinstance(image, np.ndarray):
        image = np.asarray(image if isinstance(image, Image.Image)
                           else Image.open(image).convert("RGB"))
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != image.shape[:2]:
        raise ValueError(f"mask {mask.shape} does not match image {image.shape[:2]}")
    return image[mask][:, :3]
