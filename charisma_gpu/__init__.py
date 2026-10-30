"""
charisma-gpu: the charisma colour-classification pipeline (Schwartz et al.,
R package charisma 1.0.0) in PyTorch, batched over many images so it runs on a
GPU, plus a per-pixel CLUT classifier.

    from charisma_gpu import Clut, DEFAULT_CLUT, charisma_batch, classify_pixels, png_pixels

    clut = Clut.from_csv(DEFAULT_CLUT)
    segs = [png_pixels(p) for p in paths]          # foreground pixels of cut-out PNGs
    results = charisma_batch(segs, clut)            # as R's charisma(threshold = 0, pavo = FALSE)
    shares, mean_rgb = classify_pixels(segs, clut)  # every pixel named by the CLUT
"""

from .core import (CLUT_DIR, DEFAULT_CLUT, RHIRTA_CLUT, Clut, charisma_batch,  # noqa: F401
                   classify_pixels, complete_linkage_groups, lab_to_srgb, srgb_to_hsv,
                   srgb_to_lab)
from .io import mask_pixels, png_pixels  # noqa: F401
from .petal import petal_bullseye  # noqa: F401

__version__ = "0.1.0"
