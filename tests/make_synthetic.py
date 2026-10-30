"""
Synthetic cut-out images for the parity test: blobs of colour on a transparent
background, with noise and gradients, chosen to sit in and between the default
CLUT's classes (red/brown/orange, yellow/green near their boundary, greys,
blues, purples). Deterministic.

    python tests/make_synthetic.py            # writes tests/data/synthetic_XX.png
    Rscript parity/parity.R tests/data/files.txt tests/data/r_reference.csv
"""

from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent / "data"
PALETTES = [
    [(230, 200, 20), (120, 40, 20), (40, 25, 15)],          # yellow ray, maroon, dark disk
    [(200, 30, 30), (240, 160, 30), (250, 240, 200)],       # red, orange, cream
    [(194, 185, 14), (150, 160, 40), (60, 120, 40)],        # yellow/green boundary
    [(128, 128, 128), (200, 200, 205), (40, 40, 40)],       # greys
    [(40, 60, 200), (120, 40, 160), (220, 220, 240)],       # blue, purple, white
    [(150, 75, 0), (180, 120, 60), (90, 50, 30)],           # browns
    [(255, 255, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)], # primaries
    [(250, 230, 120), (240, 210, 60), (200, 150, 20)],      # pale to golden yellows
]


def make(i, rng):
    h = w = int(rng.integers(60, 140))
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), float)
    pal = PALETTES[i % len(PALETTES)]
    labels = rng.integers(0, len(pal), size=(6, 6))
    cell = np.kron(labels, np.ones((h // 6 + 1, w // 6 + 1), int))[:h, :w]
    for k, col in enumerate(pal):
        img[cell == k] = col
    img += rng.normal(0, 4 + 4 * (i % 3), img.shape)                   # noise
    img *= (0.75 + 0.25 * xx / w)[..., None]                           # gradient
    img = np.clip(img, 0, 255).astype(np.uint8)
    r = np.hypot(yy - h / 2, xx - w / 2)
    alpha = np.where(r < 0.47 * min(h, w) * (0.8 + 0.2 * np.sin(6 * np.arctan2(yy - h / 2, xx - w / 2))),
                     255, 0)
    alpha[rng.random(alpha.shape) < 0.002] = 127                       # just below the cut
    return np.dstack([img, alpha.astype(np.uint8)])


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(20261030)
    names = []
    for i in range(16):
        name = f"synthetic_{i:02d}.png"
        Image.fromarray(make(i, rng), "RGBA").save(OUT / name)
        names.append(f"tests/data/{name}")
    (OUT / "files.txt").write_text("\n".join(names) + "\n")
    print(f"wrote {len(names)} images to {OUT}")


if __name__ == "__main__":
    main()
