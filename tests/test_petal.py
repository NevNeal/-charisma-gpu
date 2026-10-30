"""
The petal-vs-disk bullseye split on drawn flowers: a disk, eight rays, and red
either on the ray bases (a bullseye) or on the disk (a reddish disk).
"""

import numpy as np
import pytest

from charisma_gpu import RHIRTA_CLUT, Clut, petal_bullseye

YELLOW = (240, 200, 20)
MAROON = (110, 20, 20)        # dark, saturated red: bullseye-coloured
RED_BROWN = (100, 30, 15)     # a reddish-brown disk (hue 10.6): also bullseye-coloured
DARK_BROWN = (45, 30, 20)     # an ordinary disk: disk-coloured
BRIGHT_RED = (200, 30, 30)


def flower(disk_colour, base_colour, n=200, disk_r=24, ray_len=70, ray_w=14, base_frac=0.35):
    rgb = np.zeros((n, n, 3), np.uint8)
    fg = np.zeros((n, n), bool)
    yy, xx = np.mgrid[0:n, 0:n] - n / 2
    r, ang = np.hypot(yy, xx), np.arctan2(yy, xx)
    for k in range(8):
        a = k * np.pi / 4
        along = xx * np.cos(a) + yy * np.sin(a)
        across = -xx * np.sin(a) + yy * np.cos(a)
        ray = (np.abs(across) < ray_w / 2) & (along > disk_r - 4) & (along < disk_r + ray_len)
        base = ray & (along < disk_r + base_frac * ray_len)
        rgb[ray] = YELLOW
        rgb[base] = base_colour
        fg |= ray
    disk = r < disk_r
    rgb[disk] = disk_colour
    fg |= disk
    rng = np.random.default_rng(0)
    rgb = np.clip(rgb.astype(int) + rng.integers(-3, 4, rgb.shape), 0, 255).astype(np.uint8)
    return rgb, fg


@pytest.fixture(scope="module")
def clut():
    return Clut.from_csv(RHIRTA_CLUT)


def test_bullseye_on_rays_is_kept(clut):
    out = petal_bullseye(*flower(DARK_BROWN, MAROON), clut)
    s = out["shares"]
    assert s["petal_bullseye"] > 0.15
    # only the ray bases inside the disk's growth margin are counted as disk
    assert s["disk_red"] < 0.25 * s["petal_bullseye"]
    assert s["disk_area"] > 0.1


def test_reddish_disk_is_not_a_bullseye(clut):
    out = petal_bullseye(*flower(RED_BROWN, YELLOW), clut)
    s = out["shares"]
    assert s["petal_bullseye"] < 0.01
    assert s["disk_red"] > 0.12


def test_bright_ring_touching_a_dark_disk_stays_on_the_petals(clut):
    out = petal_bullseye(*flower(DARK_BROWN, BRIGHT_RED, base_frac=0.5), clut)
    assert out["shares"]["petal_bullseye"] > 0.2


def test_ring_with_no_disk_is_not_taken_for_a_disk(clut):
    rgb, fg = flower(DARK_BROWN, MAROON)
    n = fg.shape[0]
    yy, xx = np.mgrid[0:n, 0:n] - n / 2
    fg &= np.hypot(yy, xx) >= 26               # SAM3 left the disk out
    out = petal_bullseye(rgb, fg, clut)
    assert out["shares"]["petal_bullseye"] > 0.2


def test_needs_bullseye_disk_ray(tmp_path):
    from charisma_gpu import DEFAULT_CLUT
    with pytest.raises(ValueError):
        petal_bullseye(*flower(DARK_BROWN, MAROON), Clut.from_csv(DEFAULT_CLUT))
