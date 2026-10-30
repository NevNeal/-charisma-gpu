"""
charisma-gpu against R charisma 1.0.0 and against known colour-space values.
Runs on the CPU, so no GPU is needed:

    pytest tests/
"""

import colorsys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from charisma_gpu import (DEFAULT_CLUT, RHIRTA_CLUT, Clut, charisma_batch, classify_pixels,
                          complete_linkage_groups, lab_to_srgb, png_pixels, srgb_to_hsv,
                          srgb_to_lab)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"


@pytest.mark.parametrize("clut_path, reference", [
    (DEFAULT_CLUT, "r_reference.csv"),
    (RHIRTA_CLUT, "r_reference_rhirta.csv"),
])
def test_matches_r_charisma(clut_path, reference):
    """Same colour classes, same k and the same proportions as R's
    charisma(threshold = 0, pavo = FALSE) on the synthetic images."""
    ref = pd.read_csv(DATA / reference)
    paths = list(dict.fromkeys(ref["path"]))
    clut = Clut.from_csv(clut_path)
    results = charisma_batch([png_pixels(ROOT / p) for p in paths], clut, device="cpu")
    for path, py in zip(paths, results):
        r = ref[ref["path"] == path]
        r_props = dict(zip(r["colour"], r["prop"]))
        assert set(py["props"]) == set(r_props), path
        assert py["k"] == int(r["k"].iloc[0]), path
        for colour, prop in r_props.items():
            assert py["props"][colour] == pytest.approx(prop, abs=2e-3), (path, colour)


def test_lab_known_values():
    rgb = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    lab = srgb_to_lab(rgb).numpy()
    assert lab[0] == pytest.approx([100, 0, 0], abs=0.01)
    assert lab[1] == pytest.approx([53.24, 80.09, 67.20], abs=0.05)   # sRGB red, D65
    assert lab[2] == pytest.approx([0, 0, 0], abs=0.01)


def test_lab_round_trip():
    rgb = torch.rand(1000, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    back = lab_to_srgb(srgb_to_lab(rgb))
    assert torch.allclose(back, rgb, atol=1e-6)


def test_hsv_matches_colorsys():
    rgb = torch.rand(500, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    ours = srgb_to_hsv(rgb).numpy()
    for (r, g, b), (h, s, v) in zip(rgb.numpy(), ours):
        eh, es, ev = colorsys.rgb_to_hsv(r, g, b)
        assert h == pytest.approx(eh * 360, abs=1e-6) or abs(h - eh * 360) == pytest.approx(360)
        assert (s, v) == pytest.approx((es * 100, ev * 100), abs=1e-6)


def test_default_clut_names_pure_colours():
    clut = Clut.from_csv(DEFAULT_CLUT)
    hexes = {"red": "#FF0000", "yellow": "#FFFF00", "black": "#000000",
             "white": "#FFFFFF", "grey": "#808080", "blue": "#0000FF", "green": "#00FF00"}
    rgb = torch.tensor([[int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)] for h in hexes.values()],
                       dtype=torch.float64)
    names = [clut.names[i] for i in clut.label(srgb_to_hsv(rgb))]
    assert names == list(hexes)


def test_unmatched_colour_falls_back_to_first_row(tmp_path):
    path = tmp_path / "clut.csv"
    path.write_text('"color.name","default.hex","h","s","v"\n'
                    '"first","#000000","0::10","0::10","0::10"\n'
                    '"second","#FFFFFF","100::200","50::100","50::100"\n')
    clut = Clut.from_csv(path)
    hsv = torch.tensor([[300.0, 90.0, 90.0], [150.0, 60.0, 60.0]], dtype=torch.float64)
    assert clut.label(hsv).tolist() == [0, 1]       # no match -> first name, as R


def test_complete_linkage_groups():
    pts = np.array([[0.0, 0, 0], [1, 0, 0], [50, 0, 0], [51, 0, 0], [100, 0, 0]])
    assert complete_linkage_groups(pts, 5).tolist() == [0, 0, 1, 1, 2]
    # {50,51} is 50 from {100} but 51 from {0,1}, so it joins {100} first
    assert complete_linkage_groups(pts, 60).tolist() == [0, 0, 1, 1, 1]
    # complete linkage: 0-1-2 chain at spacing 4 does not merge at cutoff 5
    chain = np.array([[0.0, 0, 0], [4, 0, 0], [8, 0, 0]])
    assert len(set(complete_linkage_groups(chain, 5))) == 2


def test_pixel_mode_shares_sum_to_one():
    segs = [png_pixels(p) for p in sorted(DATA.glob("synthetic_*.png"))[:4]]
    shares, mean_rgb = classify_pixels(segs, Clut.from_csv(DEFAULT_CLUT), device="cpu")
    assert shares.shape == (4, 10)
    assert np.allclose(shares.sum(1), 1)


def test_alpha_cut_is_128(tmp_path):
    from PIL import Image
    a = np.zeros((2, 2, 4), np.uint8)
    a[..., 3] = [[127, 128], [255, 0]]
    Image.fromarray(a, "RGBA").save(tmp_path / "x.png")
    assert len(png_pixels(tmp_path / "x.png")) == 2
