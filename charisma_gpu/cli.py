#!/usr/bin/env python3
"""
charisma-gpu over a folder of cut-out images.

    charisma-gpu IMAGES_DIR --out colours.csv [--clut my_clut.csv] [--mode both]

Every *.png (and *.tif / *.tiff) under IMAGES_DIR is read; transparent pixels
are background. One row per image:

    file              path relative to IMAGES_DIR
    n_pixels          foreground pixels
    px_<colour>       share of pixels the CLUT puts in that colour (per pixel)
    px_<colour>_rgb   mean colour of those pixels, hex
    ch_k              charisma's k
    ch_<colour>       charisma's share for that colour (histogram + Lab merge,
                      each merged colour named by the CLUT; matches R
                      charisma(threshold = 0, pavo = FALSE))
    petal_bullseye    share of pixels that are bullseye-coloured AND on the
                      petals, not the disk (see petal.py); with --petal, which
                      is on by default for a CLUT with bullseye, disk and ray
    disk_red          bullseye-coloured pixels found to be on the disk
    disk_area         share of the image taken by the disk found

Rows are appended per batch; rerunning with the same --out skips files that
are already there.
"""

import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .core import DEFAULT_CLUT, RHIRTA_CLUT, Clut, charisma_batch, classify_pixels
from .petal import petal_bullseye

BUILTIN = {"default": DEFAULT_CLUT, "rudbeckia_hirta": RHIRTA_CLUT}


PETAL_COLS = ("petal_bullseye", "disk_red", "disk_area")


class _Files(torch.utils.data.Dataset):
    """Reads each cut-out (and runs the petal step, which needs the 2-D image)
    in the DataLoader's worker processes."""

    def __init__(self, files, clut, petal):
        self.files, self.clut, self.petal = files, clut, petal

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        try:
            a = np.asarray(Image.open(self.files[i]).convert("RGBA"))
            fg = a[..., 3] >= 128          # R charisma's round(alpha) >= 1
            shares = None
            if self.petal and fg.any():
                shares = petal_bullseye(a[..., :3], fg, self.clut)["shares"]
            return self.files[i], a[fg][:, :3], shares, ""
        except Exception as exc:
            return self.files[i], None, None, f"{type(exc).__name__}: {exc}"


def _items(batch):
    return batch


def _hex(rgb):
    r, g, b = (int(round(min(max(v, 0), 255))) for v in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def main(argv=None):
    p = argparse.ArgumentParser(prog="charisma-gpu", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images_dir")
    p.add_argument("--out", required=True)
    p.add_argument("--clut", default="default",
                   help="a CLUT CSV in charisma's format, or a built-in: "
                        + ", ".join(BUILTIN))
    p.add_argument("--mode", choices=("both", "pixel", "charisma"), default="both")
    p.add_argument("--bins", type=int, default=4)
    p.add_argument("--cutoff", type=float, default=20.0)
    p.add_argument("--petal", choices=("auto", "on", "off"), default="auto",
                   help="petal-vs-disk bullseye split; auto = on when the CLUT has "
                        "bullseye, disk and ray colours")
    p.add_argument("--batch", type=int, default=256, help="images per GPU batch")
    p.add_argument("--workers", type=int, default=8, help="image-reading processes")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    clut = Clut.from_csv(BUILTIN.get(args.clut, args.clut))
    can_petal = {"bullseye", "disk", "ray"} <= set(clut.names)
    if args.petal == "on" and not can_petal:
        raise SystemExit("--petal on needs a CLUT with bullseye, disk and ray colours")
    petal = args.petal == "on" or (args.petal == "auto" and can_petal)
    root = Path(args.images_dir)
    # *.png and *.PNG are the same files on case-insensitive filesystems
    found = {os.path.normcase(os.path.abspath(f)): f
             for ext in ("png", "PNG", "tif", "tiff", "TIF", "TIFF")
             for f in glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True)}
    files = sorted(found.values())
    out = Path(args.out)
    done = set(pd.read_csv(out, usecols=["file"])["file"]) if out.exists() else set()
    todo = [f for f in files if str(Path(f).relative_to(root)) not in done]
    print(f"[charisma-gpu] {len(files)} images, {len(done)} done, {len(todo)} to go | "
          f"CLUT: {', '.join(clut.names)} | {args.device}", flush=True)
    if not todo:
        return

    cols = ["file", "n_pixels", "error"]
    if args.mode in ("both", "pixel"):
        cols += [f"px_{n}" for n in clut.names] + [f"px_{n}_rgb" for n in clut.names]
    if args.mode in ("both", "charisma"):
        cols += ["ch_k"] + [f"ch_{n}" for n in clut.names]
    if petal:
        cols += list(PETAL_COLS)
    loader = torch.utils.data.DataLoader(
        _Files(todo, clut, petal), batch_size=args.batch, num_workers=args.workers, collate_fn=_items,
        prefetch_factor=4 if args.workers else None)
    header, start, n = not out.exists(), time.time(), 0
    for batch in loader:
        recs = []
        good = []
        for f, px, shares, err in batch:
            rec = {"file": str(Path(f).relative_to(root)), "error": err,
                   "n_pixels": 0 if px is None else len(px)}
            if shares:
                rec.update({k: round(float(shares[k]), 5) for k in PETAL_COLS})
            recs.append(rec)
            if px is not None and len(px):
                good.append((rec, px))
            elif not err:
                rec["error"] = "no foreground pixels"
        if good:
            segs = [px for _, px in good]
            if args.mode in ("both", "pixel"):
                shares, mean_rgb = classify_pixels(segs, clut, device=args.device)
                for j, (rec, _) in enumerate(good):
                    for c, name in enumerate(clut.names):
                        rec[f"px_{name}"] = round(float(shares[j, c]), 5)
                        rec[f"px_{name}_rgb"] = _hex(mean_rgb[j, c]) if shares[j, c] > 0 else ""
            if args.mode in ("both", "charisma"):
                res = charisma_batch(segs, clut, bins=args.bins, cutoff=args.cutoff,
                                     device=args.device)
                for (rec, _), r in zip(good, res):
                    rec["ch_k"] = r["k"]
                    for name in clut.names:
                        rec[f"ch_{name}"] = round(r["props"].get(name, 0.0), 5)
        pd.DataFrame(recs, columns=cols).to_csv(out, mode="a", header=header, index=False)
        header = False
        n += len(batch)
        rate = n / (time.time() - start)
        print(f"[charisma-gpu] {n}/{len(todo)} | {rate:.0f} images/s | "
              f"eta {(len(todo) - n) / rate / 60:.1f} min", flush=True)
    print(f"[charisma-gpu] wrote {out}")


if __name__ == "__main__":
    main()
