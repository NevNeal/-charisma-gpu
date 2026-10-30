#!/usr/bin/env python3
"""
charisma-gpu against R charisma on your own cut-out images.

    Rscript parity/parity.R paths.txt r.csv [clut.csv]
    python  parity/parity.py paths.txt r.csv [--clut clut.csv]

paths.txt lists one image per line, the same file for both. Prints every image
whose colour classes differ or whose proportions differ by more than 0.01,
then a summary and the time each side took.
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch

from charisma_gpu import DEFAULT_CLUT, Clut, charisma_batch, png_pixels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths")
    p.add_argument("r_csv")
    p.add_argument("--clut", default=str(DEFAULT_CLUT))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    paths = [line.strip() for line in open(args.paths) if line.strip()]
    clut = Clut.from_csv(args.clut)
    segs = [png_pixels(x) for x in paths]
    t0 = time.time()
    res = charisma_batch(segs, clut, device=args.device)
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    r = pd.read_csv(args.r_csv)
    same_k = same_cols = 0
    worst = []
    for path, py in zip(paths, res):
        rr = r[r["path"] == path]
        if rr.empty:
            continue
        r_props = dict(zip(rr["colour"], rr["prop"]))
        match = set(r_props) == set(py["props"])
        same_cols += match
        same_k += int(rr["k"].iloc[0]) == py["k"]
        diff = max(abs(r_props.get(c, 0) - py["props"].get(c, 0))
                   for c in set(r_props) | set(py["props"]))
        worst.append(diff)
        if not match or diff > 0.01:
            print(f"{path}\n   R  {sorted(r_props.items())}\n   py {sorted(py['props'].items())}")
    n = len(worst)
    print(f"\n{n} images | same colour classes {same_cols}/{n} | same k {same_k}/{n} | "
          f"max |proportion difference| median {np.median(worst):.4f}, max {max(worst):.4f}")
    per_image = r.drop_duplicates("path")["seconds"]
    print(f"R charisma : {per_image.sum():.1f} s ({per_image.mean():.2f} s per image)")
    print(f"charisma-gpu: {elapsed:.2f} s for the batch on {args.device}")


if __name__ == "__main__":
    main()
