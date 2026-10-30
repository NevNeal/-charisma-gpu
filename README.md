# charisma-gpu

The [charisma](https://github.com/shawntz/charisma) colour-classification
pipeline (Schwartz et al., R package charisma 1.0.0) in PyTorch, batched over
thousands of images at once so it runs on a GPU. It reproduces R's
`charisma(img, threshold = 0, pavo = FALSE)` and adds a per-pixel classifier
that uses the same colour look-up tables (CLUTs).

```python
from charisma_gpu import Clut, DEFAULT_CLUT, charisma_batch, classify_pixels, png_pixels

clut = Clut.from_csv(DEFAULT_CLUT)                 # or your own CLUT, charisma's format
segs = [png_pixels(p) for p in paths]              # foreground pixels of cut-out PNGs
res = charisma_batch(segs, clut)                   # [{k, props, centers, sizes, labels}, ...]
shares, mean_rgb = classify_pixels(segs, clut)     # [n_images, n_colours] pixel shares
```

or from the command line, over a folder of cut-outs:

```bash
pip install -e .
charisma-gpu path/to/cutouts --out colours.csv                  # default CLUT
charisma-gpu path/to/cutouts --out colours.csv --clut my.csv    # your CLUT
```

One row per image: `px_<colour>` (per-pixel share), `px_<colour>_rgb` (mean
colour of those pixels), `ch_k` and `ch_<colour>` (charisma's answer). Reruns
skip images that are already in the CSV.

## What it reproduces

R charisma, per image:

1. Background is every pixel whose rounded alpha is below 1 (alpha < 128).
2. Bin the rest into a 4 × 4 × 4 sRGB histogram (right-closed intervals). Each
   bin's colour is the mean of its pixels.
3. Drop empty bins. Cluster the bin colours in CIE Lab (D65) by complete linkage
   and cut the tree at height 20. Each group's colour is the pixel-weighted mean
   of its bins in sRGB. If that leaves 2 or fewer groups, the merge is skipped,
   as in R.
4. Reassign every pixel to its nearest colour in Lab, then move each colour to
   the Lab mean of its pixels (`recolorize::imposeColors`).
5. Name each colour with the CLUT, in HSV. The first matching row wins, and no
   match gives the first row, as in R. Then add up the proportions per name.

Steps 2, 4 and 5 run on the GPU for every image in a batch at once (scatter-add
histograms, batched nearest-centre search, vectorised CLUT tests). Step 3 sees
at most 64 colours per image and runs on the CPU. Everything is float64, as in
R: a cluster colour that lands on a CLUT boundary (value 75.99 vs 76.01) changes
the name of the whole cluster, and float32 was enough to flip one in testing.

**Not reproduced:** pavo's colour-adjacency statistics (`pavo = TRUE`),
charisma's interactive merge/replace sessions, and its diagnostic plots.

## Agreement with R

| Test | Images | Same colour classes | Same k | Max proportion difference |
|---|---|---|---|---|
| Real flower cut-outs, default CLUT | 40 | 40/40 | 40/40 | 0.0005 |
| Synthetic images (`tests/`), default CLUT | 16 | 16/16 | 16/16 | < 0.002 |
| Synthetic images, *R. hirta* CLUT | 16 | 16/16 | 16/16 | < 0.002 |

The remaining differences come from rounding at nearest-colour ties. To check
your own images:

```bash
Rscript parity/parity.R paths.txt r.csv [clut.csv]
python  parity/parity.py paths.txt r.csv [--clut clut.csv]
```

## Speed

- **R charisma:** 2.4 s per image (`pavo = FALSE`) on the 40 parity cut-outs,
  in serial.
- **charisma-gpu:** 126,653 *Rudbeckia hirta* segments (median 1,700 px) in
  about 2 minutes on one RTX 4060 laptop GPU. That's 1,057 segments/s, with
  8 processes reading the PNGs.

Reading the PNGs, not the GPU, is the bottleneck. The GPU stages alone ran at
about 230 images/s in charisma mode and about 1,400 images/s in per-pixel mode,
on 73k-pixel segments.

## Two answers per image, and when to use which

`ch_*` is charisma: the CLUT names the *merged cluster colours*, so every pixel
in a cluster gets one name. If a cluster straddles two CLUT regions, its
average decides for all of it. On a test set of *R. hirta* flowers, this
reported 0% red "bullseye" on two flowers that clearly have one: their dark
maroon pixels merged into the disk's cluster.

`px_*` names every pixel separately, so a region's share changes smoothly with
the image. Use it when you care about how much of one colour there is.

## CLUTs

`charisma_gpu/cluts/default.csv` is charisma's own `clut`, unchanged.

`charisma_gpu/cluts/rudbeckia_hirta.csv` separates the parts of a
black-eyed Susan flower head:

| name | what it captures |
|---|---|
| `bullseye` | red, maroon and chestnut ray bases |
| `disk` | dark brown, purple or black disk florets, and pinkish-grey disk tops |
| `ray` | lemon to golden-yellow rays |
| `green` | leaf or bract pixels inside the mask |
| `pale` | white or grey highlights, dried heads |
| `other` | anything else |

It was tuned on hand-boxed bullseye and disk pixels from iNaturalist photos.
The key boundary is saturation together with hue. A dark bullseye (value
10–44) must be red (hue ≤ 12° or ≥ 330°) and saturated (≥ 65%). A bright one
(value ≥ 45) may run to chestnut and orange-red (hue ≤ 26°). On the tuning
flowers:

- 84% of bullseye pixels are caught.
- 0.3% of disk pixels are wrongly called bullseye.
- Plain flowers average 1.2% bullseye pixels.

Those numbers come from the same flowers it was tuned on, so expect somewhat
worse on new data. Very dark, nearly black maroon bases are the same colour as
dark disks, and no colour rule separates them; telling them apart needs where
the pixels are, not what colour they are. CLUTs use charisma's CSV format, so
this one works in R charisma too.

## Credit

The method is charisma's and recolorize's. Please cite them; see
[NOTICE.md](NOTICE.md). The methodology write-up is in `docs/`.

MIT License.
