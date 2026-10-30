# Notice

charisma-gpu re-implements, in Python, algorithms from two R packages. No R
code is copied; the behaviour was reproduced from the published packages and
checked against them (see `tests/` and `parity/`).

- **charisma** 1.0.0: Schwartz S, Tsai W, Karan E, Juhn M, Shultz A,
  McCormack J, Smith T, Alfaro M (2025). *charisma: Reproducible Color
  Characterization of Digital Images for Biological Studies.*
  doi:10.32614/CRAN.package.charisma. MIT License.
  <https://github.com/shawntz/charisma>
  The default colour look-up table (`charisma_gpu/cluts/default.csv`) is
  charisma's `clut`, exported unchanged.

- **recolorize** 0.2.0: Weller H. *recolorize: Color-Based Image Segmentation.*
  CC BY 4.0. <https://github.com/hiweller/recolorize>
  The histogram binning, colour reclustering and `imposeColors` steps follow
  recolorize's.

If you use charisma-gpu in published work, please cite charisma (and
recolorize), since the method is theirs.
