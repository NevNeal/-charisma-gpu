"""
Build methodology.pdf: how R charisma was converted to a batched GPU pipeline,
how the conversion was checked, and how the Rudbeckia hirta CLUT was made.

    python docs/make_methodology_pdf.py                       # docs/methodology.pdf
    python docs/make_methodology_pdf.py --photos DIR --out X  # also the photo figures in DIR

The public copy has no photographs. --photos adds the class-map figures made
from iNaturalist photos (classmap_bullseye_set.png, classmap_holdout.png,
full_run_top_bullseye.png, full_run_random.png), when they are in DIR.
"""
import argparse
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (HRFlowable, Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))

styles = getSampleStyleSheet()
NAVY = colors.HexColor("#2B3A55")
LIGHT = colors.HexColor("#EAEEF5")
h1 = ParagraphStyle("h1", parent=styles["Heading1"], textColor=NAVY, fontSize=18, spaceAfter=4)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=NAVY, fontSize=13,
                    spaceAfter=4, spaceBefore=12)
sub = ParagraphStyle("sub", parent=styles["Normal"], textColor=colors.grey, fontSize=9,
                     spaceAfter=10)
body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=6)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=2, spaceAfter=3)
note = ParagraphStyle("note", parent=styles["Normal"], fontSize=8.5, textColor=colors.grey,
                      leading=11, spaceAfter=4)
code = ParagraphStyle("code", parent=styles["Code"], fontSize=8, leading=10.5, backColor=LIGHT,
                      borderPadding=6, spaceBefore=4, spaceAfter=10)
cell = ParagraphStyle("cell", parent=styles["Normal"], fontSize=8.5, leading=11)
cellb = ParagraphStyle("cellb", parent=cell, textColor=colors.white, fontName="Helvetica-Bold")

story = []
P = lambda t, s=body: story.append(Paragraph(t, s))
B = lambda t: story.append(Paragraph(t, bullet, bulletText="•"))
gap = lambda h=6: story.append(Spacer(1, h))
C = lambda t: story.append(Paragraph(t.replace("\n", "<br/>").replace("  ", "&nbsp;&nbsp;"), code))


def table(header, rows, widths):
    data = [[Paragraph(h, cellb) for h in header]] + [[Paragraph(str(c), cell) for c in r]
                                                        for r in rows]
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    story.append(t)
    gap(8)


def figure(path, caption, width=6.8):
    if not path or not os.path.exists(path):
        return
    img = Image(path)
    img.drawWidth, img.drawHeight = width * inch, width * inch * img.imageHeight / img.imageWidth
    story.append(KeepTogether([img, Paragraph(caption, note)]))


def build(photos, out):
    ph = (lambda name: os.path.join(photos, name)) if photos else (lambda name: None)

    P("Converting charisma to the GPU — Methodology", h1)
    P("charisma-gpu: R package charisma 1.0.0 (Schwartz et al.) and recolorize 0.2.0 (Weller) "
      "re-implemented in PyTorch &nbsp;|&nbsp; with a CLUT for <i>Rudbeckia hirta</i> ray, "
      "bullseye and disk colour", sub)
    P("This document records how the R pipeline was read, what the Python version does at each "
      "step and why, how the two were checked against each other, and how the colour look-up "
      "table (CLUT) for <i>Rudbeckia hirta</i> was built. Numbers marked <b>measured</b> come "
      "from runs on an RTX 4060 laptop GPU.")

    P("1. Summary", h2)
    table(["", "Result"], [
        ["Agreement with R", "40/40 real flower cut-outs: same colour classes and k, every "
                             "proportion within 0.0005. 16/16 synthetic images, with each of two "
                             "CLUTs, within 0.002 (automated tests). <b>measured</b>"],
        ["Speed", "R: 2.4 s per image, in serial (pavo = FALSE). charisma-gpu: all 126,653 "
                  "<i>R. hirta</i> segments in ~2 min, 1,057 segments/s, limited by PNG "
                  "reading. <b>measured</b>"],
        ["Not reproduced", "pavo adjacency statistics (pavo = TRUE), interactive sessions, "
                           "diagnostic plots"],
        ["Added", "Per-pixel CLUT classification, and an <i>R. hirta</i> CLUT separating "
                  "bullseye, disk and ray"],
    ], [1.4, 5.4])

    P("2. What R charisma does", h2)
    P("The source of charisma 1.0.0 and recolorize 0.2.0 was read function by function from the "
      "installed packages. With the defaults (<font face='Courier'>bins = 4, cutoff = 20, "
      "threshold = 0</font>), <font face='Courier'>charisma(img)</font> runs:")
    table(["Step", "R function", "What it does"], [
        ["Read", "charisma_readImage", "imager::load.image; values in [0, 1]"],
        ["Background", "recolorize::backgroundIndex", "a pixel is background when "
                                                      "round(alpha) &lt; 1, i.e. alpha &lt; 128 of 255"],
        ["Bin", "recolorize:::colorClustersHist", "sRGB cut into 4 equal right-closed intervals "
                                                  "per channel (64 bins); a bin's colour is the "
                                                  "mean of its pixels; bins in expand.grid order"],
        ["Merge", "charisma_recluster, charisma_hclust_color", "empty bins dropped; bin colours "
                                                               "to CIE Lab (D65); Euclidean "
                                                               "distance; hclust(complete); "
                                                               "cutree(h = 20); each group's "
                                                               "colour = size-weighted mean in "
                                                               "sRGB. If 2 or fewer groups, the "
                                                               "merge is skipped"],
        ["Refit", "recolorize::imposeColors → assignPixels", "every pixel to its nearest colour in "
                                                            "<b>Lab</b>; each colour moved to the "
                                                            "Lab mean of its pixels, back to sRGB "
                                                            "with clipping"],
        ["Name", "color2label, parse_lut", "colour to HSV (rgb2hsv); a CLUT row matches when h, s "
                                           "and v each fall in one of its lo::hi ranges "
                                           "(lo &lt;= x &lt; hi.9999); first matching name wins; "
                                           "no match gives the first row's name"],
        ["Sum", "aggregate", "pixel shares summed per name; k = number of names"],
    ], [0.8, 1.9, 4.1])
    P("Where the time goes: <font face='Courier'>assignPixels</font> measures every pixel against "
      "every colour in an interpreted <font face='Courier'>sapply</font> loop, and every image runs "
      "alone. Both are what a GPU removes.")

    P("3. The GPU design", h2)
    P("Many images are processed as one batch. All their foreground pixels go into a single "
      "[P, 3] tensor, with a parallel vector giving each pixel's image index. Each step is then "
      "one tensor operation over the whole batch:")
    table(["Step", "How it runs"], [
        ["Bin", "bin index per pixel = ceil(4x) - 1 per channel; key = image × 64 + bin; "
                "bincount and index_add give every image's 64 counts and colour sums at once (GPU)"],
        ["Merge", "at most 64 colours per image, so this runs on the CPU. Complete linkage is "
                  "written out exactly: merge the closest pair, set the new cluster's distances "
                  "to the maximum of its parts, stop when the closest pair is above the cutoff. "
                  "Complete-linkage heights never decrease, so stopping there gives the same "
                  "partition as cutting the full tree at h = 20. Groups are numbered by first "
                  "appearance, as cutree numbers them. No scipy dependency"],
        ["Refit", "merged colours padded into an [images, K, 3] tensor with invalid slots masked "
                  "to infinity; each pixel's distances to its own image's colours in one batched "
                  "operation, chunked to 2M pixels; argmin; index_add for the Lab means (GPU)"],
        ["Name", "the CLUT is parsed once into ranges; each name's test is a vectorised "
                 "OR-of-ANDs over all colours or pixels at once; first match wins via a running "
                 "'unassigned' mask (GPU)"],
    ], [0.8, 6.0])
    P("<b>Colour conversions</b> copy R's grDevices exactly. The sRGB→XYZ matrix is built the way "
      "<font face='Courier'>make.rgb</font> builds it, from the sRGB primaries and the D65 white "
      "point (x 0.3127, y 0.3290). The inverse uses the same Lab constants (ε = 216/24389, "
      "κ = 24389/27), and sRGB output is clipped to [0, 1] as with "
      "<font face='Courier'>convertColor(clip = TRUE)</font>.")
    P("<b>R behaviour kept on purpose:</b> the alpha &gt;= 128 cut; right-closed bins; "
      "expand.grid bin order; skipping the merge at 2 or fewer groups; nearest-colour search in "
      "Lab rather than sRGB; colours with zero pixels after refitting still count toward k; an "
      "unmatched colour takes the CLUT's first name.")

    P("4. Checking it against R", h2)
    P("<font face='Courier'>parity/parity.R</font> runs R charisma on a list of images. "
      "<font face='Courier'>parity/parity.py</font> runs charisma-gpu on the same images and "
      "compares colour classes, k and every proportion.")
    P("<b>The one bug found was a precision bug.</b> The first comparison, on 40 real cut-outs, "
      "matched on 39. The 40th differed by 0.147. Both versions produced the same 18 bins and the "
      "same 9 merged groups. Float32 rounding in the pixel reassignment then moved 12 of 62,531 "
      "pixels, which shifted one colour from (193.77, 185.19, 13.24) in R to (193.81, 185.21, "
      "13.21). That colour's HSV value is 75.99 in R and 76.01 in Python, and the default CLUT's "
      "yellow begins at value 76. So R named the whole 15% cluster green, and Python named it "
      "yellow. Moving the pipeline to float64, as R computes, removed the difference: the largest "
      "remaining gap is 0.0005.")
    P("The same example shows a property of charisma itself: because names go to cluster "
      "<i>averages</i>, a nudge of 0.02% in one colour can rename 15% of an image.", note)
    table(["Test", "Images", "Same classes", "Same k", "Max |Δ proportion|"], [
        ["Real cut-outs (<i>Chrysopsis</i>), default CLUT", "40", "40/40", "40/40", "0.0005"],
        ["Synthetic (tests/), default CLUT", "16", "16/16", "16/16", "&lt; 0.002"],
        ["Synthetic (tests/), <i>R. hirta</i> CLUT", "16", "16/16", "16/16", "&lt; 0.002"],
    ], [2.9, 0.7, 1.0, 0.8, 1.4])
    P("The synthetic images are made for this purpose: blocks of colour inside and between the "
      "CLUT classes, with noise, a brightness gradient, irregular alpha edges, and pixels at "
      "alpha 127 to test the cut. Their R results are committed, so "
      "<font face='Courier'>pytest</font> repeats the comparison without R. Other tests check "
      "the Lab values of white and sRGB red, a Lab round trip, HSV against Python's colorsys, the "
      "linkage, and the first-row fallback.")

    P("5. Per-pixel mode", h2)
    P("charisma gives a name to each <i>merged cluster</i>, so every pixel in a cluster shares one "
      "name. If a cluster straddles two CLUT regions, its average decides for all of it. "
      "charisma-gpu also offers <font face='Courier'>classify_pixels</font>, which names every "
      "pixel on its own with the same CLUT. On a test set of <i>R. hirta</i> flowers, the two "
      "modes disagreed most where it matters:")
    table(["Photo", "Bullseye: charisma mode", "Bullseye: per pixel", "Why"], [
        ["131410147", "0.0%", "6.0%", "maroon patches merged into the disk's cluster"],
        ["371323113", "0.0%", "7.4%", "near-black ray bases merged into the disk's cluster"],
        ["256724243", "18.9%", "9.0%", "a disk cluster's average fell in the bullseye region"],
        ["50311663", "49.1%", "47.0%", "a clear red ring; the two agree"],
    ], [1.0, 1.6, 1.4, 2.8])
    P("For \"how much of this colour is there\", the per-pixel share is the more stable number. "
      "Both are written out.")

    story.append(PageBreak())
    P("6. The <i>Rudbeckia hirta</i> CLUT", h2)
    P("The goal is to tell three parts of the flower head apart: the yellow <b>rays</b>, the "
      "red-to-maroon <b>bullseye</b> at the ray bases, and the dark <b>disk</b>. charisma's "
      "default CLUT can't. It calls maroon \"brown\" and very dark pixels \"black\", which "
      "describes both the bullseye and the disk.")
    P("<b>Data.</b> 150 research-grade iNaturalist photos of taxon 62741, spread evenly across "
      "the taxon (shard 7 of 480 in photo_id order), were segmented with SAM3 (prompt \"flower\", "
      "score &gt;= 0.5); 135 had at least one mask. Their top segments were inspected by eye. "
      "Ten had a clear bullseye, "
      "ranging from bright red rings to near-black maroon ray bases. About 22 were plain "
      "yellow-rayed flowers.")
    P("<b>Step 1: what does the bullseye add?</b> The pixel colours of the plain flowers, whose "
      "red and brown pixels can only be disk, were plotted against those of the bullseye flowers "
      "(disk plus bullseye). The bullseye adds three clusters the plain flowers lack: bright red "
      "(hue −10 to +10°, value 45–95%), red-orange or chestnut (hue 5–26°, saturation &gt;= 75%, "
      "value &gt;= 45%), and dark maroon (hue around 345–360°, value 12–45%, saturation &gt;= 70%).")
    figure(os.path.join(HERE, "hsv_plain_vs_bullseye.png"),
           "Figure 1. Pixel colours of plain (top) and bullseye (bottom) R. hirta segments, each "
           "dot in its own colour. Hue is shown from −60° so reds are continuous.")
    P("<b>Step 2: hand-picked pixels.</b> Boxes were drawn on six bullseye flowers: on the "
      "bullseye patches and on the disk of the same flower, each checked by eye on an overlay. "
      "Two things separate them. Bullseye is <b>saturated</b> (10th percentile 57–84%, against "
      "disk medians of 31–59%) and <b>red</b> (median hue 5°, against 15° for reddish-brown "
      "disks; Lab chroma 27 against 17.5).")
    P("<b>Step 3: thresholds by search.</b> Candidate rules were scored on three numbers:")
    B("recall: the share of boxed bullseye pixels caught")
    B("the share of boxed disk pixels wrongly caught")
    B("the share of all pixels on the 22 plain flowers called bullseye")
    P("A single hue/saturation rule caught 89% of bullseye pixels but put 3.7% bullseye on plain "
      "flowers (worst 11%), all of it on reddish-brown disks. Splitting the rule by brightness "
      "fixed most of that:")
    table(["Dark rule (v 10–44)", "Bright rule (v &gt;= 45)", "Recall", "Disk boxes wrong",
           "Plain flowers: mean / worst"], [
        ["h ≤ 9°, s ≥ 65", "h ≤ 26°, s ≥ 60", "0.742", "0.1%", "0.7% / 2.8%"],
        ["<b>h ≤ 12°, s ≥ 65</b>", "<b>h ≤ 26°, s ≥ 60</b>", "<b>0.839</b>", "<b>0.3%</b>",
         "<b>1.2% / 4.3%</b>"],
        ["h ≤ 15°, s ≥ 65", "h ≤ 26°, s ≥ 60", "0.865", "0.5%", "1.8% / 6.2%"],
        ["single rule, s ≥ 65, h ≤ 22°", "(same)", "0.885", "2.6%", "3.7% / 11.0%"],
    ], [1.6, 1.4, 0.7, 1.2, 1.9])
    P("(Hue ranges also include 330–360°, the red side of 0°.) The bold row is the CLUT:")
    table(["name", "h", "s", "v", "captures"], [
        ["bullseye", "0–12 | 330–360;  0–26 | 330–360", "65–100;  60–100", "10–44;  45–100",
         "maroon (dark) and red to chestnut (bright)"],
        ["green", "71–170", "25–100", "20–100", "leaf or bract inside the mask"],
        ["disk", "0–360;  0–26 | 280–360", "0–100;  15–74", "0–44;  45–79",
         "everything dark; pinkish-grey disk tops"],
        ["ray", "27–70", "25–100", "45–100", "lemon to golden yellow"],
        ["pale", "0–360", "0–24", "45–100", "white or grey highlights, dried heads"],
        ["other", "0–360", "0–100", "0–100", "the rest"],
    ], [0.75, 1.9, 1.05, 1.0, 2.1])
    P("Rows are tried in order, so saturated dark red is claimed by bullseye before the disk's "
      "catch-all \"anything dark\" row sees it. The file uses charisma's CSV format and works "
      "unchanged in R charisma, where its results also match (tested).")
    P("<b>Limits.</b> The recall and error rates above come from the flowers the thresholds were "
      "tuned on, so they're optimistic. On 12 plain flowers not used for tuning, bullseye was "
      "0–2%, except one at 4% that does have a faint red ring. Near-black maroon ray bases are "
      "the same colour as dark disks, and no colour rule can separate them. Telling them apart "
      "needs position: the disk is the central compact region, and the bullseye lies on the "
      "rays. That would be a spatial step after classification, and is not done here.", body)

    figure(ph("classmap_bullseye_set.png"),
           "Figure 2. Bullseye flowers (and two plain ones, bottom right): each cut-out beside its "
           "per-pixel class map. Red = bullseye, dark brown = disk, yellow = ray, magenta = other.")
    figure(ph("classmap_holdout.png"),
           "Figure 3. Twelve flowers not used for tuning.")

    P("7. The full <i>R. hirta</i> run", h2)
    P("All 126,653 SAM3 segments of taxon 62741's flower-detected photos (38,273 photos, cut-out "
      "PNGs, median 1,700 pixels) were classified on the RTX 4060 in about 2 minutes, with no "
      "errors. Per segment, the median per-pixel shares were 86% ray, 12% disk and 0.2% bullseye. "
      "10,655 segments have at least 5% bullseye, and 3,694 have at least 15%. The segments with "
      "the highest bullseye share are red, orange-red and mahogany-rayed forms, where the whole "
      "ray is red. A few may be other species. Segments include SAM3's low-scoring masks: filter "
      "by mask score once the run's metadata (results.csv) is joined on photo_id and mask_index.")
    figure(ph("full_run_top_bullseye.png"),
           "Figure 4. The 40 segments (>= 2,000 px) with the highest per-pixel bullseye share.")
    figure(ph("full_run_random.png"), "Figure 5. 40 random segments (>= 2,000 px).")

    P("8. Reproducing this", h2)
    C("pip install -e .\n"
      "pytest tests/                                   # parity with R, no R needed\n"
      "charisma-gpu CUTOUTS/ --out colours.csv --clut rudbeckia_hirta\n"
      "Rscript parity/parity.R paths.txt r.csv [clut.csv]   # your own parity check\n"
      "python  parity/parity.py paths.txt r.csv [--clut clut.csv]")
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.lightgrey, spaceBefore=8,
                            spaceAfter=8))
    P("The method is charisma's (Schwartz et al. 2025, doi:10.32614/CRAN.package.charisma) and "
      "recolorize's (Weller). Please cite them. Generated by docs/make_methodology_pdf.py.", note)

    SimpleDocTemplate(out, pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                      topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                      title="Converting charisma to the GPU — Methodology",
                      author="charisma-gpu").build(story)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", help="folder with the photo figures (not in the public repo)")
    ap.add_argument("--out", default=os.path.join(HERE, "methodology.pdf"))
    a = ap.parse_args()
    build(a.photos, a.out)
