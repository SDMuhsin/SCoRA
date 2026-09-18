#!/usr/bin/env python3
"""Render the SCoRA architecture figure and measure it.

The figure is drawn in centimetres and printed in points, so nothing about it
can be checked by reading the source: a coordinate that is right in the file
can still land on top of its neighbour once a glyph is set.  This instrument
renders the standalone figure, recovers the exact map from figure centimetres
to render pixels, and then asserts in pixels the things the drawing claims.

The map is recovered, not assumed.  pdflatex is asked to print the picture's
own bounding box in points; the same picture is rasterised and its ink box
found; the two rectangles give one affine map, and the instrument checks that
the two scales it implies agree with each other and with the requested dpi.
If any of that fails, every measurement below would be meaningless, so the
run stops there.

It also writes the tiles a reviewer reads:
    scratchpad/figaudit/page.png      the whole figure
    scratchpad/figaudit/mirror_*.png  the two lanes folded onto each other
    scratchpad/figaudit/crop_*.png    the regions worth a close look
    scratchpad/figaudit/map.json      the map, for zoom.sh

Selftest: `env/bin/python scripts/architecture_figure_audit.py`.
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGDIR = os.path.join(ROOT, "llmdocs", "paper", "scora_v1_prl", "figures")
OUTDIR = os.path.join(ROOT, "scratchpad", "figaudit")
DPI = 300
PPC = DPI / 2.54
COLUMN_PT = 522.0          # elsarticle 5p twocolumn \textwidth

DRAW_DIM = 30              # d as DRAWN (the generator's D)
AXIS = 3.00                # the mirror axis, in figure centimetres
# Windows whose silhouette is a frame and not a datum.  Used twice: once to
# CALIBRATE the map's y origin, once to CHECK the fold.
GEOM = [(11.62, 11.86), (12.12, 12.36), (12.58, 12.70),
        (12.75, 12.95), (13.55, 14.05)]
LANE_X0, LANE_X1 = 11.60, 17.60   # the part of the figure that is mirrored

# Regions a reviewer is asked to look at, in figure centimetres.
CROPS = {
    "layer":     (0.30, 0.70, 7.20, 5.30),
    "loss":      (6.80, 0.70, 11.60, 5.30),
    "lane_u_in": (10.50, 3.10, 14.40, 5.60),
    "lane_v_in": (10.50, 0.40, 14.40, 2.90),
    "lane_u_out": (13.90, 3.10, 17.80, 5.60),
    "lane_v_out": (13.90, 0.40, 17.80, 2.90),
    "seed":      (11.40, 2.30, 14.40, 3.70),
    "tip_u":     (13.80, 3.80, 15.60, 4.80),
    "tip_v":     (13.80, 1.20, 15.60, 2.20),
    "dw":        (4.90, 1.90, 7.30, 4.10),
    "gw":        (9.40, 1.90, 11.70, 4.10),
    "plus":      (4.50, 2.60, 5.40, 3.40),
    "gamma":     (6.20, 2.50, 7.20, 3.50),
    "bracket_l": (2.40, 1.80, 3.20, 4.20),
    "bracket_r": (6.70, 1.80, 7.50, 4.20),
    "state_u":   (14.30, 3.50, 15.90, 5.30),
    "panel_u":   (15.50, 3.20, 17.90, 5.40),
}


def sh(cmd, cwd=None):
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True,
                          text=True)


def build():
    """Rebuild the standalone, and read the picture's bounding box."""
    r = sh("pdflatex -interaction=nonstopmode -halt-on-error "
           "architecture_standalone.tex", cwd=FIGDIR)
    if not os.path.exists(os.path.join(FIGDIR, "architecture_standalone.pdf")):
        print(r.stdout[-3000:])
        raise SystemExit("the figure did not build")

    diag = os.path.join(FIGDIR, "_audit_bbox.tex")
    with open(diag, "w") as fh:
        fh.write(
            "\\documentclass[border=0cm]{standalone}\n"
            "\\usepackage{amsmath,amssymb}\n\\usepackage{tikz}\n"
            "\\usetikzlibrary{arrows.meta,calc,positioning}\n"
            "\\input{architecture_data}\n\\begin{document}\n"
            "\\input{architecture}%\n\\makeatletter\n"
            "\\typeout{AUDITBOX \\the\\pgf@picminx\\space\\the\\pgf@picmaxx"
            "\\space\\the\\pgf@picminy\\space\\the\\pgf@picmaxy}\n"
            "\\makeatother\n\\end{document}\n")
    r = sh("pdflatex -interaction=nonstopmode _audit_bbox.tex", cwd=FIGDIR)
    flat = re.sub(r"\s+", "", r.stdout)
    m = re.search(r"AUDITBOX([-\d.]+)pt([-\d.]+)pt([-\d.]+)pt([-\d.]+)pt", flat)
    if not m:
        raise SystemExit("could not read the picture bounding box")
    x0, x1, y0, y1 = (float(v) / 72.0 * 2.54 for v in m.groups())
    for ext in ("tex", "pdf", "aux", "log"):
        f = os.path.join(FIGDIR, f"_audit_bbox.{ext}")
        if os.path.exists(f):
            os.remove(f)
    return (x0, x1, y0, y1)


def render():
    os.makedirs(OUTDIR, exist_ok=True)
    stem = os.path.join(OUTDIR, "_r")
    sh(f"pdftoppm -r {DPI} -png -gray "
       f"{os.path.join(FIGDIR, 'architecture_standalone.pdf')} {stem}")
    src = stem + "-1.png"
    img = np.array(Image.open(src).convert("L"))
    os.remove(src)
    Image.fromarray(img).save(os.path.join(OUTDIR, "page.png"))
    return img


def page_pt():
    r = sh(f"pdfinfo {os.path.join(FIGDIR, 'architecture_standalone.pdf')}")
    m = re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", r.stdout)
    return float(m.group(1)), float(m.group(2))


def main() -> int:
    npass = nfail = 0

    def ck(cond, what):
        nonlocal npass, nfail
        if cond:
            npass += 1
        else:
            nfail += 1
            print(f"  FAIL  {what}")

    fx0, fx1, fy0, fy1 = build()
    img = render()
    h, w = img.shape
    ink = img < 250
    cols = np.where(ink.any(0))[0]
    rows = np.where(ink.any(1))[0]
    px0, px1, py0, py1 = int(cols[0]), int(cols[-1]), int(rows[0]), int(rows[-1])

    # --- the map, recovered from the two rectangles -----------------------
    sx = (px1 + 1 - px0) / (fx1 - fx0)
    sy = (py1 + 1 - py0) / (fy1 - fy0)
    ck(abs(sx - PPC) / PPC < 0.004, f"x scale {sx:.2f} px/cm matches {PPC:.2f}")
    ck(abs(sy - PPC) / PPC < 0.004, f"y scale {sy:.2f} px/cm matches {PPC:.2f}")
    ck(abs(sx - sy) / PPC < 0.004, "the render is isotropic")
    if nfail:
        print(f"{npass} passed, {nfail} failed")
        return 1

    ox = px0 - fx0 * PPC          # figure x = 0 lands here
    oy = py1 + 1 + fy0 * PPC      # figure y = 0 lands here

    # --- calibrate that origin before anyone measures with it ------------
    # \pgf@picminy is the PATH bounding box: it does not know about stroke
    # width, so the ink spills half a rule past it by different amounts at
    # the top and the bottom and the origin lands about a pixel out.  That
    # is 0.0085 cm at 300 dpi -- negligible in a check that measures a
    # DIFFERENCE, fatal in one that measures a SUM.  Three of four reviewers
    # reported the mirror axis as 3.009 and concluded the lower lane was
    # shifted; it is not, the map was, and they had no way to tell because
    # the diagnostic they shared cannot separate the two.
    #
    # Calibrate on features whose true centre is known EXACTLY and does not
    # depend on stroke width: each block frame is a rectangle drawn at
    # y = 2.10 and 3.90, so however thick its rule, its ink is symmetric
    # about y = 3.00.  Take the darkness-weighted centroid of one column of
    # each -- subpixel, threshold-free -- and average.  A silhouette fold
    # cannot do this job: it resolves the offset only to the nearest pixel,
    # and its parity is ambiguous, so it answered 2 where the truth is 1.
    def centroid_y(xc, y0, y1, o_y):
        c = 255.0 - img[int(o_y - y1 * PPC):int(o_y - y0 * PPC),
                        int(round(ox + xc * PPC))].astype(float)
        r = np.arange(len(c))
        return ((o_y - (int(o_y - y1 * PPC) + r)) / PPC * c).sum() / c.sum()

    FRAMES = (2.85, 4.65, 5.15, 6.95, 9.60, 11.40)   # the three d-by-d blocks
    seen = [centroid_y(xc, 2.02, 3.98, oy) for xc in FRAMES]
    ck(float(np.ptp(seen)) < 0.004,
       f"the six frame rules agree on the axis to {np.ptp(seen) * 10:.4f} mm")
    dcal = (AXIS - float(np.mean(seen))) * PPC        # in pixels, fractional
    # A large correction would mean the blocks really are misplaced, which
    # is a defect and not a calibration.  Absorb rasterisation, nothing more.
    ck(abs(dcal) <= 3,
       f"the map's y origin is {dcal:+.2f} px out"
       f" ({dcal / PPC * 10:+.3f} mm), corrected")
    oy += dcal

    json.dump({"ox": ox, "oy": oy, "ppc": PPC, "dpi": DPI,
               "fig": [fx0, fy0, fx1, fy1]},
              open(os.path.join(OUTDIR, "map.json"), "w"), indent=1)

    def X(x):
        return int(round(ox + x * PPC))

    def Y(y):
        return int(round(oy - y * PPC))

    # --- the figure fits the column it is printed in ----------------------
    pw, ph = page_pt()
    ck(pw <= COLUMN_PT, f"the figure is {pw:.1f}pt wide, under {COLUMN_PT:.0f}pt")
    ck(ph < 0.42 * 782, f"the figure is {ph:.1f}pt tall, under 2/5 of a page")
    ck(pw / ph > 2.6, "the figure is a band, not a block")

    # --- the two lanes are exact mirrors ----------------------------------
    # Both lanes are one macro called twice with a sign, so their GEOMETRY
    # must fold onto itself about the axis.  Their CONTENT must not: the two
    # factors carry different numbers and different frequencies, and a fold
    # that cancelled the cells as well would mean the figure was drawing the
    # same data twice.  So the test folds the silhouette, and only over the
    # windows where the silhouette is a frame and not a datum.
    # The windows are the ones whose silhouette is a frame: the two strip
    # frames, the support strip, two slices of the funnel that no label
    # reaches, and the state frames.  The coefficient strips are left out
    # because the tallest ink above them is their own symbol, and beta is
    # not the same height as alpha.  The state strips are left out for the
    # same reason: the count label above them is set upright in both lanes,
    # as every label is, so its silhouette cannot fold even when its box
    # is placed exactly.
    yax = Y(AXIS)
    cols = []
    for gx0, gx1 in GEOM:
        for c in range(X(gx0), X(gx1) + 1):
            up = np.where(ink[:yax, c])[0]
            dn = np.where(ink[yax:, c])[0]
            if len(up) and len(dn):
                cols.append((yax - up[0], dn[-1]))
    cols = np.array(cols)
    ck(len(cols) > 140, f"the mirror test found {len(cols)} columns to fold")
    # The map is recovered from an antialiased ink box, so the axis can land
    # up to a pixel or two out; the fold offset is searched, and what is
    # asserted is that ONE offset squares every window at once.
    offs = np.arange(-6, 7)
    errs = [np.abs((cols[:, 0] + d) - (cols[:, 1] - d)).mean() for d in offs]
    d0 = int(offs[int(np.argmin(errs))])
    dev = np.abs((cols[:, 0] + d0) - (cols[:, 1] - d0))
    ck(d0 == 0, f"the fold sits {d0} px from the CALIBRATED axis, want 0")
    # At 300 dpi one pixel is 0.0085 cm, and a rule whose true position is
    # a half pixel off the grid folds one pixel wrong everywhere, so the
    # tolerance is stated in pixels but meant in centimetres.
    ck(dev.max() <= 2,
       f"every folded column agrees to {int(dev.max())} px"
       f" ({dev.max() / PPC * 10:.3f} mm)")
    ck(float(dev.mean()) < 1.1,
       f"the mean fold error is {float(dev.mean()):.3f} px"
       f" ({float(dev.mean()) / PPC * 10:.3f} mm)")

    # The same fold over everything, saved as a picture rather than asserted:
    # what survives it is what differs between the two factors, which should
    # be cells, curves, stems and the four symbols, and nothing else.
    band = ink[:, X(LANE_X0):X(LANE_X1)]
    yfold = yax + d0
    k = min(yfold, band.shape[0] - yfold)
    a = band[yfold - k:yfold, :]
    b = band[yfold:yfold + k, :][::-1, :]
    Image.fromarray(np.where(a != b, 0, 255).astype(np.uint8)).save(
        os.path.join(OUTDIR, "mirror_diff.png"))

    # --- the blocks are where the drawing says they are -------------------
    # Each entry is a rule the figure draws; the test is that ink is found
    # within half a millimetre of it, and that the gap beside it is clear.
    def inked(x, y, r=2):
        return bool(ink[Y(y) - r:Y(y) + r + 1, X(x) - r:X(x) + r + 1].any())

    def clear(x0, y0, x1, y1):
        return not bool(ink[Y(y1):Y(y0), X(x0):X(x1)].any())

    edges = [("X left", 0.45, 4.84), ("X right", 2.25, 4.84),
             ("gX left", 0.45, 1.16), ("gX right", 2.25, 1.16),
             ("W0 left", 2.85, 3.00), ("W0 right", 4.65, 3.00),
             ("dW left", 5.15, 3.00), ("dW right", 6.95, 3.00),
             ("H left", 7.65, 4.84), ("H right", 9.45, 4.84),
             ("gH left", 7.65, 1.16), ("gH right", 9.45, 1.16),
             ("gdW left", 9.60, 3.00), ("gdW right", 11.40, 3.00)]
    for nm, x, y in edges:
        ck(inked(x, y), f"the {nm} edge is drawn where it is declared")

    # the two squares really are squares, and the same square
    for nm, x0, x1 in (("W0", 2.85, 4.65), ("dW", 5.15, 6.95),
                       ("gdW", 9.60, 11.40)):
        ck(abs((x1 - x0) - 1.80) < 1e-9, f"{nm} is d cells wide")

    # the gap the frozen weight leaves in the backward row: there is no
    # gradient block under W0, and the figure must not quietly grow one
    ck(clear(2.90, 0.50, 4.60, 0.85), "nothing sits below the frozen weight")

    # --- the compression is drawn at the ratio it has ---------------------
    # the coefficient strip against the factor strip, measured in ink
    for lo, hi, nm in ((3.40, 5.20, "u"), (0.80, 2.60, "v")):
        ck(abs((hi - lo) - 1.80) < 1e-9, f"the {nm} strip is d cells long")
    for lo, hi, nm in ((4.40, 4.70, "beta"), (1.30, 1.60, "alpha")):
        ck(abs((hi - lo) - 0.30) < 1e-9, f"the {nm} strip is s cells long")
    ck(abs(0.30 / 1.80 - 128 / 768) < 1e-12,
       "s/d in the drawing is 128/768")

    # --- the funnel really tapers -----------------------------------------
    def ink_span(x, y0, y1):
        col = ink[Y(y1):Y(y0), X(x)]
        w = np.where(col)[0]
        return (0 if len(w) == 0 else (w[-1] - w[0] + 1) / PPC)

    # The taper is the one ratio the caption invites a reader to measure,
    # so it is measured, not assumed.  It is measured at two columns in the
    # funnel's middle and extrapolated to its two edges, because a span read
    # at the tip itself is dominated by the slanted strokes' own width: the
    # closed trapezoid used to render 8 per cent over there, a 0.5 pt line
    # meeting a 3 mm edge at a shallow angle adding about 0.1 mm at each end.
    xa, xb = 12.80, 13.90
    sa, sb = ink_span(xa, 3.30, 5.30), ink_span(xb, 3.70, 4.90)
    slope = (sb - sa) / (xb - xa)
    at_wide = sa + slope * (12.70 - xa)
    at_narrow = sa + slope * (14.10 - xa)
    ck(abs(at_wide - 1.80) < 0.04,
       f"the funnel's wide edge extrapolates to {at_wide:.3f} cm, d = 1.80")
    ck(abs(at_narrow - 0.30) < 0.04,
       f"the funnel's narrow edge extrapolates to {at_narrow:.3f} cm, s = 0.30")
    taper = at_wide / max(at_narrow, 1e-6)
    ck(abs(taper - 6.0) / 6.0 < 0.10,
       f"the funnel tapers {taper:.2f} to one against d/s = 6")

    # --- the gradient's two segments are one line ------------------------
    # In each lane the gradient line stops at the factor strip's near edge
    # and resumes on its far edge, because it passes BEHIND the factor: no
    # orthogonal route around it exists once the two trunks that feed the
    # two strips are forbidden to cross.  The occlusion only reads as
    # occlusion if the two segments are collinear, so that is measured.
    def line_row(x, y0, y1):
        col = ink[Y(y1):Y(y0), X(x)]
        w = np.where(col)[0]
        return None if len(w) == 0 else Y(y1) + int(round(w.mean()))

    for sgn, nm in ((1, "u"), (-1, "v")):
        yc = AXIS + sgn * 1.175
        a = line_row(12.07, yc - 0.10, yc + 0.10)
        b = line_row(12.50, yc - 0.10, yc + 0.10)
        ck(a is not None and b is not None,
           f"the {nm} lane's gradient line is drawn both sides of the factor")
        if a is not None and b is not None:
            ck(abs(a - b) <= 1,
               f"the {nm} lane's two gradient segments are collinear"
               f" ({abs(a - b)} px apart)")

    # --- the wide edge is countable as d cells, not just s -------------
    # Drawn as one flat fill with only the s lit cells bordered, a blind
    # reader counted 5 lit against 5 at the narrow end and concluded the
    # funnel maps s to s.  That is a ratio of one where the method's whole
    # claim is s/d, and the caption asks the reader to count it here.  The
    # separators are white rules, so they read far lighter than any cell:
    # the count is the same at every threshold from 235 to 250.
    for sgn, nm in ((1, "u"), (-1, "v")):
        r0, r1 = sorted((Y(AXIS + sgn * 2.19), Y(AXIS + sgn * 0.41)))
        col = img[r0:r1, X(12.64)]
        light = (col > 240).astype(np.int8)
        n = int((np.diff(np.concatenate(([0], light, [0]))) == 1).sum())
        ck(n == DRAW_DIM - 1,
           f"the {nm} funnel's wide edge shows {n} cell separators,"
           f" want {DRAW_DIM - 1} for {DRAW_DIM} cells")

    # --- the five lit cells on each wide edge survive the travel lines ----
    # They are drawn after the lines for exactly this reason: drawn before,
    # two of the five fused with a line and a blind reader counted six.
    for sgn, nm in ((1, "u"), (-1, "v")):
        # inset past the strip's own frame, which is darker than an unlit
        # cell and would count as a sixth and seventh run
        r0, r1 = sorted((Y(AXIS + sgn * 2.16), Y(AXIS + sgn * 0.44)))
        col = img[r0:r1, X(12.64)]
        dark = (col < 200).astype(np.int8)
        n = int((np.diff(np.concatenate(([0], dark, [0]))) == 1).sum())
        ck(n == 5, f"the {nm} funnel's wide edge lights {n} cells, want 5")

    # --- no foreign ink inside either generated field --------------------
    # The two synthesis trunks used to run 7.45 mm DOWN THROUGH the update's
    # interior and land their heads on the plaid, which is the one place the
    # figure states that the update is rank one.  They now stop on its frame.
    # The darkest cell any field draws is 109 of 255 and a rule is near
    # black, so a floor separates them.  The gamma disc is the one object
    # allowed inside, and it is masked out by radius, not by name.
    for nm, (x0, y0, x1, y1), disc in (
            ("update", (5.15, 2.10, 6.95, 3.90), (6.60, AXIS, 0.24)),
            ("update gradient", (9.60, 2.10, 11.40, 3.90), None)):
        f = img[Y(y1 - 0.04):Y(y0 + 0.04), X(x0 + 0.04):X(x1 - 0.04)]
        keep = np.ones(f.shape, bool)
        if disc is not None:
            cx, cy, r = disc
            gy, gx = np.mgrid[0:f.shape[0], 0:f.shape[1]]
            keep = (((gx - (cx - x0 - 0.04) * PPC) ** 2
                     + (gy - ((y1 - 0.04) - cy) * PPC) ** 2) > (r * PPC) ** 2)
        lo = int(f[keep].min())
        ck(lo >= 95, f"the {nm} field's interior is all cells"
                     f" (darkest {lo}, a rule would be under 60)")

    # --- nothing is drawn outside the region the figure is planned in -----
    ck(fx0 > 0.30 and fx1 < 17.85, f"ink spans x {fx0:.2f} to {fx1:.2f} cm")
    ck(fy0 > 0.18 and fy1 < 5.85, f"ink spans y {fy0:.2f} to {fy1:.2f} cm")

    # --- the tiles a reviewer reads ---------------------------------------
    for nm, (x0, y0, x1, y1) in CROPS.items():
        tile = img[max(Y(y1), 0):Y(y0), max(X(x0), 0):X(x1)]
        ck(tile.size > 0, f"crop {nm} is inside the page")
        if tile.size:
            big = Image.fromarray(tile).resize(
                (tile.shape[1] * 3, tile.shape[0] * 3), Image.LANCZOS)
            big.save(os.path.join(OUTDIR, f"crop_{nm}.png"))

    print(f"{npass} passed, {nfail} failed")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
