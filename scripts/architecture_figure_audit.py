#!/usr/bin/env python3
"""Mechanical audit of the SCoRA architecture figure.

A drawing this dense is not reviewable by reading its source, and the
blemishes that matter -- an arrowhead a hair inside the shape it points at, a
frame a pixel off its mirror, a curve touching its own border -- are exactly
the ones an eye skims past.  So they are measured here, on the rendered page,
and the numbers are the finding.

Checks:
  mirror     the figure is drawn about y = 3.60 cm; everything right of the
             fork is one macro called twice, so the render must be symmetric
             there up to the data and the labels.  Emits a diff image.
  bounds     the drawing fits the 522 pt column it is destined for.
  ink        no ink touches the page border, so nothing is clipped.
  crops      writes zoomed tiles so a reviewer can look at one joint at a time.

Run: env/bin/python scripts/architecture_figure_audit.py [--outdir DIR]
Test: env/bin/python scripts/architecture_figure_audit.py --selftest
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGDIR = os.path.join(ROOT, "llmdocs", "paper", "scora_v1_prl", "figures")
DRIVER = "architecture_standalone"

# the column the figure is drawn for, and the border the standalone class adds
COLUMN_PT = 522.0
BORDER_CM = 0.2

# the drawing's own frame, from the geometry block of architecture.tex
FIG_X0, FIG_X1 = 0.50, 17.35
FIG_Y0, FIG_Y1 = 0.95, 6.25
# The topmost ink is a 0.5 pt frame CENTRED on FIG_Y1, so the page standalone
# crops to is half a stroke taller than the declared frame.  Without this the
# map reads 0.09 mm low everywhere and the balance check reports a lean the
# drawing does not have.
STROKE_CM = 0.5 / 72 * 2.54 / 2
AXIS_Y = 3.60
FORK_X = 7.25

DPI = 300

# the joints worth looking at closely, in figure centimetres:
# name -> (x0, y0, x1, y1)
CROPS = {
    "layer": (0.20, 0.80, 5.60, 6.40),
    "sum": (2.20, 4.30, 5.50, 6.40),
    "neck": (4.90, 2.90, 8.10, 4.40),
    "fork": (6.30, 1.90, 8.40, 5.30),
    "lane_u_head": (7.40, 4.10, 12.90, 5.90),
    "lane_v_head": (7.40, 1.30, 12.90, 3.10),
    "lane_u_tail": (12.60, 4.00, 17.50, 5.90),
    "lane_v_tail": (12.60, 1.30, 17.50, 3.20),
    "seed": (10.20, 2.60, 12.80, 4.60),
    "budget": (15.30, 2.05, 17.50, 5.25),
    "spectrum_u": (12.80, 4.20, 16.10, 5.30),
    "deltaw": (2.90, 2.30, 5.50, 4.90),
}


def cm_to_px(v: float) -> float:
    return v / 2.54 * DPI


def render(outdir: str) -> np.ndarray:
    """Build the standalone figure and return the page as a grey array."""
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                    DRIVER + ".tex"], cwd=FIGDIR, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    png = os.path.join(outdir, "page")
    subprocess.run(["pdftoppm", "-r", str(DPI), "-png", "-gray", "-singlefile",
                    os.path.join(FIGDIR, DRIVER + ".pdf"), png], check=True)
    return read_png_gray(png + ".png")


def read_png_gray(p: str) -> np.ndarray:
    """Decode a greyscale PNG without pulling in an image library."""
    out = subprocess.run(["pdftoppm", "-v"], capture_output=True)
    del out
    import zlib
    import struct
    raw = open(p, "rb").read()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, w = 8, b"", None
    while pos < len(raw):
        ln = struct.unpack(">I", raw[pos:pos + 4])[0]
        typ = raw[pos + 4:pos + 8]
        dat = raw[pos + 8:pos + 8 + ln]
        if typ == b"IHDR":
            w, h, depth, ctype = struct.unpack(">IIBB", dat[:10])
            # pdftoppm hands back grey as 1 byte and colour as 3; both are
            # accepted so the reader does not depend on a renderer flag
            assert depth == 8 and ctype in (0, 2), (depth, ctype)
            nch = 1 if ctype == 0 else 3
        elif typ == b"IDAT":
            idat += dat
        pos += 12 + ln
    buf = zlib.decompress(idat)
    stride = w * nch
    img = np.zeros((h, stride), dtype=np.int16)
    prev = np.zeros(stride, dtype=np.int16)
    i = 0
    for y in range(h):
        ft = buf[i]
        i += 1
        line = np.frombuffer(buf[i:i + stride], dtype=np.uint8).astype(np.int16)
        i += stride
        if ft == 0:
            cur = line.copy()
        elif ft == 1:
            cur = line.copy()
            for x in range(nch, stride):
                cur[x] = (cur[x] + cur[x - nch]) & 0xFF
        elif ft == 2:
            cur = (line + prev) & 0xFF
        elif ft == 3:
            cur = line.copy()
            for x in range(stride):
                a = cur[x - nch] if x >= nch else 0
                cur[x] = (cur[x] + ((a + prev[x]) >> 1)) & 0xFF
        elif ft == 4:
            cur = line.copy()
            for x in range(stride):
                a = int(cur[x - nch]) if x >= nch else 0
                b = int(prev[x])
                c = int(prev[x - nch]) if x >= nch else 0
                pp = a + b - c
                pa, pb, pc = abs(pp - a), abs(pp - b), abs(pp - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                cur[x] = (cur[x] + pr) & 0xFF
        else:
            raise ValueError(f"filter {ft}")
        img[y] = cur
        prev = cur
    if nch == 1:
        return img.astype(np.uint8)
    rgb = img.reshape(h, w, 3).astype(np.float64)
    lum = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    return np.clip(lum, 0, 255).astype(np.uint8)


def write_png_gray(p: str, a: np.ndarray) -> None:
    import zlib
    import struct
    h, w = a.shape
    raw = b"".join(b"\x00" + a[y].astype(np.uint8).tobytes() for y in range(h))

    def chunk(t, d):
        return (struct.pack(">I", len(d)) + t + d
                + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    open(p, "wb").write(png)


def fig_to_img(img: np.ndarray, x: float, y: float):
    """Figure centimetres -> pixel (col, row).

    Anchored on the page ORIGIN and the scale, not on the page width: TikZ
    reserves a little empty space past the rightmost ink for the arrow tips
    it has declared, so a mapping that stretched the declared frame across
    the whole page would be off by that slack everywhere.
    """
    del img
    ppc = DPI / 2.54
    return ((x - (FIG_X0 - BORDER_CM - STROKE_CM)) * ppc,
            ((FIG_Y1 + BORDER_CM + STROKE_CM) - y) * ppc)


def audit(outdir: str, verbose: bool = True):
    os.makedirs(outdir, exist_ok=True)
    img = render(outdir)
    h, w = img.shape
    findings = []

    def say(*a):
        if verbose:
            print(*a)

    # -- the page the figure will occupy
    pdf = subprocess.run(["pdfinfo", os.path.join(FIGDIR, DRIVER + ".pdf")],
                         capture_output=True, text=True).stdout
    size = [l for l in pdf.splitlines() if l.startswith("Page size")][0]
    wpt = float(size.split()[2])
    drawn = wpt - 2 * BORDER_CM / 2.54 * 72
    say(f"page      {size.split(':')[1].strip()}")
    say(f"drawing   {drawn:.1f} pt wide against a {COLUMN_PT:.0f} pt column"
        f"  ({drawn / COLUMN_PT * 100:.1f}%)")
    if drawn > COLUMN_PT:
        findings.append(f"the drawing is {drawn:.1f} pt, wider than the column")

    # -- nothing clipped: the border must be blank all the way round
    b = int(cm_to_px(BORDER_CM) * 0.6)
    edges = [img[:b, :], img[-b:, :], img[:, :b], img[:, -b:]]
    touch = [int((e < 250).sum()) for e in edges]
    say(f"border    ink in the {b}px margin, t/b/l/r: {touch}")
    if any(t > 0 for t in touch):
        findings.append(f"ink reaches the page border {touch}; something is"
                        f" outside the frame the geometry block declares")

    # -- the ink must sit inside the frame the geometry block declares, or
    #    every coordinate quoted in a finding below points at the wrong thing
    ink = img < 250
    cols = np.where(ink.any(axis=0))[0]
    rows = np.where(ink.any(axis=1))[0]
    ppc = DPI / 2.54
    ix0 = cols.min() / ppc + (FIG_X0 - BORDER_CM - STROKE_CM)
    ix1 = cols.max() / ppc + (FIG_X0 - BORDER_CM - STROKE_CM)
    iy1 = (FIG_Y1 + BORDER_CM + STROKE_CM) - rows.min() / ppc
    iy0 = (FIG_Y1 + BORDER_CM + STROKE_CM) - rows.max() / ppc
    say(f"ink       x {ix0:.2f} .. {ix1:.2f} cm,  y {iy0:.2f} .. {iy1:.2f} cm"
        f"   (declared {FIG_X0} .. {FIG_X1},  {FIG_Y0} .. {FIG_Y1})")
    for got, want, what in ((ix0, FIG_X0, "left"), (ix1, FIG_X1, "right"),
                            (iy0, FIG_Y0, "bottom"), (iy1, FIG_Y1, "top")):
        if abs(got - want) > 0.05:
            findings.append(f"the {what} edge of the ink is at {got:.2f} cm,"
                            f" not the declared {want:.2f}: the geometry"
                            f" block in architecture.tex is out of date")
    mid = (iy0 + iy1) / 2
    say(f"balance   the ink is centred on y = {mid:.3f} cm,"
        f" the axis is {AXIS_Y}")
    if abs(mid - AXIS_Y) > 0.04:
        findings.append(f"the ink is centred on y = {mid:.3f}, not on the"
                        f" axis {AXIS_Y}: the figure is top or bottom heavy")

    # -- the mirror.  Everything right of the fork is one macro called twice,
    #    so the only differences there may be the data and the labels.
    axr = int(round(fig_to_img(img, 0, AXIS_Y)[1]))
    say(f"axis      y = {AXIS_Y} cm is image row {axr} of {h}")
    fx = int(round(fig_to_img(img, FORK_X, 0)[0]))
    # Fold about the row that actually minimises the difference, and report
    # how far that is from the axis.  A whole-pixel search, because the axis
    # lands between two pixel rows and folding on the wrong one prints a
    # 1 px ghost along every horizontal rule in the picture -- an artefact of
    # the measurement that would otherwise read as a crooked drawing.
    def fold(ar):
        hh = min(ar, h - ar)
        a = img[ar - hh:ar, fx:].astype(np.int16)
        b = img[ar:ar + hh, fx:][::-1, :].astype(np.int16)
        return np.abs(a - b)
    cand = sorted(((int((fold(ar) > 60).sum()), ar)
                   for ar in range(axr - 6, axr + 7)))
    bestn, bestr = cand[0]
    say(f"fold      best at row {bestr}, axis maps to {axr}"
        f"  (delta {bestr - axr} px = {(bestr - axr) / DPI * 2.54 * 10:.2f} mm)")
    if abs(bestr - axr) > 2:
        findings.append(f"the two lanes fold best about row {bestr} but the"
                        f" axis is row {axr}: they are not mirror images")
    right, flip = None, None
    diff = fold(bestr)
    diff = diff.astype(np.uint8)
    # a structural asymmetry is a run of dark pixels in one column; the data
    # and the labels differ everywhere, so report where the diff is a LINE
    strong = (diff > 60)
    per_col = strong.sum(axis=0)
    write_png_gray(os.path.join(outdir, "mirror_diff.png"), 255 - diff)
    say(f"mirror    {strong.sum()} pixels differ by >60/255 across the fold"
        f" (data and labels included)")
    say(f"          worst columns: "
        + ", ".join(f"x={(fx + c - cm_to_px(BORDER_CM)) / DPI * 2.54 + FIG_X0:.2f}cm"
                    f":{per_col[c]}" for c in np.argsort(per_col)[-5:][::-1]))

    # -- crops, so a reviewer can look at one joint at a time
    for name, (x0, y0, x1, y1) in CROPS.items():
        c0, r1 = fig_to_img(img, x0, y0)
        c1, r0 = fig_to_img(img, x1, y1)
        tile = img[max(0, int(r0)):int(r1), max(0, int(c0)):int(c1)]
        write_png_gray(os.path.join(outdir, f"crop_{name}.png"), tile)
    say(f"crops     {len(CROPS)} tiles in {outdir}")

    if findings:
        print("\nFINDINGS")
        for f in findings:
            print("  *", f)
    else:
        print("\nno mechanical finding")
    return findings


def selftest() -> int:
    ok = bad = 0

    def ck(c, label):
        nonlocal ok, bad
        if c:
            ok += 1
        else:
            bad += 1
            print(f"  FAIL  {label}")

    # the png reader has to be right, or every measurement below is fiction
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        a = (np.arange(64 * 40).reshape(40, 64) % 251).astype(np.uint8)
        p = os.path.join(td, "t.png")
        write_png_gray(p, a)
        b = read_png_gray(p)
        ck(b.shape == a.shape, "png round trip keeps the shape")
        ck(np.array_equal(a, b), "png round trip is lossless")

    # the coordinate map must put the declared corners on the declared pixels
    img = np.zeros((676, 2054), dtype=np.uint8)
    for x, y in ((FIG_X0, FIG_Y0), (FIG_X1, FIG_Y1)):
        c, r = fig_to_img(img, x, y)
        ck(0 < c < img.shape[1] and 0 < r < img.shape[0],
           f"({x},{y}) cm lands inside the page")
    c0, r0 = fig_to_img(img, FIG_X0, FIG_Y1)
    c1, r1 = fig_to_img(img, FIG_X1, FIG_Y0)
    ck(abs((c1 - c0) - cm_to_px(FIG_X1 - FIG_X0)) < 2.0,
       "the mapped width is the declared width")
    ck(abs((r1 - r0) - cm_to_px(FIG_Y1 - FIG_Y0)) < 2.0,
       "the mapped height is the declared height")
    ck(abs(c0 - cm_to_px(BORDER_CM + STROKE_CM)) < 1.5,
       "the declared left edge lands on the standalone border")
    ck(abs(r0 - cm_to_px(BORDER_CM + STROKE_CM)) < 1.5,
       "the declared top edge lands on the standalone border")
    ck(0.004 < STROKE_CM < 0.010,
       "the half-stroke correction is a half stroke, not a fudge")
    for name, (x0, y0, x1, y1) in CROPS.items():
        ck(x0 < x1 and y0 < y1, f"crop {name} is not inside out")
        ck(FIG_X0 - 0.4 <= x0 and x1 <= FIG_X1 + 0.4,
           f"crop {name} is inside the drawing horizontally")
        ck(FIG_Y0 - 0.4 <= y0 and y1 <= FIG_Y1 + 0.4,
           f"crop {name} is inside the drawing vertically")
    print(f"{ok} passed, {bad} failed")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(ROOT, "scratchpad",
                                                     "figaudit"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    audit(a.outdir)
    return selftest()


if __name__ == "__main__":
    sys.exit(main())
