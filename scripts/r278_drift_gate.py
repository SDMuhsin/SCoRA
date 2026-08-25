#!/usr/bin/env python
"""[R.278] STANDING HARNESS-DRIFT GATE — every grid cell that has a banked seed-41 twin.

[R.250 4] and [R.257] each did this ONCE, by hand, for one arm.  The check is the strongest
evidence CONTEXT 2's headline is not a harness artifact, so it should be a standing gate over
EVERY available twin rather than two ad-hoc verifications.  It also scores [R.263] P11
(QWHA parity) automatically when those cells land.

⛔ A twin is only admitted after its DRIVER has been grepped for the flags attributed to it
([R.250]'s rule, and [R.244]'s three-check-in error).  Each entry cites the driver line
checked.  ⭐ [R.257]'s rider: grep the driver's FLAGS, not its HEADER -- r68's header is
stale copy-paste from r65.

Determinism: [R.257] measured |diff| = 0.000e+00 across ~3 days and every [R.236] harness
edit, so the tolerance is EXACT.  A non-zero diff is a finding, not a rounding artifact.
"""
import argparse, glob, importlib.util, os, re, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_s = importlib.util.spec_from_file_location("r237_read", os.path.join(ROOT, "scripts", "r237_read.py"))
R = importlib.util.module_from_spec(_s); _s.loader.exec_module(R)
PH = os.path.join(ROOT, "scratchpad", "phaseR")

# grid cell -> (banked log, driver verified, what was checked)
TWINS = {
    "fftm-lr5e-2-s150": ("r68/k256warm-s41.log", "r68_warmup_k256.sh",
                         "lr 5e-2, clf 5e-3, wd .01, warm 140, bs 32, 30ep, k256, qv; "
                         "scaling/seed/support implicit -> defaults 150/777/scattered [R.257]"),
    # ⛔ the first version of this table guessed the filename "r82_slr_zero_warm.sh", which
    # does not exist -- fixture D7 caught it.  That is [R.244]'s error exactly: attributing
    # flags to a driver never opened.  The real file is r82_slr_zero_warmup.sh.
    "scora-lr5e-2":     ("r82/slrzerowarm-s41.log", "r82_slr_zero_warmup.sh",
                         "lr 5e-2, clf 5e-3, wd .01, warm 140, bs 32, 30ep, qv, rank 1, "
                         "s 128, seed 777, zero init [r82:3-6]; also verified in [R.250 4]"),
    "lyra-lr5e-2-e3.0": ("r161/lyra-s41.log", "r161_rte_pareto.sh",
                         "lr 5e-2, clf 5e-3, wd .01, warm 140, bs 32, 30ep, qv, p=q=16, "
                         "scaling 0.2, d_init 0.07, geometric, exp 3.0 [r161:4-8]"),
    "qwha-lr5e-2-s150.0": ("r188/qwha-s41.log", "r188_qwha.sh",
                           "lr 5e-2, clf 5e-3, wd .01, warm 140, bs 32, 30ep, qv, k256, "
                           "scaling 150.0, seed 777, init_weights 0 [r188:5-8]"),
    "qwha-lr5e-2-s106.0660": ("r232/qwhac-s41.log", "r232_qwha_corrected.sh",
                              "as r188 with scaling 150.0 -> 106.0660, the ONLY changed flag "
                              "[R.232 2] -- this cell scores [R.263] P11"),
}


def banked(rel):
    p = os.path.join(PH, rel)
    if not os.path.exists(p):
        return None
    v = [float(x) for x in re.findall(r"'accuracy': ([0-9.eE+-]+)", open(p, errors="ignore").read())]
    return max(v) if len(v) >= 30 else None


def report(cells, stream=sys.stdout):
    p = lambda *a: print(*a, file=stream)
    p("=" * 78)
    p("[R.278] HARNESS-DRIFT GATE -- grid cells vs their banked seed-41 twins")
    p("=" * 78)
    p("  tolerance: EXACT.  [R.257] measured |diff| = 0.000e+00; a non-zero diff is a finding.")
    p("")
    n_ok = n_bad = n_wait = 0
    for cell, (rel, drv, note) in sorted(TWINS.items()):
        g = cells.get(cell)
        b = banked(rel)
        if g is None or b is None:
            n_wait += 1
            p(f"  {cell:24s} ⏳ {'grid cell' if g is None else 'banked twin'} not available "
              f"({rel})")
            continue
        d = g["acc"] - b
        ok = d == 0.0
        n_ok += ok; n_bad += (not ok)
        p(f"  {cell:24s} grid {g['acc']:.16f}")
        p(f"  {'':24s} bank {b:.16f}   |diff| {abs(d):.3e}  "
          + ("✅ BIT-IDENTICAL" if ok else "⛔⛔ DRIFT"))
        p(f"  {'':24s} driver {drv}: {note}")
    p("")
    p(f"  {n_ok} bit-identical, {n_bad} drifted, {n_wait} pending")
    if n_bad:
        p("  ⛔⛔ DRIFT DETECTED. CONTEXT 2's 'not a harness artifact' argument rests on these")
        p("     twins. Escalate before any further reading of the grid.")
    elif n_ok >= 2:
        p(f"  ✅ {n_ok} independent arms reproduce bit-identically -> no harness drift.")
    return {"ok": n_ok, "bad": n_bad, "pending": n_wait}


def selftest():
    ok, bad = [], []
    def chk(n, c, d=""):
        (ok if c else bad).append(n); print(f"  {'PASS' if c else 'FAIL'}  {n}" + (f"  {d}" if d else ""))
    import io, tempfile
    H = lambda a: {"acc": a, "collapsed": False, "near_floor": False}
    real = banked("r68/k256warm-s41.log")
    chk("D1 the banked [R.68] twin parses to its published seed-41 value",
        real is not None and abs(real - 0.7545126353790613) < 1e-15, str(real))
    buf = io.StringIO(); r = report({"fftm-lr5e-2-s150": H(real)}, buf)
    chk("D2 an exact match is BIT-IDENTICAL", r["ok"] == 1 and "BIT-IDENTICAL" in buf.getvalue())
    buf = io.StringIO(); r = report({"fftm-lr5e-2-s150": H(real + 1e-12)}, buf)
    chk("D3 a 1e-12 difference is DRIFT, not 'close enough'",
        r["bad"] == 1 and "DRIFT DETECTED" in buf.getvalue())
    buf = io.StringIO(); r = report({}, buf)
    chk("D4 missing cells are PENDING, never a pass", r["ok"] == 0 and r["pending"] == len(TWINS))
    chk("D5 a nonexistent banked log yields None rather than crashing",
        banked("nosuchdir/nope.log") is None)
    chk("D6 the P11 twin is registered", "qwha-lr5e-2-s106.0660" in TWINS)
    chk("D7 every twin cites a driver that EXISTS on disk",
        all(os.path.exists(os.path.join(PH, v[1])) for v in TWINS.values()),
        str([v[1] for v in TWINS.values() if not os.path.exists(os.path.join(PH, v[1]))]))
    print(f"\n  selftest: {len(ok)} passed, {len(bad)} failed")
    if bad: print("  FAILED: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    report(R.load(os.path.join(PH, "r237", "csv")))
