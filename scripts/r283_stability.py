#!/usr/bin/env python
"""[R.283] Per-arm TRAINING STABILITY on the grid: degenerate-epoch rate at matched ladder.

[R.282] left the DFT > WHT > Haar accuracy ordering unexplained and pointed at [R.34]'s
unpredicted hint -- the DCT frame showed 0/150 degenerate epochs vs a random frame's 14/150,
i.e. the axis may be STABILITY rather than representation.  This tests that on the grid.

⛔⛔ THE CONFOUND THIS FILE EXISTS TO CONTROL.  A raw per-arm dead-cell count is dominated by
LADDER EXTENT, not by the arm: FourierFT's lr ladder was extended to 5e-1 by [R.253] and ALL
of its dead cells live on that rung, which no other arm ran.  Uncontrolled, FourierFT looks
like the LEAST stable arm; restricted to the shared ladder it is the MOST stable.  The first
reading was wrong and the control reversed it.

Degenerate epoch := eval accuracy <= RTE majority + 1e-4 ([R.222]'s metric-correct rule).
It mixes SLOW IGNITION (early epochs, [R.227]) with COLLAPSE (high LR); this file does not
separate them and says so.
"""
import argparse, glob, importlib.util, os, re, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_s = importlib.util.spec_from_file_location("r237_read", os.path.join(ROOT, "scripts", "r237_read.py"))
R = importlib.util.module_from_spec(_s); _s.loader.exec_module(R)
SHARED_LR = {"5e-3", "1.5e-2", "5e-2", "1.5e-1"}     # the ladder EVERY arm ran


def epoch_stats(label, logdir):
    tot = deg = 0
    for p in glob.glob(os.path.join(logdir, label + "*.log")):
        v = [float(x) for x in re.findall(r"'accuracy': ([0-9.eE+-]+)",
                                          open(p, errors="ignore").read())]
        tot += len(v)
        deg += sum(1 for x in v if x <= R.RTE_MAJORITY + 1e-4)
    return deg, tot


def report(cells, logdir, restrict=True, stream=sys.stdout):
    p = lambda *a: print(*a, file=stream)
    p("=" * 78)
    p(f"[R.283] per-arm stability  ({'SHARED lr ladder only' if restrict else '⛔ UNCONTROLLED'})")
    p("=" * 78)
    if not restrict:
        p("  ⛔ ladder extent is NOT controlled -- dead cells track which arm got an extra rung.")
    rows = {}
    p(f"  {'arm':8s} {'cells':>5s} {'dead':>5s} {'degenerate epochs':>20s}  {'mean acc':>9s}")
    for arm in ("fftm", "loca", "lyra", "wave1", "wave2", "qwha", "scora"):
        sub = {k: v for k, v in cells.items()
               if R.arm_of(k) == arm and "-ofat-" not in k
               and (not restrict or R._knobs(k).get("lr") in SHARED_LR)}
        if not sub:
            continue
        dead = sum(1 for v in sub.values() if v["collapsed"] or v.get("near_floor"))
        deg = tot = 0
        for k in sub:
            d, t = epoch_stats(k, logdir); deg += d; tot += t
        rate = deg / tot if tot else float("nan")
        acc = sum(v["acc"] for v in sub.values()) / len(sub)
        rows[arm] = (len(sub), dead, rate, acc)
        p(f"  {arm:8s} {len(sub):5d} {dead:5d} {deg:7d}/{tot:<7d} {100*rate:5.1f}%  {acc:9.4f}")
    if rows:
        best = min(rows, key=lambda a: rows[a][2])
        p(f"\n  most stable: {best} ({100*rows[best][2]:.1f}% degenerate epochs)"
          f"   highest mean acc: {max(rows, key=lambda a: rows[a][3])}")
    return rows


def selftest():
    ok, bad = [], []
    def chk(n, c, d=""):
        (ok if c else bad).append(n); print(f"  {'PASS' if c else 'FAIL'}  {n}" + (f"  {d}" if d else ""))
    import io, tempfile
    H = lambda a: {"acc": a, "collapsed": False, "near_floor": False}
    DEAD = {"acc": R.RTE_MAJORITY, "collapsed": True, "near_floor": False}
    cells = {"fftm-lr5e-1-s150": DEAD, "fftm-lr5e-2-s150": H(0.75),
             "wave1-lr5e-2-fs150": H(0.65)}
    with tempfile.TemporaryDirectory() as td:
        buf = io.StringIO(); r = report(cells, td, restrict=True, stream=buf)
        chk("B1 the 5e-1 rung is EXCLUDED by the shared-ladder restriction",
            r["fftm"][0] == 1 and r["fftm"][1] == 0, str(r))
        buf = io.StringIO(); r2 = report(cells, td, restrict=False, stream=buf)
        chk("B2 uncontrolled, the same arm shows the dead cell", r2["fftm"][1] == 1, str(r2))
        chk("B2b and the uncontrolled mode warns about the confound",
            "NOT controlled" in buf.getvalue())
        chk("B3 ⭐ the control CHANGES the answer -- that is why it exists",
            r["fftm"][1] != r2["fftm"][1])
        open(os.path.join(td, "fftm-lr5e-2-s150.log"), "w").write(
            "'accuracy': 0.52708\n'accuracy': 0.75\n'accuracy': 0.76\n")
        d, t = epoch_stats("fftm-lr5e-2-s150", td)
        chk("B4 degenerate epochs are counted metric-correctly ([R.222])", (d, t) == (1, 3), f"{d}/{t}")
        chk("B5 a label with no log yields 0/0, not a crash", epoch_stats("nope", td) == (0, 0))
    print(f"\n  selftest: {len(ok)} passed, {len(bad)} failed")
    if bad: print("  FAILED: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--uncontrolled", action="store_true")
    a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    d = os.path.join(ROOT, "scratchpad", "phaseR", "r237")
    report(R.load(os.path.join(d, "csv")), os.path.join(d, "logs"), restrict=not a.uncontrolled)
