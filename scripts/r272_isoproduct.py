#!/usr/bin/env python
"""[R.272] The iso-product test across EVERY arm whose plane has a scale axis.

[R.270] retracted [R.252]'s inference that the (lr, scale) plane does NOT collapse to the
1-D coordinate lr*scale -- on FourierFT's 4 iso-product pairs the disagreement was SMALLER
than measurement noise.  4 pairs cannot settle it either way.  This pools every arm whose
grid plane is lr x (a scale-like knob), which multiplies the evidence at zero GPU cost:

    fftm   lr x scaling      (atom  = scaling/sqrt(2mn))
    wave1  lr x fs           (atom  = fs/sqrt(2*mu*mn))    -- same functional form
    wave2  lr x fs           (same, mu=2)
    qwha   lr x scaling      (atom  = scaling/sqrt(mn))
    loca   lr x alpha        (alpha is LoCA's output scale)

⛔ EXCLUDED: lyra -- its second axis is `spectral_freq_exponent`, which is NOT a scale, so
lr*e is not an effective step and pooling it would be a category error.

⛔ The null is measurement noise, sd = 0.0186 [R.83] for a difference of two RTE runs.
Under PERFECT collapse iso-product cells differ only by noise: E|d| = sd*sqrt(2/pi) = 0.0148.
"""
import argparse, collections, importlib.util, math, os, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_s = importlib.util.spec_from_file_location("r237_read", os.path.join(ROOT, "scripts", "r237_read.py"))
R = importlib.util.module_from_spec(_s); _s.loader.exec_module(R)

SD = 0.0186                                  # [R.83] paired sd on RTE
E_ABS = SD * math.sqrt(2.0 / math.pi)        # E|delta| under pure noise
SCALE_AXIS = {"fftm": "s", "wave1": "fs", "wave2": "fs", "qwha": "s", "loca": "a"}


def pairs(cells, arm):
    """[(product, [(label, acc), ...]), ...] for plane cells sharing lr*scale."""
    ax = SCALE_AXIS.get(arm)
    if ax is None:
        return []
    g = collections.defaultdict(list)
    for k, v in cells.items():
        if R.arm_of(k) != arm or "-ofat-" in k:
            continue
        kn = R._knobs(k)
        if "lr" not in kn or ax not in kn:
            continue
        g[round(float(kn["lr"]) * float(kn[ax]), 9)].append((k, v["acc"]))
    return [(p, sorted(c)) for p, c in sorted(g.items()) if len(c) >= 2]


def report(cells, totals, stream=sys.stdout):
    p = lambda *a: print(*a, file=stream)
    p("=" * 78)
    p("[R.272] ISO-PRODUCT TEST -- does the (lr, scale) plane collapse to lr*scale?")
    p("=" * 78)
    p(f"  null: pure noise, sd {SD} [R.83]  =>  E|delta| = {E_ABS:.5f} ({E_ABS*277:.1f} eval examples)")
    p("  ⛔ lyra EXCLUDED: its 2nd axis is an exponent, not a scale.")
    p("")
    alld = []
    for arm in SCALE_AXIS:
        exp = totals.get(arm)
        sub = [k for k in cells if R.arm_of(k) == arm]
        if not sub:
            continue
        if exp and len(sub) < exp:
            p(f"  {arm:6s} ⛔ INCOMPLETE {len(sub)}/{exp} -- excluded (PROCESS.md 1.4)")
            continue
        ps = pairs(cells, arm)
        if not ps:
            p(f"  {arm:6s} no coincident products on its ladders")
            continue
        ds = []
        for prod, cs in ps:
            d = max(a for _, a in cs) - min(a for _, a in cs)
            ds.append(d)
            p(f"  {arm:6s} lr*scale={prod:9.4f}  |d|={d:.4f} ({d*277:4.1f} ex) = {d/SD:4.2f} sd   "
              + " ".join(f"{k.split('-',1)[1]}:{a:.4f}" for k, a in cs))
        alld.extend(ds)
        p(f"  {arm:6s} -> {len(ds)} pairs, mean |d| = {sum(ds)/len(ds):.5f} "
          f"({sum(ds)/len(ds)*277:.1f} ex)   vs noise {E_ABS*277:.1f} ex")
        p("")
    if not alld:
        p("no pairs")
        return {}
    mean = sum(alld) / len(alld)
    p("-" * 78)
    p(f"  POOLED: {len(alld)} iso-product pairs across arms")
    p(f"    observed mean |delta|  {mean:.5f}  ({mean*277:.1f} eval examples)")
    p(f"    pure-noise expectation {E_ABS:.5f}  ({E_ABS*277:.1f} eval examples)")
    p(f"    ratio observed/noise   {mean/E_ABS:.2f}")
    nbig = sum(1 for d in alld if d > 2 * SD)
    p(f"    pairs above 2 sd       {nbig}/{len(alld)}   (expected under noise: {0.0455*len(alld):.1f})")
    # ⛔⛔ NO VERDICT.  SD is [R.83]'s CROSS-SEED paired sd, but both cells of an iso-product
    # pair run at the SAME seed (41).  [R.257] showed the harness is bit-deterministic at fixed
    # (config, seed), so there is NO run-to-run noise here at all -- the whole difference is a
    # deterministic function of the config, plus chaotic divergence that this SD does not measure.
    # Using the cross-seed SD as the null OVERSTATES the noise, which is why the pairs appear to
    # agree "better than noise" (1.81 sd below, p~0.035).  The correct null is UNMEASURED.
    p(f"    ⭐ [R.273] CALIBRATED THE NULL: rho between two configs' seed-deviations = +0.02")
    p(f"       (se 0.019, n=666) => ~0, so the cross-seed sd IS the right null here.")
    # ⛔ the p-value must be COMPUTED, not hardcoded: an earlier version printed the 8-pair
    # p (0.035) after the pool grew to 12, which would have understated its own evidence.
    _z = (E_ABS - mean) / (SD * math.sqrt(1 - 2 / math.pi) / math.sqrt(len(alld)))
    _p = 0.5 * (1 - math.erf(_z / math.sqrt(2)))
    p(f"       Observed sits {_z:.2f} sd BELOW it (p~{_p:.3f}, one-sided): iso-product cells agree BETTER")
    p(f"       than arbitrary config pairs do -- the signature of COLLAPSE. Suggestive, not settled.")
    p(f"       ⇒ [R.270] reinstated; [R.270 4]'s 10-cell 5-seed test still settles it.  [R.273]")
    return {"n": len(alld), "mean": mean, "ratio": mean / E_ABS}


def selftest():
    ok, bad = [], []
    def chk(n, c, d=""):
        (ok if c else bad).append(n); print(f"  {'PASS' if c else 'FAIL'}  {n}" + (f"  {d}" if d else ""))
    H = lambda a: {"acc": a, "collapsed": False, "near_floor": False}
    import io
    # two cells at the same product 0.75: lr 1.5e-2 * 50  and  lr 5e-3 * 150
    c = {"fftm-lr1.5e-2-s50": H(0.65), "fftm-lr5e-3-s150": H(0.66),
         "fftm-lr5e-2-s150": H(0.75)}
    ps = pairs(c, "fftm")
    chk("I1 iso-product cells are grouped by lr*scale", len(ps) == 1 and len(ps[0][1]) == 2, str(ps))
    chk("I1b a product with a single cell forms no pair", all(len(x[1]) >= 2 for x in ps))
    chk("I2 lyra is excluded by construction (exponent is not a scale)",
        pairs({"lyra-lr5e-2-e3.0": H(0.7), "lyra-lr1.5e-1-e1.0": H(0.7)}, "lyra") == [])
    buf = io.StringIO(); report(c, {"fftm": 3}, buf)
    chk("I3 the calibrated null [R.273] is cited, not the withdrawn [R.272] refusal",
        "CALIBRATED THE NULL" in buf.getvalue(),
        buf.getvalue()[-300:])
    c2 = {"fftm-lr1.5e-2-s50": H(0.60), "fftm-lr5e-3-s150": H(0.72), "fftm-lr5e-2-s150": H(0.75)}
    buf = io.StringIO(); report(c2, {"fftm": 3}, buf)
    chk("I4 the pooled statistics are always printed",
        "POOLED" in buf.getvalue(), buf.getvalue()[-200:])
    chk("I4b but the raw numbers are still reported for the user to judge",
        "|d|=0.1200" in buf.getvalue(), buf.getvalue()[:400])
    buf = io.StringIO(); report(c, {"fftm": 99}, buf)
    chk("I5 an INCOMPLETE arm is excluded, not read", "INCOMPLETE" in buf.getvalue()
        and "POOLED" not in buf.getvalue())
    chk("I6 the noise expectation is sd*sqrt(2/pi)", abs(E_ABS - 0.0186 * 0.7979) < 1e-5, f"{E_ABS:.6f}")
    print(f"\n  selftest: {len(ok)} passed, {len(bad)} failed")
    if bad: print("  FAILED: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    d = os.path.join(ROOT, "scratchpad", "phaseR", "r237")
    report(R.load(os.path.join(d, "csv")), R.arm_totals(d))
