#!/usr/bin/env python
"""[R.273] Calibrate the NULL for a same-seed comparison of two configs, from banked 5-seed blocks.

[R.272] argued that [R.83]'s sigma (a CROSS-SEED paired sd) is the wrong null for the grid's
SAME-SEED iso-product pairs, because [R.257] shows the harness is bit-deterministic at fixed
(config, seed).  That argument conflates two different things:

  * REPRODUCIBILITY at fixed (config, seed): exact.  [R.257], |diff| = 0.
  * the SEED-IDIOSYNCRATIC DEVIATION eps_i(s) = x_i(s) - mean_s x_i(s): real, and exactly what
    sigma measures.  Two DIFFERENT configs at the same seed still have different eps.

The null for |x_1(41) - x_2(41)| under "the two configs are equivalent" is
    Var = 2 sigma^2 (1 - rho),   rho = corr(eps_1, eps_2) at a shared seed.
[R.272] assumed rho >> 0 without measuring it.  This measures both terms from 38 banked 5-seed
RTE configs, and needs no GPU.

Usage:  env/bin/python scripts/r273_null_calibration.py [--selftest]
"""
import argparse, collections, glob, itertools, math, os, re, statistics, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PH = os.path.join(ROOT, "scratchpad", "phaseR")
MAJ = 146 / 277
NEAR = 3.0 / 277


def load_blocks():
    """{(block, config): {seed: best-epoch accuracy}} over COMPLETE 30-epoch RTE logs."""
    B = collections.defaultdict(dict)
    for p in glob.glob(os.path.join(PH, "*", "*.log")):
        nm = os.path.basename(p)[:-4]
        m = re.search(r"s(4[1-5])$", nm)
        if not m:
            continue
        v = [float(x) for x in re.findall(r"'accuracy': ([0-9.eE+-]+)",
                                          open(p, errors="ignore").read())]
        if len(v) >= 30:
            B[(os.path.basename(os.path.dirname(p)), nm[:m.start()])][int(m.group(1))] = max(v)
    return {k: v for k, v in B.items() if len(v) == 5 and "mrpc" not in k[1]}


def devs(v):
    m = statistics.mean(v.values())
    return [v[s] - m for s in range(41, 46)]


def corr(a, b):
    na = math.sqrt(sum(t * t for t in a)); nb = math.sqrt(sum(t * t for t in b))
    return sum(p * q for p, q in zip(a, b)) / (na * nb) if na > 0 and nb > 0 else float("nan")


def report(full, stream=sys.stdout):
    p = lambda *a: print(*a, file=stream)
    byblk = collections.defaultdict(list)
    for blk, nm in full:
        byblk[blk].append(nm)
    p("=" * 78)
    p("[R.273] NULL CALIBRATION for a same-seed comparison of two configs")
    p("=" * 78)
    sds, rs = [], []
    for blk, nms in sorted(byblk.items()):
        if len(nms) != 2:
            continue
        a, b = sorted(nms)
        va, vb = full[(blk, a)], full[(blk, b)]
        if min(min(va.values()), min(vb.values())) <= MAJ + NEAR:
            p(f"  {blk:8s} ⛔ excluded -- a dead/near-floor seed ([R.222]/[R.255])")
            continue
        d = [va[s] - vb[s] for s in range(41, 46)]
        sd = statistics.stdev(d); r = corr(devs(va), devs(vb))
        sds.append(sd); rs.append(r)
        p(f"  {blk:8s} {a:14s} vs {b:14s}  paired sd {sd:.4f} ({sd*277:4.1f} ex)  rho {r:+.2f}")
    if not sds:
        return {}
    med = statistics.median(sds)
    # rho over ALL config pairs, not just same-block ones
    allr = [corr(devs(full[a]), devs(full[b])) for a, b in itertools.combinations(sorted(full), 2)]
    allr = [x for x in allr if not math.isnan(x)]
    mr = statistics.mean(allr)
    # a single pair has no sampling sd; report se=nan rather than crashing (fixture N4)
    se = statistics.stdev(allr) / math.sqrt(len(allr)) if len(allr) > 1 else float("nan")
    p("")
    p(f"  MEDIAN paired sd, {len(sds)} matched same-block pairs : {med:.4f}  ({med*277:.1f} eval ex)")
    p(f"  [R.83]'s banked RTE paired sd                        : 0.0186  (5.2 eval ex)")
    p(f"  mean rho over {len(allr)} config pairs                    : {mr:+.4f}  (se {se:.4f})")
    p("")
    if abs(mr) < 0.1 or (se == se and abs(mr) < 3 * se):
        p("  ⇒ ⭐ rho is INDISTINGUISHABLE FROM ZERO: two different configs at a shared seed have")
        p("     essentially independent seed-deviations.  ⇒ the CROSS-SEED paired sd IS the correct")
        p("     null for a same-seed comparison, and [R.272]'s objection does not hold. [R.273 3]")
    else:
        p(f"  ⇒ ⛔ rho = {mr:+.3f} is material: the null must be scaled by sqrt(1-rho) = "
          f"{math.sqrt(max(0,1-mr)):.3f}")
    return {"median_sd": med, "rho": mr, "rho_se": se, "n_pairs": len(sds)}


def selftest():
    ok, bad = [], []
    def chk(n, c, d=""):
        (ok if c else bad).append(n); print(f"  {'PASS' if c else 'FAIL'}  {n}" + (f"  {d}" if d else ""))
    import io
    chk("N1 devs sum to zero", abs(sum(devs({41: .7, 42: .8, 43: .6, 44: .75, 45: .65}))) < 1e-12)
    chk("N2 corr of a series with itself is 1", abs(corr([1, -1, 0, 2, -2], [1, -1, 0, 2, -2]) - 1) < 1e-12)
    chk("N3 corr of a series with its negation is -1",
        abs(corr([1, -1, 0, 2, -2], [-1, 1, 0, -2, 2]) + 1) < 1e-12)
    f = {("b1", "x-"): {41: .70, 42: .72, 43: .68, 44: .74, 45: .71},
         ("b1", "y-"): {41: .60, 42: .62, 43: .58, 44: .64, 45: .61}}
    buf = io.StringIO(); r = report(f, buf)
    chk("N4 a perfectly parallel pair has paired sd 0", abs(r["median_sd"]) < 1e-12, str(r))
    chk("N4b and rho = +1", abs(r["rho"] - 1.0) < 1e-9, str(r))
    chk("N4c which is reported as MATERIAL, not waved through", "is material" in buf.getvalue())
    f2 = {("b1", "x-"): {41: .70, 42: .72, 43: .68, 44: .74, 45: .71},
          ("b1", "y-"): {41: MAJ, 42: .62, 43: .58, 44: .64, 45: .61}}
    buf = io.StringIO(); report(f2, buf)
    chk("N5 a pair containing a DEAD seed is excluded", "excluded" in buf.getvalue())
    print(f"\n  selftest: {len(ok)} passed, {len(bad)} failed")
    if bad: print("  FAILED: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else (report(load_blocks()) and 0))
