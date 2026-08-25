#!/usr/bin/env python
"""[R.260] THE TUNED-BASELINE TABLE — the deliverable of [R.237], emitted mechanically.

CONTEXT 5: "produce the tuned-baseline table."  The reader [R.237] ranks cells per arm;
the confirmation generator [R.237c] emits the 5-seed block.  Neither produces the TABLE,
and a table hand-assembled from a printed cell list is exactly how [R.254]'s fragility
counts went wrong ([R.256] 4).  So it is assembled here, from the same tested primitives.

EVERY column is mechanical and traceable:
  best / acc            r237_read.load()      -- the arm's argmax cell
  bracketed?            r237_read.edge_report()   (a) of the CONTEXT 2 standard
  worst adjacent        r237_read.fragility_report()  (b) of it, [R.256]
  centre gap            r237_read.centre_validity()   -- the [R.259] confound
  search dim            jobs.tsv                      -- plane cells and swept axes
  published point       PUBLISHED below, [R.258]      -- is the arm's OWN config in the box?

⛔ WHAT IT REFUSES
  * to emit a row for an arm that is not 100% complete (PROCESS.md 1.4)
  * to drop the SCREENING caption.  Every number here is n=1, seed 41.  The table is
    a table of CANDIDATES until the confirmation block runs.
  * to compare arms.  It prints per-arm rows and the qualifiers; the cross-arm verdict
    is the 5/5 gate's job, never this file's.

Usage:
  env/bin/python scripts/r260_table.py [--md]
  env/bin/python scripts/r260_table.py --selftest
"""
import argparse, importlib.util, io, os, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_spec = importlib.util.spec_from_file_location(
    "r237_read", os.path.join(ROOT, "scripts", "r237_read.py"))
R = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(R)

# [R.258] each arm's OWN published operating point, and whether the grid's ladders
# contain it.  A baseline tuned outside its published box is not "at its own optimum".
PUBLISHED = {
    "loca":  ("lr 5e-3, a=1.0  [N.1 0.1]",      True,  "ladder extended DOWN to 5e-4 for this"),
    "fftm":  ("a in {29..141}, lr 1e-2..1.2e-1", True,  "spanned by lr 5e-3..5e-1, s 50..300"),
    "lyra":  ("lr 2e-2, gamma=1, exp 3.0",       True,  "interior on both swept axes"),
    "scora": ("none -- this repo's method",      None,  "scale DERIVED a-priori, PROCESS 5 test 4"),
    "wave1": ("lambda=25, lr 1e-4  [N.1 0.1]",  False, "[R.258] lr 50x above bottom rung, atom 91x below top"),
    "wave2": ("lambda=25, lr 1e-4  [N.1 0.1]",  False, "[R.258] same; mu=2 is this repo's variant anyway"),
    "qwha":  ("no published RoBERTa value",      None,  "[R.236 3.5] derived ladder, correctly"),
}


# [R.277] flops/token per arm, [measured, src/bench_adapter_cost.py] at each arm's OWN
# floor, d=768, k=256, B=4096, UNMERGED regime [R.107] -- the same cost model and the same
# conventions as scratchpad/phaseR/frontier.py, so the two cannot drift apart.
# ⛔ wave1 and wave2 are DIFFERENT configurations and carry DIFFERENT costs: pairing mu=2's
# accuracy with mu=1's cost was a real defect, found 2026-08-20 [R.194 2].  Fixture C3 guards it.
COST_KEY = {"fftm": ("fft_fast_rfft", {}), "fftstock": ("fft_fast_rfft", {}),
            "wave1": ("waveft_factored", {"r": 1}), "wave2": ("waveft_factored", {"r": 2}),
            "qwha": ("qwha", {}), "loca": ("loca_factored", {}),
            "lyra": ("lyra_factored", {}), "scora": ("slr_factored", {})}


def flops_per_token(arm, d=768, k=256, B=4096):
    """fwd adapter flops/token for `arm`'s method at the shared budget, or None."""
    ent = COST_KEY.get(arm)
    if ent is None:
        return None
    import sys as _s
    _s.path.insert(0, os.path.join(ROOT, "src"))
    from bench_adapter_cost import theoretical_flops as T
    return T(ent[0], d, d, B, k, **ent[1])["fwd_adapter"] / B


def search_dim(grid_dir):
    """{arm: (plane_cells, ofat_cells, [axes])} from the frozen job list.  [R.259 2]"""
    import re, collections
    plane, ofat, axes = collections.Counter(), collections.Counter(), collections.defaultdict(set)
    p = os.path.join(grid_dir, "jobs.tsv")
    if not os.path.exists(p):
        return {}
    for line in open(p):
        lab = line.split("\t")[0].strip()
        if not lab:
            continue
        arm = R.arm_of(lab)
        if "-ofat-" in lab:
            ofat[arm] += 1
            continue
        plane[arm] += 1
        for k in R._knobs(lab):
            axes[arm].add(k)
    return {a: (plane[a], ofat[a], sorted(axes[a])) for a in set(plane) | set(ofat)}


# [R.264] the comparable statistic across arms is the RATIO observed_gain / null_curse,
# NOT the raw gain: a 9-cell search has a ~1.3x smaller null than a 25-cell one before any
# real effect exists.  sigma = [R.83]'s RTE paired sd 0.0186 / sqrt(2) = single-run sd.
# [R.275] sigma is now DIRECTLY measured, not derived: the median within-config sd across
# seeds over 34 healthy banked 5-seed RTE configs is 0.01397.  [R.264] used 0.0186/sqrt2 =
# 0.01315 (a 6% underestimate, which INFLATES the ratios).  Both are reported there; no
# verdict changes.  ⚠️ the MEAN within-config sd is 0.0231, inflated by one config at 0.0988
# -- the median is the right statistic and the earlier apparent conflict was mean-vs-median.
CURSE_SIGMA = 0.01397


def null_curse(k, sigma=CURSE_SIGMA, trials=40000, seed=12345):
    """E[max of k iid N(0,sigma) - one fixed member].  [R.264], same estimator."""
    import random
    if k < 2:
        return 0.0
    rnd = random.Random(seed)
    tot = 0.0
    for _ in range(trials):
        d = [rnd.gauss(0.0, sigma) for _ in range(k)]
        tot += max(d) - d[0]
    return tot / trials


def curse_line(ncells, gain):
    """The [R.264] ratio row for one arm, or '' when it cannot be formed."""
    if gain is None or ncells < 2:
        return ""
    nc = null_curse(ncells)
    if nc <= 0:
        return ""
    r = gain / nc
    v = ("⭐ EXCEEDS the curse -- real structure" if r >= 2 else
         "⚠️ COMPARABLE to selection noise -- do not quote without confirmation" if r >= 1 else
         "⛔ BELOW the curse -- this surface is flatter than noise")
    return f"null E[gain] {nc:+.4f} over {ncells} cells   ratio {r:.2f}x   {v}"


def _one_line(msgs, *keys):
    """First message containing any of `keys`, stripped; '' if none."""
    for m in msgs:
        if any(k in m for k in keys):
            return m.strip()
    return ""


def build(cells, totals, dims, stream=sys.stdout, md=False, shared_cost=None):
    p = lambda *a: print(*a, file=stream)
    p("=" * 78)
    p("[R.260] TUNED-BASELINE TABLE -- RTE / roberta-base / q+v / seed 41")
    p("=" * 78)
    p("⛔ SCREENING: every value is n=1 on a task with paired sd ~0.02-0.03 and metric")
    p("   quantised at 1/277 = 0.0036.  These are CANDIDATES.  A tuned baseline exists")
    p("   only after the 5-seed confirmation block (scripts/r237_confirm_gen.py).")
    p("⛔ Rows are NOT to be compared across arms here -- that is the 5/5 gate's job.")
    p("")
    rows = []
    for arm, title in R.ARMS.items():
        sub = {k: v for k, v in cells.items() if R.arm_of(k) == arm}
        exp = totals.get(arm)
        if not sub or not exp or len(sub) < exp:
            if exp:
                rows.append((arm, None, len(sub), exp))
            continue
        best, bv = max(sub.items(), key=lambda kv: kv[1]["acc"])
        edge = R.edge_report(sub, best, complete=True)
        frag = R.fragility_report(sub, best, bv["acc"])
        cenl = R.centre_validity(sub, R.CENTRE.get(arm, ""), bv["acc"])
        cen = sub.get(R.CENTRE.get(arm, ""), {}).get("acc")
        pub = PUBLISHED.get(arm, ("?", None, ""))
        d = dims.get(arm, (0, 0, []))
        p("-" * 78)
        p(f"{title}   [{arm}]   {len(sub)}/{exp} cells  *** COMPLETE ***")
        p(f"  best cell         {best}   acc {bv['acc']:.4f}")
        if cen is not None:
            p(f"  shared centre     {R.CENTRE[arm]}   acc {cen:.4f}   "
              f"delta {bv['acc'] - cen:+.4f} ({(bv['acc'] - cen) * 277:+.1f} eval ex)")
        # ⛔ An EMPTY edge list means UNDEFINED (the argmax is an OFAT cell and lies on
        # no ladder), NOT "not bracketed".  The first version read empty as negative and
        # printed "NOT bracketed" for SCoRA, whose PLANE argmax is in fact interior
        # ([R.213]/[R.254]) -- i.e. the bug understated OUR OWN arm.  [R.260 4]
        if not edge:
            plane = {k: v for k, v in sub.items() if R._knobs(k)}
            pbest = max(plane, key=lambda k: plane[k]["acc"]) if plane else None
            pedge = R.edge_report(plane, pbest, complete=True) if pbest else []
            p(f"  (a) bracketed?    n/a -- argmax is an OFAT cell, on no ladder")
            if pbest:
                p(f"       plane argmax {pbest}: "
                  + ("INTERIOR on every swept ladder" if any("INTERIOR" in m for m in pedge)
                     else "⚠️ NOT bracketed -- LOWER BOUND"))
                for m in pedge:
                    if "LOWER BOUND" in m:
                        p(f"       {m.strip()}")
        else:
            p(f"  (a) bracketed?    "
              + ("INTERIOR on every swept ladder" if any("INTERIOR" in m for m in edge)
                 else "⚠️ NOT bracketed -- LOWER BOUND"))
            for m in edge:
                if "LOWER BOUND" in m:
                    p(f"       {m.strip()}")
        w = _one_line(frag, "fragility:")
        v = _one_line(frag, "CLIFF-EDGE", "STEEP", "FLAT", "no fragility verdict")
        p(f"  (b) worst adj.    {w[len('fragility: '):] if w.startswith('fragility:') else (w or 'n/a')}")
        p(f"       verdict      {v or 'n/a'}")
        if any("OFAT cell" in m for m in frag):
            p("       ⚠️ argmax is an OFAT cell -- (b) is the PLANE argmax's, a different cell")
        p(f"  centre validity   "
          + ("⛔ " + cenl[0].strip().lstrip('⛔ ') if cenl else "✅ centre is at/near this arm's argmax"))
        cl = curse_line(len(sub), (bv["acc"] - cen) if cen is not None else None)
        if cl:
            p(f"  winner's curse    {cl}   [R.264]")
        fl = flops_per_token(arm)
        if fl is not None:
            ref = flops_per_token("fftm")
            p(f"  flops/token       {fl:,.0f}   ({fl/ref:.2f}x FourierFT)   [R.277], unmerged [R.107]")
        p(f"  search dimension  {d[0]} plane cells ({' x '.join(d[2]) or 'n/a'}) + {d[1]} OFAT   [R.259]")
        p(f"  published point   {pub[0]}")
        p(f"       in the box?  "
          + {True: "✅ yes", False: "⛔⛔ NO", None: "n/a"}[pub[1]] + f"   -- {pub[2]}")
        if cen is not None:
            plane_only = {k: v for k, v in sub.items() if R._knobs(k)}
            if plane_only:
                pb = max(plane_only.values(), key=lambda v: v["acc"])["acc"]
                (shared_cost if shared_cost is not None else {})[arm] = pb - cen
        rows.append((arm, best, bv["acc"], exp))
    # [R.279] the SHARED-PROTOCOL COST: what each arm loses by being tuned at the shared
    # centre instead of its own optimum.  [R.259] showed the centre sits at SCoRA's own
    # optimum, so this is the fairness number that concern needs.
    costs = {a: g for a, g in (shared_cost or {}).items()}
    if len(costs) >= 2:
        p("")
        p("=" * 78)
        p("SHARED-PROTOCOL COST -- accuracy lost by using the shared centre, per arm  [R.279]")
        p("=" * 78)
        base = {a: g for a, g in costs.items() if a != "scora"}
        for a, g in sorted(costs.items(), key=lambda kv: -kv[1]):
            p(f"  {a:8s} {g:+.4f}  ({g*277:5.1f} eval examples)"
              + ("   ⭐ OUR ARM -- the centre IS its optimum" if a == "scora" else ""))
        if base:
            med = sorted(base.values())[len(base) // 2]
            p(f"\n  baselines: {len(base)} arms, median {med:+.4f} ({med*277:.1f} eval examples),"
              f" range {min(base.values()):+.4f}..{max(base.values()):+.4f}")
            p(f"  SCoRA: {costs.get('scora', float('nan')):+.4f}")
            p("  ⇒ the shared protocol costs every BASELINE and costs OUR arm nothing [R.259].")
    p("")
    p("=" * 78)
    inc = [(a, n, e) for (a, b, n, e) in rows if b is None]
    if inc:
        p("NOT YET EMITTED (arm incomplete -- PROCESS.md 1.4):")
        for a, n, e in inc:
            p(f"  {a:9s} {n}/{e}")
    p(f"complete arms in this table: {sum(1 for r in rows if r[1] is not None)}")
    return rows


def selftest():
    ok, bad = [], []

    def chk(name, cond, detail=""):
        (ok if cond else bad).append(name)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))

    H = lambda a: {"acc": a, "collapsed": False, "near_floor": False}
    DEAD = {"acc": R.RTE_MAJORITY, "collapsed": True, "near_floor": False}
    cells = {"fftm-lr5e-3-s150": H(0.70), "fftm-lr5e-2-s150": H(0.72),
             "fftm-lr1.5e-1-s150": H(0.77), "fftm-lr1.5e-1-s100": DEAD,
             "loca-lr5e-2-a1.0": H(0.68)}
    dims = {"fftm": (4, 5, ["lr", "s"]), "loca": (25, 7, ["a", "lr"])}

    buf = io.StringIO()
    rows = build(cells, {"fftm": 4, "loca": 9}, dims, stream=buf)
    out = buf.getvalue()
    chk("T1 a COMPLETE arm is emitted", "fftm-lr1.5e-1-s150" in out)
    chk("T2 an INCOMPLETE arm is refused, and said so",
        "NOT YET EMITTED" in out and "loca" in out.split("NOT YET EMITTED")[1], out[-300:])
    chk("T2b and no loca row was printed", "  best cell         loca" not in out)
    chk("T3 the SCREENING caption is not droppable", "SCREENING" in out and "CANDIDATES" in out)
    chk("T4 fragility (b) is carried through", "CLIFF-EDGE" in out, out)
    chk("T5 bracketing (a) is carried through", "NOT bracketed" in out or "INTERIOR" in out)
    chk("T6 search dimension is printed", "plane cells" in out and "[R.259]" in out)
    chk("T7 published-point coverage is printed", "published point" in out)
    chk("T8 WaveFT is marked OUT of the box", PUBLISHED["wave1"][1] is False)
    chk("T8b and FourierFT/LoCA/LYRA are marked in it",
        all(PUBLISHED[a][1] is True for a in ("fftm", "loca", "lyra")))
    chk("T9 arms with no published value are n/a, not 'yes'",
        PUBLISHED["qwha"][1] is None and PUBLISHED["scora"][1] is None)
    chk("T10 build() returns one entry per known arm state", len(rows) == 2, str(rows))

    # T13-T16 [R.264] the curse ratio column
    chk("T13 k=1 forms no ratio (no selection happened)", curse_line(1, 0.05) == "")
    chk("T13b a missing centre forms no ratio", curse_line(25, None) == "")
    big = curse_line(25, 0.0650)
    chk("T14 a gain well above the null is EXCEEDS", "EXCEEDS" in big, big)
    chk("T15 a gain near the null is COMPARABLE", "COMPARABLE" in curse_line(25, 0.0300),
        curse_line(25, 0.0300))
    chk("T16 a gain below the null is BELOW", "BELOW" in curse_line(9, 0.0072), curse_line(9, 0.0072))
    chk("T16b the null grows with cell count [R.264]", null_curse(9) < null_curse(25))
    chk("T17 the ratio reaches the printed table", "winner's curse" in out, out[:200])

    # C1-C4 [R.277] the cost column
    f1, f2 = flops_per_token("wave1"), flops_per_token("wave2")
    chk("C1 every grid arm has a cost", all(flops_per_token(a) for a in COST_KEY))
    chk("C2 SCoRA is the cheapest arm", flops_per_token("scora") == min(
        flops_per_token(a) for a in COST_KEY), f"{flops_per_token('scora'):,.0f}")
    chk("C3 ⛔ wave1 and wave2 carry DIFFERENT costs ([R.194] defect guard)",
        f1 != f2 and f2 > f1, f"mu1 {f1:,.0f} vs mu2 {f2:,.0f}")
    chk("C3b and mu=2 costs exactly 2k more flops/token than mu=1 ([R.165])",
        abs((f2 - f1) - 2 * 256) < 1e-6, f"{f2-f1:.1f} vs {2*256}")
    chk("C4 an unknown arm yields no cost rather than a wrong one",
        flops_per_token("nosucharm") is None)
    chk("C5 the cost reaches the printed table", "flops/token" in out, out[:200])

    # P1-P3 [R.279] shared-protocol cost
    sc = {}
    cc = {"fftm-lr5e-2-s150": H(0.70), "fftm-lr1.5e-1-s150": H(0.77), "fftm-lr5e-3-s150": H(0.66),
          "scora-lr5e-2": H(0.74), "scora-lr5e-3": H(0.70), "scora-lr1.5e-1": H(0.69)}
    buf2 = io.StringIO()
    build(cc, {"fftm": 3, "scora": 3}, {"fftm": (3, 0, ["lr", "s"]), "scora": (3, 0, ["lr"])},
          stream=buf2, shared_cost=sc)
    chk("P1 the shared-protocol cost is computed per arm",
        abs(sc.get("fftm", 0) - 0.07) < 1e-9 and abs(sc.get("scora", 1)) < 1e-9, str(sc))
    chk("P2 an arm whose centre IS its argmax costs zero and is marked as ours",
        "OUR ARM" in buf2.getvalue(), buf2.getvalue()[-400:])
    chk("P3 the pooled line excludes our arm from the baseline median",
        "baselines: 1 arms" in buf2.getvalue(), buf2.getvalue()[-400:])

    # T11 an OFAT argmax must carry the [R.256] warning into the table
    c2 = {"scora-lr5e-3": H(0.69), "scora-lr5e-2": H(0.74), "scora-lr1.5e-1": DEAD,
          "scora-ofat-clf1e-2": H(0.75)}
    buf = io.StringIO()
    build(c2, {"scora": 4}, {"scora": (3, 1, ["lr"])}, stream=buf)
    chk("T11 an OFAT argmax is flagged as a different cell in (b)",
        "argmax is an OFAT cell" in buf.getvalue(), buf.getvalue())
    o2 = buf.getvalue()
    chk("T11b (a) for an OFAT argmax is n/a, NOT 'NOT bracketed'",
        "(a) bracketed?    n/a" in o2 and "(a) bracketed?    ⚠️" not in o2, o2)
    chk("T11c and the PLANE argmax's bracketing is reported instead",
        "plane argmax scora-lr5e-2: INTERIOR" in o2, o2)

    # T12 search_dim parses a real jobs.tsv if present
    d = search_dim(os.path.join(ROOT, "scratchpad", "phaseR", "r237"))
    if d:
        chk("T12 scora's plane is 1-D and smaller than every baseline's [R.259]",
            d["scora"][0] < min(d[a][0] for a in ("loca", "fftm", "lyra") if a in d), str(d))
    print("\n" + "=" * 60)
    print(f"  selftest: {len(ok)} passed, {len(bad)} failed")
    if bad:
        print("  FAILED: " + ", ".join(bad))
    print("=" * 60)
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dir", default=os.path.join(ROOT, "scratchpad", "phaseR", "r237"))
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    build(R.load(os.path.join(a.dir, "csv")), R.arm_totals(a.dir), search_dim(a.dir),
          shared_cost={})
