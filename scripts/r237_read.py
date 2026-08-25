#!/usr/bin/env python
"""[R.237] READER -- baseline hyperparameter grid, RTE / roberta-base.

PROCESS.md 6: "Test the reader before the spend, not after."  Run
`env/bin/python scripts/r237_read.py --selftest` and require 100% before
believing any verdict this prints.

WHAT IT DOES
  * concatenates the per-cell CSVs under scratchpad/phaseR/r237/csv/
  * applies [R.222]'s CORRECTED collapse detector.  RTE is an ACCURACY task, so
    collapse is predicting the MAJORITY CLASS (146/277 = 0.52708), NOT metric==0.
    Every RTE reader in this repo tested `metric == 0` and therefore NEVER fired.
  * per arm: reports the best cell, the centre cell, and the per-knob deltas
  * emits the 5-seed CONFIRMATION commands for each arm's winner

WHAT IT REFUSES TO DO
  * call any single-seed cell a result.  n=1, RTE sd ~0.02-0.03, metric quantised
    at 1/277 = 0.0036.  Everything here is a CANDIDATE for confirmation.
  * compare across arms.  This file tunes each arm against ITSELF.  The
    cross-arm comparison happens only after the confirmation block, under the
    5/5 gate, and never from these numbers.
"""
import argparse, glob, os, sys, math, tempfile, csv

RTE_MAJORITY = 146 / 277        # 0.527076..., [R.222]
COLLAPSE_TOL = 1e-4
NEAR_FLOOR_EX = 3      # [R.255] within this many eval examples of the floor = effectively dead
NEAR_FLOOR_EPS = 1e-9  # [R.261] float64: (MAJ + 3/277) - MAJ EXCEEDS 3/277, so a run exactly at the
                       # documented band edge was scored healthy.  The band is inclusive; make it so.
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                 "scratchpad", "phaseR", "r237")

ARMS = {
    "loca":     "LoCA",
    "fftm":     "FourierFT (merged == stock == fast)",
    "scora":    "SCoRA (ours)",
    "lyra":     "LYRA",
    "wave1":    "WaveFT mu=1 (as published)",
    "wave2":    "WaveFT mu=2 (repo rank fix)",
    "qwha":     "QWHA",
    "fftstock": "FourierFT stock (parity spot-check)",
}
# the shared centre cell per arm -- the one every OFAT delta is one knob from
CENTRE = {
    "loca":  "loca-lr5e-2-a1.0",
    "fftm":  "fftm-lr5e-2-s150",
    "scora": "scora-lr5e-2",
    "lyra":  "lyra-lr5e-2-e3.0",
    "wave1": "wave1-lr5e-2-fs150",
    "wave2": "wave2-lr5e-2-fs150",
    "qwha":  "qwha-lr5e-2-s106.0660",
}


def argmax_cell(sub):
    """Deterministic argmax over {label: {'acc':...}}: highest acc, ties broken by the
    LEXICOGRAPHICALLY SMALLEST label.  [R.289]

    ⛔ WHY THIS EXISTS.  `max(d, key=...)` returns the FIRST maximal key in iteration order,
    and this reader builds its dict from a sorted glob while r237_confirm_gen.py builds its
    from a different source.  On an exact tie the two disagreed: scora-ofat-clf1e-2 and
    scora-ofat-cosine both score 0.7509025270758123, and the DELIVERABLE named one while the
    CONFIRMATION BLOCK would have run the other.  Ties are not rare here -- RTE accuracy is
    quantised at 1/277.
    """
    return min((k for k in sub if sub[k]["acc"] == max(v["acc"] for v in sub.values())))


def tied_with_argmax(sub):
    """Every label sharing the argmax's accuracy (including it).  len>1 => an arbitrary pick."""
    if not sub:
        return []
    top = max(v["acc"] for v in sub.values())
    return sorted(k for k in sub if sub[k]["acc"] == top)


def arm_of(label):
    return label.split("-")[0]


def load(csv_dir):
    """Return {label: {'acc': float, 'collapsed': bool}} from per-cell CSVs."""
    out = {}
    for p in sorted(glob.glob(os.path.join(csv_dir, "*.csv"))):
        label = os.path.basename(p)[:-4]
        try:
            with open(p) as f:
                rows = list(csv.DictReader(f))
        except Exception:
            continue
        if not rows:
            continue
        r = rows[-1]
        try:
            acc = float(r.get("accuracy", "nan"))
        except (TypeError, ValueError):
            continue
        if math.isnan(acc):
            continue
        # [R.255] TWO failure modes, not one.
        #  * "collapsed": best epoch IS the majority class -- [R.222]'s corrected rule.
        #  * "near-floor": best epoch is within NEAR_FLOOR_EX eval examples of it.
        # ⛔ THE GAP THIS CLOSES: [R.222]'s detector tests the BEST epoch for EQUALITY
        # with the floor.  A run dead for 16/30 epochs that scrapes ONE example above it
        # (0.5307 vs 0.5271) passed as healthy.  Measured instance: fftm-lr5e-1-s150.
        # On RTE, 1-3 examples above the majority class is not learning, it is noise.
        d = acc - RTE_MAJORITY
        out[label] = {"acc": acc,
                      "collapsed": abs(d) < COLLAPSE_TOL,
                      "near_floor": COLLAPSE_TOL <= d <= NEAR_FLOOR_EX / 277.0 + NEAR_FLOOR_EPS}
    return out


def _knobs(label):
    """Split a grid label into its swept (key, value) pairs.

    'loca-lr5e-3-a2.0' -> {'lr': '5e-3', 'a': '2.0'}
    'qwha-lr5e-2-s106.0660' -> {'lr': '5e-2', 's': '106.0660'}
    OFAT labels ('fftm-ofat-wd0') carry no numeric ladder and return {}.
    """
    import re as _re
    parts = label.split("-")[1:]
    if parts and parts[0] == "ofat":
        return {}
    out, i = {}, 0
    # rejoin fragments split by '-' inside a number, e.g. lr '5e-3'
    joined, buf = [], ""
    for p_ in parts:
        if buf:
            joined.append(buf + "-" + p_); buf = ""
        elif _re.fullmatch(r"[a-z]+[0-9.]*e", p_):
            buf = p_
        else:
            joined.append(p_)
    if buf:
        joined.append(buf)
    for p_ in joined:
        m = _re.fullmatch(r"([a-z]+)([0-9][0-9.e+-]*)", p_)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def edge_report(sub, best_label, complete):
    """Is the argmax at an extreme of any swept ladder? [R.244 8]

    A boundary argmax reports 'best within the swept range', NOT an optimum --
    PROCESS.md 5 test 5 requires the baseline at its OWN optimum.
    """
    bk = _knobs(best_label)
    if not bk:
        return []
    ladders = {}
    for lab in sub:
        for k, v in _knobs(lab).items():
            ladders.setdefault(k, set()).add(v)
    msgs = []
    for k, v in bk.items():
        vals = sorted(ladders.get(k, set()), key=lambda x: float(x))
        if len(vals) < 2:
            continue
        if v == vals[-1]:
            msgs.append(f"   {'⚠️' if complete else '(provisional)'} argmax is at the TOP of the "
                        f"'{k}' ladder {vals} -> NOT bracketed above; report as a LOWER BOUND")
        elif v == vals[0]:
            msgs.append(f"   {'⚠️' if complete else '(provisional)'} argmax is at the BOTTOM of the "
                        f"'{k}' ladder {vals} -> NOT bracketed below; report as a LOWER BOUND")
    if not msgs and complete:
        msgs.append(f"   ✅ argmax is INTERIOR on every swept ladder -> a genuine bracketed optimum")
    return msgs


GATE_MIN_EFFECT = 0.021   # smallest effect the 5/5 gate certifies (PROCESS 1.3)


def _ladders(sub):
    """{knob: sorted list of values} over an arm's plane cells."""
    lad = {}
    for lab in sub:
        for k, v in _knobs(lab).items():
            lad.setdefault(k, set()).add(v)
    return {k: sorted(v, key=lambda x: float(x)) for k, v in lad.items()}


def neighbours(sub, label):
    """Cells one rung away from `label` on exactly one swept ladder.  [R.256]

    Returns [(neighbour_label, knob, value)].  Empty for an OFAT label -- an OFAT
    cell is not on any ladder and therefore HAS no neighbours.
    """
    bk = _knobs(label)
    if not bk:
        return []
    lad = _ladders(sub)
    out = []
    for other in sub:
        if other == label:
            continue
        ok = _knobs(other)
        if not ok or set(ok) != set(bk):
            continue
        diff = [k for k in bk if ok[k] != bk[k]]
        if len(diff) != 1:
            continue
        k = diff[0]
        vals = lad.get(k, [])
        try:
            i, j = vals.index(bk[k]), vals.index(ok[k])
        except ValueError:
            continue
        if abs(i - j) == 1:
            out.append((other, k, ok[k]))
    return out


def fragility_report(sub, best_label, best_acc):
    """(b) of the CONTEXT 2 reporting standard: the worst adjacent cell.  [R.256]

    (a) 'is the argmax bracketed' is edge_report().  (b) had NO instrument at all
    until this unit -- [R.254]'s fragility verdicts were hand-read off the printed
    list.  Rule frozen in llmdocs/R256_fragility_instrument_prereg.md 2.
    """
    msgs = []
    lab, acc, prefix = best_label, best_acc, ""
    if not _knobs(best_label):
        plane = {k: v for k, v in sub.items() if _knobs(k)}
        msgs.append("   ⚠️ argmax is an OFAT cell -- it lies on no ladder, so its adjacent-cell "
                    "fragility is UNDEFINED")
        if not plane:
            return msgs
        lab, v = max(plane.items(), key=lambda kv: kv[1]["acc"])
        acc, prefix = v["acc"], "plane-argmax "
        msgs.append(f"   -> reporting the PLANE argmax instead: {lab} ({acc:.4f}) "
                    f"-- NOT the same cell as the arm's best")
    nb = neighbours(sub, lab)
    if len(nb) < 2:
        msgs.append(f"   ⚠️ only {len(nb)} adjacent cell(s) complete for {lab} -- "
                    f"no fragility verdict")
        return msgs
    dead = [(o, k, v) for (o, k, v) in nb
            if sub[o]["collapsed"] or sub[o].get("near_floor")]
    worst = min(nb, key=lambda t: sub[t[0]]["acc"])
    wacc = sub[worst[0]]["acc"]
    drop = acc - wacc
    if dead:
        verdict = (f"⛔ CLIFF-EDGE -- {len(dead)}/{len(nb)} adjacent cell(s) DEAD "
                   f"({', '.join(o for o, _, _ in dead)})")
    elif drop > GATE_MIN_EFFECT:
        verdict = (f"⚠️ STEEP -- worst adjacent drop {drop:.4f} exceeds the {GATE_MIN_EFFECT} "
                   f"the 5/5 gate can certify")
    else:
        verdict = f"✅ FLAT -- worst adjacent drop {drop:.4f} is within gate resolution"
    msgs.append(f"   {prefix}fragility: {len(nb)} adjacent cell(s); worst = {worst[0]} "
                f"{wacc:.4f} ({-drop:+.4f}, {-drop * 277:+.1f} eval examples)")
    msgs.append(f"   {verdict}")
    return msgs


def arm_totals(grid_dir):
    """{arm: expected cell count} from the frozen job list.  [R.251 4.2]

    Per-ARM completeness, not global.  The first version compared each arm's
    cell count to the GRID total, so a COMPLETED arm was still treated as
    partial and its validity flags (edge / centre) never fired until all 152
    cells were in -- i.e. exactly when they would be useless.
    """
    p = os.path.join(grid_dir, "jobs.tsv")
    out = {}
    if not os.path.exists(p):
        return out
    for line in open(p):
        lab = line.split("\t")[0].strip()
        if lab:
            out[arm_of(lab)] = out.get(arm_of(lab), 0) + 1
    return out


def centre_validity(sub, centre_label, best_acc, tol=0.02):
    """Is this arm's OFAT centre anywhere near its own optimum?  [R.251 4.1]

    Every OFAT cell is a ONE-KNOB DELTA FROM THE CENTRE.  If the centre sits far
    below the arm's plane argmax -- or is collapsing -- those deltas measure
    'does this knob rescue a bad config', NOT 'does this knob help at the
    optimum'.  Measured instance: LoCA's centre (lr 5e-2, a=1.0) scores 0.6859
    with 3/30 dead epochs against a plane argmax of 0.7726.
    """
    cen = sub.get(centre_label)
    if cen is None:
        return []
    gap = best_acc - cen["acc"]
    msgs = []
    if cen["collapsed"]:
        msgs.append("   ⛔ OFAT CENTRE IS COLLAPSED (majority class) -- every OFAT delta for this "
                    "arm is measured from a dead run and is UNINTERPRETABLE as tuning evidence")
    elif gap > tol:
        msgs.append(f"   ⛔ OFAT centre is {gap:+.4f} BELOW this arm's plane argmax "
                    f"({cen['acc']:.4f} vs {best_acc:.4f}) -> its OFAT deltas measure "
                    f"'does this knob rescue a bad config', NOT 'does it help at the optimum'")
    return msgs


def report(cells, jobs_total=None, stream=sys.stdout, totals=None):
    p = lambda *a: print(*a, file=stream)
    p("=" * 78)
    p("[R.237] BASELINE GRID -- RTE / roberta-base / q+v / seed 41 / SCREENING ONLY")
    p("=" * 78)
    n = len(cells)
    if jobs_total:
        p(f"cells complete: {n}/{jobs_total}"
          + ("   *** GRID PARTIAL -- per-ARM completeness is marked on each arm "
             "below (PROCESS.md 1.4) ***" if n < jobs_total else ""))
    else:
        p(f"cells complete: {n}")
    ncol = sum(1 for v in cells.values() if v["collapsed"])
    nnf = sum(1 for v in cells.values() if v.get("near_floor"))
    p(f"collapsed cells (acc == RTE majority {RTE_MAJORITY:.5f}): {ncol}"
      + ("   <- KEPT, not dropped ([R.222])" if ncol else ""))
    if nnf:
        p(f"near-floor cells (within {NEAR_FLOOR_EX} eval examples of the floor): {nnf}"
          f"   <- effectively dead, [R.255]; KEPT")
    p("")

    winners = {}
    for arm, title in ARMS.items():
        sub = {k: v for k, v in cells.items() if arm_of(k) == arm}
        if not sub:
            continue
        p("-" * 78)
        p(f"{title}   [{arm}]   {len(sub)} cells"
          + (f" / {exp0} expected" if (exp0 := (totals or {}).get(arm)) else "")
          + ("   *** COMPLETE ***" if exp0 and len(sub) >= exp0 else ""))
        best = (argmax_cell(sub), sub[argmax_cell(sub)])
        exp = (totals or {}).get(arm)
        partial_arm = exp is not None and len(sub) < exp
        winners[arm] = best
        cen = CENTRE.get(arm)
        cen_acc = sub.get(cen, {}).get("acc") if cen else None
        for k, v in sorted(sub.items(), key=lambda kv: -kv[1]["acc"]):
            flag = ""
            if k == best[0]:
                flag += "  <-- best"
            if cen and k == cen:
                flag += "  (centre)"
            if v["collapsed"]:
                flag += "  ** COLLAPSED (majority class) **"
            elif v.get("near_floor"):
                flag += (f"  ** NEAR-FLOOR ({(v['acc']-RTE_MAJORITY)*277:.0f} example(s) above "
                         f"majority) -- effectively dead **")
            p(f"   {v['acc']:.4f}  {k}{flag}")
        _tied = tied_with_argmax(sub)
        if len(_tied) > 1:
            p(f"   ⚠️ {len(_tied)} cells TIE at the argmax ({', '.join(_tied)}) -- the winner is an "
              f"ARBITRARY pick among them [R.289]")
        if cen:
            for m in centre_validity(sub, cen, best[1]["acc"]):
                p(m)
        for m in edge_report(sub, best[0], complete=(jobs_total is None or len(sub) > 0 and not partial_arm)):
            p(m)
        for m in fragility_report(sub, best[0], best[1]["acc"]):
            p(m)
        if cen_acc is not None:
            d = best[1]["acc"] - cen_acc
            p(f"   centre {cen_acc:.4f} -> best {best[1]['acc']:.4f}   "
              f"delta = {d:+.4f}  ({d * 277:+.1f} eval examples)")
            if abs(d) < 1 / 277:
                p("   ⚠️ delta is BELOW one eval example -- not a real difference")
        else:
            p("   ⚠️ centre cell not complete: no delta reportable")
        p("")

    p("=" * 78)
    p("CONFIRMATION BLOCK -- required before ANY of the above enters a table")
    p("=" * 78)
    p("Each winner must be re-run at 5 seeds against that arm's own centre.")
    p("A 1-seed argmax carries winner's curse; it inflates every BASELINE, i.e.")
    p("it runs against our own arm.  That is the conservative direction, and it")
    p("is why the screen is acceptable -- it is NOT a licence to quote these.")
    for arm, (label, v) in sorted(winners.items()):
        if arm == "fftstock":
            continue
        p(f"  {arm:9s} winner={label:28s} acc={v['acc']:.4f}"
          f"   vs centre {CENTRE.get(arm, '?')}")
    return winners


# ---------------------------------------------------------------------------
def selftest():
    ok, bad = [], []

    def chk(name, cond, detail=""):
        (ok if cond else bad).append(name)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))

    with tempfile.TemporaryDirectory() as td:
        def cell(label, acc):
            with open(os.path.join(td, label + ".csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["name", "accuracy", "seed"])
                w.writeheader()
                w.writerow({"name": label, "accuracy": acc, "seed": 41})

        # S1 collapse detection on an ACCURACY task -- the [R.222] defect
        cell("fftm-lr5e-2-s150", 0.7509)
        cell("fftm-lr5e-2-s300", RTE_MAJORITY)      # dead run, metric != 0
        cell("fftm-lr5e-3-s150", 0.7112)
        c = load(td)
        chk("S1 majority-class collapse detected (metric != 0)",
            c["fftm-lr5e-2-s300"]["collapsed"] is True)
        chk("S2 a healthy cell is NOT flagged",
            c["fftm-lr5e-2-s150"]["collapsed"] is False)
        chk("S3 collapsed cell is KEPT in the pool, not dropped",
            len(c) == 3)

        # S4 argmax picks the best, not the first or last
        import io
        buf = io.StringIO()
        w = report(c, jobs_total=3, stream=buf)
        chk("S4 winner is the argmax", w["fftm"][0] == "fftm-lr5e-2-s150")

        # S5 a collapsed cell can never win when a healthy one exists
        cell("wave1-lr5e-2-fs150", RTE_MAJORITY)
        cell("wave1-lr5e-3-fs150", 0.6101)
        c = load(td)
        buf = io.StringIO()
        w = report(c, jobs_total=5, stream=buf)
        chk("S5 collapsed cell does not win", w["wave1"][0] == "wave1-lr5e-3-fs150")

        # S6 PARTIAL is announced when cells are missing
        buf = io.StringIO()
        report(c, jobs_total=99, stream=buf)
        chk("S6 partial state announced", "PARTIAL" in buf.getvalue())

        # S7 sub-one-example deltas are called out
        cell("scora-lr5e-2", 0.7256)
        cell("scora-lr1.5e-2", 0.7256 + 0.002)      # < 1/277
        c = load(td)
        buf = io.StringIO()
        report(c, jobs_total=7, stream=buf)
        chk("S7 sub-quantum delta flagged", "BELOW one eval example" in buf.getvalue())

        # S10-S13 [R.244 8] ladder-edge detection -- the new code path.  Without
        # these the function is untested, which is the [R.162] defect exactly.
        chk("S10 label parses into knobs",
            _knobs("loca-lr5e-3-a2.0") == {"lr": "5e-3", "a": "2.0"}, str(_knobs("loca-lr5e-3-a2.0")))
        chk("S10b OFAT labels carry no ladder", _knobs("fftm-ofat-wd0") == {})
        chk("S10c qwha's decimal scale parses",
            _knobs("qwha-lr5e-2-s106.0660") == {"lr": "5e-2", "s": "106.0660"},
            str(_knobs("qwha-lr5e-2-s106.0660")))
        ladder = ["loca-lr5e-3-a0.5", "loca-lr5e-3-a1.0", "loca-lr5e-3-a2.0", "loca-lr5e-3-a4.0"]
        msgs = edge_report(ladder, "loca-lr5e-3-a4.0", complete=True)
        chk("S11 TOP-of-ladder argmax is flagged", any("TOP of the" in m for m in msgs), str(msgs))
        msgs = edge_report(ladder, "loca-lr5e-3-a0.5", complete=True)
        chk("S12 BOTTOM-of-ladder argmax is flagged", any("BOTTOM of the" in m for m in msgs), str(msgs))
        msgs = edge_report(ladder, "loca-lr5e-3-a2.0", complete=True)
        chk("S13 interior argmax is certified, not flagged",
            any("INTERIOR" in m for m in msgs) and not any("LOWER BOUND" in m for m in msgs), str(msgs))
        msgs = edge_report(ladder, "loca-lr5e-3-a4.0", complete=False)
        chk("S13b incomplete arm is marked provisional",
            any("(provisional)" in m for m in msgs), str(msgs))

        # S14-S16 [R.251 4.1] OFAT-centre validity -- the new code path
        good = {"x-lr5e-2-a1.0": {"acc": 0.77, "collapsed": False},
                "x-lr1.5e-2-a2.0": {"acc": 0.7726, "collapsed": False}}
        chk("S14 a centre near the argmax is NOT flagged",
            centre_validity(good, "x-lr5e-2-a1.0", 0.7726) == [], str(centre_validity(good, "x-lr5e-2-a1.0", 0.7726)))
        bad_c = {"x-lr5e-2-a1.0": {"acc": 0.6859, "collapsed": False}}
        m = centre_validity(bad_c, "x-lr5e-2-a1.0", 0.7726)
        chk("S15 a centre far below the argmax IS flagged", any("BELOW this arm" in x for x in m), str(m))
        dead_c = {"x-lr5e-2-a1.0": {"acc": RTE_MAJORITY, "collapsed": True}}
        m = centre_validity(dead_c, "x-lr5e-2-a1.0", 0.7726)
        chk("S16 a COLLAPSED centre is flagged as uninterpretable",
            any("CENTRE IS COLLAPSED" in x for x in m), str(m))
        chk("S16b missing centre yields no claim", centre_validity({}, "nope", 0.77) == [])

        # S17 [R.251 4.2] per-ARM completeness, not global
        import io as _io
        # 3 lr rungs so the argmax can be genuinely INTERIOR
        c2 = {"loca-lr5e-3-a1.0": {"acc": 0.70, "collapsed": False},
              "loca-lr1.5e-2-a1.0": {"acc": 0.7726, "collapsed": False},
              "loca-lr5e-2-a1.0": {"acc": 0.68, "collapsed": False},
              "fftm-lr5e-3-s150": {"acc": 0.64, "collapsed": False}}
        buf = _io.StringIO()
        report(c2, jobs_total=152, stream=buf, totals={"loca": 3, "fftm": 21})
        out = buf.getvalue()
        chk("S17 a COMPLETE arm is marked complete even mid-grid", "*** COMPLETE ***" in out)
        chk("S17b and its edge verdict is NOT provisional",
            "INTERIOR" in out and "(provisional)" not in out.split("FourierFT")[0], out[:200])

        # S18-S20 [R.255] near-floor detection -- the gap [R.222]'s exact rule left
        cell("wave2-lr5e-2-fs150", RTE_MAJORITY + 1.0 / 277)     # 1 example above floor
        cell("wave2-lr5e-3-fs150", 0.7100)
        c = load(td)
        chk("S18 a run 1 example above the floor is NEAR-FLOOR",
            c["wave2-lr5e-2-fs150"]["near_floor"] is True)
        chk("S18b and is NOT mislabelled as collapsed",
            c["wave2-lr5e-2-fs150"]["collapsed"] is False)
        cell("wave2-lr1.5e-2-fs150", RTE_MAJORITY + 3.0 / 277)   # [R.261] the band EDGE
        c = load(td)
        chk("S18c a run EXACTLY 3 examples above the floor is NEAR-FLOOR (band is inclusive)",
            c["wave2-lr1.5e-2-fs150"]["near_floor"] is True,
            "float64 edge: (MAJ+3/277)-MAJ > 3/277")
        chk("S19 a healthy run is neither", not c["wave2-lr5e-3-fs150"]["near_floor"]
            and not c["wave2-lr5e-3-fs150"]["collapsed"])
        buf = io.StringIO(); report(c, jobs_total=99, stream=buf)
        chk("S20 near-floor is announced and kept", "NEAR-FLOOR" in buf.getvalue()
            and "effectively dead" in buf.getvalue())

        # S21-S27 [R.256] adjacent-cell fragility -- the new code path.
        # Rule frozen in llmdocs/R256_fragility_instrument_prereg.md 2.
        H = lambda a: {"acc": a, "collapsed": False, "near_floor": False}
        DEAD = {"acc": RTE_MAJORITY, "collapsed": True, "near_floor": False}
        plane = {"z-lr5e-3-a1.0": H(0.70), "z-lr1.5e-2-a1.0": H(0.77),
                 "z-lr5e-2-a1.0": H(0.68), "z-lr1.5e-2-a0.5": H(0.75),
                 "z-lr1.5e-2-a2.0": H(0.60), "z-lr5e-2-a2.0": H(0.55)}
        nb = sorted(o for o, _, _ in neighbours(plane, "z-lr1.5e-2-a1.0"))
        chk("S21 neighbours are one rung away on exactly one ladder",
            nb == ["z-lr1.5e-2-a0.5", "z-lr1.5e-2-a2.0", "z-lr5e-2-a1.0", "z-lr5e-3-a1.0"], str(nb))
        chk("S21b a two-knob-away cell is NOT a neighbour",
            "z-lr5e-2-a2.0" not in nb, str(nb))
        m = fragility_report(plane, "z-lr1.5e-2-a1.0", 0.77)
        chk("S22 a large live drop is STEEP", any("STEEP" in x for x in m), str(m))
        chk("S22b and the worst neighbour is named",
            any("z-lr1.5e-2-a2.0" in x for x in m), str(m))
        flat = {"z-lr5e-3-a1.0": H(0.765), "z-lr1.5e-2-a1.0": H(0.77),
                "z-lr5e-2-a1.0": H(0.76)}
        m = fragility_report(flat, "z-lr1.5e-2-a1.0", 0.77)
        chk("S23 a small drop is FLAT", any("FLAT" in x for x in m), str(m))
        cliff = dict(flat); cliff["z-lr5e-2-a1.0"] = DEAD
        m = fragility_report(cliff, "z-lr1.5e-2-a1.0", 0.77)
        chk("S24 a DEAD neighbour is CLIFF-EDGE, outranking the drop size",
            any("CLIFF-EDGE" in x for x in m), str(m))
        nf = dict(flat)
        nf["z-lr5e-2-a1.0"] = {"acc": RTE_MAJORITY + 1.0 / 277,
                               "collapsed": False, "near_floor": True}
        m = fragility_report(nf, "z-lr1.5e-2-a1.0", 0.77)
        chk("S24b a NEAR-FLOOR neighbour also counts as dead",
            any("CLIFF-EDGE" in x for x in m), str(m))
        m = fragility_report({"z-ofat-wd0": H(0.75), "z-lr5e-3-a1.0": H(0.70),
                              "z-lr1.5e-2-a1.0": H(0.74), "z-lr5e-2-a1.0": H(0.69)},
                             "z-ofat-wd0", 0.75)
        chk("S25 an OFAT argmax yields UNDEFINED, not a silent substitution",
            any("UNDEFINED" in x for x in m), str(m))
        chk("S25b and the plane fallback is labelled as a different cell",
            any("PLANE argmax" in x and "NOT the same cell" in x for x in m), str(m))
        chk("S25c and the fallback verdict is the plane cell's",
            any("plane-argmax fragility" in x for x in m), str(m))
        m = fragility_report({"z-lr5e-3-a1.0": H(0.70), "z-lr1.5e-2-a1.0": H(0.77)},
                             "z-lr1.5e-2-a1.0", 0.77)
        chk("S26 fewer than 2 adjacent cells refuses a verdict",
            any("no fragility verdict" in x for x in m)
            and not any("FLAT" in x or "STEEP" in x or "CLIFF" in x for x in m), str(m))
        # NB: report() only prints arms listed in ARMS, so this fixture must use a
        # REAL arm prefix.  The first version used 'z-' and failed -- the code was
        # right and the fixture was wrong (meta-rule 4, again).
        real = {"fftm-lr5e-3-s150": H(0.765), "fftm-lr1.5e-2-s150": H(0.77),
                "fftm-lr5e-2-s150": dict(DEAD)}
        buf = io.StringIO()
        report(real, jobs_total=3, stream=buf, totals={"fftm": 3})
        chk("S27 fragility is printed by report(), not just callable",
            "fragility:" in buf.getvalue() and "CLIFF-EDGE" in buf.getvalue(),
            buf.getvalue()[:400])

        # S28-S30 [R.289] deterministic argmax under exact ties
        tie = {"scora-ofat-cosine": {"acc": 0.7509, "collapsed": False, "near_floor": False},
               "scora-ofat-clf1e-2": {"acc": 0.7509, "collapsed": False, "near_floor": False},
               "scora-lr5e-2": {"acc": 0.7437, "collapsed": False, "near_floor": False}}
        chk("S28 an exact tie resolves to the lexicographically smallest label",
            argmax_cell(tie) == "scora-ofat-clf1e-2", argmax_cell(tie))
        chk("S28b and it does NOT depend on dict insertion order",
            argmax_cell(dict(reversed(list(tie.items())))) == "scora-ofat-clf1e-2")
        chk("S29 every tied label is reported, not just the winner",
            tied_with_argmax(tie) == ["scora-ofat-clf1e-2", "scora-ofat-cosine"],
            str(tied_with_argmax(tie)))
        buf = io.StringIO(); report(tie, jobs_total=3, stream=buf, totals={"scora": 3})
        chk("S30 the tie is ANNOUNCED as an arbitrary pick", "TIE at the argmax" in buf.getvalue())

        # S8 an empty / headerless CSV is skipped, not crashed on
        open(os.path.join(td, "qwha-lr5e-2-s150.csv"), "w").close()
        c = load(td)
        chk("S8 empty CSV skipped without crashing", "qwha-lr5e-2-s150" not in c)

        # S9 a NaN accuracy row is skipped
        cell("qwha-lr5e-3-s150", float("nan"))
        c = load(td)
        chk("S9 NaN accuracy skipped", "qwha-lr5e-3-s150" not in c)

    print("\n" + "=" * 60)
    print(f"  selftest: {len(ok)} passed, {len(bad)} failed")
    if bad:
        print("  FAILED: " + ", ".join(bad))
    print("=" * 60)
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dir", default=D)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    jobs = os.path.join(a.dir, "jobs.tsv")
    total = sum(1 for _ in open(jobs)) if os.path.exists(jobs) else None
    report(load(os.path.join(a.dir, "csv")), jobs_total=total,
           totals=arm_totals(a.dir))
