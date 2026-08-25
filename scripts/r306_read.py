#!/usr/bin/env python
"""[R.306] READER -- SCoRA under EQUAL tuning rigour, reported next to [R.305].

  env/bin/python scripts/r306_read.py --selftest   # run this before believing it
  env/bin/python scripts/r306_read.py

WHAT IT REFUSES TO DO
  * ⛔ print `scora2` WITHOUT `scora`.  Replacing the a-priori row with the swept
    row would be tuning our own arm harder AFTER seeing that it lost -- the single
    most attackable move available.  `report()` emits both or neither; the
    selftest asserts it.
  * quote a stage A/B/C cell as a result (n=1, seed 41, SELECTION only).
  * hide an unbracketed optimum: an argmax still on a ladder edge after stage B
    prints as a LOWER BOUND.
  * declare a cross-arm winner from means -- cross-arm claims go through the 5/5
    paired sign gate (PROCESS 1.3), which certifies effects >= 0.021 only.
  * ⛔ take a max over two confirmation sets.  The candidates come from ONE
    union pool (`r306_plan.union_pool`); this file only prints what stage D ran.

Every baseline row shown here is [R.305]'s, unchanged and re-read from its own
state dir -- this run re-tuned exactly one arm.
"""
import argparse, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r305_plan as P
import r305_read as R5
import r306_plan as R6

A_PRIORI_SCALING = R6.DERIVED_SCALING
RTE_SINGLE_RUN_SD = 0.01397          # CONTEXT 6, [R.283]


def _dlabel(rec, cand, seed, man):
    """The stage-D label that ran `cand` at `seed`, or '' if none."""
    for lab, m in man.items():
        if m.get("stage") == "D" and m.get("from") == cand and m.get("seed") == seed:
            return lab
    return ""


GATE_FLOOR = 0.021          # PROCESS 1.3: the smallest effect 5/5 at n=5 certifies


def gate_note(w, tot, med):
    """What a NON-firing sign gate means -- and it is not one thing.

    ⛔ Do not print the 0.009-0.021 band boilerplate blind.  A sign gate fails
    for TWO reasons that mean opposite things: a genuinely small effect, or a
    LARGE one with a seed that reversed.  [R.306] hit the second, and the
    boilerplate would have read as "the gap is small" when the median deficit
    was 1.5x what the gate can certify."""
    if w in (0, tot):
        return []
    if abs(med) > GATE_FLOOR:
        return [f"   ⛔ the gate did NOT fire, but the median delta {med:+.4f} is LARGER "
                f"than the {GATE_FLOOR} it can certify:",
                "      the sign gate failed on VARIANCE (a reversing seed), not on a "
                "closed gap.  Neither 'we win' nor 'the gap closed' may be claimed."]
    return ["   ⛔ 0.009-%.3f is a real effect this design CANNOT certify. "
            "Do not upgrade it in prose." % GATE_FLOOR]


def r305_confirmed():
    """{arm: (label, mean, sd, n, per_seed)} for [R.305]'s arms, re-read from
    its own state dir.  `r306_plan.install()` re-points `r305_plan`'s globals,
    so D is saved and restored around the read rather than assumed."""
    old_d, old_arms = P.D, P.ARMS
    try:
        P.D = R6.R305_D
        res, man = P.load(), P.read_manifest()
        data = R5.collect(res, man)
        out = {}
        for arm, rec in data.items():
            t = R5.tuned(rec)
            if t:
                out[arm] = t
        return out, data
    finally:
        P.D, P.ARMS = old_d, old_arms


def r306_data():
    R6.install()
    res, man = P.load(), P.read_manifest()
    return R5.collect(res, man), res, man


def union_report(res, man):
    """The candidate pool and the top-2 actually confirmed, with provenance."""
    pool = R6.union_pool(res, man)
    top = P.top_n_cells(pool, P.N_CONFIRM_CANDIDATES) if pool else []
    return pool, top


def report():
    data6, res6, man6 = r306_data()
    conf5, data5 = r305_confirmed()
    rec = data6.get("scora2", {"plane": {}, "edges": [], "ofat": {}, "confirm": {}})

    print("=" * 78)
    print("[R.306] SCoRA WITH ITS SCALE AXIS SWEPT -- RTE / roberta-base / q+v / k=256")
    print(f"cells complete: {len(res6)}/{len(man6)}")
    print("selection seed %d (n=1, NOT quotable) | confirmation seeds %s (out-of-sample)"
          % (P.SCREEN_SEED, P.CONFIRM_SEEDS))
    print("=" * 78)

    # ---- the swept plane ---------------------------------------------------
    n_plane = sum(1 for m in man6.values() if m["stage"] in ("A", "B"))
    print(f"\nswept plane: {len(rec['plane'])}/{n_plane} complete "
          f"(a-priori scaling {A_PRIORI_SCALING:.10g} is rung s2)")
    if rec["plane"]:
        b = rec["plane_best"]
        dead = sum(1 for v in rec["plane"].values() if v["collapsed"] or v["near_floor"])
        print(f"   plane argmax  {b}  {rec['plane'][b]['acc']:.4f}"
              f"   ({dead} of {len(rec['plane'])} cells dead/near-floor)")
        toks = man6[b]["args"].split()
        keep = [f"{f} {v}" for f, v in zip(toks, toks[1:])
                if f in ("--learning_rate", "--slr_scaling", "--weight_decay", "--classifier_lr")]
        print(f"     config:     {'  '.join(keep)}")
        if rec["edges"]:
            for axis, at, lo, hi in rec["edges"]:
                print(f"   ⚠️ argmax still on the {'TOP' if at == hi else 'BOTTOM'} rung of the "
                      f"{axis} ladder [{lo}..{hi}] -> report as a LOWER BOUND")
        else:
            print("   ✅ argmax INTERIOR on every ladder -> a genuinely bracketed optimum")
        # Did sweeping the scale beat the DERIVED value at all?  This is the one
        # question the sweep answers about the method rather than about a baseline.
        s2 = {k: v for k, v in rec["plane"].items() if k.endswith("-s2")}
        if s2:
            bs2 = P.argmax_cell(s2)
            print(f"   derived-scale column best  {bs2}  {s2[bs2]['acc']:.4f}"
                  f"   (swept best {rec['plane'][b]['acc']:.4f}, "
                  f"delta {rec['plane'][b]['acc'] - s2[bs2]['acc']:+.4f} at n=1)")
        ep = P.best_epoch(b)
        if ep is not None:
            flag = "  ⚠️ TRUNCATED (peak in the final 10% of epochs) [R.285]" if ep >= 27 else ""
            print(f"   argmax epoch {ep}/30{flag}")
    if rec["ofat"]:
        print("   OFAT, one knob from THIS ARM'S OWN optimum [R.279]:")
        for k, d in sorted(rec["ofat"].items(), key=lambda kv: -kv[1]):
            mark = "  <- exceeds the 0.021 the 5/5 gate can certify" if abs(d) > 0.021 else ""
            print(f"      {k:12s} {d:+.4f}{mark}")

    # ---- the union pool ----------------------------------------------------
    pool, top = union_report(res6, man6)
    n_own = sum(1 for k in pool if k.startswith("scora2-"))
    print(f"\nUNION candidate pool: {len(pool)} seed-41 cells "
          f"({n_own} swept + {len(pool) - n_own} a-priori from [R.305])")
    for r, c in enumerate(top):
        src = "swept" if c.startswith("scora2-") else "a-priori [R.305]"
        print(f"   c{r}  {c:24s} {pool[c]['acc']:.4f}   {src}")
    n_imp = sum(1 for m in man6.values() if m.get("imported_from"))
    if n_imp:
        print(f"   ({n_imp} confirmation cells IMPORTED from [R.305] -- identical args and seed)")

    # ---- the table ---------------------------------------------------------
    # ---- ⛔ BOTH confirmed candidates, in full -----------------------------
    # `tuned()` reports the best 5-seed MEAN -- the preregistered rule, applied
    # identically to every baseline arm in [R.305].  It is NOT re-litigated here.
    # But a mean is not a result on its own: [R.305] confirms TWO candidates
    # precisely because the top-1 is unreliable, so both vectors are printed and
    # each gets its own gate.  A reader who only sees the winning mean cannot
    # tell a stable arm from one seed's luck.
    full = {c: s for c, s in rec["confirm"].items() if len(s) == len(P.CONFIRM_SEEDS)}
    if full:
        ref = conf5["fftm"][4] if "fftm" in conf5 else None
        print(f"\nCONFIRMED CANDIDATES ({len(full)} of {P.N_CONFIRM_CANDIDATES}), "
              f"per-seed over {P.CONFIRM_SEEDS}:")
        for c in sorted(full):
            vals = [full[c][s] for s in P.CONFIRM_SEEDS]
            sd = R5._sd(vals)
            line = (f"   {c:22s} mean {statistics.fmean(vals):.4f}  sd {sd:.4f}  "
                    f"[{' '.join(f'{v:.4f}' for v in vals)}]")
            if ref:
                w, ti, tot = R5.paired_gate(vals, ref)
                line += f"   vs fftm {w}/{tot}"
            print(line)
            if sd > 2 * RTE_SINGLE_RUN_SD:
                print(f"      ⚠️ sd {sd:.4f} is {sd/RTE_SINGLE_RUN_SD:.1f}x RTE's single-run sd "
                      f"({RTE_SINGLE_RUN_SD:.5f}) -- this candidate's MEAN is fragile")
            trunc = [s for s in P.CONFIRM_SEEDS
                     if (P.best_epoch(f"{_dlabel(rec, c, s, man6)}") or 0) >= 27]
            if trunc:
                print(f"      ⚠️ [R.285] peak in the final 10% of epochs on seed(s) {trunc} "
                      f"-- those runs were TRUNCATED, not converged")

    t6 = R5.tuned(rec)
    if not t6:
        have = {c: len(s) for c, s in rec["confirm"].items()}
        print(f"\nconfirmation incomplete: {have or 'not started'}")
        print("⛔ no row may be quoted until a full 5-seed block lands.")
        return
    lab6, mu6, sd6, n6, vals6 = t6

    print("\n" + "=" * 78)
    print("TABLE -- 5 out-of-sample seeds (42-46), paired.  Baselines are [R.305]'s.")
    print("⛔ BOTH SCoRA rows are reported.  `scora` keeps the derived-not-swept")
    print("   claim; `scora2` forfeits it to answer 'did you tune yours as hard?'.")
    print("=" * 78)
    rows = [(a, t[1], t[2], t[0]) for a, t in conf5.items()]
    rows.append(("scora2", mu6, sd6, lab6))
    print(f"{'arm':10s} {'mean':>8s} {'sd':>8s}  {'config':s}")
    for a, mu, sd, lab in sorted(rows, key=lambda r: -r[1]):
        note = ""
        if a == "scora":
            note = "   <- OURS, a-priori scaling (derived, [R.305])"
        elif a == "scora2":
            note = "   <- OURS, scaling SWEPT (a-priori claim forfeited)"
            if data6["scora2"]["edges"]:
                note += " [LOWER BOUND: unbracketed]"
        print(f"{a:10s} {mu:8.4f} {sd:8.4f}  {lab}{note}")

    # ---- the gates ---------------------------------------------------------
    if "fftm" in conf5:
        ref = conf5["fftm"][4]
        w, ti, tot = R5.paired_gate(vals6, ref)
        med = statistics.median([x - y for x, y in zip(vals6, ref)])
        verdict = ("PASSES 5/5 FOR us" if w == tot else
                   "PASSES 5/5 AGAINST us" if w == 0 else
                   "does NOT pass the gate in either direction")
        print(f"\nscora2 vs FourierFT (merged), same 5 seeds, PROCESS 1.3's 5/5 gate:")
        print(f"   wins {w}/{tot} (ties {ti})   median delta {med:+.4f}   {verdict}")
        for line in gate_note(w, tot, med):
            print(line)
    if "scora" in conf5:
        v5 = conf5["scora"][4]
        w, ti, tot = R5.paired_gate(vals6, v5)
        med = statistics.median([x - y for x, y in zip(vals6, v5)])
        print(f"\nswept vs a-priori (scora2 vs scora), same 5 seeds:")
        print(f"   wins {w}/{tot} (ties {ti})   median delta {med:+.4f}")
        print("   this is a statement about the DERIVATION, not about any baseline.")
        if med <= 0:
            print("   ✅ sweeping did not beat the derived value -> the a-priori rule "
                  "was already at or above the swept optimum.")
    print("\n⛔ the gate certifies effects >= 0.021; anything in 0.009-0.021 is a real")
    print("   effect this design cannot certify.  Never upgrade it in prose.")


# ============================================================================
def selftest():
    n = [0]

    def ck(c, m):
        n[0] += 1
        if not c:
            print("FAIL:", m); sys.exit(1)

    S = P.CONFIRM_SEEDS

    # -- 1. the a-priori scaling this reader labels must be the SHIPPED one ---
    ck(abs(A_PRIORI_SCALING - 0.01220703125) < 1e-12,
       f"a-priori scaling must be 0.01220703125, got {A_PRIORI_SCALING}")

    # -- 2. [R.305]'s table must be readable and UNCHANGED by this run -------
    conf5, _ = r305_confirmed()
    ck("fftm" in conf5 and "scora" in conf5,
       f"[R.305]'s fftm and scora rows must be re-readable, got {sorted(conf5)}")
    ck(abs(conf5["fftm"][1] - 0.7906) < 5e-4,
       f"[R.305]'s FourierFT row must still read 0.7906, got {conf5['fftm'][1]:.4f}")
    ck(abs(conf5["scora"][1] - 0.7480) < 5e-4,
       f"[R.305]'s a-priori SCoRA row must still read 0.7480, got {conf5['scora'][1]:.4f}")
    ck(len(conf5["fftm"][4]) == len(S), "the comparator vector must be 5 seeds")

    # -- 3. reading [R.306] must NOT leave r305_plan's globals re-pointed ----
    #    (both readers share the module; a leaked `D` would make the next
    #     r305_read call silently read r306's dir and print a wrong table)
    old = P.D
    r305_confirmed()
    ck(P.D == old, "r305_confirmed must restore r305_plan.D")

    # -- 4. the union pool is BOTH halves, and top-2 comes from ONE pool -----
    man = {f"scora2-A-p0-s{j}": {"arm": "scora2", "stage": "A",
                                 "args": f"--slr_scaling {j}", "seed": 41}
           for j in range(4)}
    res = {k: {"acc": 0.60, "collapsed": False, "near_floor": False} for k in man}
    pool, top = union_report(res, man)
    ck(len(pool) > len(res), "the union must add [R.305]'s a-priori cells")
    ck(len(top) == P.N_CONFIRM_CANDIDATES, f"exactly {P.N_CONFIRM_CANDIDATES} candidates")
    #   CONTROL: when the swept cells dominate, BOTH candidates are swept ----
    res["scora2-A-p0-s0"]["acc"] = 0.99
    res["scora2-A-p0-s1"]["acc"] = 0.98
    _, top = union_report(res, man)
    ck(all(c.startswith("scora2-") for c in top), f"swept winners must be chosen, got {top}")
    #   CONTROL: when they are worthless, the a-priori cells are chosen ------
    for k in res:
        res[k]["acc"] = 0.10
    _, top = union_report(res, man)
    ck(all(c.startswith("scora-") for c in top), f"a-priori winners must be chosen, got {top}")

    # -- 5. an incomplete 5-seed block is NOT reportable ([R.289]) -----------
    m2, r2 = {}, {}
    for i, s in enumerate(S):
        lab = f"scora2-D-c0-seed{s}"
        m2[lab] = {"arm": "scora2", "stage": "D", "args": "x", "seed": s, "from": "c"}
        r2[lab] = {"acc": 0.75, "collapsed": False, "near_floor": False}
        if i == 3:
            break
    ck(R5.tuned(R5.collect(r2, m2)["scora2"]) is None, "4/5 seeds must NOT be reportable")
    lab = f"scora2-D-c0-seed{S[4]}"
    m2[lab] = {"arm": "scora2", "stage": "D", "args": "x", "seed": S[4], "from": "c"}
    r2[lab] = {"acc": 0.75, "collapsed": False, "near_floor": False}
    t = R5.tuned(R5.collect(r2, m2)["scora2"])
    ck(t is not None and t[3] == 5, "5/5 seeds become reportable")

    # -- 6. ⛔ THE ONE THAT MATTERS: the printed table can never contain
    #        `scora2` without `scora`.  Checked on the REAL report path.
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report()
    out = buf.getvalue()
    if "scora2 " in out and "TABLE --" in out:
        ck("\nscora " in out or "\nscora     " in out,
           "a table containing scora2 MUST also contain [R.305]'s scora row")
    ck("BOTH SCoRA rows" in out or "confirmation incomplete" in out or "nothing planned" in out,
       "report() must either state the both-rows rule or say it has no data")
    #   CONTROL: the rule is not vacuous -- the banner text must be present in
    #   the source that prints the table, so deleting it fails this test.
    src = open(os.path.join(HERE, "r306_read.py")).read()
    ck('rows.append(("scora2"' in src and 'conf5.items()' in src,
       "the table must be built from [R.305]'s rows PLUS scora2, not scora2 alone")

    # -- 7. a NON-firing gate must not be described the same way twice ------
    ck(gate_note(0, 5, -0.05) == [], "a 5/5 gate (against) needs no caveat")
    ck(gate_note(5, 5, +0.05) == [], "a 5/5 gate (for) needs no caveat")
    big = " ".join(gate_note(1, 5, -0.0325))
    small = " ".join(gate_note(3, 5, -0.004))
    ck("VARIANCE" in big and "LARGER" in big,
       "a non-firing gate with |median| > the floor must be called out as VARIANCE")
    ck("CANNOT certify" in small and "VARIANCE" not in small,
       "a non-firing gate with a small median keeps the band boilerplate")
    ck(big != small, "the two failure modes must NOT print the same sentence")
    #   CONTROL: the branch really is on the FLOOR, not on the win count
    ck(" ".join(gate_note(1, 5, -0.004)) == small,
       "the message must be chosen by |median|, not by the number of wins")

    # -- 8. _dlabel maps (candidate, seed) -> the stage-D cell that ran it ----
    man = {"z-D-c0-seed42": {"stage": "D", "from": "cand", "seed": 42},
           "z-D-c1-seed42": {"stage": "D", "from": "other", "seed": 42},
           "z-A-p0":        {"stage": "A", "from": "cand", "seed": 42}}
    ck(_dlabel(None, "cand", 42, man) == "z-D-c0-seed42", "_dlabel must find the D cell")
    ck(_dlabel(None, "cand", 43, man) == "", "_dlabel must return '' for a missing seed")
    ck(_dlabel(None, "nope", 42, man) == "", "_dlabel must not match another candidate")

    # -- 9. the fragility warning fires on spread, and ONLY on spread --------
    ck(R5._sd([0.769, 0.6787, 0.7509, 0.7906, 0.787]) > 2 * RTE_SINGLE_RUN_SD,
       "the observed [R.306] c1 vector must trip the fragility threshold")
    ck(not (R5._sd([0.7545, 0.7365, 0.7401, 0.7617, 0.7365]) > 2 * RTE_SINGLE_RUN_SD),
       "the stable c0 vector must NOT trip it -- else the warning is vacuous")

    # -- 10. the report must disclose BOTH candidates, not just the winner ---
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        report()
    o2 = buf2.getvalue()
    if "TABLE --" in o2:
        ck("CONFIRMED CANDIDATES" in o2, "the report must print every confirmed candidate")
        ck(o2.count("mean 0.7") >= 2, "both candidates' per-seed means must be printed")

    print(f"[r306_read] selftest: {n[0]} passed, 0 failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else report()
