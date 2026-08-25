#!/usr/bin/env python
"""[R.305] READER -- the deliverable: fairly tuned frequency-domain baselines.

  env/bin/python scripts/r305_read.py --selftest   # run this before believing it
  env/bin/python scripts/r305_read.py

WHAT IT REFUSES TO DO
  * quote a stage A/B/C cell as a result.  Those are n=1, seed 41, SELECTION
    only.  RTE's paired sd is 0.0186 (~5.2 eval examples) and the metric is
    quantised at 1/277 = 0.0036.
  * hide an unbracketed optimum.  If an arm's argmax still sits on a ladder
    edge after stage B, the row is printed as a LOWER BOUND and says so.
  * declare a cross-arm winner from means.  Cross-arm claims go through the
    5/5 paired sign gate (PROCESS 1.3), which certifies effects >= 0.021 only.
"""
import argparse, math, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r305_plan as P


def _sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else float("nan")


def collect(results=None, manifest=None):
    """-> {arm: {...}} with the plane verdict, OFAT deltas and confirmation."""
    results = P.load() if results is None else results
    manifest = P.read_manifest() if manifest is None else manifest
    out = {}
    arms = sorted({m["arm"] for m in manifest.values()})
    for arm in arms:
        sel = {k: v for k, v in results.items()
               if manifest.get(k, {}).get("arm") == arm
               and manifest[k]["stage"] in ("A", "B", "C")}
        plane = {k: v for k, v in sel.items() if manifest[k]["stage"] in ("A", "B")}
        rec = {"select": sel, "plane": plane, "edges": [], "ofat": {}, "confirm": {}}
        if plane:
            best = P.argmax_cell(plane)
            rec["plane_best"] = best
            bi, bj = P._idx(best)
            pis = sorted({P._idx(k)[0] for k in plane})
            if bi in (min(pis), max(pis)):
                rec["edges"].append(("P", bi, min(pis), max(pis)))
            if bj is not None:
                sjs = sorted({P._idx(k)[1] for k in plane})
                if bj in (min(sjs), max(sjs)):
                    rec["edges"].append(("scale", bj, min(sjs), max(sjs)))
            for k, v in sel.items():
                if manifest[k]["stage"] == "C":
                    rec["ofat"][k.split("-C-")[1]] = v["acc"] - plane[best]["acc"]
        # confirmation: {candidate_label: {seed: acc}}
        for k, v in results.items():
            m = manifest.get(k, {})
            if m.get("arm") == arm and m.get("stage") == "D":
                rec["confirm"].setdefault(m["from"], {})[m["seed"]] = v["acc"]
        out[arm] = rec
    return out


def tuned(rec):
    """The arm's reported number: the candidate with the best 5-seed MEAN, over
    seeds DISJOINT from selection.  Ties break on the smallest label [R.289].
    Returns (label, mean, sd, n, per_seed) or None if unconfirmed."""
    full = {c: s for c, s in rec["confirm"].items()
            if len(s) == len(P.CONFIRM_SEEDS)}
    if not full:
        return None
    best = min(full, key=lambda c: (-statistics.fmean(full[c].values()), c))
    vals = [full[best][s] for s in P.CONFIRM_SEEDS]
    return best, statistics.fmean(vals), _sd(vals), len(vals), vals


def paired_gate(a, b):
    """PROCESS 1.3's 5/5 gate on two arms' per-seed vectors (same seeds)."""
    wins = sum(1 for x, y in zip(a, b) if x > y)
    ties = sum(1 for x, y in zip(a, b) if x == y)
    return wins, ties, len(a)


def report():
    results, manifest = P.load(), P.read_manifest()
    if not manifest:
        print("[r305] nothing planned yet")
        return
    data = collect(results, manifest)
    done = len(results)
    print("=" * 78)
    print("[R.305] FAIRLY TUNED FREQUENCY-DOMAIN BASELINES -- RTE / roberta-base / q+v / k=256")
    print(f"cells complete: {done}/{len(manifest)}")
    print("selection seed %d (n=1, NOT quotable) | confirmation seeds %s (out-of-sample)"
          % (P.SCREEN_SEED, P.CONFIRM_SEEDS))
    print("=" * 78)

    for arm in sorted(data):
        rec = data[arm]
        title = P.ARMS[arm]["title"] if arm in P.ARMS else arm
        n_plane = sum(1 for m in manifest.values()
                      if m["arm"] == arm and m["stage"] in ("A", "B"))
        print(f"\n{'-'*78}\n{title}   [{arm}]   plane {len(rec['plane'])}/{n_plane} complete")
        if not rec["plane"]:
            print("   (no plane cells yet)")
        else:
            b = rec["plane_best"]
            dead = sum(1 for v in rec["plane"].values() if v["collapsed"] or v["near_floor"])
            print(f"   plane argmax  {b}  {rec['plane'][b]['acc']:.4f}"
                  f"   ({dead} of {len(rec['plane'])} cells dead/near-floor)")
            toks = manifest[b]["args"].split()
            keep = [f"{f} {v}" for f, v in zip(toks, toks[1:])
                    if f in ("--learning_rate", "--weight_decay", "--classifier_lr")
                    or ("scal" in f or f.endswith("_scale"))]
            print(f"     config:     {'  '.join(keep)}")
            if rec["edges"]:
                for axis, at, lo, hi in rec["edges"]:
                    print(f"   ⚠️ argmax still on the {'TOP' if at == hi else 'BOTTOM'} rung of the "
                          f"{axis} ladder [{lo}..{hi}] -> report as a LOWER BOUND")
            else:
                print("   ✅ argmax INTERIOR on every ladder -> a genuinely bracketed optimum")
            ep = P.best_epoch(b)
            if ep is not None:
                flag = "  ⚠️ TRUNCATED (peak in the final 10% of epochs) [R.285]" if ep >= 27 else ""
                print(f"   argmax epoch {ep}/30{flag}")
        if rec["ofat"]:
            print("   OFAT, one knob from THIS ARM'S OWN optimum (not a shared centre) [R.279]:")
            for k, d in sorted(rec["ofat"].items(), key=lambda kv: -kv[1]):
                mark = "  <- exceeds the 0.021 the 5/5 gate can certify" if abs(d) > 0.021 else ""
                print(f"      {k:12s} {d:+.4f}{mark}")
        t = tuned(rec)
        if t:
            lab, mu, sd, n, vals = t
            print(f"   ⭐ CONFIRMED  {lab}   {mu:.4f} +- {sd:.4f}  (n={n}, "
                  f"per-seed {' '.join(f'{v:.4f}' for v in vals)})")
        else:
            have = {c: len(s) for c, s in rec["confirm"].items()}
            print(f"   confirmation incomplete: {have or 'not started'}")

    # ---- the table ---------------------------------------------------------
    conf = {a: tuned(r) for a, r in data.items()}
    conf = {a: t for a, t in conf.items() if t}
    if conf:
        print("\n" + "=" * 78)
        print("TUNED TABLE -- 5 seeds, out-of-sample, each arm at its OWN optimum")
        print("=" * 78)
        print(f"{'arm':10s} {'mean':>8s} {'sd':>8s}  {'config':s}")
        for a, (lab, mu, sd, n, _) in sorted(conf.items(), key=lambda kv: -kv[1][1]):
            edge = " [LOWER BOUND: unbracketed]" if data[a]["edges"] else ""
            print(f"{a:10s} {mu:8.4f} {sd:8.4f}  {lab}{edge}")
        # paired 5/5 gate against FourierFT, the comparator both STANDINGs rest on
        if "fftm" in conf:
            ref = conf["fftm"][4]
            print("\npaired vs FourierFT (same 5 seeds), PROCESS 1.3's 5/5 gate:")
            for a, (lab, mu, sd, n, vals) in sorted(conf.items()):
                if a == "fftm":
                    continue
                w, ti, tot = paired_gate(vals, ref)
                verdict = ("PASSES 5/5" if w == tot else
                           "PASSES 5/5 (other direction)" if w == 0 else
                           "does NOT pass the gate")
                print(f"   {a:10s} wins {w}/{tot} (ties {ti})  median delta "
                      f"{statistics.median([x-y for x, y in zip(vals, ref)]):+.4f}   {verdict}")
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
    man, res = {}, {}

    def cell(lab, arm, stage, acc, seed=P.SCREEN_SEED, frm=None):
        man[lab] = {"arm": arm, "stage": stage, "args": "--learning_rate 1", "seed": seed}
        if frm:
            man[lab]["from"] = frm
        res[lab] = {"acc": acc, "collapsed": False, "near_floor": False}

    # a 2x2 plane with an INTERIOR-impossible argmax (every 2x2 argmax is on an
    # edge) plus a 3x3 one where it is interior -- the edge detector must
    # distinguish them, or "hyperparameter-optimal" means nothing.
    for i in (-1, 0, 1):
        for j in (0, 1, 2):
            cell(f"x-A-p{i}-s{j}", "x", "A", 0.60)
    res["x-A-p0-s1"]["acc"] = 0.80
    d = collect(res, man)
    ck(d["x"]["plane_best"] == "x-A-p0-s1", "interior argmax")
    ck(d["x"]["edges"] == [], "interior argmax must report NO edge")
    res["x-A-p0-s1"]["acc"] = 0.60
    res["x-A-p1-s2"]["acc"] = 0.80
    d = collect(res, man)
    ck({e[0] for e in d["x"]["edges"]} == {"P", "scale"}, "corner argmax = TWO edges")

    # OFAT deltas are measured from the arm's OWN plane optimum, not a centre
    cell("x-C-wd0", "x", "C", 0.83)
    d = collect(res, man)
    ck(abs(d["x"]["ofat"]["wd0"] - 0.03) < 1e-12, "OFAT delta is vs the plane optimum")

    # confirmation: only a COMPLETE 5-seed block may be reported.  [R.289]:
    # never quote partial work -- a 4-of-5 block must read as incomplete.
    for s in S[:4]:
        cell(f"x-D-c0-seed{s}", "x", "D", 0.70, seed=s, frm="x-A-p1-s2")
    ck(tuned(collect(res, man)["x"]) is None, "4/5 seeds must NOT be reportable")
    cell(f"x-D-c0-seed{S[4]}", "x", "D", 0.75, seed=S[4], frm="x-A-p1-s2")
    t = tuned(collect(res, man)["x"])
    ck(t is not None and t[3] == 5, "5/5 seeds become reportable")
    ck(abs(t[1] - statistics.fmean([0.70]*4 + [0.75])) < 1e-12, "mean over the 5 seeds")

    # two candidates -> the better MEAN wins, and it can differ from the
    # single-seed screening winner.  That is the whole point of stage D.
    for s in S:
        cell(f"x-D-c1-seed{s}", "x", "D", 0.90, seed=s, frm="x-A-p0-s1")
    t = tuned(collect(res, man)["x"])
    ck(t[0] == "x-A-p0-s1", "the higher 5-seed mean must win, not the screening argmax")

    # the 5/5 paired gate
    ck(paired_gate([1, 1, 1, 1, 1], [0, 0, 0, 0, 0]) == (5, 0, 5), "5/5 wins")
    ck(paired_gate([1, 1, 1, 1, 0], [0, 0, 0, 0, 0]) == (4, 1, 5), "4/5 does not pass")
    ck(paired_gate([0, 0, 0, 0, 0], [0, 0, 0, 0, 0]) == (0, 5, 5), "ties counted")
    print(f"[r305_read] selftest: {n[0]} passed, 0 failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else report()
