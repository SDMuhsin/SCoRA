#!/usr/bin/env python
"""[R.237] CONFIRMATION-BLOCK GENERATOR.

The screening grid does NOT establish a tuned baseline.  It is n=1 per cell on a task
whose paired sd is ~0.02-0.03 and whose metric is quantised at 1/277 = 0.0036, so its
per-arm argmax is a CANDIDATE carrying winner's curse.  A baseline is established only
when that candidate is re-run at 5 seeds against the arm's own centre and actually wins.

This emits that confirmation block mechanically from the completed grid:
  for each arm  ->  winner config and centre config, 5 seeds each, same protocol.

⛔ It REFUSES to emit for any arm that is not 100% complete (PROCESS.md 1.4).
⛔ It only reads; it launches nothing.

Usage:
  env/bin/python scripts/r237_confirm_gen.py            # emit for complete arms
  env/bin/python scripts/r237_confirm_gen.py --selftest
"""
import argparse, csv, glob, math, os, sys, collections

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GRID = os.path.join(ROOT, "scratchpad", "phaseR", "r237")
MAJORITY = 146 / 277
CENTRE = {                       # from R237_baseline_grid_prereg.md 2
    "loca":  "loca-lr5e-2-a1.0",
    "fftm":  "fftm-lr5e-2-s150",
    "scora": "scora-lr5e-2",
    "lyra":  "lyra-lr5e-2-e3.0",
    "wave1": "wave1-lr5e-2-fs150",
    "wave2": "wave2-lr5e-2-fs150",
    "qwha":  "qwha-lr5e-2-s106.0660",
}


def load_jobs(d):
    """label -> arg string, from the frozen job list."""
    p = os.path.join(d, "jobs.tsv")
    out = {}
    if not os.path.exists(p):
        return out
    for line in open(p):
        if "\t" in line:
            lab, args = line.rstrip("\n").split("\t", 1)
            out[lab] = args
    return out


def load_done(d):
    dd = os.path.join(d, "done")
    return set(os.listdir(dd)) if os.path.isdir(dd) else set()


def load_acc(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "csv", "*.csv"))):
        try:
            rows = list(csv.DictReader(open(p)))
        except Exception:
            continue
        if not rows:
            continue
        try:
            v = float(rows[-1].get("accuracy", "nan"))
        except (TypeError, ValueError):
            continue
        if not math.isnan(v):
            out[os.path.basename(p)[:-4]] = v
    return out


def generate(d, stream=sys.stdout):
    p = lambda *a: print(*a, file=stream)
    jobs, done, acc = load_jobs(d), load_done(d), load_acc(d)
    by_arm = collections.defaultdict(list)
    for lab in jobs:
        by_arm[lab.split("-")[0]].append(lab)

    emitted = {}
    p("# [R.237] CONFIRMATION BLOCK -- generated, not launched.")
    p("# Each arm's screening winner vs its own centre, 5 seeds, same protocol.")
    # ⛔ [R.265] REPLACED the rule this header used to state ("a baseline counts as TUNED only
    # if the winner beats the centre").  That rule quotes the WINNER even when it loses to the
    # centre, which UNDERSTATES a baseline -- the one direction never permitted [R.196].
    p("# READING RULE (frozen, [R.265]):")
    p("#   table value = max(median(winner), median(centre)) -- never understate a baseline.")
    p("#     residual max-of-2 bias at n=5 is 0.0033 = 0.91 eval examples, vs 7.2 for the screen.")
    p("#   'tuning improved arm X' may be claimed ONLY if the paired winner-centre delta clears")
    p("#     the 5/5 sign gate; otherwise say 'not separated by this gate'.")
    p("#   NO re-selection: a disappointing winner is NOT replaced by the runner-up.")
    p("#   POWER, precomputed [R.265 3]: P(5/5) = 1.00 loca . 1.00 fftm . 0.81 lyra . 0.12 scora")
    p("#     => SCoRA's row will read 'not separated by this gate' whatever happens.")
    for arm in sorted(by_arm):
        labs = by_arm[arm]
        missing = [l for l in labs if l not in done]
        if missing:
            p(f"\n# {arm}: INCOMPLETE ({len(labs)-len(missing)}/{len(labs)}) -> REFUSED "
              f"(PROCESS.md 1.4)")
            continue
        scored = {l: acc[l] for l in labs if l in acc}
        if not scored:
            p(f"\n# {arm}: complete but no parsable accuracies -> REFUSED")
            continue
        # [R.289] deterministic tie-break, IDENTICAL to r237_read.argmax_cell: highest acc,
        # ties to the lexicographically smallest label.  ⛔ Before this, max() picked the first
        # maximal key in iteration order and this file disagreed with the reader on scora's
        # exact tie (ofat-clf1e-2 vs ofat-cosine, both 0.7509025270758123) -- the DELIVERABLE
        # named one config and this block would have RUN THE OTHER.
        _top = max(scored.values())
        _tied = sorted(l for l in scored if scored[l] == _top)
        win = _tied[0]
        if len(_tied) > 1:
            p(f"\n# ⚠️ {arm}: {len(_tied)} cells TIE at the argmax ({', '.join(_tied)});"
              f" the winner is an ARBITRARY pick [R.289]")
        cen = CENTRE.get(arm)
        if arm == "fftstock":
            p(f"\n# {arm}: parity spot-check, not a tuning arm -> no confirmation needed")
            continue
        if cen is None or cen not in jobs:
            p(f"\n# {arm}: centre '{cen}' not in the job list -> REFUSED")
            continue
        if win == cen:
            p(f"\n# {arm}: the CENTRE is already the winner ({scored[cen]:.4f}) "
              f"-> nothing to confirm; the shipped config stands")
            emitted[arm] = None
            continue
        p(f"\n# {arm}: winner {win} ({scored[win]:.4f}) vs centre {cen} ({scored.get(cen, float('nan')):.4f})")
        for lab in (win, cen):
            p(f"#   {lab}")
            p(f"R237C_LABEL={lab} R237C_ARGS='{jobs[lab]}'   # x5 seeds 41-45")
        emitted[arm] = (win, cen)
    if not any(emitted.values()):
        p("\n# nothing to confirm yet -- no arm is both complete and beaten by a non-centre cell")
    return emitted


def selftest():
    ok, bad = [], []
    def chk(n, c, d=""):
        (ok if c else bad).append(n); print(f"  {'PASS' if c else 'FAIL'}  {n}" + (f"  {d}" if d else ""))
    import io, tempfile

    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "csv")); os.makedirs(os.path.join(td, "done"))
        labs = {"scora-lr5e-2": "--A", "scora-lr5e-3": "--B", "scora-lr1.5e-2": "--C"}
        with open(os.path.join(td, "jobs.tsv"), "w") as f:
            for k, v in labs.items():
                f.write(f"{k}\t{v}\n")
        def cell(lab, v, mark=True):
            with open(os.path.join(td, "csv", lab + ".csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["name", "accuracy", "seed"]); w.writeheader()
                w.writerow({"name": lab, "accuracy": v, "seed": 41})
            if mark:
                open(os.path.join(td, "done", lab), "w").close()

        # 1 of 3 done -> refuse
        cell("scora-lr5e-2", 0.7473)
        buf = io.StringIO(); e = generate(td, buf)
        chk("T1 incomplete arm refused", "INCOMPLETE" in buf.getvalue() and "REFUSED" in buf.getvalue())

        # all done, a non-centre cell wins -> emit both configs
        cell("scora-lr5e-3", 0.7600); cell("scora-lr1.5e-2", 0.7500)
        buf = io.StringIO(); e = generate(td, buf)
        chk("T2 complete arm emits winner+centre", e.get("scora") == ("scora-lr5e-3", "scora-lr5e-2"), str(e))
        chk("T3 both configs appear in the output",
            "R237C_LABEL=scora-lr5e-3" in buf.getvalue() and "R237C_LABEL=scora-lr5e-2" in buf.getvalue())

        # centre itself wins -> nothing to confirm
        cell("scora-lr5e-3", 0.7000); cell("scora-lr1.5e-2", 0.7100)
        buf = io.StringIO(); e = generate(td, buf)
        chk("T4 centre-wins -> nothing to confirm",
            e.get("scora") is None and "CENTRE is already the winner" in buf.getvalue())

    print("\n" + "=" * 56); print(f"  selftest: {len(ok)} passed, {len(bad)} failed")
    if bad: print("  FAILED: " + ", ".join(bad))
    print("=" * 56); return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dir", default=GRID); a = ap.parse_args()
    sys.exit(selftest() if a.selftest else (generate(a.dir) and 0))
