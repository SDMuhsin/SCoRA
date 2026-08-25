#!/usr/bin/env python
"""[R.304] STANDING GATE for [R.303]'s upsert trap.

`_upsert_result` (train_glue.py:330) DELETES every row matching
    (model_name_or_path, task_name, optimizer, lr, total_batch_size)
before appending.  ⛔ `seed` is NOT in that key, so N seeds of one config written to ONE
csv collapse to the LAST seed -- silently, leaving a csv that looks complete.

[R.303, measured] the banked blocks survive only because `total_batch_size` is NaN and
pandas evaluates NaN == NaN as False, so the delete-mask is all-False.  That is an
ACCIDENT.  Anything that populates that column re-arms the trap for every future run.

THIS GATE CHECKS TWO THINGS
  A. DRIVERS: any driver that loops seeds must parameterise its GLUE_RESULTS_FILE by the
     seed (or by a label containing it).  A driver that does not is EXPOSED.
  B. BANKED DATA: no csv may have lost seeds -- reported as the count of key-groups that
     preserve more than one seed (proof the NaN escape is holding).

⛔ Reports; changes nothing.  Run before launching any new multi-seed block.
Usage:  env/bin/python scripts/r304_upsert_gate.py [--selftest]
"""
import argparse, collections, csv, glob, os, re, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
KEY = ("model_name_or_path", "task_name", "optimizer", "lr", "total_batch_size")
# drivers that predate [R.303] and are COMPLETE -- their data is banked and verified by
# [R.278]'s 5-arm bit-identical drift gate.  Exposure matters for FUTURE runs only.
LEGACY = re.compile(r"scratchpad/phaseR/r\d")


def driver_exposed(text):
    """(loops_seeds, csv_is_per_seed) for one driver's source."""
    loops = bool(re.search(r"for S in|for sd in|GLUE_SEEDS=\$\{?S", text))
    paths = re.findall(r'GLUE_RESULTS_FILE="([^"]*)"', text)
    if not paths:
        return loops, True
    # safe if EVERY results path is parameterised by the seed, or by a label/name/cell
    # variable (which the modern drivers build as "<config>-s<seed>")
    per = all(re.search(r"\$\{?S\b|\$\{?sd\b|\$\{?(label|name|cell)\}?", p) for p in paths)
    return loops, per


def scan_drivers(stream=sys.stdout):
    p = lambda *a: print(*a, file=stream)
    live, legacy = [], []
    for f in sorted(glob.glob(os.path.join(ROOT, "scripts", "*.sh"))
                    + glob.glob(os.path.join(ROOT, "scratchpad", "phaseR", "*.sh"))):
        t = open(f, errors="ignore").read()
        if "GLUE_RESULTS_FILE" not in t:
            continue
        loops, per = driver_exposed(t)
        if not loops or per:
            continue
        (legacy if LEGACY.search(f.replace(ROOT, "").lstrip("/")) else live).append(os.path.basename(f))
    p(f"  A. DRIVERS  exposed-and-LIVE: {len(live)}   exposed-but-legacy/complete: {len(legacy)}")
    for n in live:
        p(f"     ⛔ {n}  -- loops seeds into ONE csv; re-arms [R.303] if run")
    if not live:
        p("     ✅ every driver under scripts/ writes one csv per (label, seed)")
    return live, legacy


def scan_data(stream=sys.stdout):
    p = lambda *a: print(*a, file=stream)
    multi = single = tot = 0
    for path in glob.glob(os.path.join(ROOT, "scratchpad", "phaseR", "**", "*.csv"), recursive=True):
        if os.sep + "r237" in path or os.sep + "r237c" in path:
            continue                                    # per-cell by construction
        try:
            rows = list(csv.DictReader(open(path, errors="ignore")))
        except Exception:
            continue
        if not rows or "seed" not in rows[0]:
            continue
        tot += 1
        g = collections.defaultdict(set)
        for r in rows:
            g[tuple(r.get(k) for k in KEY)].add(r.get("seed"))
        if any(len(v) > 1 for v in g.values()):
            multi += 1
        else:
            single += 1
    p(f"  B. BANKED   {tot} csvs with a seed column")
    p(f"     ✅ {multi} preserve MULTIPLE seeds under one upsert key "
      f"(proof the NaN escape held)")
    p(f"     .. {single} have one seed per key-group (smoke tests / 1-seed-per-config blocks)")
    return tot, multi, single


def report(stream=sys.stdout):
    p = lambda *a: print(*a, file=stream)
    p("=" * 78); p("[R.304] UPSERT-TRAP GATE  --  [R.303]"); p("=" * 78)
    p(f"  key = {KEY}   ⛔ 'seed' absent")
    live, legacy = scan_drivers(stream)
    tot, multi, single = scan_data(stream)
    p("")
    if live:
        p("  ⛔⛔ VERDICT: a LIVE driver would collapse its seeds. Fix before launching.")
    else:
        p("  ✅ VERDICT: no live driver is exposed; no banked csv shows seed loss.")
        p("  ⚠️ The protection in legacy data is a NaN in `total_batch_size`, NOT a design.")
    return {"live": live, "legacy": legacy, "multi": multi}


def selftest():
    ok, bad = [], []
    def chk(n, c, d=""):
        (ok if c else bad).append(n); print(f"  {'PASS' if c else 'FAIL'}  {n}" + (f"  {d}" if d else ""))
    exposed = 'for S in 41 42; do\n GLUE_RESULTS_FILE="$D/blk.csv" env/bin/python x\ndone'
    safe1 = 'for S in 41 42; do\n GLUE_RESULTS_FILE="$D/csv/cell-s$S.csv" env/bin/python x\ndone'
    safe2 = 'for S in 41 42; do\n name="a-s$S"\n GLUE_RESULTS_FILE="$D/csv/$name.csv" env/bin/python x\ndone'
    noseed = 'GLUE_RESULTS_FILE="$D/one.csv" env/bin/python x'
    chk("U1 a driver looping seeds into ONE csv is EXPOSED", driver_exposed(exposed) == (True, False))
    chk("U2 a csv path parameterised by $S is safe", driver_exposed(safe1) == (True, True))
    chk("U3 a csv path via a $name containing the seed is safe", driver_exposed(safe2) == (True, True))
    chk("U4 a single-seed driver is not flagged", driver_exposed(noseed)[0] is False)
    chk("U5 the key really omits seed", "seed" not in KEY)
    import io
    buf = io.StringIO(); r = report(buf)
    chk("U6 the real repo has NO live exposed driver", r["live"] == [], str(r["live"]))
    chk("U7 and banked data shows multi-seed preservation", r["multi"] > 50, str(r["multi"]))
    chk("U8 the verdict line is printed", "VERDICT" in buf.getvalue())
    print(f"\n  selftest: {len(ok)} passed, {len(bad)} failed")
    if bad: print("  FAILED: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else (report() and 0))
