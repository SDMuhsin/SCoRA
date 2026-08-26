#!/usr/bin/env python
"""[fir] Read the MRPC hyperparameter grid.  Reports COVERAGE before it reports a winner.

⛔⛔ THE TWO WAYS THIS TABLE CAN LIE, both guarded:
  1. READING THE WRONG COLUMN.  MRPC's primary metric is **F1**, not accuracy
     (`train_glue._METRIC_FOR_TASK`, parsed -- never copied).  A cell that trained
     fine still writes an `accuracy` value, so the failure would be silent.
  2. RANKING A GRID WITH HOLES.  A sweep with 40 dead cells still prints a
     best-of, and it looks exactly like a complete one.  Coverage is printed
     FIRST, every missing cell is named, and the header says INCOMPLETE until the
     grid is whole.

⚠ ONE SEED.  [R.273]'s null gives the seed-to-seed sigma on this family; adjacent
  cells inside that band are NOT ordered by this grid.  The output prints the top
  cells as a REGION, and says so.  Confirming a point needs seeds -- a separate spend.

Usage:
    env/bin/python scripts/fir_hp_read.py --run-root <dir> [--arm fftm] [--top 15]
    env/bin/python scripts/fir_hp_read.py --selftest
"""
import argparse, csv, glob, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import fir_hp_plan as H                                                # noqa: E402
import r310_read as R                                                  # noqa: E402  (metric map + floor)


def load(run_root):
    """{cell_id: {'val','metric','best_epoch'}} -- the metric chosen by the CSV's
    OWN task column, so a mis-filed CSV cannot be scored against the wrong metric."""
    out = {}
    for p in sorted(glob.glob(os.path.join(run_root, "csv", "*.csv"))):
        cid = os.path.basename(p)[:-4]
        try:
            rows = list(csv.DictReader(open(p)))
        except Exception:
            continue
        if not rows:
            continue
        row = rows[-1]
        task = (row.get("task_name") or "").strip()
        col = R.metric_of(task)
        try:
            val = float(row.get(col, "nan"))
        except (TypeError, ValueError):
            continue
        if math.isnan(val):
            continue
        try:
            be = float(row.get("best_epoch", "nan"))
        except (TypeError, ValueError):
            be = float("nan")
        out[cid] = {"val": val, "metric": col, "best_epoch": be, "task": task}
    return out


def coverage(run_root, arms=None):
    want = [H.cell_id(c) for c in H.cells(arms)]
    got = load(run_root)
    done = {c for c in want if c in got}
    failed = {os.path.basename(p) for p in glob.glob(os.path.join(run_root, "fail", "*"))}
    missing = [c for c in want if c not in done]
    return want, got, done, failed, missing


def report(run_root, arms=None, top=15):
    want, got, done, failed, missing = coverage(run_root, arms)
    floor = R.collapse_value(H.TASK)
    lines = []
    lines.append(f"MRPC hyperparameter grid  |  digest {H.digest()}  |  "
                 f"{H.TASK} / {H.TARGETS} / {H.EPOCHS} epochs / seed {H.SEED}")
    lines.append(f"  coverage: {len(done)}/{len(want)} cells have a result   "
                 f"failed: {len(failed)}   missing: {len(missing)}")
    if missing:
        lines.append("  ⛔ INCOMPLETE -- the ranking below is over the cells that FINISHED.")
        shown = [m for m in missing[:8]]
        lines.append(f"     missing e.g.: {', '.join(shown)}"
                     + (f"  (+{len(missing)-len(shown)} more)" if len(missing) > len(shown) else ""))
        if failed:
            lines.append(f"     of those, {len(failed)} recorded a FAILURE -- read "
                         f"{os.path.join(run_root, 'fail')}/<cell>")
    if not done:
        lines.append("  nothing to rank yet.")
        return lines
    metric = got[next(iter(done))]["metric"]
    lines.append(f"  metric: {metric} (MRPC's own primary metric)"
                 + (f"   collapse floor {floor:.4f}" if floor is not None else ""))
    dead = [c for c in done if floor is not None and got[c]["val"] <= floor + 1e-9]
    if dead:
        lines.append(f"  ⚠ {len(dead)}/{len(done)} finished cells are AT OR BELOW the "
                     f"collapse floor -- they ran, they did not learn.")
    rank = sorted(done, key=lambda c: -got[c]["val"])
    lines.append("")
    lines.append(f"  top {min(top, len(rank))} of {len(done)}  "
                 f"⚠ ONE SEED: read this as a REGION, not an order")
    lines.append(f"    {'cell':58s} {metric:>8s}  best_ep")
    for c in rank[:top]:
        lines.append(f"    {c:58s} {got[c]['val']:8.4f}  {got[c]['best_epoch']:.0f}")
    # per-arm best, because the two arms are separate baselines
    for arm in (arms or H.ARMS):
        sub = [c for c in rank if f"-{arm}-" in c]
        if sub:
            lines.append(f"  best {arm:9s}: {got[sub[0]]['val']:.4f}   {sub[0]}")
    # ⭐ EDGE REPORT: an optimum on the boundary means the grid was too narrow, and
    #   that is a finding about the SEARCH, not about the method.
    best = rank[0]
    bc = H.parse_cell_id(best)
    edges = []
    if bc["lr"] in (min(H.LRS), max(H.LRS)):
        edges.append(f"lr={bc['lr']:g} is at the grid EDGE")
    if bc["scaling"] in (min(H.SCALINGS), max(H.SCALINGS)):
        edges.append(f"scaling={bc['scaling']:g} is at the grid EDGE")
    if bc["classifier_lr"] in (min(H.CLF_LRS), max(H.CLF_LRS)):
        edges.append(f"classifier_lr={bc['classifier_lr']:g} is at the grid EDGE")
    if edges:
        lines.append("  ⛔ " + "; ".join(edges) +
                     " -- the optimum may lie OUTSIDE this grid. Widen before quoting it.")
    else:
        lines.append("  ✅ the best cell is INTERIOR on every axis (the grid brackets it)")
    return lines


# ---------------------------------------------------------------------------
def selftest():
    import tempfile, shutil
    ok, bad = [], []

    def ck(c, l):
        (ok if c else bad).append(l)

    ck(R.metric_of("mrpc") == "f1", "MRPC's primary metric is F1, read from train_glue")
    ck(R.metric_of("mrpc") != "accuracy", "CONTROL: it is NOT accuracy (the silent-wrong-column trap)")

    d = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(d, "csv")); os.makedirs(os.path.join(d, "fail"))
        cs = H.cells()

        def write(cid, f1, acc=0.9, ep=3):
            with open(os.path.join(d, "csv", cid + ".csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["task_name", "f1", "accuracy", "best_epoch"])
                w.writeheader()
                w.writerow({"task_name": "mrpc", "f1": f1, "accuracy": acc, "best_epoch": ep})

        # a partial grid must say INCOMPLETE and must still rank what it has
        write(H.cell_id(cs[0]), 0.80)
        write(H.cell_id(cs[1]), 0.85)
        L = report(d)
        ck(any("INCOMPLETE" in l for l in L), "a partial grid is labelled INCOMPLETE")
        ck(any("coverage: 2/160" in l for l in L), "coverage is counted and printed first")
        ck(any(H.cell_id(cs[1]) in l for l in L), "the finished cells are still ranked")
        ck(any("ONE SEED" in l for l in L), "the one-seed caveat is printed with the table")

        # the WRONG-COLUMN control: f1 and accuracy differ, the reader must use f1
        ck(load(d)[H.cell_id(cs[1])]["val"] == 0.85, "reads the F1 column, not accuracy")

        # the collapse floor fires
        write(H.cell_id(cs[2]), R.collapse_value("mrpc"))
        L = report(d)
        ck(any("collapse floor" in l for l in L), "the metric-aware floor is reported")
        ck(any("did not learn" in l for l in L), "CONTROL: a floor-value cell is flagged")

        # a failure marker is surfaced
        open(os.path.join(d, "fail", H.cell_id(cs[3])), "w").write("exit=1 elapsed=9s")
        L = report(d)
        ck(any("recorded a FAILURE" in l for l in L), "failed cells are named, not dropped")

        # EDGE control: a winner on the boundary must be called out...
        write(H.cell_id(cs[0]), 0.99)                      # cs[0] is lr min, sc min, clr min
        ck(any("grid EDGE" in l for l in report(d)), "CONTROL: an edge optimum is flagged")
        # ...and an interior winner must NOT be
        interior = next(c for c in cs if c["lr"] == 0.5 and c["scaling"] == 142
                        and c["classifier_lr"] == 5e-3)
        write(H.cell_id(interior), 0.999)
        ck(any("INTERIOR on every axis" in l for l in report(d)),
           "CONTROL: an interior optimum is NOT flagged (the check can pass)")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    for l in ok:
        print(f"  ✅ {l}")
    for l in bad:
        print(f"  ⛔ {l}")
    print(f"selftest: {len(ok)} passed, {len(bad)} failed")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.run_root:
        raise SystemExit("FAIL CLOSED: --run-root is required")
    for l in report(a.run_root, [a.arm] if a.arm else None, a.top):
        print(l)


if __name__ == "__main__":
    main()
