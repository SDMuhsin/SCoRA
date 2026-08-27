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
import re, argparse, csv, glob, math, os, sys

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
            val = float("nan")
        try:
            be = float(row.get("best_epoch", "nan"))
        except (TypeError, ValueError):
            be = float("nan")
        try:
            acc = float(row.get("accuracy", "nan"))
        except (TypeError, ValueError):
            acc = float("nan")
        # ⛔ A NaN METRIC IS A RESULT, NOT AN ABSENCE. The first version skipped it,
        #   so a cell that RAN AND DIVERGED was counted as "missing" -- while its
        #   `done` marker said it succeeded. Two parts of the same report would have
        #   disagreed, and the reading "the sweep did not finish" is the opposite of
        #   the truth, which is "this hyperparameter blows the model up". lr=4.0 is
        #   deliberately on this grid, so this is the expected outcome for some cells.
        out[cid] = {"val": val, "metric": col, "best_epoch": be, "task": task,
                    "accuracy": acc, "diverged": math.isnan(val)}
    return out


def _accuracy_floor():
    """MRPC's majority-class rate -- the accuracy a constant predictor scores.
    Read from the MEASURED dataset sizes, never typed in."""
    S = H.FP.sizes()
    lc = S.get(H.TASK, {}).get("label_counts")
    if not lc:
        return None
    counts = {int(k): v for k, v in lc.items()}
    return max(counts.values()) / sum(counts.values())


def coverage(run_root, arms=None):
    want = [H.cell_id(c) for c in H.cells(arms)]
    got = load(run_root)
    done = {c for c in want if c in got}          # includes diverged cells: they RAN
    failed = {os.path.basename(p) for p in glob.glob(os.path.join(run_root, "fail", "*"))}
    missing = [c for c in want if c not in done]
    return want, got, done, failed, missing


def report(run_root, arms=None, top=15):
    want, got, done, failed, missing = coverage(run_root, arms)
    floor = R.collapse_value(H.TASK)
    lines = []
    lines.append(f"MRPC hyperparameter grid {H.GRID_NAME}  |  digest {H.digest()}  |  "
                 f"{H.TASK} / {H.TARGETS} / {H.EPOCHS} epochs / seed {H.SEED}"
                 + (f"  |  arms {', '.join(arms or H.ARMS)}"))
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
    diverged = sorted(c for c in done if got[c]["diverged"])
    scored = [c for c in done if not got[c]["diverged"]]
    acc_floor = _accuracy_floor()
    lines.append(f"  metric: {metric} (MRPC's own primary metric)"
                 + (f"   collapse floor {floor:.4f}" if floor is not None else ""))
    # ⚠ ON MRPC THE F1 FLOOR IS HIGH: predicting all-positive scores F1 0.8122
    #   because 279/408 eval pairs are positive. So F1 SEPARATES POORLY here --
    #   0.85 is 0.04 above a constant predictor. Accuracy's floor is the 0.6838
    #   majority rate, so it discriminates far better. The RANKING stays on the
    #   task's declared primary metric; accuracy is printed beside it so a cell
    #   near the F1 floor cannot be mistaken for a cell that learned.
    if acc_floor is not None:
        lines.append(f"  ⚠ that floor is HIGH ({floor:.4f}): an all-positive predictor scores it. "
                     f"accuracy is shown too (its floor is {acc_floor:.4f}).")
    if diverged:
        lines.append(f"  ⛔ {len(diverged)} cell(s) RAN AND DIVERGED (no finite {metric}) -- "
                     f"that is a result about the hyperparameter, not a missing cell:")
        for c in diverged[:6]:
            lines.append(f"       {c}")
    dead = [c for c in scored if floor is not None and got[c]["val"] <= floor + 1e-9]
    if dead:
        lines.append(f"  ⚠ {len(dead)}/{len(scored)} scored cells are AT OR BELOW the "
                     f"collapse floor -- they ran, they did not learn.")
    rank = sorted(scored, key=lambda c: -got[c]["val"])
    if not rank:
        lines.append("  no cell produced a finite metric.")
        return lines
    lines.append("")
    lines.append(f"  top {min(top, len(rank))} of {len(scored)}  "
                 f"⚠ ONE SEED: read this as a REGION, not an order")
    pcol = "  P/P_ref" if H.COORD == "p" else ""
    lines.append(f"    {'cell':58s} {metric:>8s} {'acc':>7s}  best_ep{pcol}")
    for c in rank[:top]:
        a = got[c]["accuracy"]
        astr = "    n/a" if math.isnan(a) else f"{a:7.4f}"
        # ⭐ on a P grid the cell id carries the DERIVED lr; the swept coordinate is
        #   P, so print it or the table cannot be read in the coordinate it was
        #   searched in.
        pstr = f"  {H.parse_cell_id(c)['p_mult']:>7g}" if H.COORD == "p" else ""
        lines.append(f"    {c:58s} {got[c]['val']:8.4f} {astr}  "
                     f"{got[c]['best_epoch']:.0f}{pstr}")
    # per-arm best, because the two arms are separate baselines
    for arm in (arms or H.ARMS):
        sub = [c for c in rank if f"-{arm}-" in c]
        if sub:
            lines.append(f"  best {arm:9s}: {got[sub[0]]['val']:.4f}   {sub[0]}")
    # ⭐ EDGE REPORT: an optimum on the boundary means the grid was too narrow, and
    #   that is a finding about the SEARCH, not about the method.
    best = rank[0]
    bc = H.parse_cell_id(best)
    edges, untestable = [], []
    # ⛔ AN AXIS WITH FEWER THAN 3 VALUES HAS NO INTERIOR, so "the optimum is at the
    #   edge" is true by construction there and carries no information. Flagging it
    #   anyway would fire on EVERY run of a 2-value axis and train the reader to skim
    #   past the one warning that matters. Say it is untestable instead.
    # ⛔ THE AXES COME FROM THE GRID, NOT FROM THIS FILE. w1 sweeps P = lr*atom and
    #   DERIVES lr, so lr takes one value per (P, scaling) pair -- 24 of them -- and
    #   an edge test on it would be meaningless. H.axes() is the single declaration.
    for name, key, axis in H.axes():
        val = bc[key]
        if len(set(axis)) < 3:
            untestable.append(f"{name} ({len(set(axis))} values)")
        elif val in (min(axis), max(axis)):
            edges.append(f"{name}={val:g} is at the grid EDGE")
    if edges:
        lines.append("  ⛔ " + "; ".join(edges) +
                     " -- the optimum may lie OUTSIDE this grid. Widen before quoting it.")
    else:
        lines.append("  ✅ the best cell is INTERIOR on every testable axis "
                     "(the grid brackets it)")
    if untestable:
        lines.append(f"  ⚠ no interior to test on: {', '.join(untestable)} -- an axis with "
                     f"<3 values cannot be bracketed, by construction.")
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
        ck(any(f"coverage: 2/{len(cs)}" in l for l in L),
           "coverage is counted and printed first")   # ⚠ from the planner, not a literal
        ck(any(H.cell_id(cs[1]) in l for l in L), "the finished cells are still ranked")
        ck(any("ONE SEED" in l for l in L), "the one-seed caveat is printed with the table")

        # the WRONG-COLUMN control: f1 and accuracy differ, the reader must use f1
        ck(load(d)[H.cell_id(cs[1])]["val"] == 0.85, "reads the F1 column, not accuracy")

        # the collapse floor fires
        write(H.cell_id(cs[2]), R.collapse_value("mrpc"))
        L = report(d)
        ck(any("collapse floor" in l for l in L), "the metric-aware floor is reported")
        ck(any("did not learn" in l for l in L), "CONTROL: a floor-value cell is flagged")

        # ⛔ A DIVERGED CELL (NaN metric) IS A RESULT, NOT A MISSING CELL
        with open(os.path.join(d, "csv", H.cell_id(cs[4]) + ".csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["task_name", "f1", "accuracy", "best_epoch"])
            w.writeheader(); w.writerow({"task_name": "mrpc", "f1": "nan",
                                         "accuracy": "nan", "best_epoch": 1})
        L = report(d)
        ck(any("RAN AND DIVERGED" in l for l in L), "a NaN cell is reported as DIVERGED")
        ck(any(H.cell_id(cs[4]) in l for l in L), "the diverged cell is NAMED")
        ck(not any(f"missing e.g.: {H.cell_id(cs[4])}" in l for l in L),
           "CONTROL: it is NOT also counted as missing (done/ says it ran)")
        ck(any("its floor is" in l for l in L), "the accuracy floor is printed beside F1's")

        # a failure marker is surfaced
        open(os.path.join(d, "fail", H.cell_id(cs[3])), "w").write("exit=1 elapsed=9s")
        L = report(d)
        ck(any("recorded a FAILURE" in l for l in L), "failed cells are named, not dropped")

        # EDGE control: a winner on the boundary must be called out...
        write(H.cell_id(cs[0]), 0.99)                      # cs[0] is lr min, sc min, clr min
        ck(any("grid EDGE" in l for l in report(d)), "CONTROL: an edge optimum is flagged")
        # ...and an interior winner must NOT be. ⚠ Interior is computed FROM THE GRID
        #   (a hardcoded point stopped existing the day the grid was replaced), and
        #   only axes with >=3 values have an interior at all.
        def _mid(axis):
            vs = sorted(set(axis))
            return vs[len(vs) // 2] if len(vs) >= 3 else vs[0]
        want = {k: _mid(v) for _n, k, v in H.axes()}
        interior = next(c for c in cs if all(c[k] == v for k, v in want.items()))
        write(H.cell_id(interior), 0.999)
        L = report(d)
        ck(any("INTERIOR on every testable axis" in l for l in L),
           "CONTROL: an interior optimum is NOT flagged (the check can pass)")
        if len(set(H.CLF_LRS)) < 3:
            ck(any("no interior to test on" in l for l in L),
               "a <3-value axis is declared untestable, not silently flagged as an edge")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    for l in ok:
        print(f"  ✅ {l}")
    for l in bad:
        print(f"  ⛔ {l}")
    print(f"selftest: {len(ok)} passed, {len(bad)} failed")
    return 1 if bad else 0



def _selftest_every_grid():
    """⭐ RUN THE CHECKS FOR EVERY GRID, NOT JUST THE SELECTED ONE.

    ⛔ Module globals are bound to FIR_HP_GRID at IMPORT, so one process can only
      ever check one grid -- and `run_all_gates.py` sets no env var, so for two
      grids the suite was green while a *different* grid's checks had never run in
      it. That is this repo's Law 1 in miniature: a check must exercise what the
      job will actually run. Re-exec once per grid and aggregate.
    """
    import subprocess
    if os.environ.get("FIR_HP_GRID"):
        print(f"  ⚠ FIR_HP_GRID={os.environ['FIR_HP_GRID']} is set; checking ALL grids anyway.")
    tot_p = tot_f = 0
    for g in sorted(H.GRIDS):
        print(f"--- grid {g} " + "-" * 50)
        env = dict(os.environ, FIR_HP_GRID=g, FIR_HP_ONE_GRID="1")
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--selftest"],
                           capture_output=True, text=True, env=env,
                           cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        m = None
        for m in re.finditer(r"selftest:\s*(\d+) passed, (\d+) failed", r.stdout):
            pass
        if m is None:
            print(f"  ⛔ grid {g} produced no selftest line (rc={r.returncode})")
            tot_f += 1
            continue
        tot_p += int(m.group(1)); tot_f += int(m.group(2))
        if r.returncode != 0 and int(m.group(2)) == 0:
            tot_f += 1     # ⛔ fail closed: a crash after a green line is still a failure
    print("=" * 62)
    print(f"selftest: {tot_p} passed, {tot_f} failed  (all {len(H.GRIDS)} grids)")
    return 1 if tot_f else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root")
    ap.add_argument("--arm", default=None)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        # ⛔ EVERY GRID, ALWAYS -- deliberately IGNORING FIR_HP_GRID. The first
        #   version skipped the fan-out when that var was set, so an operator who
        #   had exported a grid for a submit got a suite that silently covered ONE
        #   grid instead of three, with a smaller green total and no warning. A
        #   check must not cover LESS because of an unrelated environment variable.
        if not os.environ.get("FIR_HP_ONE_GRID"):
            sys.exit(_selftest_every_grid())
        sys.exit(selftest())
    if not a.run_root:
        raise SystemExit("FAIL CLOSED: --run-root is required")
    for l in report(a.run_root, [a.arm] if a.arm else None, a.top):
        print(l)


if __name__ == "__main__":
    main()
