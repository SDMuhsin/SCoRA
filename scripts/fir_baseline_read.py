#!/usr/bin/env python
"""
STAGE 06's READER — read the baseline SEARCH and write the PROXY.

  FIR_BASE_TASK=mrpc env/bin/python scripts/fir_baseline_read.py --run-root <root>
  FIR_BASE_TASK=mrpc env/bin/python scripts/fir_baseline_read.py --run-root <root> --write-proxy
  env/bin/python scripts/fir_baseline_read.py --selftest

⛔⛔ WHY THIS FILE EXISTS, AND WHAT IT REPLACES.
   Until now stage 06 had NO reader -- `06_baseline.sh --status` says so in as many
   words ("stage 06 has no dedicated reader yet"). The consequence was a round trip
   baked into the protocol: run the 12-cell search, rsync the CSVs off the cluster,
   have someone READ them, EDIT `fir_baseline_plan.PROXY` by hand, commit, push,
   pull on the cluster, and only then submit the 30 final cells. Two of this
   project's most expensive incidents were a fix that was pushed to a ref nothing
   pulls and a cluster running older code than the log implied; a protocol with a
   mandatory hand-edit-and-push in the middle of it invites both.

   ⇒ The winner is read ON THE CLUSTER and written to a FILE. `fir_baseline_plan`
     loads that file. Nothing is hand-edited and nothing has to travel.

⛔ THE FILE IS NOT A SHORTCUT PAST THE RULES. It is written only when:
     * EVERY search cell is present -- selecting on a partial ladder is selecting
       on whichever cells happened to finish first;
     * every cell read is at the SEARCH SEED -- picking on all five seeds and then
       reporting those same five is selection on the test set;
     * the winner is an ACTUAL CELL of the declared ladder (its lr is in LRS and
       its batch in BATCHES), so a corrupt or hand-written file fails closed.
   And `--write-proxy` REFUSES to overwrite an existing proxy: the value is
   measured once and then it is history. Use --force only to redo a search you
   have deliberately discarded, and bump nothing silently (FIR_SETUP Law 8).
"""
import argparse, csv, glob, json, math, os, socket, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import fir_baseline_plan as H                                          # noqa: E402
import r310_read as R                                                  # noqa: E402


def proxy_path():
    """⚠ ONE PROXY PER CLUSTER STAGE DIRECTORY, not one per repo. The value is
    selected from cells measured on a particular machine; a second cluster's
    search must not silently inherit it."""
    d = os.environ.get("LRS_STAGE_DIR", "sbatch/fir")
    return os.path.join(ROOT, d, "baseline_proxy.json")


def read_search(run_root):
    """{cell_id: record} for the SEARCH cells only, scored on the task's OWN metric.

    ⛔ The metric comes from the CSV's own `task_name`, never from the filename: a
    mis-filed CSV then fails to score rather than scoring against the wrong column.
    """
    want = {H.cell_id(c): c for c in H.cells(task=H.SELECTION_TASK, stage="search")}
    out = {}
    for p in sorted(glob.glob(os.path.join(run_root, "csv", "*.csv"))):
        cid = os.path.basename(p)[:-4]
        if cid not in want:
            continue
        try:
            rows = list(csv.DictReader(open(p)))
        except Exception:
            continue
        if not rows:
            continue
        row = rows[-1]
        task = (row.get("task_name") or "").strip()
        col = R.metric_of(task)

        def f(k):
            try:
                return float(row.get(k, "nan"))
            except (TypeError, ValueError):
                return float("nan")

        seed_in_csv = (row.get("seed") or "").strip()
        out[cid] = {"val": f(col), "metric": col, "task": task,
                    "best_epoch": f("best_epoch"), "peak_mem_mib": f("peak_mem_mib"),
                    "seed_in_csv": seed_in_csv, "cell": want[cid],
                    "diverged": math.isnan(f(col))}
    return out


def show(run_root):
    recs = read_search(run_root)
    cells = H.cells(task=H.SELECTION_TASK, stage="search")
    metric = R.metric_of(H.SELECTION_TASK)
    print(f"STAGE 06 SEARCH — task {H.SELECTION_TASK}  metric {metric}  "
          f"seed {H.SEARCH_SEED}  epochs {H.EPOCHS[H.SELECTION_TASK]}")
    print(f"  run root : {run_root}")
    print(f"  cells    : {len(recs)} / {len(cells)} present")
    print()
    hdr = "  bs    " + "".join(f"{lr:>10g}" for lr in H.LRS)
    print(hdr)
    for b in H.BATCHES:
        line = f"  {b:<6}"
        for lr in H.LRS:
            c = {"arm": H.ARM, "task": H.SELECTION_TASK, "lr": lr, "batch": b,
                 "seed": H.SEARCH_SEED, "epochs": H.EPOCHS[H.SELECTION_TASK]}
            r = recs.get(H.cell_id(c))
            line += "  --------" if r is None else f"{r['val']:>10.4f}"
        print(line)
    print()
    return recs, cells


def pick(run_root, strict=True):
    """The winner, with every reason it could be refused stated as a refusal."""
    recs = read_search(run_root)
    cells = H.cells(task=H.SELECTION_TASK, stage="search")
    problems = []
    if len(recs) != len(cells):
        missing = [H.cell_id(c) for c in cells if H.cell_id(c) not in recs]
        problems.append(f"the ladder is INCOMPLETE: {len(recs)}/{len(cells)} cells. "
                        f"Missing: {', '.join(missing[:6])}"
                        f"{' ...' if len(missing) > 6 else ''}")
    bad_seed = [c for c, r in recs.items()
                if r["seed_in_csv"] and r["seed_in_csv"] != str(H.SEARCH_SEED)]
    if bad_seed:
        problems.append(f"⛔ {len(bad_seed)} cell(s) are NOT at the search seed "
                        f"{H.SEARCH_SEED}: {bad_seed[:3]}")
    live = {c: r for c, r in recs.items() if not r["diverged"]}
    if not live:
        problems.append("every cell read as NaN — nothing to select from")
    if problems and strict:
        return None, problems
    if not live:
        return None, problems
    best_id = max(live, key=lambda c: live[c]["val"])
    best = live[best_id]
    cell = best["cell"]
    # ⛔ THE WINNER MUST BE A CELL OF THE DECLARED LADDER. Cheap, and it is what
    #   makes a corrupted or hand-made file fail closed instead of steering 30 cells.
    if cell["lr"] not in H.LRS or cell["batch"] not in H.BATCHES:
        problems.append(f"⛔ the winning cell {best_id} is not on the declared ladder")
        return None, problems
    # ⚠ EDGE WARNING, not a refusal. [R.298]/[R.258]: an optimum on the boundary of
    #   the swept box means the box may be in the wrong place, and this project has
    #   already shipped one proxy whose winner sat at the lowest rung of both axes.
    edges = []
    if cell["lr"] in (min(H.LRS), max(H.LRS)):
        edges.append(f"lr {cell['lr']:g} is the {'lowest' if cell['lr'] == min(H.LRS) else 'highest'} rung")
    if cell["batch"] in (min(H.BATCHES), max(H.BATCHES)):
        edges.append(f"batch {cell['batch']} is the {'smallest' if cell['batch'] == min(H.BATCHES) else 'largest'} rung")
    best["edges"] = edges
    best["cell_id"] = best_id
    return best, problems


def write_proxy(run_root, force=False):
    best, problems = pick(run_root, strict=True)
    if problems:
        print("⛔ REFUSING to write the proxy:")
        for p in problems:
            print(f"   - {p}")
        print("   A proxy selected from a partial or mis-seeded ladder steers all 30")
        print("   final cells. Finish the search (re-running the stage IS the recovery")
        print("   procedure -- it submits only cells with no done marker) and re-read.")
        return 1
    path = proxy_path()
    if os.path.exists(path) and not force:
        print(f"⛔ REFUSING: {path} already exists.")
        print("   The proxy is MEASURED ONCE and then it is history. Overwriting it")
        print("   silently re-points every final cell at a different operating point")
        print("   while the cells already run keep the old one -- two protocols in one")
        print("   column. If you have deliberately discarded that search, pass --force.")
        print("   --- the existing proxy ---")
        print("   " + open(path).read().replace("\n", "\n   "))
        return 1
    cell = best["cell"]
    payload = {
        "lr": cell["lr"], "batch": cell["batch"],
        "selection_task": H.SELECTION_TASK, "search_seed": H.SEARCH_SEED,
        "metric": best["metric"], "value": best["val"],
        "cell_id": best["cell_id"], "best_epoch": best["best_epoch"],
        "n_cells": len(H.cells(task=H.SELECTION_TASK, stage="search")),
        "edges": best["edges"],
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(), "run_root": run_root,
        "_comment": ("Written by scripts/fir_baseline_read.py from measured CSVs. "
                     "DO NOT EDIT BY HAND: fir_baseline_plan asserts this (lr, batch) "
                     "is an actual cell of the declared ladder and refuses otherwise."),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"✅ wrote {path}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if best["edges"]:
        print()
        print("⚠⚠ THE WINNER IS ON AN EDGE OF THE LADDER: " + "; ".join(best["edges"]))
        print("   The optimum may lie outside the swept box, so this proxy is a")
        print("   BOUND on that axis, not a maximum. [R.298] It must be stated")
        print("   wherever the row appears -- the RoBERTa fp16 baseline has exactly")
        print("   this defect and carries exactly this note.")
    return 0


def selftest():
    ok, bad = [], []

    def ck(c, m):
        (ok if c else bad).append(m)

    import tempfile
    metric = R.metric_of(H.SELECTION_TASK)
    ck(metric == "f1", f"the selection task's metric is read from train_glue ({metric})")

    cells = H.cells(task=H.SELECTION_TASK, stage="search")
    ck(len(cells) == len(H.LRS) * len(H.BATCHES),
       f"the search is {len(cells)} cells on one task")

    def fixture(tmp, subset, vals, seed=None, task=None):
        os.makedirs(os.path.join(tmp, "csv"), exist_ok=True)
        for c, v in zip(subset, vals):
            cid = H.cell_id(c)
            with open(os.path.join(tmp, "csv", cid + ".csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["task_name", "seed", metric,
                                                   "best_epoch", "peak_mem_mib"])
                w.writeheader()
                w.writerow({"task_name": task or H.SELECTION_TASK,
                            "seed": str(seed if seed is not None else c["seed"]),
                            metric: f"{v}", "best_epoch": "3",
                            "peak_mem_mib": "38311"})

    # --- the happy path
    with tempfile.TemporaryDirectory() as tmp:
        vals = [0.80 + 0.001 * i for i in range(len(cells))]
        vals[5] = 0.95                                     # a unique, interior-ish winner
        fixture(tmp, cells, vals)
        best, probs = pick(tmp)
        ck(not probs, f"a COMPLETE ladder reads with no problems ({probs})")
        ck(best is not None and best["cell_id"] == H.cell_id(cells[5]),
           "the winner is the argmax cell")

    # --- ⛔ CONTROL: a partial ladder must REFUSE
    with tempfile.TemporaryDirectory() as tmp:
        fixture(tmp, cells[:5], [0.9] * 5)
        best, probs = pick(tmp)
        ck(best is None and any("INCOMPLETE" in p for p in probs),
           "⛔ CONTROL: an INCOMPLETE ladder is refused, not silently selected from")

    # --- ⛔ CONTROL: cells at the wrong seed must REFUSE
    with tempfile.TemporaryDirectory() as tmp:
        fixture(tmp, cells, [0.9] * len(cells), seed=43)
        best, probs = pick(tmp)
        ck(best is None and any("search seed" in p for p in probs),
           "⛔ CONTROL: cells not at the search seed are refused "
           "(selecting on the reported seeds is selection on the test set)")

    # --- ⛔ CONTROL: an all-NaN ladder must REFUSE rather than pick a NaN
    with tempfile.TemporaryDirectory() as tmp:
        fixture(tmp, cells, ["nan"] * len(cells))
        best, probs = pick(tmp)
        ck(best is None and any("NaN" in p for p in probs),
           "⛔ CONTROL: an all-diverged ladder is refused (a NaN-passing max reducer "
           "is a defect this repo has already shipped once)")

    # --- ⛔ CONTROL: write_proxy refuses to clobber
    with tempfile.TemporaryDirectory() as tmp:
        vals = [0.80 + 0.001 * i for i in range(len(cells))]
        fixture(tmp, cells, vals)
        d = os.path.join(tmp, "stage")
        os.makedirs(d)
        old = os.environ.get("LRS_STAGE_DIR")
        os.environ["LRS_STAGE_DIR"] = os.path.relpath(d, ROOT)
        try:
            ck(write_proxy(tmp) == 0, "a first --write-proxy succeeds")
            ck(write_proxy(tmp) == 1,
               "⛔ CONTROL: a SECOND --write-proxy REFUSES rather than re-pointing "
               "cells that have already run")
            ck(write_proxy(tmp, force=True) == 0, "--force overrides deliberately")
            got = json.load(open(os.path.join(d, "baseline_proxy.json")))
            ck(got["lr"] in H.LRS and got["batch"] in H.BATCHES,
               "the written proxy is an ACTUAL cell of the declared ladder")
            ck(got["search_seed"] == H.SEARCH_SEED, "the proxy records its search seed")
        finally:
            if old is None:
                os.environ.pop("LRS_STAGE_DIR", None)
            else:
                os.environ["LRS_STAGE_DIR"] = old

    # --- ⭐ the edge warning must FIRE (Law 7: verify a gate fires)
    with tempfile.TemporaryDirectory() as tmp:
        vals = [0.0] * len(cells)
        edge = [c for c in cells if c["lr"] == min(H.LRS) and c["batch"] == min(H.BATCHES)][0]
        vals[cells.index(edge)] = 0.99
        fixture(tmp, cells, vals)
        best, _ = pick(tmp)
        ck(best is not None and len(best["edges"]) == 2,
           "⛔ CONTROL: a winner at the corner of the box raises BOTH edge warnings")

    for m in ok:
        print(f"  ✅ {m}")
    for m in bad:
        print(f"  ⛔ {m}")
    print(f"\nselftest: {len(ok)} passed, {len(bad)} failed")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default=None)
    ap.add_argument("--write-proxy", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.run_root:
        return print("--run-root is required (the stage-06 sweep root)") or 2
    show(a.run_root)
    if a.write_proxy:
        return write_proxy(a.run_root, force=a.force)
    best, probs = pick(a.run_root)
    if probs:
        print("⚠ not ready to write a proxy:")
        for p in probs:
            print(f"   - {p}")
        return 1
    print(f"WINNER: {best['cell_id']}   {best['metric']}={best['val']:.4f}")
    print(f"  -> lr {best['cell']['lr']:g}  batch {best['cell']['batch']}")
    if best["edges"]:
        print("  ⚠ ON AN EDGE: " + "; ".join(best["edges"]))
    print("\nwrite it with:  --write-proxy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
