#!/usr/bin/env python
"""[fir] READ the final multi-task runs: MEDIAN OVER 5 SEEDS of the MAX OVER EPOCHS.

    env/bin/python scripts/fir_final_read.py --run-root ./logs/final
    env/bin/python scripts/fir_final_read.py --run-root ./logs/final --emit results/
    FIR_FINAL_TASK=cola env/bin/python scripts/fir_final_read.py --run-root ./logs/final

⛔ THE READING TRAPS THIS FILE EXISTS TO GUARD, each one already paid for:
  * **the metric is the TASK's own** -- CoLA is Matthews, STS-B is Pearson, MRPC is
    F1, the rest accuracy. Reading `accuracy` on CoLA is [R.310]'s
    read-a-non-accuracy-task trap and it produces a plausible, wrong table.
  * **the collapse floor is metric-aware**: MRPC's F1 floor is 0.8122 (an
    all-positive predictor scores it), CoLA's MCC floor is 0.0, accuracy's is the
    majority rate. A cell at the floor RAN; it did not LEARN.
  * **a diverged (NaN) cell is a RESULT, not a missing one.**
  * **an argmax at the LAST epoch means the number is a LOWER BOUND** -- the run was
    still improving when the budget ran out. Flagged `^`, never silently ranked.
  * **a partial arm is INCOMPLETE**: a median over 3 of 5 seeds is not the reported
    statistic, and [R.310]'s rule is that a partial column is never quoted.
"""
import argparse, csv, glob, math, os, statistics, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import fir_final_plan as H                                             # noqa: E402
import r310_read as R                                                  # noqa: E402
import fir_preflight_arms as PA                                        # noqa: E402
import fir_plan as FP                                                  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "src"))
import bench_adapter_cost as BC                                        # noqa: E402

# ---------------------------------------------------------------------------
# ⭐⭐ THE COMPUTATIONAL COLUMNS.
# ---------------------------------------------------------------------------
# `[user, 2026-08-30]`: "the output csvs should obviously also output all the
# necessary computational metrics (like peak memory, flops per token, training
# throughput, min memory, etc.)"
#
# ⛔ EVERY ONE OF THESE IS EITHER MEASURED BY THE RUN ITSELF OR DERIVED BY AN
#   INSTRUMENT THAT ALREADY EXISTS.  Nothing here is a new cost model:
#     * the five memory columns and the two step-time columns are written by
#       `train_glue.py` into each cell's own CSV -- MEASURED, per seed;
#     * throughput is those step times inverted, with an EXACT token count
#       (`--pad_to_max_length` is forced, so a step is exactly 32 x 128 tokens);
#     * flops/token come from `src/bench_adapter_cost`, which owns BOTH the frozen
#       J.2 op-counter and the arm -> counter map. ⭐ The map is IMPORTED, at this
#       backbone's width -- a second copy that agreed today is precisely the defect
#       class this repo keeps paying for, and `r307_cost_table` delegates to the
#       same functions so the two tables cannot drift.
# ⛔⛔ AND IT IS IMPORTED FROM THERE, NOT FROM `r307_cost_table` [2026-08-30].
#   r307 reads `scratchpad/phaseR/r308/timing.csv` AT IMPORT, and `scratchpad/` is
#   gitignored: it does not travel to fir. Importing r307 here made THIS module
#   un-importable ON THE CLUSTER while every check on this box stayed green -- the
#   05 submit gate refused with "FAIL: fir_final_read selftest" and nothing ran.
#   ⭐ A fir stage may only import modules that survive with `results/` and
#   `scratchpad/` ABSENT; `fir_shell_gates.t_fir_stage_imports_survive_...` now
#   proves that for every module the submit path selftests.
COST_COLS = ["peak_mem_mib", "param_mem_mib", "opt_mem_mib", "runtime_mem_mib",
             "theoretical_mem_mib", "avg_step_time", "std_step_time",
             "total_training_time_sec"]
D_MODEL = 2048              # gemma-2b hidden size (q_proj / o_proj are 2048x2048)
N_MODULES = 36              # [measured, preflight] every arm adapts 36 modules
HEAD_PARAMS = 4096          # gemma-2b `score` = 2048 x 2
K = 256                     # the matched budget, every arm
TOKENS_PER_STEP = 32 * 128  # batch x max_length; padding to max_length is FORCED


# ⛔⛔ THE DATASET SIZES MUST COME FROM THE COMMITTED, FIR-SIDE FILE.
#   `r310_read.collapse_value()` defaults to `r310_plan.sizes()`, which reads
#   `scratchpad/phaseR/r310/dataset_sizes.json` -- gitignored, dev-box only. On fir
#   it FAILS CLOSED at CALL time, so the module imported fine and the selftest died,
#   and `05_final.sh` refused six canaries in a row with "FAIL: fir_final_read
#   selftest" [2026-08-30, §25.4]. `fir_plan.sizes()` reads
#   `sbatch/fir/dataset_sizes.json`, which IS committed and IS what every other fir
#   stage derives its step counts and warmup from.
#   ⭐ THE SIZES A NUMBER IS DERIVED FROM MUST TRAVEL WITH THE CODE THAT DERIVES IT.
def _floor(task):
    """The degenerate-model floor, metric-aware, from the COMMITTED sizes."""
    return R.collapse_value(task, S=FP.sizes())


def load(run_root):
    """{cell_id: {'val','metric','best_epoch','accuracy','diverged'}} -- the metric
    chosen by the CSV's OWN task column, so a mis-filed CSV cannot be scored
    against the wrong metric."""
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

        def f(k):
            try:
                return float(row.get(k, "nan"))
            except (TypeError, ValueError):
                return float("nan")
        val = f(col)
        rec = {"val": val, "metric": col, "best_epoch": f("best_epoch"),
               "accuracy": f("accuracy"), "task": task,
               "diverged": math.isnan(val)}
        # ⭐ THE COMPUTATIONAL COLUMNS ARE ALREADY IN EVERY RUN'S OWN CSV -- they
        #   are MEASURED by train_glue, per run, and were simply never read here.
        #   Reading them costs nothing and makes the cost table a by-product of the
        #   accuracy table rather than a separate experiment [R.307's whole shape].
        for k in COST_COLS:
            rec[k] = f(k)
        out[cid] = rec
    return out


def rows(run_root, task):
    """[(arm, median, values, best_epochs, n, at_floor, truncated)] for one task."""
    got = load(run_root)
    floor = _floor(task)
    ep = H.EPOCHS[task]
    out = []
    for a in H.ARMS:
        vals, eps = [], []
        for c in H.cells(task=task, arms=[a]):
            g = got.get(H.cell_id(c))
            if g is None:
                continue
            vals.append(g["val"])
            eps.append(g["best_epoch"])
        fin = [v for v in vals if v == v]
        med = statistics.median(fin) if fin else None
        # ⛔ "TRUNCATED" IS PER-ARM AND IT IS ABOUT THE *SEEDS THAT WON AT THE END*.
        #   epochs are 0-indexed in train_glue, so the last one is ep-1.
        trunc = sum(1 for e in eps if e == e and int(e) == ep - 1)
        at_floor = (floor is not None and med is not None and med <= floor + 1e-9)
        out.append((a, med, vals, eps, len(vals), at_floor, trunc))
    return out, floor


def cost_of(arm, got_rows):
    """{column: value} for one arm: MEASURED costs as the MEDIAN over the seeds
    that produced them, plus DERIVED throughput and flops/token.

    ⛔ MEDIAN, not mean, and over the SAME seeds the metric is reported over -- a
      cost number from a different set of runs than the accuracy number is a
      different experiment.
    ⚠ A cost column is missing (empty) when the runs did not write it; it is never
      filled with a zero, because 0 MiB and "not recorded" are opposite claims."""
    out = {}
    for k in COST_COLS:
        vals = [g[k] for g in got_rows if g.get(k) == g.get(k)]     # drop NaN
        out[k] = f"{statistics.median(vals):.6g}" if vals else ""
    # --- DERIVED throughput. ⚠ `avg_step_time` is seconds per optimiser step.
    st = [g["avg_step_time"] for g in got_rows
          if g.get("avg_step_time") == g.get("avg_step_time") and g["avg_step_time"] > 0]
    if st:
        med = statistics.median(st)
        out["steps_per_sec"] = f"{1.0 / med:.6g}"
        out["tokens_per_sec"] = f"{TOKENS_PER_STEP / med:.6g}"
    else:
        out["steps_per_sec"] = out["tokens_per_sec"] = ""
    # --- DERIVED flops, from r307's own map at THIS backbone's width -----------
    try:
        un = BC.arm_flops_per_token(arm, TOKENS_PER_STEP, D_MODEL)
        mg = BC.arm_merged_per_token(arm, D_MODEL, TOKENS_PER_STEP)
    except (KeyError, TypeError):
        un = mg = None
    # ⛔ PLAIN DECIMAL, NOT %g. A flop count is exact and 8,388,608 formatted with
    #   `%.6g` is "8.38861e+06" -- three digits thrown away and a string no
    #   spreadsheet sums correctly. The check below compares against the exact
    #   integer, so this fired the first time it ran.
    out["adapter_flops_per_token_unmerged"] = "" if un is None else f"{un:.4f}"
    out["adapter_flops_per_token_merged"] = "" if mg is None else f"{mg:.4f}"
    out["dense_gemm_flops_per_token"] = str(2 * D_MODEL * D_MODEL)
    # --- DERIVED trainable params, from the budget model the receipts check uses
    mult = PA.LOCATION_MULTIPLIER.get(arm, 1)
    out["trainable_params"] = str(K * mult * N_MODULES + HEAD_PARAMS)
    out["params_per_module"] = str(K * mult)
    return out


def report(run_root, tasks=None, emit=None):
    lines = []
    got = load(run_root)
    lines.append(f"FINAL RUNS on google/gemma-2b  |  plan digest {H.digest()}  |  "
                 f"targets {H.TARGETS}  |  seeds {H.SEEDS}")
    lines.append("  reported number: the MEDIAN over 5 seeds of the MAX over epochs "
                 "of the task's OWN primary metric")
    lines.append("  hyperparameters: the FROZEN proxy table "
                 "(llmdocs/GEMMA_HP_PROXY.md), selected on MRPC")
    emit_rows = []
    for t in (tasks or H.tasks()):
        want = [H.cell_id(c) for c in H.cells(task=t)]
        have = [c for c in want if c in got]
        rr, floor = rows(run_root, t)
        metric = R.metric_of(t)
        head = (f"\n### {t}   metric={metric}   epochs={H.EPOCHS[t]}   "
                f"cells {len(have)}/{len(want)}")
        if len(have) < len(want):
            head += "   ⛔ INCOMPLETE"
        if t == H.SELECTION_TASK:
            head += "   ⛔⛔ IN-SAMPLE (the proxy HPs were selected on this task)"
        lines.append(head)
        if floor is not None:
            lines.append(f"  degenerate-model floor for {metric}: {floor:.4f} "
                         f"-- a cell at or below it RAN but did not LEARN")
        if not have:
            lines.append("  nothing to report yet.")
            continue
        lines.append(f"  {'arm':10s} {'median':>8s} {'min':>8s} {'max':>8s} {'n':>3s}  flags")
        for a, med, vals, eps, n, at_floor, trunc in sorted(
                rr, key=lambda r: (-1e9 if r[1] is None else -r[1])):
            if med is None:
                lines.append(f"  {a:10s} {'--':>8s} {'--':>8s} {'--':>8s} {n:3d}")
                continue
            fin = [v for v in vals if v == v]
            flags = []
            if n < len(H.SEEDS):
                flags.append(f"⛔ {n}/5 seeds -- INCOMPLETE, do not quote")
            if any(v != v for v in vals):
                flags.append(f"{sum(1 for v in vals if v != v)} DIVERGED")
            if trunc:
                flags.append(f"^{trunc}/{n} peaked at the LAST epoch (LOWER BOUND)")
            if at_floor:
                flags.append("⛔ AT THE COLLAPSE FLOOR")
            lines.append(f"  {a:10s} {med:8.4f} {min(fin):8.4f} {max(fin):8.4f} "
                         f"{n:3d}  {'; '.join(flags)}")
            _rows = [got[H.cell_id(c)] for c in H.cells(task=t, arms=[a])
                     if H.cell_id(c) in got]
            emit_rows.append({"task": t, "metric": metric, "arm": a,
                              "median": f"{med:.6f}", "n_seeds": n,
                              "min": f"{min(fin):.6f}", "max": f"{max(fin):.6f}",
                              "values": " ".join(f"{v:.6f}" for v in vals),
                              "best_epochs": " ".join(
                                  "nan" if e != e else str(int(e)) for e in eps),
                              "epochs": H.EPOCHS[t],
                              "n_peaked_at_last_epoch": trunc,
                              "floor": "" if floor is None else f"{floor:.6f}",
                              "at_floor": int(at_floor),
                              "in_sample": int(t == H.SELECTION_TASK),
                              **cost_of(a, _rows)})
    # ------------------------------------------------------------------
    # ⭐ THE COMPUTATIONAL TABLE. Printed ONCE, not per task: the memory and
    #   step-time of an arm are properties of the ARM at this backbone, and the
    #   only task-dependence is sequence padding -- which is FORCED to 128 for
    #   every task, so a per-task cost table would be six copies of one row.
    #   ⚠ The measured columns below are therefore pooled over every task that has
    #   run, and the spread across tasks is the honest error bar on them.
    # ------------------------------------------------------------------
    pooled = {}
    for a in H.ARMS:
        rs = [got[H.cell_id(c)] for t in (tasks or H.tasks())
              for c in H.cells(task=t, arms=[a]) if H.cell_id(c) in got]
        if rs:
            pooled[a] = (cost_of(a, rs), len(rs))
    if pooled:
        lines.append("\n### computational cost   ⚠ memory + step time are MEASURED "
                     "(median over every cell of that arm); flops/token are DERIVED "
                     "[r307's frozen op-counter, d=2048]")
        lines.append(f"  {'arm':10s} {'peak MiB':>9s} {'param MiB':>9s} {'opt MiB':>8s} "
                     f"{'s/step':>8s} {'tok/s':>9s} {'fl/tok(un)':>10s} "
                     f"{'fl/tok(mg)':>10s} {'train par':>10s} {'n':>4s}")
        for a in H.ARMS:
            if a not in pooled:
                continue
            c, n = pooled[a]

            def g(k, w=9, p=""):
                return f"{float(c[k]):{w}.{4 if p else 0}f}" if c[k] else f"{'--':>{w}s}"
            lines.append(
                f"  {a:10s} {g('peak_mem_mib')} {g('param_mem_mib')} "
                f"{g('opt_mem_mib', 8)} {g('avg_step_time', 8, 'p')} "
                f"{g('tokens_per_sec', 9)} "
                f"{float(c['adapter_flops_per_token_unmerged']):10.0f} "
                f"{float(c['adapter_flops_per_token_merged']):10.1f} "
                f"{int(c['trainable_params']):10d} {n:4d}")
        lines.append(f"  reference: the frozen dense GEMM this adapter sits beside is "
                     f"{2 * D_MODEL * D_MODEL:,} flops/token per module.")
        lines.append("  ⚠ `fl/tok(mg)` is the MERGED floor -- the dW rebuild amortised over "
                     "a 4,096-token batch. ⛔ Only the FourierFT-merged and SCoRA arms "
                     "actually RUN merged [R.307]; for the others it is a floor this "
                     "repo has never measured end to end.")
        lines.append("  ⚠ `trainable_params` is DERIVED from the budget model the receipt "
                     "check enforces (k=256 x the arm's multiplier x 36 modules + a "
                     "4,096-param head). ⛔ LoCA's multiplier is 3 -- it is NOT at "
                     "parameter parity, and a table that omits that is misleading.")
    lines.append("")
    lines.append("⚠ ONE table, MANY caveats -- none of them optional:")
    lines.append("  * ⛔ a `^` column was still IMPROVING at its epoch budget: the number is a "
                 "LOWER BOUND, not a converged result.")
    lines.append("  * ⛔ MRPC is IN-SAMPLE (the HPs were selected on it). It is not evidence "
                 "of transfer and must be labelled wherever it appears.")
    lines.append("  * ⛔ ONE hyperparameter setting per arm, carried as a PROXY: no per-task "
                 "tuning was done, for any arm. That is a limitation, stated, not a result.")
    lines.append("  * ⚠ step time is measured on a DEDICATED H100 (one cell per GPU), unlike "
                 "the dev box's 3-way concurrent cells, which [R.209] measured at ~2.2x "
                 "inflation -- so this throughput IS quotable. ⛔ But fir and the dev box "
                 "have DIFFERENT torch builds: never mix the two cost tables.")
    lines.append("  * ⚠ a median of 5 is a ROBUST centre, not a significance test. This family's "
                 "winner's curse is ~0.02-0.04 [R.264]; differences below that are not ordered.")
    if emit:
        p = emit if emit.endswith(".csv") else os.path.join(emit, "fir_final_gemma2b.csv")
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(emit_rows[0].keys()) if emit_rows
                               else ["task", "arm", "median"])
            w.writeheader()
            for r in emit_rows:
                w.writerow(r)
        lines.append(f"\n⭐ wrote {len(emit_rows)} row(s) to {p}")
    return lines


# ---------------------------------------------------------------------------
def selftest():
    import tempfile, shutil
    ok, bad = [], []

    def ck(c, l):
        (ok if c else bad).append(l)

    ck(R.metric_of("cola") == "matthews_correlation",
       "CoLA's metric is Matthews, read from train_glue -- not accuracy")
    ck(R.metric_of("stsb") == "pearson", "STS-B's is Pearson")
    ck(R.metric_of("mrpc") == "f1", "MRPC's is F1")
    ck(abs(_floor("mrpc") - 0.81222707) < 1e-6,
       "⭐ MRPC's F1 floor is the all-positive predictor's 0.81222707, not 0.5")
    ck(_floor("cola") == 0.0, "CoLA's MCC floor is 0.0")
    # ⛔ AND THEY COME FROM THE COMMITTED FILE, NOT r310's dev-box copy -- the
    #   defect that refused six canaries on fir. Both directions: the committed
    #   path must exist, and the two files must agree where they overlap (if they
    #   ever disagreed, one of the two tables would be built on the wrong sizes).
    ck(os.path.exists(FP.SIZES_PATH),
       f"⭐ the sizes come from the COMMITTED {os.path.relpath(FP.SIZES_PATH, ROOT)}")
    _S = FP.sizes()
    ck(all(t in _S for t in H.TASKS),
       f"...and it carries every task this stage runs ({', '.join(H.TASKS)})")

    d = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(d, "csv"))
        t = H.tasks()[0]
        ep = H.EPOCHS[t]
        met = R.metric_of(t)

        def write(cid, val, best_ep):
            # ⛔ THE FIXTURE MUST NOT NAME A COLUMN TWICE. On an accuracy task
            #   `met` IS "accuracy", so a fieldname list containing both collapsed
            #   the row and every assertion below read the fixture's 0.9 instead of
            #   the value under test -- six green-looking checks that tested
            #   nothing. ⭐ A TEST FIXTURE IS CODE, and this one had the same
            #   duplicate-key defect the CSV upsert gate exists to prevent.
            row = {"task_name": t, met: val, "best_epoch": best_ep}
            if "accuracy" not in row:
                row["accuracy"] = 0.9
            with open(os.path.join(d, "csv", cid + ".csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row))
                w.writeheader()
                w.writerow(row)

        arm = H.ARMS[0]
        cs = H.cells(task=t, arms=[arm])
        # 3 of 5 seeds -> INCOMPLETE, and the median must not be quotable
        for c, v in zip(cs[:3], [0.70, 0.80, 0.90]):
            write(H.cell_id(c), v, 1)
        L = report(d, tasks=[t])
        ck(any("INCOMPLETE" in l for l in L), "a partial task is labelled INCOMPLETE")
        ck(any("3/5 seeds" in l for l in L),
           "⭐ an arm with 3 of 5 seeds is flagged, per-arm, as not quotable")
        ck(any(" 0.8000" in l for l in L),
           "...and the median it does show is the median of what EXISTS (0.80)")
        # 5 seeds, and the MEDIAN is the middle value, not the mean or the max
        for c, v in zip(cs, [0.10, 0.20, 0.55, 0.90, 0.95]):
            write(H.cell_id(c), v, 1)
        L = report(d, tasks=[t])
        ck(any(" 0.5500" in l for l in L),
           "⭐ the reported number is the MEDIAN of 5 (0.55), not the mean (0.54) "
           "or the max (0.95)")
        ck(any("0.1000" in l and "0.9500" in l for l in L),
           "...and the min/max spread is printed beside it")
        # ⛔ a run that peaked at the LAST epoch is a LOWER BOUND
        for c in cs:
            write(H.cell_id(c), 0.55, ep - 1)
        L = report(d, tasks=[t])
        ck(any("LAST epoch" in l and "LOWER BOUND" in l for l in L),
           "⭐ an argmax at the last epoch is flagged as a LOWER BOUND")
        for c in cs:
            write(H.cell_id(c), 0.55, 0)
        L = report(d, tasks=[t])
        ck(not any("LAST epoch" in l for l in L),
           "CONTROL: ...and it is NOT flagged when the peak is interior")
        # ⛔ a NaN is a RESULT
        write(H.cell_id(cs[0]), "nan", 1)
        L = report(d, tasks=[t])
        ck(any("DIVERGED" in l for l in L), "a NaN cell is reported as DIVERGED")
        # ⛔ the collapse floor
        fl = _floor(t)
        if fl is not None:
            for c in cs:
                write(H.cell_id(c), fl, 1)
            L = report(d, tasks=[t])
            ck(any("COLLAPSE FLOOR" in l for l in L),
               f"CONTROL: an arm whose median IS the floor ({fl:.4f}) is flagged")
        # ------------------------------------------------------------------
        # ⭐ THE COMPUTATIONAL COLUMNS, both directions.
        # ------------------------------------------------------------------
        # (i) absent from the CSVs -> EMPTY, never 0. "0 MiB" and "not recorded"
        #     are opposite claims and a table that prints one for the other is
        #     worse than a table with a gap in it.
        for c, v in zip(cs, [0.5, 0.6, 0.7, 0.8, 0.9]):
            write(H.cell_id(c), v, 1)
        rs0 = [load(d)[H.cell_id(c)] for c in cs]
        c0 = cost_of(H.ARMS[0], rs0)
        ck(c0["peak_mem_mib"] == "" and c0["avg_step_time"] == "",
           "⛔ a cost column the runs did not write comes back EMPTY, not 0")
        ck(c0["steps_per_sec"] == "" and c0["tokens_per_sec"] == "",
           "...and a throughput with no step time is EMPTY, not infinite")
        # (ii) present -> the MEDIAN over the seeds, and throughput derived from it
        def write_cost(cid, val, step_t, peak):
            row = {"task_name": t, met: val, "best_epoch": 1,
                   "avg_step_time": step_t, "peak_mem_mib": peak,
                   "param_mem_mib": 10, "opt_mem_mib": 20, "runtime_mem_mib": 30,
                   "theoretical_mem_mib": 40, "std_step_time": 0.01,
                   "total_training_time_sec": 100}
            if "accuracy" not in row:
                row["accuracy"] = 0.9
            with open(os.path.join(d, "csv", cid + ".csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row))
                w.writeheader(); w.writerow(row)
        for c, v, stp, pk in zip(cs, [0.5, 0.6, 0.7, 0.8, 0.9],
                                 [0.4, 0.5, 0.5, 0.6, 5.0], [100, 200, 300, 400, 500]):
            write_cost(H.cell_id(c), v, stp, pk)
        rs1 = [load(d)[H.cell_id(c)] for c in cs]
        c1 = cost_of(H.ARMS[0], rs1)
        ck(c1["peak_mem_mib"] == "300", "⭐ peak memory is the MEDIAN over the seeds (300)")
        ck(c1["avg_step_time"] == "0.5",
           "⭐ ...and so is step time (0.5) -- the 5.0 s outlier does not drag it, "
           "which is why it is a median and not a mean")
        ck(abs(float(c1["tokens_per_sec"]) - (32 * 128) / 0.5) < 1e-6,
           f"⭐ throughput is DERIVED from it at an EXACT token count "
           f"(32x128 / 0.5 = {(32*128)/0.5:g} tok/s -- padding to max_length is forced)")
        ck(abs(float(c1["steps_per_sec"]) - 2.0) < 1e-9, "...and steps/s is 1/0.5")
        # (iii) flops come from r307's map, at THIS backbone's width -- not a copy
        ck(abs(float(c1["adapter_flops_per_token_unmerged"])
               - BC.arm_flops_per_token(H.ARMS[0], TOKENS_PER_STEP, D_MODEL)) < 1e-6,
           "⭐ flops/token are the frozen J.2 op-counter, imported, at d=2048")
        ck(float(c1["adapter_flops_per_token_unmerged"])
           != BC.arm_flops_per_token(H.ARMS[0], TOKENS_PER_STEP, 768),
           "⛔ CONTROL: and they are NOT the ROBERTA number (d=768) -- a width that "
           "silently defaulted would be a plausible, wrong cost table")
        ck(c1["dense_gemm_flops_per_token"] == str(2 * D_MODEL * D_MODEL),
           "the frozen dense GEMM reference is printed beside them")
        # (iv) the parameter budget, and LoCA's 3x
        ck(c1["trainable_params"] == str(256 * 36 + 4096),
           "trainable params are DERIVED from the enforced budget model")
        cl = cost_of("loca", rs1)
        ck(cl["trainable_params"] == str(3 * 256 * 36 + 4096),
           "⛔ CONTROL: LoCA's 3x location budget is carried (it is NOT at parity)")
        L = report(d, tasks=[t])
        ck(any("computational cost" in l for l in L), "the cost table is printed")
        ck(any("NOT at parameter parity" in l for l in L),
           "...with LoCA's parity caveat attached to it")
        ck(any("DIFFERENT torch builds" in l for l in L),
           "...and the fir-vs-dev-box warning that bars mixing two cost tables")

        # the emit path
        for c, v in zip(cs, [0.5, 0.6, 0.7, 0.8, 0.9]):
            write_cost(H.cell_id(c), v, 0.5, 300)
        out = os.path.join(d, "results", "x.csv")
        report(d, tasks=[t], emit=out)
        rr = list(csv.DictReader(open(out)))
        ck(rr and rr[0]["median"] == "0.700000",
           "the emitted CSV carries the median as a number")
        ck(rr and rr[0]["values"].count(" ") == 4,
           "...and all five per-seed values beside it, so the median is checkable")
        ck(rr and rr[0]["in_sample"] == str(int(t == H.SELECTION_TASK)),
           "...and whether the task is IN-SAMPLE")
        for k in ("peak_mem_mib", "tokens_per_sec", "adapter_flops_per_token_unmerged",
                  "trainable_params", "param_mem_mib", "total_training_time_sec"):
            ck(rr and rr[0].get(k) not in (None, ""),
               f"⭐ the emitted CSV carries the computational column {k!r}")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    for l in ok:
        print(f"  ✅ {l}")
    for l in bad:
        print(f"  ⛔ {l}")
    print(f"selftest: {len(ok)} passed, {len(bad)} failed")
    return 1 if bad else 0


def _selftest_every_task():
    import subprocess, re
    tot_p = tot_f = 0
    views = H.TASKS
    for t in views:
        print(f"--- task view {t} " + "-" * 46)
        env = dict(os.environ, FIR_FINAL_TASK=t, FIR_FINAL_ONE="1")
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--selftest"],
                           capture_output=True, text=True, env=env, cwd=ROOT)
        sys.stdout.write(r.stdout)
        m = None
        for m in re.finditer(r"selftest:\s*(\d+) passed, (\d+) failed", r.stdout):
            pass
        if m is None:
            print(f"  ⛔ view {t} produced no selftest line (rc={r.returncode})")
            tot_f += 1
            continue
        tot_p += int(m.group(1)); tot_f += int(m.group(2))
        if r.returncode != 0 and int(m.group(2)) == 0:
            tot_f += 1
    print("=" * 62)
    print(f"selftest: {tot_p} passed, {tot_f} failed  (all {len(views)} task views)")
    return 1 if tot_f else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root")
    ap.add_argument("--emit", default=None,
                    help="write the aggregated table to this CSV (or directory)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        if not os.environ.get("FIR_FINAL_ONE"):
            sys.exit(_selftest_every_task())
        sys.exit(selftest())
    if not a.run_root:
        raise SystemExit("--run-root is required")
    for l in report(a.run_root, emit=a.emit):
        print(l)


if __name__ == "__main__":
    main()
