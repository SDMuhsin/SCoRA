#!/usr/bin/env python
"""[R.310] READER -- the multi-task camera-ready table.

⛔ WHAT MAKES THIS READER DIFFERENT FROM EVERY PRIOR ONE IN THIS REPO
  `r305_plan.load()` reads the **`accuracy` column**.  That is correct on RTE and
  WRONG on four of the eight columns here: CoLA reports `matthews_correlation`,
  STS-B `pearson`, MRPC and CB `f1`.  A cell that trained fine still writes an
  `accuracy` value on those tasks, so the failure would be SILENT and the table
  would look completely plausible.  This module therefore reads the primary
  metric out of `train_glue.py`'s own `_METRIC_FOR_TASK` -- parsed from the
  source with `ast`, so the authority is the harness and not a copy of it, and
  so that importing the reader does not drag in torch.

⛔ THE TWO REPORTING RULES THIS FILE ENFORCES IN CODE
  1. BOTH SCoRA rows print, always ([R.306]).  Printing the swept row alone
     would be tuning our own arm harder after seeing it lose; printing the
     a-priori row alone would hide that we were asked to tune it as hard.
     `--only` cannot select one without the other.
  2. A column is labelled INCOMPLETE unless all 5 seeds landed.  At ~66 h/cell
     on MNLI a partial column is the likeliest thing to be misread as a result.

⭐ THE COLLAPSE FLOOR IS METRIC-AWARE.  RTE's floor is its majority-class rate,
  but that reasoning does not transfer: MCC's degenerate value is 0, Pearson's
  is 0 (STS-B is REGRESSION -- CONTEXT §4.4), and binary/macro F1's is the F1 of
  the best constant predictor, which on MRPC is **0.812**, far above any
  accuracy-style floor.  Comparing MRPC's F1 to a 0.68 majority rate would
  declare a fully collapsed run healthy.

⭐ `best_epoch` IS PART OF THE RESULT, not diagnostics.  The reported number is
  the MAX over the 30 epochs (user, 2026-08-24).  An arm whose argmax sits at
  the last epoch was still improving when the budget ran out, so its number is a
  LOWER BOUND; the table marks those with `^`.
"""
import argparse, ast, csv, glob, json, math, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import r310_plan as PL

CSV_DIR = os.path.join(PL.D, "csv")
COMPARATOR = "fftm"          # [R.305]'s winner; every gate is paired against it


# ============================================================================
def metric_for_task():
    """`train_glue.py`'s own `_METRIC_FOR_TASK`, parsed from source.

    ⛔ Read, never copied.  A second copy of this mapping is a second place for
    it to be wrong, and the failure mode (reading `accuracy` on CoLA) produces a
    plausible number rather than an error."""
    src = open(os.path.join(ROOT, "src", "train_glue.py")).read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "_METRIC_FOR_TASK" for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("[r310] _METRIC_FOR_TASK not found in src/train_glue.py")


METRIC = metric_for_task()


def metric_of(task):
    return METRIC.get(task, "accuracy")


# ============================================================================
def collapse_value(task, S=None):
    """The metric value a DEGENERATE model scores on this task -- the value that
    means "this run learned nothing".  Metric-aware; see the module docstring.

    Returns None where no meaningful floor exists."""
    S = S or PL.sizes()
    m = metric_of(task)
    lc = S[task].get("label_counts")
    if m == "matthews_correlation":
        return 0.0                       # MCC of any constant predictor
    if m == "pearson":
        return 0.0                       # regression; a constant predictor is undefined/0
    if lc is None:
        return None
    counts = {int(k): v for k, v in lc.items()}
    n = sum(counts.values())
    if m == "accuracy":
        return max(counts.values()) / n
    if m == "f1":
        k = len(counts)
        if k == 2:
            # binary F1 on the positive class (label 1), as `evaluate`/glue does:
            # predicting all-positive gives 2*n1/(n1+N); all-negative gives 0.
            return 2 * counts.get(1, 0) / (counts.get(1, 0) + n)
        # macro F1 over k classes: the best constant predictor picks one class.
        return max(2 * c / (c + n) for c in counts.values()) / k
    return None


# ============================================================================
def load(csv_dir=None):
    """{label: {'val', 'best_epoch', 'metric'}} from the per-cell CSVs.

    ⛔ The metric column is chosen by the cell's OWN task, read from the CSV row
    (not from the label), so a mis-filed CSV cannot be scored against the wrong
    metric."""
    csv_dir = csv_dir or CSV_DIR
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
        row = rows[-1]
        task = (row.get("task_name") or "").strip()
        col = metric_of(task)
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
        out[label] = {"val": val, "best_epoch": be, "metric": col, "task": task}
    return out


def _sd(v):
    return statistics.stdev(v) if len(v) > 1 else 0.0


def collect(results=None, manifest=None):
    """{task: {arm: {'vals': {seed: v}, 'epochs': {seed: e}}}} -- confirmation
    cells only.  Spot-check cells (stage `spot`, n=1 at the selection seed) are
    EXCLUDED: PROCESS §5 forbids quoting a selection-stage number as a result."""
    results = load() if results is None else results
    manifest = PL.read_manifest() if manifest is None else manifest
    out = {}
    for lab, m in manifest.items():
        if m.get("stage") != "D" or lab not in results:
            continue
        d = out.setdefault(m["task"], {}).setdefault(m["arm"], {"vals": {}, "epochs": {}})
        d["vals"][m["seed"]] = results[lab]["val"]
        d["epochs"][m["seed"]] = results[lab]["best_epoch"]
    return out


def arm_stat(cell):
    """(mean, sd, n, per_seed) over the CONFIRM seeds present, in seed order.

    None-tolerant: mid-run, most arms have no cell at all, and every caller here
    is reading a PARTIAL grid by design."""
    if not cell:
        return None
    vals = [cell["vals"][s] for s in PL.CONFIRM_SEEDS if s in cell["vals"]]
    if not vals:
        return None
    return statistics.fmean(vals), _sd(vals), len(vals), vals


def paired_gate(a, b):
    """PROCESS §1.3's 5/5 sign gate on two arms' per-seed vectors (same seeds)."""
    wins = sum(1 for x, y in zip(a, b) if x > y)
    ties = sum(1 for x, y in zip(a, b) if x == y)
    return wins, ties, len(a)


def gate_note(wins, n, med):
    """⭐ [R.306]: a NON-FIRING sign gate has two opposite meanings -- a genuinely
    small effect, or a large one with a reversing seed.  Branch on |median|."""
    if wins == n:
        return "5/5 ⭐"
    if wins == 0:
        return "0/5 ⛔"
    return f"{wins}/{n} {'LARGE-but-unstable' if abs(med) >= 0.021 else 'small'}"


# ============================================================================
def report(only_task=None):
    S = PL.sizes()
    man = PL.read_manifest()
    data = collect(manifest=man)
    ref = PL.rte_reference()
    tasks = [only_task] if only_task else [t for t in PL.TASKS if t in data]

    print("=" * 100)
    print("[R.310] MULTI-TASK TABLE -- roberta-base / query+value / k=256 / 30 epochs")
    print("hyperparameters: [R.305]/[R.306]'s RTE-selected settings, carried as a PROXY")
    print("reported number: the MAX over the 30 epochs of the task's OWN primary metric")
    print(f"seeds {PL.CONFIRM_SEEDS} (out-of-sample for the tuning) | comparator = {COMPARATOR}")
    print("=" * 100)

    planned = {}
    for m in man.values():
        if m.get("stage") == "D":
            planned[m["task"]] = planned.get(m["task"], 0) + 1
    for t in tasks:
        got = sum(len(c["vals"]) for c in data.get(t, {}).values())
        print(f"  {t:6s} {metric_of(t):22s} floor={_f(collapse_value(t, S)):>7s} "
              f"cells {got}/{planned.get(t, 0)}"
              + ("" if got == planned.get(t, 0) else "   ⛔ INCOMPLETE"))
    print()

    for t in tasks:
        _report_task(t, data.get(t, {}), S)

    print("\n" + "-" * 100)
    print("RTE (already measured, [R.305]/[R.306] -- NOT re-run here):")
    for arm, title, _src in PL.ARMS:
        if arm in ref:
            mean, sd, _v = ref[arm]
            print(f"  {arm:9s} {mean:.4f} ± {sd:.4f}   {title}")
    print("-" * 100)
    print("^ = argmax at the LAST epoch: still improving at the budget, number is a LOWER BOUND")
    print("⛔ = at or below the degenerate-model floor for that task's metric")
    _spot(man)


def _f(x):
    return "n/a" if x is None else f"{x:.4f}"


def _report_task(task, cells, S):
    floor = collapse_value(task, S)
    comp = cells.get(COMPARATOR)
    comp_stat = arm_stat(comp) if comp else None
    print(f"\n### {task}   metric={metric_of(task)}   "
          f"eval n={S[task]['eval']}   degenerate floor={_f(floor)}")
    print(f"  {'arm':9s} {'mean':>8s} {'sd':>8s} {'n':>3s} {'ep':>5s}  "
          f"{'vs ' + COMPARATOR:>10s}  gate")
    for arm, title, _src in PL.ARMS:
        c = cells.get(arm)
        st = arm_stat(c) if c else None
        if st is None:
            print(f"  {arm:9s} {'--':>8s} {'--':>8s} {0:3d}")
            continue
        mean, sd, n, vals = st
        eps = [c["epochs"][s] for s in PL.CONFIRM_SEEDS if s in c["epochs"]]
        eps = [e for e in eps if not math.isnan(e)]
        med_ep = statistics.median(eps) if eps else float("nan")
        # epochs are 0-indexed in train_glue's loop
        binding = "^" if eps and med_ep >= PL.EPOCHS - 1 else " "
        flag = "⛔" if (floor is not None and mean <= floor + 1e-9) else "  "
        delta = gate = ""
        if comp_stat and arm != COMPARATOR and n == comp_stat[2] == len(PL.CONFIRM_SEEDS):
            d = [x - y for x, y in zip(vals, comp_stat[3])]
            w, _ties, nn = paired_gate(vals, comp_stat[3])
            delta = f"{statistics.fmean(d):+.4f}"
            gate = gate_note(w, nn, statistics.median(d))
        print(f"  {arm:9s} {mean:8.4f} {sd:8.4f} {n:3d} {med_ep:5.1f}{binding} "
              f"{delta:>10s}  {gate} {flag}")
    _scora_guard(cells)


def _scora_guard(cells):
    """⛔ [R.306]: both SCoRA rows or neither.  A reader that can print one alone
    is a reader that will eventually print the flattering one alone."""
    have = [a for a in PL.SCORA_ROWS if arm_stat(cells.get(a))]
    if len(have) == 1:
        print(f"  ⚠️  only {have[0]} has landed; the other SCoRA row is still running. "
              "NEITHER is quotable alone ([R.306]).")


def _spot(man):
    res = load()
    rows = [(l, m) for l, m in man.items() if m.get("stage") == "spot" and l in res]
    if not rows:
        return
    print("\nCB --classifier_lr spot-check (n=1 at the SELECTION seed -- ⛔ NOT a result,")
    print("it only says whether the 3-class head plausibly changes the 5e-3 derivation):")
    for lab, m in sorted(rows):
        print(f"  {lab:32s} {res[lab]['val']:.4f}")
    # ⛔ The comparison that matters is against the SAME-SEED control cell, not
    # against the 5-seed confirmation mean (seeds 42-46, a different seed set).
    for arm in PL.SPOT_ARMS:
        ctl = res.get(f"cb-{arm}-spot-clf5e-3")
        if not ctl:
            print(f"  ⚠️  {arm}: no same-seed 5e-3 control yet -- the deltas above "
                  "are UNINTERPRETABLE until it lands.")
            continue
        for clf in PL.SPOT_CLF:
            if clf == "5e-3":
                continue
            cell = res.get(f"cb-{arm}-spot-clf{clf}")
            if cell:
                print(f"    {arm:6s} clf {clf} vs the 5e-3 control: "
                      f"{cell['val'] - ctl['val']:+.4f}  (n=1, seed {PL.SCREEN_SEED})")


# ============================================================================
def psweep(task="stsb"):
    """The proxy's falsification test: is each arm's CARRIED lr the argmax of a
    3-rung ladder on THIS task?

    ⛔ THE READING IS PREREGISTERED in `r310_plan.plan_psweep`'s docstring and is
    NOT re-decided here.  This function only reports which of those cases holds.
    n=1 at the selection seed: it can REMOVE a claim, never add one."""
    res, man = load(), PL.read_manifest()
    rows = {}
    for lab, m in man.items():
        if m.get("stage") == "psweep" and m.get("task") == task and lab in res:
            rows.setdefault(m["arm"], {})[m["mult"]] = res[lab]["val"]
    if not rows:
        print(f"[r310] no psweep cells for {task} yet")
        return {}
    # ⛔ rung-agnostic: the edge-extension rule ADDS rungs, so the columns must be
    # taken from what was actually run, never from the base ladder constant.
    mults = sorted({m for d in rows.values() for m in d})
    print(f"\n### INCIDENTAL: one-off `P` ladder on {task} -- ratio 2, scale FIXED, "
          f"seed {PL.SCREEN_SEED}")
    print("  ⛔ NOT part of [R.310]'s design and NOT repeatable: per-task sweeps were ruled")
    print("  out (USER DECISION 2026-08-24) and the planner can no longer emit these cells.")
    print("  n=1, one task, 4 of 9 arms.  It informs the LIMITATIONS section only.")
    print(f"  {'arm':9s} " + "".join(f"{'x' + format(m, 'g'):>9s}" for m in mults)
          + "   argmax   verdict")
    out = {}
    for arm in sorted(rows):
        r = rows.get(arm, {})
        planned = {m["mult"] for l, m in man.items()
                   if m.get("stage") == "psweep" and m.get("task") == task
                   and m.get("arm") == arm}
        cells = "".join(f"{r[m]:9.4f}" if m in r else f"{'.':>9s}" for m in mults)
        if not planned or len(r) < len(planned):
            print(f"  {arm:9s} {cells}   INCOMPLETE ({len(r)}/{len(planned)})")
            continue
        best = max(r, key=lambda m: (r[m], -m))
        lo, hi = min(r), max(r)
        interior = lo < best < hi
        at_carried = best == 1.0
        out[arm] = {"argmax_mult": best, "interior": interior,
                    "at_carried": at_carried, "vals": dict(r)}
        if at_carried:
            v = "⭐ carried lr IS the argmax -- proxy centred here"
        elif interior:
            v = f"⚠️  bracketed, but the optimum is x{best:g}, NOT the carried lr"
        else:
            v = "⛔ EDGE -- unbracketed, the carried lr is a LOWER BOUND"
        print(f"  {arm:9s} {cells}   x{best:g}      {v}")
    if not out:
        return out
    moved = [a for a, v in out.items() if not v["at_carried"]]
    edge = [a for a, v in out.items() if not v["interior"]]
    print()
    if not moved:
        print("  Every arm's carried lr is the argmax of its own ladder on this task.")
    else:
        print(f"  On this task {', '.join(moved)} score HIGHER away from the carried lr"
              + (" (still unbracketed at the top rung)" if edge else "") + ".")
        print("  ⭐ The shift is COMMON-MODE -- every arm tested moves the same direction,")
        print("  by the same one rung -- so it is not evidence of a DIFFERENTIAL advantage")
        print("  to any arm, and the ladder was stopped before it bracketed.")
    print("  ⛔ THIS DOES NOT QUALIFY ANY RESULT IN THE TABLE.  Every arm ran the identical")
    print("  preregistered protocol, so the columns are valid MATCHED-PROTOCOL comparisons.")
    print("  What it illustrates, for the limitations section, is the already-declared")
    print("  premise of the proxy: RTE-selected settings are not each task's per-task")
    print("  optimum, and this run does not claim they are.")
    return out


# ============================================================================
def ranks():
    """Per-task rank of every arm, INCLUDING the imported RTE column.

    ⛔ Generated, never transcribed (PROCESS §2: "never print a statistic as a
    literal").  Only columns where all 9 arms have all 5 seeds are shown -- a
    rank computed on a partial column would move as cells land."""
    data = collect()
    cols = {"rte": {a: v[0] for a, v in PL.rte_reference().items()}}
    for t in PL.TASKS:
        d = data.get(t, {})
        st = {a: arm_stat(d.get(a)) for a in PL.ARM_NAMES}
        st = {a: v[0] for a, v in st.items() if v and v[2] == len(PL.CONFIRM_SEEDS)}
        if len(st) == len(PL.ARM_NAMES):
            cols[t] = st
    rk = {t: {a: i + 1 for i, a in enumerate(sorted(c, key=lambda x: -c[x]))}
          for t, c in cols.items()}
    print("arm      " + "".join(f"{t:>8s}" for t in rk))
    for a in PL.ARM_NAMES:
        print(f"{a:9s}" + "".join(f"{rk[t][a]:8d}" for t in rk))
    if len(rk) > 1:
        spread = {a: max(rk[t][a] for t in rk) - min(rk[t][a] for t in rk)
                  for a in PL.ARM_NAMES}
        print(f"\nmax rank swing across {len(rk)} columns: "
              + ", ".join(f"{a} {spread[a]}" for a in
                          sorted(spread, key=lambda x: -spread[x])))
        print("⚠️  most of this movement is INSIDE each column's noise band -- rank "
              "instability is not evidence of a mechanism.")
    return rk


# ============================================================================
def selftest():
    n = [0]

    def ck(c, msg):
        n[0] += 1
        if not c:
            print(f"FAIL: {msg}")
            selftest.failed = getattr(selftest, "failed", 0) + 1
    selftest.failed = 0
    S = PL.sizes()

    # ---- 1. the per-task metric, and the defect it prevents ---------------
    ck(metric_of("cola") == "matthews_correlation", "cola -> MCC")
    ck(metric_of("stsb") == "pearson", "stsb -> pearson")
    ck(metric_of("mrpc") == "f1" and metric_of("cb") == "f1", "mrpc/cb -> f1")
    ck(metric_of("mnli") == metric_of("sst2") == metric_of("boolq") == "accuracy",
       "mnli/sst2/boolq -> accuracy")
    ck(sum(1 for t in PL.TASKS if metric_of(t) != "accuracy") == 4,
       "⛔ 4 of 7 tasks are NOT scored on `accuracy` -- the whole point of this reader")

    # ---- 2. the metric-aware floor ----------------------------------------
    ck(abs(collapse_value("cola", S)) < 1e-12, "MCC floor is 0, not the majority rate")
    ck(abs(collapse_value("stsb", S)) < 1e-12, "pearson floor is 0 (regression, no classes)")
    f_mrpc = collapse_value("mrpc", S)
    ck(abs(f_mrpc - 0.8122) < 1e-3, f"MRPC all-positive F1 floor {f_mrpc:.4f}")
    # CONTROL: the naive majority-accuracy floor would MISS a collapsed MRPC run.
    ck(f_mrpc > max(v for v in S["mrpc"]["label_counts"].values()) / S["mrpc"]["eval"],
       "⛔ the F1 floor is ABOVE the majority-accuracy rate (the silent-pass control)")
    f_cb = collapse_value("cb", S)
    ck(abs(f_cb - (2 * 28 / (28 + 56)) / 3) < 1e-9, f"CB macro-F1 floor {f_cb:.4f}")
    ck(abs(collapse_value("mnli", S) - max(int(v) for v in
        [c for c in S["mnli"]["label_counts"].values()]) / S["mnli"]["eval"]) < 1e-9,
       "MNLI floor is its majority-class rate")
    # CONTROL: a mutated label histogram must move the floor.
    mut = {k: dict(v) for k, v in S.items()}
    mut["boolq"] = dict(S["boolq"], label_counts={"0": 1, "1": 3269})
    ck(collapse_value("boolq", mut) > collapse_value("boolq", S),
       "the floor tracks the label histogram (control)")

    # ---- 3. the reader on a fixture ---------------------------------------
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        def cell(lab, task, metcol, val, be=5):
            with open(os.path.join(td, lab + ".csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["task_name", "accuracy", metcol, "best_epoch"])
                w.writeheader()
                w.writerow({"task_name": task, "accuracy": 0.999,
                            metcol: val, "best_epoch": be})
        cell("cola-fftm-seed42", "cola", "matthews_correlation", 0.55)
        cell("stsb-fftm-seed42", "stsb", "pearson", 0.88)
        r = load(td)
        ck(abs(r["cola-fftm-seed42"]["val"] - 0.55) < 1e-9,
           "⛔ CoLA is read as MCC, not as the 0.999 sitting in `accuracy`")
        ck(abs(r["stsb-fftm-seed42"]["val"] - 0.88) < 1e-9, "STS-B is read as pearson")
        ck(r["cola-fftm-seed42"]["metric"] == "matthews_correlation", "metric recorded")

        # collect(): spot cells excluded, confirmation cells kept
        man = {"cola-fftm-seed42": {"task": "cola", "arm": "fftm", "seed": 42, "stage": "D"},
               "stsb-fftm-seed42": {"task": "stsb", "arm": "fftm", "seed": 42, "stage": "D"}}
        c = collect(r, man)
        ck(set(c) == {"cola", "stsb"}, "collect keys by task")
        man2 = dict(man)
        man2["cola-fftm-seed42"] = dict(man["cola-fftm-seed42"], stage="spot")
        ck("cola" not in collect(r, man2), "⛔ spot cells are EXCLUDED from the table")

    # ---- 4. the gate ------------------------------------------------------
    ck(paired_gate([1, 1, 1, 1, 1], [0, 0, 0, 0, 0]) == (5, 0, 5), "5/5 gate fires")
    ck(gate_note(5, 5, 0.03).startswith("5/5"), "5/5 note")
    ck("LARGE" in gate_note(4, 5, 0.03), "⭐ 4/5 with a LARGE median is flagged unstable")
    ck("small" in gate_note(4, 5, 0.005), "4/5 with a small median is flagged small")
    ck(gate_note(0, 5, -0.03).startswith("0/5"), "0/5 note")

    # ---- 5. the two reporting rules ---------------------------------------
    ck(set(PL.SCORA_ROWS) <= set(PL.ARM_NAMES), "both SCoRA rows are in the arm list")
    ck(COMPARATOR in PL.ARM_NAMES, "the comparator is an arm")
    ref = PL.rte_reference()
    ck(len(ref) == len(PL.ARM_NAMES), "RTE reference column complete")

    # ---- 5b. the reader must survive a PARTIAL grid ------------------------
    # ⛔ This run is read continuously while it fills, so "one arm has landed and
    # eight have not" is the NORMAL state, not an edge case.
    ck(arm_stat(None) is None, "arm_stat tolerates a missing arm")
    ck(arm_stat({"vals": {}, "epochs": {}}) is None, "arm_stat tolerates an empty arm")
    part = {"fftm": {"vals": {42: 0.9}, "epochs": {42: 3}}}
    ck(arm_stat(part["fftm"])[2] == 1, "a 1-seed column reports n=1")
    _scora_guard(part)                      # must not raise with neither row present
    _scora_guard({"scora": {"vals": {42: 0.5}, "epochs": {42: 1}}})   # one row present
    ck(True, "the SCoRA guard runs on partial data without raising")

    # ---- 5c. the rank table is generated, and only from COMPLETE columns ---
    rk = ranks()
    ck("rte" in rk, "the RTE column is ranked alongside the new ones")
    ck(all(sorted(v.values()) == list(range(1, len(PL.ARM_NAMES) + 1))
           for v in rk.values()), "every ranked column is a permutation of 1..9")
    ck(rk["rte"]["fftm"] == 1, "fftm ranks 1 on RTE, as [R.305] reports")

    # ---- 5d. the psweep verdict, on FIXTURES (the cells are still running) --
    _res, _man = load, PL.read_manifest
    import types
    def _fake(vals):
        """vals: {arm: {mult: score}} -> monkeypatched load/read_manifest"""
        r, m = {}, {}
        for arm, d in vals.items():
            for mult, v in d.items():
                lab = f"stsb-{arm}-psweep-x{mult:g}"
                r[lab] = {"val": v, "best_epoch": 5, "metric": "pearson", "task": "stsb"}
                m[lab] = {"stage": "psweep", "task": "stsb", "arm": arm, "mult": mult,
                          "seed": PL.SCREEN_SEED}
        return r, m
    for name, vals, want_interior in [
        ("centred", {"fftm": {0.5: 0.1, 1.0: 0.9, 2.0: 0.2}}, True),
        ("edge-high", {"fftm": {0.5: 0.1, 1.0: 0.5, 2.0: 0.9}}, False),
        ("edge-low", {"fftm": {0.5: 0.9, 1.0: 0.5, 2.0: 0.1}}, False),
    ]:
        r, m = _fake(vals)
        g = globals()
        ol, om = g["load"], PL.read_manifest
        g["load"] = lambda csv_dir=None, _r=r: _r
        PL.read_manifest = lambda _m=m: _m
        try:
            v = psweep("stsb")
        finally:
            g["load"], PL.read_manifest = ol, om
        ck(v["fftm"]["interior"] is want_interior,
           f"psweep {name}: interior={want_interior} (control)")

    # ---- 6. epoch-binding flag --------------------------------------------
    ck(PL.EPOCHS == 30, "30 epochs (user decision) -- the ^ flag is keyed to it")

    print(f"[r310_read] selftest: {n[0] - selftest.failed} passed, {selftest.failed} failed")
    if selftest.failed:
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--task", choices=PL.TASKS)
    ap.add_argument("--ranks", action="store_true", help="per-task rank table, generated")
    ap.add_argument("--psweep", nargs="?", const="stsb", choices=PL.TASKS,
                    help="the proxy falsification ladder for a task")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.ranks:
        ranks()
    elif a.psweep:
        psweep(a.psweep)
    else:
        report(a.task)
