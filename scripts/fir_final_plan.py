#!/usr/bin/env python
"""[fir] THE FINAL MULTI-TASK RUNS on gemma-2b, at the FROZEN proxy hyperparameters.

`[user, 2026-08-30]`: "run full experiments on baselines and SCoRA (and SCoRA2)
with these hyperparameters for tasks MRPC, CoLA, STSB, RTE, SST-2 and QNLI ...
track the best task performance value out of all epochs and median of 5 values.
Parallelize across seeds AND tasks. We need canaries for these too."

  9 arms x 6 tasks x 5 seeds = 270 cells.  ONE cell = one (arm, task, seed).

⭐ WHY ONE SEED PER CELL AND NOT train_glue's OWN Mo5.  `src/train_glue.py` can run
  five seeds itself and log the median row -- that is the "Mo5" protocol it was
  written for.  Splitting the seeds into separate cells gives the SAME number (a
  median of five per-seed maxima either way) while making the seed axis
  parallelisable, which is what was asked for.  The median is then taken by
  `fir_final_read.py`, over cells.

⛔⛔ THE HYPERPARAMETERS ARE TYPED CONSTANTS, AND THEY MUST BE.  They are a MEASURED
  selection (the best cell of each arm's search) and measured results live in
  gitignored `logs/`, which does not travel to fir.  A planner that read them there
  would emit an EMPTY table on the cluster, silently -- exactly the
  `r310_plan.selected_args()` trap.  So each row is typed here WITH THE CELL ID IT
  CAME FROM, and the selftest asserts that id is a real cell of that arm's search
  and that our command reproduces that cell's swept flags EXACTLY.  That assert is
  what fires if a grid is ever edited.

Usage:
    env/bin/python scripts/fir_final_plan.py --selftest
    FIR_FINAL_TASK=rte env/bin/python scripts/fir_final_plan.py --list
    FIR_FINAL_TASK=rte env/bin/python scripts/fir_final_plan.py --show
    env/bin/python scripts/fir_final_plan.py --cmd <cell-id>
"""
import argparse, hashlib, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import fir_arms as FA                                                  # noqa: E402
import fir_plan as FP                                                  # noqa: E402
import fir_hp_plan as HP                                               # noqa: E402

TARGETS = "q_o"
SEEDS = [42, 43, 44, 45, 46]          # [R.310]'s seeds, unchanged
ARMS = ["fftm", "fftstock", "wave1", "wave2", "loca", "qwha", "lyra",
        "scora", "scora2"]

# ---------------------------------------------------------------------------
# ⭐⭐ THE FROZEN PROXY HYPERPARAMETERS.  Human record + justification:
#   `llmdocs/GEMMA_HP_PROXY.md`.  Search: `llmdocs/FIR_GEMMA_PORT.md` §23.
# ---------------------------------------------------------------------------
#   `cell`  the SEARCH cell this row is, verbatim.  Not decoration: the selftest
#           resolves it against `fir_hp_plan` and refuses a row that is not a cell
#           the search actually ran.
#   `lr` / `scaling` / `clf_lr`  exactly as the winning cell ran them.  ⚠ `:g` is
#           how `_set_flag` formats a flag, so these ARE the values the command
#           carried (qwha's scale really was `294.611`, not the full float).
#   `extra` the 4th-axis knob, for the one arm that has one.
SELECTED = {
    "fftm":     {"lr": 1.5,       "scaling": 400.0,     "clf_lr": 5e-4,
                 "cell": "mrpc-fftm-q_o-lr1p5-sc400-clr0p0005-seed42"},
    "fftstock": {"lr": 1.0,       "scaling": 400.0,     "clf_lr": 5e-4,
                 "cell": "mrpc-fftstock-q_o-lr1-sc400-clr0p0005-seed42"},
    "wave1":    {"lr": 0.5,       "scaling": 1200.0,    "clf_lr": 5e-4,
                 "cell": "mrpc-wave1-q_o-lr0p5-sc1200-clr0p0005-seed42"},
    "wave2":    {"lr": 0.64,      "scaling": 600.0,     "clf_lr": 2e-3,
                 "cell": "mrpc-wave2-q_o-lr0p64-sc600-clr0p002-seed42"},
    "loca":     {"lr": 0.013125,  "scaling": 4.0,       "clf_lr": 2e-3,
                 "cell": "mrpc-loca-q_o-lr0p013125-sc4-clr0p002-seed42"},
    "qwha":     {"lr": 2.25013,   "scaling": 294.611,   "clf_lr": 5e-3,
                 "cell": "mrpc-qwha-q_o-lr2p25013-sc294p611-clr0p005-seed42"},
    "lyra":     {"lr": 0.0375,    "scaling": 0.8,       "clf_lr": 5e-3,
                 "extra": {"freq_exponent": 5.0},
                 "cell": "mrpc-lyra-q_o-lr0p0375-sc0p8-clr0p005-ex5-seed42"},
    # ⛔ scora has NO scale flag: its scale is derived a priori from --slr_s
    #   (fir_arms: "DO NOT ADD ONE").  `scaling: None` is a DECLARED kind here,
    #   exactly as `no_scale` is in the search planner -- not a missing key.
    "scora":    {"lr": 0.0451498, "scaling": None,      "clf_lr": 2e-3,
                 "cell": "mrpc-scora-q_o-lr0p0451498-clr0p002-seed42"},
    "scora2":   {"lr": 0.0705466, "scaling": 0.0244141, "clf_lr": 5e-4,
                 "cell": "mrpc-scora2-q_o-lr0p0705466-sc0p0244141-clr0p0005-seed42"},
}

# ---------------------------------------------------------------------------
# ⭐⭐ THE EPOCH SCHEDULE.  The one number that was NOT measured for us, so here is
#   the reasoning, in full, in one place -- change this dict, not the callers.
# ---------------------------------------------------------------------------
# ⛔ 5 EPOCHS TRUNCATES, AND THAT IS MEASURED. Across the 9 arms' TOP-10 search
#   cells (90 runs), the argmax epoch is the LAST one in 48/90 and the
#   second-to-last in 29/90 -- i.e. 86% were still improving when the 5-epoch
#   budget ran out. Every §23 number is therefore a LOWER BOUND, and a final table
#   built at 5 epochs would be measuring the budget, not the method.
#
# ⛔ AND EPOCHS ARE NOT THE INVARIANT -- STEPS ARE. QNLI has 28x MRPC's rows, so
#   "30 epochs everywhere" (which is what [R.310] could afford on roberta-base)
#   is 3,274 steps/epoch x 30 = 98k steps on an H100, per cell, per seed. The
#   schedule below equalises OPTIMISATION STEPS to ~1.5k-10k and keeps at least
#   3 evaluation points on every task, because the reported number is a MAX OVER
#   EPOCHS and a max over two points is not a max.
#     task   rows    steps/ep   epochs   total steps   eval points
#     rte     2490      78         20       1,560          20
#     mrpc    3668     115         20       2,300          20
#     stsb    5749     180         15       2,700          15
#     cola    8551     268         12       3,216          12
#     sst2   67349    2105          5      10,525           5
#     qnli  104743    3274          3       9,822           3
# ⚠ SST-2 and QNLI get the FEWEST evaluation points and the MOST steps. That is the
#   honest trade at this backbone's cost, and the reader flags an argmax on the last
#   epoch (`^`) so a truncated column cannot be read as a converged one.
# ⚠ train_glue evaluates ONCE PER EPOCH (there is no --eval_steps), so "evaluation
#   points" is exactly the epoch count. Raising it is the only way to get more.
EPOCHS = {"rte": 20, "mrpc": 20, "stsb": 15, "cola": 12, "sst2": 5, "qnli": 3}
TASKS = ["rte", "mrpc", "stsb", "cola", "sst2", "qnli"]

# ⛔⛔ MRPC IS IN-SAMPLE. The proxy HPs were selected ON MRPC, so its column is not
#   an out-of-sample measurement. Declared here as data so the reader cannot forget
#   to label it -- [R.310] has the same shape with RTE and labels it.
SELECTION_TASK = "mrpc"

TASK_NAME = os.environ.get("FIR_FINAL_TASK", "all")
if TASK_NAME not in TASKS + ["all"]:
    raise SystemExit(f"FAIL CLOSED: FIR_FINAL_TASK={TASK_NAME!r} is not one of "
                     f"{TASKS + ['all']}")


def tasks():
    """The SELECTED tasks.  ⭐ `all` is a READING/PLANNING view; the submit path
    refuses it, because --time is per-array and the per-task wall-clock spans 6x."""
    return list(TASKS) if TASK_NAME == "all" else [TASK_NAME]


def is_all():
    return TASK_NAME == "all"


def cells(task=None, arms=None, seeds=None):
    """Every cell, in a DETERMINISTIC order -- the Slurm array index IS this order.

    ⭐ The outer loop is the TASK so that a per-task submission's indices are a
      contiguous block of this same enumeration when `all` is selected, and the
      inner loops are (arm, seed) so a canary picking one arm never lands on a
      single seed by accident."""
    out = []
    for t in ([task] if task else tasks()):
        for a in (arms or ARMS):
            for s in (seeds or SEEDS):
                out.append({"arm": a, "task": t, "targets": TARGETS, "seed": s,
                            "epochs": EPOCHS[t]})
    return out


def cell_id(c):
    """⛔ APPEND-ONLY once anything has run: CSVs, `done` and `fail` markers on fir
    are named by this string.  `-final` distinguishes these from the SEARCH cells,
    which live in another root but share the (task, arm, targets, seed) shape."""
    return f"{c['task']}-{c['arm']}-{c['targets']}-final-ep{c['epochs']}-seed{c['seed']}"


def parse_cell_id(cid):
    """The inverse.  Used by the array task, so a typo cannot silently run a
    DIFFERENT cell than the one whose CSV will carry the name.

    ⛔ SEARCHES EVERY TASK, not the selected one: the array body pins
      FIR_FINAL_TASK, but a cell id must mean one thing regardless of what is
      selected, or `--status` under `all` could not resolve its own cells."""
    for c in cells(task=None, arms=None, seeds=None) if is_all() else \
            [x for t in TASKS for x in cells(task=t)]:
        if cell_id(c) == cid:
            return c
    raise SystemExit(f"FAIL CLOSED: {cid!r} is not a cell of the final runs")


def cell_cmd(c, model=None):
    """The full `src/train_glue.py` command for one cell.

    ⭐ Built by the SAME planner every other fir stage uses (`fir_plan.cell_cmd`),
      then the frozen knobs are overridden -- so the module-name port, the dtype,
      the batch size and the derived warmup cannot drift away from the rest of the
      port.  `port_mode='derived'` is passed for form only: both values it sets are
      overridden below, exactly as in the search."""
    sel = SELECTED[c["arm"]]
    kw = {"model": model} if model else {}
    cmd = FP.cell_cmd(c["arm"], c["task"], c["seed"], c["targets"], "derived",
                      c["epochs"], **kw)
    cmd = HP._set_flag(cmd, "--learning_rate", sel["lr"])
    cmd = HP._set_flag(cmd, "--classifier_lr", sel["clf_lr"])
    sf = FA.ARM_SCALE_FLAG[c["arm"]]
    if sel["scaling"] is None:
        # ⛔ AND IT MUST BE THE ARM THAT HAS NO SCALE, NOT A ROW THAT FORGOT ONE.
        if sf:
            raise SystemExit(f"FAIL CLOSED: {c['arm']} HAS a scale flag ({sf}) but its "
                             f"SELECTED row carries no scaling")
    else:
        if not sf:
            raise SystemExit(f"FAIL CLOSED: {c['arm']} has no scale flag to set")
        cmd = HP._set_flag(cmd, sf, sel["scaling"])
    for k, v in (sel.get("extra") or {}).items():
        cmd = HP._set_flag(cmd, HP.EXTRA_KEYS[k], v)
    cmd[cmd.index("--name") + 1] = cell_id(c)
    return cmd


def cell_env(c, run_root):
    """⛔ ONE CSV PER CELL.  `train_glue._upsert_result`'s key OMITS the seed
    (scripts/r304_upsert_gate.py enforces that), so two seeds pointed at one CSV
    COLLAPSE INTO ONE ROW, silently.  The seed lives in the FILENAME.
    ⚠ And the seed is delivered by GLUE_SEEDS -- `--seed` is IGNORED by train_glue."""
    e = FP.cell_env(c["arm"], c["task"], c["seed"], c["targets"], run_root)
    e["GLUE_RESULTS_FILE"] = os.path.join(run_root, "csv", cell_id(c) + ".csv")
    return e


def steps_per_cell(task):
    return FP.total_steps(task, EPOCHS[task])


def digest():
    return hashlib.sha1(json.dumps(
        [cell_id(c) for c in cells()], sort_keys=True).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
def canary_cells():
    """⭐ THE COVERING CANARY: every ARM smoked exactly once, every TASK smoked at
    least once, in 9 cells.

    ⛔ WHY NOT "one cell per arm on one task", and not "one cell per task on one
      arm".  Two different things are unproven here and they fail in different
      places: an ARM's flag string reaching a decoder (proven on MRPC by 658 cells,
      but its selected HP has never run at ANOTHER task's step count) and a TASK's
      data/metric/collapse path (never run on this backbone AT ALL -- every gemma
      cell to date is MRPC).  A canary that covers only one axis leaves the other
      to be discovered by 270 queued cells.
    ⛔ AND IT MUST MEASURE THE WALL PER TASK, because --time is per array and the
      per-task cost spans ~6x. Every task therefore appears.
    ⭐ The 3 spare slots (9 arms - 6 tasks) go to the CHEAPEST task, so covering the
      arm axis costs the least GPU it can. Assignment is deterministic and asserted
      below, never hand-written."""
    order = sorted(TASKS, key=lambda t: steps_per_cell(t))     # cheapest first
    assign = []
    for i, a in enumerate(ARMS):
        assign.append((a, order[i] if i < len(order) else order[0]))
    seed = SEEDS[0]
    return [{"arm": a, "task": t, "targets": TARGETS, "seed": seed,
             "epochs": EPOCHS[t]} for a, t in assign]


def canary_indices(ids=None):
    """The covering canary as indices into cells() -- the SELECTED task's slice.

    ⛔ A per-task submission can only run the canary cells that belong to that task,
      and it must run ALL of them (that is how the arm axis gets covered)."""
    ids = ids or [cell_id(c) for c in cells()]
    want = [cell_id(c) for c in canary_cells() if c["task"] in tasks()]
    out = []
    for w in want:
        if w not in ids:
            raise SystemExit(f"FAIL CLOSED: canary cell {w} is not in the plan")
        out.append(ids.index(w))
    if not out:
        raise SystemExit("FAIL CLOSED: no canary cell for this task")
    return sorted(out)


def show():
    print(f"FINAL RUNS  |  digest {digest()}  |  google/gemma-2b / {TARGETS} / "
          f"seeds {SEEDS}")
    print(f"  task selector FIR_FINAL_TASK={TASK_NAME}  -> {tasks()}")
    print(f"  {len(ARMS)} arms x {len(tasks())} task(s) x {len(SEEDS)} seeds = "
          f"{len(cells())} cells")
    print("  task   epochs  steps/cell  warmup  metric        in-sample?")
    for t in tasks():
        import r310_read as R
        print(f"  {t:6s} {EPOCHS[t]:6d}  {steps_per_cell(t):10d}  "
              f"{FP.warmup_for(t, EPOCHS[t]):6d}  {R.metric_of(t):13s} "
              f"{'⛔ YES (HPs selected here)' if t == SELECTION_TASK else 'no'}")
    print("  canary (covering: every arm once, every task >=once):")
    for c in canary_cells():
        mark = "  <- this task" if c["task"] in tasks() else ""
        print(f"    {cell_id(c)}{mark}")


# ---------------------------------------------------------------------------
def selftest():
    ok, bad = [], []

    def ck(c, l):
        (ok if c else bad).append(l)

    import r310_read as R
    cs = cells()
    n_exp = len(ARMS) * len(tasks()) * len(SEEDS)
    ck(len(cs) == n_exp,
       f"{TASK_NAME}: {len(ARMS)} arms x {len(tasks())} task(s) x {len(SEEDS)} "
       f"seeds = {len(cs)} cells")
    ck(len(cells(task=None)) == n_exp if not is_all() else len(cs) == 270,
       f"the FULL plan is 9 x 6 x 5 = 270 cells (this view: {len(cs)})")
    ids = [cell_id(c) for c in cs]
    ck(len(set(ids)) == len(ids), "every cell id is unique")
    ck(cells() == cells(), "cell order is deterministic (the array index is stable)")
    ck(all(parse_cell_id(i) is not None for i in ids[:3]), "cell ids round-trip")
    try:
        parse_cell_id("rte-fftm-q_o-final-ep99-seed42")
        ck(False, "CONTROL: an unknown cell id is refused")
    except SystemExit:
        ck(True, "CONTROL: an unknown cell id is refused")

    # ------------------------------------------------------------------
    # ⭐⭐ THE HYPERPARAMETERS ARE THE SEARCH'S OWN WINNERS -- ASSERTED, NOT CLAIMED.
    #   This is the check that fires if a grid is ever edited under us.
    # ------------------------------------------------------------------
    for a in ARMS:
        sel = SELECTED[a]
        # the id must be a cell of SOME grid that searched this arm
        hp = None
        for n, g in HP.GRIDS.items():
            if g.get("union") or a not in g.get("arms", []):
                continue
            for c in HP._cells_of(g, [a]):
                if HP.cell_id(c) == sel["cell"]:
                    hp = c
                    break
            if hp:
                break
        ck(hp is not None,
           f"⭐ {a}: its frozen row IS a cell of the search ({sel['cell']})")
        if hp is None:
            continue
        # ⛔ AND THE FLAGS MUST MATCH THE COMMAND THAT ACTUALLY RAN, token for
        #   token -- not just the id. A typed `lr` that formats differently from the
        #   searched one would run a DIFFERENT configuration under the same name.
        want = HP.cell_cmd(hp)
        got = cell_cmd({"arm": a, "task": "rte", "targets": TARGETS, "seed": 42,
                        "epochs": EPOCHS["rte"]})

        def val(toks, flag):
            return toks[toks.index(flag) + 1] if flag in toks else None
        flags = ["--learning_rate", "--classifier_lr"]
        sf = FA.ARM_SCALE_FLAG[a]
        if sf:
            flags.append(sf)
        for k in (sel.get("extra") or {}):
            flags.append(HP.EXTRA_KEYS[k])
        for f in flags:
            ck(val(want, f) is not None and val(want, f) == val(got, f),
               f"⭐ {a}: {f} is BYTE-IDENTICAL to the searched winner "
               f"({val(want, f)!r} vs {val(got, f)!r})")
        # ⛔ CONTROL, both directions: a row perturbed by one ulp must FAIL the
        #   comparison, or "byte-identical" is vacuous.
        bogus = HP._set_flag(list(got), "--learning_rate", sel["lr"] * 1.01)
        ck(val(bogus, "--learning_rate") != val(want, "--learning_rate"),
           f"CONTROL: {a}: a 1% lr perturbation is DETECTED by that comparison")
        # the no-scale arm must acquire no scale flag, in both directions
        s = " ".join(got)
        if sel["scaling"] is None:
            ck(sf is None and "--slr_scaling" not in s,
               f"{a}: NO scale flag appears (its scale is derived a priori)")
        else:
            ck(f"{sf} {sel['scaling']:g}" in s, f"{a}: its scale flag carries {sel['scaling']:g}")

    # --- every task's own knobs
    ck(sorted(EPOCHS) == sorted(TASKS), "the epoch schedule covers exactly the 6 tasks")
    ck(SELECTION_TASK in TASKS,
       "⛔ the SELECTION task is in the table and is declared IN-SAMPLE")
    for t in tasks():
        e = EPOCHS[t]
        st = steps_per_cell(t)
        ck(e >= 3, f"{t}: {e} epochs -- at least 3 evaluation points for a MAX-over-epochs")
        ck(st >= 1500, f"{t}: {st} optimisation steps (>=1500)")
        ck(FP.warmup_for(t, e) == int(round(FP.WARMUP_RATIO * st)),
           f"{t}: warmup {FP.warmup_for(t, e)} is derived from ITS OWN step count")
        ck(R.metric_of(t) is not None, f"{t}: has a declared primary metric "
                                       f"({R.metric_of(t)})")
    # ⛔ 5 EPOCHS IS THE THING THIS SCHEDULE EXISTS TO FIX -- assert it is nowhere.
    ck(all(v > 5 or steps_per_cell(k) > 5000 for k, v in EPOCHS.items()),
       "no task keeps the SEARCH's 5-epoch budget unless it buys >5000 steps by size")

    # --- the canary
    cc = canary_cells()
    ck(len(cc) == len(ARMS), f"the canary is {len(ARMS)} cells, one per ARM")
    ck(sorted(c["arm"] for c in cc) == sorted(ARMS),
       "⭐ every arm appears EXACTLY once in the canary")
    ck(set(c["task"] for c in cc) == set(TASKS),
       "⭐ every task appears at least once (--time is per-array and spans ~6x)")
    cheap = min(TASKS, key=steps_per_cell)
    ck([c["task"] for c in cc].count(cheap) == len(ARMS) - len(TASKS) + 1,
       f"the {len(ARMS) - len(TASKS)} spare slots go to the CHEAPEST task ({cheap})")
    ck(all(cell_id(c) in {cell_id(x) for t in TASKS for x in cells(task=t)}
           for c in cc), "every canary cell is a real cell of the plan")
    idx = canary_indices(ids)
    ck(len(idx) == len(set(idx)) and all(0 <= i < len(cs) for i in idx),
       f"the canary resolves to {len(idx)} distinct in-range index(es) for {TASK_NAME}")
    ck([parse_cell_id(ids[i])["task"] for i in idx] == [TASK_NAME] * len(idx)
       or is_all(), "...and every one of them belongs to the SELECTED task")

    # --- the command really carries what it claims, for every arm
    for a in ARMS:
        c = {"arm": a, "task": tasks()[0], "targets": TARGETS, "seed": 43,
             "epochs": EPOCHS[tasks()[0]]}
        s = " ".join(cell_cmd(c))
        ck(s.count("--learning_rate") == 1, f"{a}: exactly ONE --learning_rate")
        ck(s.count("--classifier_lr") == 1, f"{a}: exactly ONE --classifier_lr")
        ck("query" not in s and "value" not in s, f"{a}: no RoBERTa module name survives")
        ck(f"--adapter_target_modules {FP.TARGET_SETS[TARGETS]}" in s,
           f"{a}: the generic target override is present")
        ck("--dtype float32" in s, f"{a}: float32 (the bf16 default is machine-dependent)")
        ck(f"--num_train_epochs {c['epochs']}" in s, f"{a}: this task's epoch count")
        ck(cell_id(c) in s, f"{a}: the cell id is the run name")
        e = cell_env(c, "/tmp/x")
        ck(e["GLUE_SEEDS"] == "43", f"{a}: the seed is delivered by GLUE_SEEDS")
        ck(e["GLUE_RESULTS_FILE"].endswith(cell_id(c) + ".csv"),
           f"{a}: one CSV per cell, named by the cell id")
    # ⛔ ONE CSV PER CELL, over the WHOLE plan -- two seeds on one CSV collapse
    #   into one row silently.
    seen = {cell_env(c, "/tmp/x")["GLUE_RESULTS_FILE"] for t in TASKS
            for c in cells(task=t)}
    ck(len(seen) == 270, f"the 270 cells map to {len(seen)} distinct CSVs")

    for l in ok:
        print(f"  ✅ {l}")
    for l in bad:
        print(f"  ⛔ {l}")
    print(f"selftest: {len(ok)} passed, {len(bad)} failed")
    return 1 if bad else 0


def _selftest_every_task():
    """⭐ EVERY TASK VIEW, NOT JUST THE SELECTED ONE.  Module globals bind
    FIR_FINAL_TASK at import, so one process can only check one view -- and
    `run_all_gates.py` sets no env var.  Re-exec once per view and aggregate.
    (The same defect `fir_hp_plan --selftest` had when it skipped its fan-out.)"""
    import subprocess, re
    if os.environ.get("FIR_FINAL_TASK"):
        print(f"  ⚠ FIR_FINAL_TASK={os.environ['FIR_FINAL_TASK']} is set; "
              f"checking ALL views anyway.")
    tot_p = tot_f = 0
    views = TASKS + ["all"]
    for t in views:
        print(f"--- task view {t} " + "-" * 46)
        env = dict(os.environ, FIR_FINAL_TASK=t, FIR_FINAL_ONE="1")
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--selftest"],
                           capture_output=True, text=True, env=env, cwd=ROOT)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
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
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--cmd", default=None)
    a = ap.parse_args()
    if a.selftest:
        if not os.environ.get("FIR_FINAL_ONE"):
            sys.exit(_selftest_every_task())
        sys.exit(selftest())
    if a.list:
        for c in cells():
            print(cell_id(c))
        return
    if a.cmd:
        print(" ".join(cell_cmd(parse_cell_id(a.cmd))))
        return
    show()


if __name__ == "__main__":
    main()
