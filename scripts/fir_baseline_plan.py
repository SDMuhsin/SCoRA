#!/usr/bin/env python
"""
STAGE 06 -- the FP16 FULL-FINE-TUNING BASELINE on gemma-2b (fir).

  FIR_BASE_TASK=rte  env/bin/python scripts/fir_baseline_plan.py --show
  FIR_BASE_TASK=rte  env/bin/python scripts/fir_baseline_plan.py --list
  env/bin/python scripts/fir_baseline_plan.py --selftest

WHAT THIS IS.  Stage 05 compared nine PEFT arms to each other.  It carries no
FULL-FINE-TUNING row -- the one row every LoRA-family paper reports.  This stage
adds it, on the SAME backbone, the SAME six tasks, the SAME seeds and the SAME
per-task epoch budget, so the number can be printed in stage 05's own table.

⛔ THE RECIPE AND ITS THREE DELIBERATE DEVIATIONS are documented ONCE, in
   scripts/fp16_baseline_plan.py's module docstring and llmdocs/FP16_BASELINE.md.
   This module imports the recipe constants from there rather than restating them
   -- two copies of a protocol are two protocols.

⛔ MIXED PRECISION, NOT A CAST.  `--dtype float32 --mixed_precision fp16`: fp32
   master weights and fp32 optimizer state, fp16 compute, loss scaler.  [measured
   2026-09-02] `--dtype float16` leaves full FT at its INIT accuracy across seven
   lr rungs spanning 1e-5..1e-2, because gradients underflow fp16's exponent range.

⛔ THE SWEEP IS AUTHORISED FOR THIS ARM ONLY [user, 2026-09-02] and is a budget
   asymmetry in the BASELINE's favour.  Report it [Dodge et al. 2019 §6]: a
   baseline WIN is then not attributable to method; a baseline LOSS is clean.

⚠ EVERY COST NUMBER HERE IS AN EXTRAPOLATION FROM THE PEFT ARMS.  This repo has
   burned that exact mistake ("a WaveFT cell is ~1.7x a FourierFT one"; [measured]
   1.03x).  ⛔ SIZE `--time` FROM THIS STAGE'S OWN CANARY, never from stage 05.
"""
import argparse, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import fir_plan as FP                                                  # noqa: E402
import fp16_baseline_plan as RECIPE                                    # noqa: E402

ARM = "base"                       # the name fir_hp_run_cell.BASELINE_ARMS knows
MODEL = "google/gemma-2b"
SEEDS = [42, 43, 44, 45, 46]       # stage 05's seeds, unchanged

# ⭐ STAGE 05's OWN PER-TASK EPOCHS, not the recipe's flat 10.  The point of this
#   row is to sit in stage 05's table; an epoch budget that differs from the arms
#   it is compared against would confound method with training length.  That IS a
#   deviation from the published recipe and is disclosed as one.
TASKS = ["rte", "mrpc", "stsb", "cola", "sst2", "qnli"]
EPOCHS = {"rte": 20, "mrpc": 20, "stsb": 15, "cola": 12, "sst2": 5, "qnli": 3}

# ⭐ THE LADDER IS SHIFTED DOWN relative to roberta-base's.  A 2.5B model full-FT
#   does not take a 3e-4 step; published full-FT lrs for models this size sit at
#   1e-6..5e-5.  The recipe's own three rungs {1e-5, 2e-5, 3e-5} are all INSIDE
#   this ladder -- `--selftest` asserts it, the same admissibility rule stage 04
#   held every PEFT grid to ([R.258]: the published operating point must be inside
#   the swept range).
LRS = [5e-6, 1e-5, 2e-5, 3e-5, 5e-5, 1e-4]
BATCHES = [16, 32]                 # the recipe's own two rungs
SEARCH_SEED = 42

# ⭐⭐ ONE SWEEP, CARRIED AS A PROXY -- the protocol this program already uses
#   [user, 2026-09-02; memory: hp-transfer-proxy].  The ladder is swept on ONE task
#   and the winner is carried UNCHANGED to all six columns.
# ⛔ THE SELECTION TASK IS **MRPC**, because stage 05's nine PEFT arms were tuned on
#   MRPC (GEMMA_HP_PROXY.md) and carried to these same six columns.  Baseline and
#   arms therefore share ONE in-sample task, which stage 05's reader already labels.
#   Choosing a different one would put two different in-sample tasks in one table.
SELECTION_TASK = "mrpc"

# ⭐ THE PROXY: (lr, batch) from the sweep. EMPTY until the search has been read --
#   the stage-06 analogue of GEMMA_HP_PROXY.md: measured, written once, never guessed.
# ⛔ Selection is on the SEARCH SEED ONLY; picking on all five and reporting those
#   same five is selection on the test set.
PROXY: tuple = ()


def _load_proxy():
    """⭐ THE PROXY IS READ FROM A FILE THE READER WROTE, NOT FROM A HAND-EDIT.

    It used to be a literal in this module, which put a human edit-commit-push-pull
    cycle in the MIDDLE of the protocol: run the search on the cluster, copy the
    CSVs home, edit this line, push, pull, submit the final cells. This repo has
    already lost a round trip to a fix pushed to a ref nothing pulls, and another to
    a cluster quietly running older code than its log implied. So the winner is now
    read where it was measured (`scripts/fir_baseline_read.py --write-proxy`) and
    lands in `<LRS_STAGE_DIR>/baseline_proxy.json`.

    ⛔ FAIL CLOSED, LOUDLY. The file steers all 30 final cells, so a value that is
      not an actual cell of the declared ladder is refused rather than used. That is
      what makes a hand-written or truncated file safe: it cannot quietly become the
      operating point.
    ⚠ Absent file => () => the final stage emits ZERO cells, which 06_baseline.sh
      turns into an explicit refusal. That is the correct early state.
    """
    d = os.environ.get("LRS_STAGE_DIR", "sbatch/fir")
    path = os.path.join(ROOT, d, "baseline_proxy.json")
    if not os.path.exists(path):
        return ()
    import json
    try:
        p = json.load(open(path))
    except Exception as e:
        raise SystemExit(f"FAIL CLOSED: {path} is not readable JSON: {e}")
    for k in ("lr", "batch", "selection_task", "search_seed"):
        if k not in p:
            raise SystemExit(f"FAIL CLOSED: {path} has no {k!r}. It must be written by "
                             f"scripts/fir_baseline_read.py --write-proxy, not by hand.")
    if p["lr"] not in LRS or p["batch"] not in BATCHES:
        raise SystemExit(
            f"FAIL CLOSED: {path} names lr={p['lr']!r} batch={p['batch']!r}, which is "
            f"NOT a cell of this stage's ladder (lr in {LRS}, batch in {BATCHES}). "
            f"A proxy that was never searched cannot be carried to six columns.")
    if p["selection_task"] != SELECTION_TASK:
        raise SystemExit(
            f"FAIL CLOSED: {path} was selected on {p['selection_task']!r} but this "
            f"stage's selection task is {SELECTION_TASK!r}. Two in-sample tasks in "
            f"one row is exactly what SELECTION_TASK exists to prevent.")
    if p["search_seed"] != SEARCH_SEED:
        raise SystemExit(
            f"FAIL CLOSED: {path} was selected at seed {p['search_seed']}, not the "
            f"search seed {SEARCH_SEED}. Selecting on the reported seeds is selection "
            f"on the test set.")
    return (p["lr"], p["batch"])


PROXY = _load_proxy()

TASK_NAME = os.environ.get("FIR_BASE_TASK", "all")
if TASK_NAME not in TASKS + ["all"]:
    raise SystemExit(f"FAIL CLOSED: FIR_BASE_TASK={TASK_NAME!r} is not one of "
                     f"{TASKS + ['all']}")
GRID_NAME = TASK_NAME


def tasks():
    """`all` is a READING/PLANNING view; the submit path refuses it, because
    `--time` is per-array and the per-task wall-clock spans ~11x."""
    return list(TASKS) if TASK_NAME == "all" else [TASK_NAME]


def is_all():
    return TASK_NAME == "all"


def steps(task, batch, epochs):
    n = FP.sizes()[task]["train"]
    return -(-n // batch) * epochs


def warmup(task, batch, epochs):
    return max(1, round(RECIPE.WARMUP_FRAC * steps(task, batch, epochs)))


def cells(task=None, stage="search", seeds=None):
    """Deterministic order -- the Slurm array index IS this order.

    `search` sweeps (batch, lr) at ONE seed; `final` runs the remaining seeds at a
    winner supplied by the reader.  ⛔ Selection uses the SEARCH SEED ONLY: picking
    on all five and then reporting those same five is selection on the test set."""
    out = []
    for t in ([task] if task else tasks()):
        if stage == "search":
            # ⛔ THE SWEEP RUNS ON THE SELECTION TASK ONLY.
            if t != SELECTION_TASK:
                continue
            for b in BATCHES:
                for lr in LRS:
                    out.append({"arm": ARM, "task": t, "lr": lr, "batch": b,
                                "seed": SEARCH_SEED, "epochs": EPOCHS[t]})
        else:
            # ⛔⛔ FAIL CLOSED UNTIL THE SEARCH HAS BEEN READ. An earlier draft of this
            #   function enumerated the WHOLE ladder at the remaining four seeds --
            #   288 gemma-2b full-FT cells, several hundred GPU-h, for a stage whose
            #   final row needs SIX configurations. `--selftest` now asserts that an
            #   empty PROXY yields ZERO final cells, so the accident cannot
            #   be submitted.
            if not PROXY:
                continue
            lr, b = PROXY
            for s in (seeds or SEEDS):
                out.append({"arm": ARM, "task": t, "lr": lr, "batch": b,
                            "seed": s, "epochs": EPOCHS[t]})
    return out


def cell_id(c):
    """⛔ APPEND-ONLY once anything has run -- CSVs and markers on fir are named by
    it.  `-base-` and the lr/bs fields keep it disjoint from stage 05's ids."""
    lr = f"{c['lr']:g}".replace(".", "p").replace("-", "m")
    return (f"{c['task']}-base-lr{lr}-bs{c['batch']}"
            f"-ep{c['epochs']}-seed{c['seed']}")


def parse_cell_id(cid):
    """⛔ Searches EVERY task and BOTH stages, so a cell id means one thing
    regardless of what is selected."""
    for stage in ("search", "final"):
        for c in ([x for t in TASKS for x in cells(task=t, stage=stage)]):
            if cell_id(c) == cid:
                return c
    raise SystemExit(f"FAIL CLOSED: {cid!r} is not a cell of the baseline runs")


def cell_cmd(c, model=None):
    """⛔ NO adapter flags, NO --classifier_lr, NO --adapter_target_modules.
    `--optimizer adamw` with no suffix IS full fine-tuning (train_glue.py:686), and
    an absent --classifier_lr means ONE param group at --learning_rate (:2038)."""
    # ⛔⛔ THE FIRST ELEMENT IS THE SCRIPT, NEVER AN INTERPRETER.
    #   fir_hp_run_cell.run() invokes this as
    #       subprocess.run([python or sys.executable] + cell_cmd(...))
    #   so it supplies the interpreter itself. This function used to return
    #       ["env/bin/python", "-u", "src/train_glue.py", ...]
    #   which produced `env/bin/python env/bin/python -u src/train_glue.py`, i.e.
    #   python handed its OWN BINARY as a script -- narval job 2689499_9 died in 1s:
    #       File ".../env/bin/python", line 1
    #           ELF
    #       SyntaxError: source code cannot contain null bytes
    #   ⚠ fir_hp_plan and fir_final_plan have always returned "src/train_glue.py"
    #     first and between them ran 1,500 cells. This planner is the only one that
    #     never executed a cell, so it is the only one where the mismatch survived.
    #   ⚠ `-u` went with it: the runner captures output and writes the log at the
    #     end, so unbuffering changed nothing, and the two planners that produced
    #     every banked cell do not pass it either.
    return ["src/train_glue.py",
            "--model_name_or_path", model or MODEL,
            "--task_name", c["task"],
            "--dtype", "float32",
            "--mixed_precision", "fp16",
            # ⛔⛔ TWO MEMORY FLAGS, AND THEY ARE PART OF THE PROTOCOL, NOT OF A
            #   CLUSTER. [MEASURED 2026-09-08, A40 44.4 GiB] a gemma-2b full
            #   fine-tune holds params + grads + two fp32 AdamW moments = 37.3 GiB
            #   of state before any activation, and torch's DEFAULT AdamW then asks
            #   `torch._foreach_sqrt` for a full-size temporary on top. It OOMs on
            #   every one of {bs 16, bs 32} x {checkpointing on, off}. `fused`
            #   removes the temporaries (one in-place CUDA kernel) and
            #   checkpointing removes the activation peak; BOTH are needed:
            #       foreach + ckpt  OOM     | single + ckpt 42,258 MiB
            #       fused  + no ckpt 44,090 | ⭐ fused + ckpt 38,311 MiB
            #   ⭐ THEY ARE SET UNCONDITIONALLY, NOT PER CLUSTER. Stage 06 has never
            #     produced a cell on any machine, so nothing is invalidated by
            #     choosing them now -- and a flag that depends on which cluster ran
            #     the cell would make the six columns of ONE row two protocols.
            #   ⚠ Neither changes the update rule: `fused` is the same AdamW maths
            #     in one kernel, and checkpointing recomputes activations exactly.
            #     [measured, MRPC 1 epoch, seed 42, lr 2e-5] `single` scored
            #     f1 0.9046 -- the fused arm is compared against it in
            #     llmdocs/NARVAL_PORT.md rather than assumed equal.
            #   ⛔ BUT THEY DO CHANGE THE COST COLUMNS. Stage 05's nine PEFT arms ran
            #     WITHOUT checkpointing, so this row's s/step and peak-memory are NOT
            #     comparable to theirs. Say so wherever the row is printed.
            "--adam_impl", "fused",
            "--gradient_checkpointing",
            "--per_device_train_batch_size", str(c["batch"]),
            "--num_train_epochs", str(c["epochs"]),
            "--num_warmup_steps", str(warmup(c["task"], c["batch"], c["epochs"])),
            "--max_length", str(RECIPE.MAX_LENGTH),
            "--optimizer", "adamw",
            "--learning_rate", f"{c['lr']:g}",
            "--weight_decay", str(RECIPE.WEIGHT_DECAY),
            "--name", cell_id(c)]


def cell_env(c, run_root):
    """⛔ ONE CSV PER CELL. `_upsert_result`'s key omits the seed, so two seeds on
    one CSV collapse into one row silently -- the seed lives in the FILENAME."""
    return {"GLUE_SEEDS": str(c["seed"]),
            "GLUE_RESULTS_FILE": os.path.join(run_root, "csv", cell_id(c) + ".csv")}


def canary_indices(stage="search"):
    """Indices into `cells(stage)` for the canary.

    ⛔ ONE PER SELECTED TASK, at a MIDDLE lr and the larger batch -- a canary must be
      CENTRAL (an edge rung may legitimately collapse, which tells you nothing about
      the wall) and it must smoke the thing that is actually new here: whether a 2.5B
      full fine-tune with fp32 master weights and fp32 Adam state FITS on the GPU.
      [estimate] ~40 GB of states before activations, against stage 05's measured
      27-29 GB peak. ⚠ ESTIMATE. That is exactly what the canary is for.
    ⛔ AND IT SIZES `--time` PER TASK: the per-task wall spans ~11x in stage 05."""
    cs = cells(stage=stage)
    mid = LRS[len(LRS) // 2]
    out, seen = [], set()
    for i, c in enumerate(cs):
        if c["task"] in seen:
            continue
        if c["lr"] == mid and c["batch"] == max(BATCHES):
            out.append(i)
            seen.add(c["task"])
    return out


def show():
    print(f"BASELINE (stage 06) on {MODEL}  |  arm={ARM}  |  FIR_BASE_TASK={TASK_NAME}")
    print(f"  mixed precision: fp16 (autocast + GradScaler, fp32 master weights)")
    print(f"  lr ladder      : {', '.join(f'{x:g}' for x in LRS)}")
    print(f"  batch          : {BATCHES}   epochs: per task, stage 05's own")
    print(f"  seeds          : {SEEDS} (search on {SEARCH_SEED} only)")
    ns = len(cells(stage='search'))
    nf = len(cells(stage='final'))
    print(f"  search cells   : {ns}     final cells: {nf}")
    print(f"  task   epochs  steps(bs32)  warmup  head")
    for t in tasks():
        print(f"  {t:6} {EPOCHS[t]:6}  {steps(t,32,EPOCHS[t]):11}  "
              f"{warmup(t,32,EPOCHS[t]):6}")
    print("\n⚠ MRPC is IN-SAMPLE for stage 05's PEFT arms (their HPs were selected on")
    print("  it). This baseline is tuned per task, so MRPC is not in-sample FOR IT --")
    print("  ⛔ but the column still cannot be read as a like-for-like comparison.")


def selftest():
    global PROXY
    ok, bad = [], []

    def ck(c, m):
        (ok if c else bad).append(m)

    for lr in RECIPE.PUBLISHED_LRS:
        ck(lr in LRS, f"[R.258] the PUBLISHED rung {lr:g} is inside the swept ladder")
    ck(min(LRS) < min(RECIPE.PUBLISHED_LRS), "the ladder extends BELOW the published grid "
                                             "(a 2.5B full FT takes a smaller step)")
    ck(EPOCHS == {"rte": 20, "mrpc": 20, "stsb": 15, "cola": 12, "sst2": 5, "qnli": 3},
       "epochs are stage 05's own, per task -- the row must be comparable to it")
    ck(SEEDS == [42, 43, 44, 45, 46], "stage 05's seeds, unchanged")

    for t in TASKS:
        c = {"arm": ARM, "task": t, "lr": 2e-5, "batch": 32,
             "seed": 43, "epochs": EPOCHS[t]}
        s = " ".join(cell_cmd(c))
        ck("--mixed_precision fp16" in s, f"{t}: real AMP, not a cast")
        ck("--dtype float32" in s, f"{t}: fp32 master weights")
        ck("--classifier_lr" not in s, f"{t}: NO separate head lr (full FT is one group)")
        # ⛔ the two memory flags are protocol. A silent removal would not fail any
        #   other check here -- it would simply OOM on a 40 GiB GPU, hours later.
        ck("--adam_impl fused" in s, f"{t}: fused AdamW (no full-size foreach temporaries)")
        ck("--gradient_checkpointing" in s, f"{t}: gradient checkpointing")
        ck("adapter" not in s and "target_modules" not in s, f"{t}: no adapter flag")
        ck("query" not in s and "value" not in s, f"{t}: no RoBERTa module name survives")
        ck(s.count("--learning_rate") == 1, f"{t}: exactly ONE --learning_rate")
        ck(f"--num_train_epochs {EPOCHS[t]}" in s, f"{t}: this task's epoch count")
        ck(cell_id(c) in s, f"{t}: the cell id is the run name")
        e = cell_env(c, "/tmp/x")
        ck(e["GLUE_SEEDS"] == "43", f"{t}: the seed is delivered by GLUE_SEEDS")
        ck(e["GLUE_RESULTS_FILE"].endswith(cell_id(c) + ".csv"), f"{t}: one CSV per cell")
        w = warmup(t, 32, EPOCHS[t])
        ck(abs(w / steps(t, 32, EPOCHS[t]) - RECIPE.WARMUP_FRAC) < 0.02,
           f"{t}: warmup is {RECIPE.WARMUP_FRAC:g} of THIS cell's steps")

    # ids: unique, round-trip, and disjoint from stage 05's
    allc = [c for t in TASKS for st in ("search", "final") for c in cells(task=t, stage=st)]
    ids = [cell_id(c) for c in allc]
    ck(len(ids) == len(set(ids)), f"{len(ids)} cell ids, all distinct")
    ck(all(parse_cell_id(i) is not None for i in ids[:20]), "cell ids round-trip")
    try:
        parse_cell_id("mrpc-fftm-q_o-final-ep20-seed42")
        ck(False, "CONTROL: a STAGE 05 cell id is refused")
    except SystemExit:
        ck(True, "⛔ CONTROL: a STAGE 05 cell id is refused by this planner")
    try:
        parse_cell_id("nonsense")
        ck(False, "CONTROL: a nonsense id is refused")
    except SystemExit:
        ck(True, "⛔ CONTROL: a nonsense cell id fails closed")
    # ⭐ ONE SWEEP, CARRIED
    ck(SELECTION_TASK in TASKS,
       f"the selection task ({SELECTION_TASK}) IS one of the six -- stage 05's arms are "
       f"in-sample on the same task, so the table has ONE in-sample column, not two")
    srch = [c for t in TASKS for c in cells(task=t, stage="search")]
    ck(len(srch) == len(LRS) * len(BATCHES),
       f"the sweep is {len(LRS)}x{len(BATCHES)} = {len(srch)} cells, on ONE task")
    ck({c["task"] for c in srch} == {SELECTION_TASK},
       f"⛔ CONTROL: every swept cell is on {SELECTION_TASK}")
    for t in TASKS:
        if t != SELECTION_TASK:
            ck(cells(task=t, stage="search") == [],
               f"⛔ CONTROL: asking to sweep {t} yields ZERO cells")
    # ⛔ CONTROL: test the MECHANISM, not the ambient state. This check used to read
    #   the live PROXY, so the day the reader legitimately wrote one on the cluster
    #   the control would have started FAILING for the one reason that is not a
    #   defect. Force the empty case, exactly as the populated case below is forced.
    _saved = PROXY
    PROXY = ()
    ck(len([c for t in TASKS for c in cells(task=t, stage="final")]) == 0,
       "⛔ CONTROL: with an EMPTY proxy the final stage emits ZERO cells -- it cannot "
       "be submitted before the search is read")
    PROXY = (2e-5, 32)
    fin = [c for t in TASKS for c in cells(task=t, stage="final")]
    ck(len(fin) == len(TASKS) * len(SEEDS),
       f"...and with a proxy declared it is {len(TASKS)} columns x {len(SEEDS)} seeds = {len(fin)}")
    ck({(c["lr"], c["batch"]) for c in fin} == {(2e-5, 32)},
       "⛔ CONTROL: EVERY final cell carries the SAME (lr, batch) -- a per-task winner "
       "here would be a silent sweep")
    ck({c["task"] for c in fin} == set(TASKS), "final covers all six columns")
    PROXY = _saved

    ci = canary_indices()
    # ⛔ ONE PER TASK **PRESENT IN THIS STAGE'S CELLS**, not per selected task. The
    #   search stage only ever contains the selection task, so a check written against
    #   tasks() said "expected 6, got 1" -- the CHECK was stale after the protocol
    #   changed from six sweeps to one, not the picker.
    _st = {c["task"] for c in cells(stage="search")}
    ck(len(ci) == len(_st),
       f"canary picks exactly one cell per task PRESENT in the stage ({len(ci)} of {len(_st)})")
    ck(len({cells()[i]["task"] for i in ci}) == len(ci), "canary cells are on DISTINCT tasks")
    ck(all(cells()[i]["lr"] not in (min(LRS), max(LRS)) for i in ci),
       "⛔ canary cells are CENTRAL in lr -- an edge rung may collapse and tell you nothing")

    # one CSV per cell across the WHOLE plan
    seen = {cell_env(c, "/tmp/x")["GLUE_RESULTS_FILE"] for c in allc}
    ck(len(seen) == len(allc), f"the {len(allc)} cells map to {len(seen)} distinct CSVs")

    for l in ok:
        print(f"  ✅ {l}")
    for l in bad:
        print(f"  ⛔ {l}")
    print(f"\nselftest: {len(ok)} passed, {len(bad)} failed")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--stage", default="search", choices=["search", "final"])
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.show:
        show(); return
    if a.list:
        for c in cells(stage=a.stage):
            print(cell_id(c))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
