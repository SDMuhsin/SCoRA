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

# ⭐ THE SEARCH'S WINNERS -- (lr, batch) per task, EMPTY until the search has run
#   and been read.  This is the stage-06 analogue of GEMMA_HP_PROXY.md: a measured
#   table, written once, never guessed.  ⛔ Selection is on the SEARCH SEED ONLY.
WINNERS: dict = {}

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
            for b in BATCHES:
                for lr in LRS:
                    out.append({"arm": ARM, "task": t, "lr": lr, "batch": b,
                                "seed": SEARCH_SEED, "epochs": EPOCHS[t]})
        else:
            # ⛔⛔ FAIL CLOSED UNTIL THE SEARCH HAS BEEN READ. An earlier draft of this
            #   function enumerated the WHOLE ladder at the remaining four seeds --
            #   288 gemma-2b full-FT cells, several hundred GPU-h, for a stage whose
            #   final row needs SIX configurations. `--selftest` now asserts that an
            #   empty WINNERS table yields ZERO final cells, so the accident cannot
            #   be submitted.
            w = WINNERS.get(t)
            if w is None:
                continue
            lr, b = w
            for s in (seeds or SEEDS):
                if s == SEARCH_SEED:
                    continue
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
    return ["env/bin/python", "-u", "src/train_glue.py",
            "--model_name_or_path", model or MODEL,
            "--task_name", c["task"],
            "--dtype", "float32",
            "--mixed_precision", "fp16",
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
    ck(len(cells(stage="final")) == 0 if not WINNERS else True,
       "⛔ CONTROL: with an EMPTY winners table the final stage emits ZERO cells "
       "-- it cannot be submitted before the search is read")
    _saved = dict(WINNERS)
    WINNERS.update({"rte": (2e-5, 32)})
    ck(len(cells(task="rte", stage="final")) == len(SEEDS) - 1,
       f"...and with one winner declared it emits exactly {len(SEEDS)-1} seeds for that task")
    ck(all(c["seed"] != SEARCH_SEED for c in cells(task="rte", stage="final")),
       "⛔ CONTROL: the final stage never re-runs the search seed")
    WINNERS.clear(); WINNERS.update(_saved)

    ci = canary_indices()
    ck(len(ci) == len(tasks()), f"canary picks exactly one cell per selected task ({len(ci)})")
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
