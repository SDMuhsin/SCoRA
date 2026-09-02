#!/usr/bin/env python
"""
The FP16 FULL-FINE-TUNING BASELINE grid -- the planner, and the ONLY place the
recipe lives.

  env/bin/python scripts/fp16_baseline_plan.py --selftest
  env/bin/python scripts/fp16_baseline_plan.py --generate <task> [--stage search|final]
  env/bin/python scripts/fp16_baseline_plan.py --read [--task T]

WHY THIS EXISTS  (llmdocs/FP16_BASELINE.md)
  Every PEFT arm in this repo is compared against other PEFT arms.  Neither table
  carries the row that LoRA-family papers always report: FULL FINE-TUNING.

THE RECIPE, AND ITS PROVENANCE
  ⭐ Neither LoRA nor FourierFT reproduces its own full-FT baseline -- both CITE it.
  LoRA Table 2 footnotes FT as "numbers published in prior works" and its GLUE setup
  follows Liu et al. 2019; FourierFT's FF row carries no +- where every other row does.
  So the recipe copied here is RoBERTa's own GLUE appendix (Liu et al. 2019, Table 10).

⛔ THREE DEVIATIONS FROM THE PUBLISHED RECIPE, ALL DELIBERATE, ALL DISCLOSED
  1. max_length 128, NOT the papers' 512.  Every PEFT cell in both of this repo's
     tables used the 128 default.  A 512 baseline beside 128 arms is a confound AND
     ~4x the cost.  Comparability inside our own table wins; say so wherever the row
     appears.
  2. AdamW betas/eps are torch defaults (0.9, 0.999)/1e-8, not the recipe's
     (0.9, 0.98)/1e-6 -- train_glue.py exposes no flag for them, and every PEFT arm
     in both tables also ran on the torch defaults.  Consistent within our table.
  3. The lr ladder is WIDER than the published {1e-5, 2e-5, 3e-5}.  ⭐ [measured,
     CB, 7 rungs] float32's optimum is 1e-4 (0.9107) against 0.8035 at 3e-5: the
     published grid does NOT contain the optimum at 10 epochs / max_len 128.  A
     search confined to it would understate the baseline -- i.e. would flatter US.
     The published rungs are all still IN the ladder; `--selftest` asserts it.

⛔ THE SWEEP IS AUTHORISED.  CONTEXT 2.4 forbids per-task hyperparameter sweeps for
  the nine PEFT arms.  [user, 2026-09-02] the fp16 baseline is an EXPLICIT EXCEPTION.
  That is a budget asymmetry in the BASELINE's favour and it must be reported as one
  [Dodge et al. 2019 6]: if the baseline WINS, the win is not attributable to method;
  if it LOSES despite the larger budget, that direction IS clean.

⛔ MIXED PRECISION IS NOT `--dtype float16`.  [measured 2026-09-02] casting the model
  makes full FT untrainable -- fp16 sits at its INIT accuracy across lr 1e-5..1e-2
  (7 rungs), because gradients underflow fp16's exponent range to zero.  These cells
  pass `--dtype float32 --mixed_precision fp16`: fp32 master weights + fp32 optimizer
  state, half-precision compute, loss scaler.  [measured] that recovers float32's
  curve (0.875 vs 0.9107 at the optimum) at 1.58x the speed and 0.82x the memory.
"""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "scratchpad", "phaseR", "fp16base")
SIZES = os.path.join(ROOT, "scratchpad", "phaseR", "r310", "dataset_sizes.json")

# --- the recipe -------------------------------------------------------------
MODEL       = "roberta-base"
# ⛔⛔ 30, NOT THE RECIPE'S 10 -- and this is a DELIBERATE deviation, corrected
#   2026-09-02 after it distorted a column.
#   [R.310]'s nine PEFT arms run 30 epochs and report the MAX over them. Giving the
#   baseline 10 handicaps it against every arm it sits beside.
#   ⭐ [measured, CB seed 42, at the proxy] 10 epochs -> F1 0.4739; 30 epochs -> 0.8045.
#   **+0.33 from the epoch budget alone** -- larger than any effect this table is
#   trying to measure, and it made CB look like a baseline collapse when it was a
#   budget artefact.
#   ⚠ Stream B (fir_baseline_plan) already used stage 05's per-task epochs for exactly
#   this reason; stream A did not, and that inconsistency was mine.
EPOCHS      = 30
WARMUP_FRAC = 0.06
WEIGHT_DECAY = 0.1
MAX_LENGTH  = 128
SEEDS       = [42, 43, 44, 45, 46]
SEARCH_SEED = 42

# the published rungs, kept verbatim, plus the extension the measurement forced
PUBLISHED_LRS = [1e-5, 2e-5, 3e-5]
LRS           = [1e-5, 2e-5, 3e-5, 5e-5, 1e-4, 2e-4, 3e-4]
BATCHES       = [16, 32]           # the recipe's own two rungs

TASKS = ["cb", "mrpc", "stsb", "cola", "boolq", "sst2"]   # the REPORTED columns
# stream B's six, for the fir planner to import (fir has no dev-box state dir)
TASKS_FIR = ["rte", "mrpc", "stsb", "cola", "sst2", "qnli"]

# ⭐⭐ ONE SWEEP, CARRIED AS A PROXY -- the protocol this program already uses
#   [user, 2026-09-02; memory: hp-transfer-proxy].  The ladder is swept on ONE task
#   and the winner is carried unchanged to every reported column.
# ⛔ AND THE SELECTION TASK IS **RTE**, NOT ONE OF THE SIX.  [R.305]/[R.306] selected
#   the nine PEFT arms' RoBERTa hyperparameters on RTE and [R.310] carries them as a
#   proxy to exactly these columns.  Selecting the baseline on RTE too means:
#     * baseline and PEFT arms share ONE selection task, so the tuning is symmetric;
#     * and ALL SIX reported columns stay OUT-OF-SAMPLE for both.
#   Selecting on MRPC instead would have made MRPC in-sample for the baseline while
#   the arms beside it are in-sample on RTE -- two different in-sample tasks in one
#   table, which is not a comparison anybody can read.
# ⚠ RTE is therefore SWEPT but NOT REPORTED as a baseline column here.
SELECTION_TASK = "rte"

# ⛔ NO per-task carrying rule is needed. For the PEFT arms the transferable quantity
#   is P = lr * atom, because the atom depends on model WIDTH. Full fine-tuning has no
#   adapter and no atom: the model is identical across tasks, so the RAW (lr, batch)
#   is what transfers. Nothing to rescale, and nothing that silently should have been.


def sizes():
    with open(SIZES) as f:
        return json.load(f)


def steps_per_cell(task, batch):
    n = sizes()[task]["train"]
    return -(-n // batch) * EPOCHS          # ceil-div, then epochs


def warmup(task, batch):
    return max(1, round(WARMUP_FRAC * steps_per_cell(task, batch)))


def label(task, lr, batch, seed):
    """⛔⛔ THE EPOCH COUNT IS IN THE ID, and it must stay there.
    `done/<label>` is the resume key. When EPOCHS moved 10 -> 30 the old ids were
    IDENTICAL to the new ones, so a re-run would have SKIPPED all 68 finished cells
    and served 10-epoch numbers inside a 30-epoch table -- silently, with every
    marker and CSV looking perfectly healthy. A budget that changes the result
    belongs in the identity of the cell."""
    return f"{task}-base-lr{lr:g}-bs{batch}-ep{EPOCHS}-seed{seed}"


def cell_args(task, lr, batch):
    """The complete arg string. ⛔ No --classifier_lr: absent => ONE param group at
    --learning_rate (train_glue.py:2038), which is what full fine-tuning means.
    ⛔ No --adapter_target_modules and no adapter flags: `--optimizer adamw` with no
    suffix IS the full-FT arm (train_glue.py:686)."""
    return (f"--model_name_or_path {MODEL} --task_name {task}"
            f" --dtype float32 --mixed_precision fp16"
            f" --per_device_train_batch_size {batch}"
            f" --num_train_epochs {EPOCHS}"
            f" --num_warmup_steps {warmup(task, batch)}"
            f" --max_length {MAX_LENGTH}"
            f" --optimizer adamw"
            f" --learning_rate {lr:g}"
            f" --weight_decay {WEIGHT_DECAY}")


def cells(task, stage="search", winner=None):
    """search: the full lr x batch ladder on the SELECTION TASK at one seed.
    final:  that winner carried UNCHANGED to every reported column, all five seeds."""
    out = []
    if stage == "search":
        # ⛔ THE SWEEP RUNS ON THE SELECTION TASK ONLY. An earlier version of this
        #   function swept EVERY task -- 84 cells and six different winners, which is
        #   not a proxy protocol at all. [user] corrected it.
        if task not in (None, SELECTION_TASK):
            return []
        for batch in BATCHES:
            for lr in LRS:
                out.append((label(SELECTION_TASK, lr, batch, SEARCH_SEED), SEARCH_SEED,
                            cell_args(SELECTION_TASK, lr, batch)))
    else:
        # ⛔ FAIL CLOSED: the proxy comes from the SELECTION TASK's sweep, read off
        #   disk. No sweep, no final cells -- never a guessed default.
        if winner is None:
            winner = best_config(SELECTION_TASK)
        if winner is None:
            return []
        lr, batch = winner
        for t in ([task] if task else TASKS):
            for seed in SEEDS:
                out.append((label(t, lr, batch, seed), seed, cell_args(t, lr, batch)))
    return out


# --- reading ----------------------------------------------------------------
# ⛔⛔ THE METRIC IS NOT TYPED HERE. It is `train_glue.py`'s own `_METRIC_FOR_TASK`,
#   parsed from source by `r310_read.metric_for_task()` -- the SAME authority the
#   [R.310] table uses, so a baseline row and the arm rows beside it cannot be two
#   different quantities.
# ⛔ THE BUG THIS FIXES [2026-09-02]: this map was hand-written with "cb": "accuracy".
#   **CB'S METRIC IS F1.** The baseline's CB number was therefore an ACCURACY being
#   compared against the nine arms' F1 -- and the memory `r310-multitask-launched`
#   records this exact trap ("the read-`accuracy`-on-a-non-accuracy-task trap") from
#   a previous occurrence. ⭐ Re-typing a table that already exists is how a class
#   that was closed once reopens. Ask the authority; never restate it.
def _metric_map():
    import r310_read as R
    return R.metric_for_task()


class _MetricView(dict):
    """Reads through to train_glue's map, defaulting to accuracy like r310_read does."""
    def __missing__(self, k):
        return _metric_map().get(k, "accuracy")


METRIC = _MetricView(_metric_map())


def _rows(task):
    import csv
    d = os.path.join(STATE, "csv")
    out = []
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        if not f.startswith(f"{task}-base-") or not f.endswith(".csv"):
            continue
        try:
            rs = list(csv.DictReader(open(os.path.join(d, f))))
        except Exception:
            continue
        if not rs:
            continue
        r = rs[-1]
        v = r.get(METRIC[task])
        if v in (None, "", "N/A"):
            continue
        stem = f[:-4]
        # ⛔ ONLY CELLS AT THE CURRENT EPOCH BUDGET. The 10-epoch cells from before the
        #   2026-09-02 correction are still on disk (deliberately -- they are the
        #   evidence for the +0.33 epoch effect), and mixing them into this table would
        #   be the very confound the correction removed.
        if "-ep" not in stem:
            continue                      # pre-correction id, no budget recorded at all
        try:
            ep = int(stem.split("-ep")[1].split("-seed")[0])
        except (IndexError, ValueError):
            continue
        if ep != EPOCHS:
            continue
        lr = float(stem.split("-lr")[1].split("-bs")[0])
        bs = int(stem.split("-bs")[1].split("-ep")[0])
        sd = int(stem.split("-seed")[1])
        out.append(dict(lr=lr, bs=bs, seed=sd, epochs=ep, score=float(v), label=stem))
    return out


def best_config(task):
    """⛔ Selection uses the SEARCH SEED ONLY. Picking on all seeds and then reporting
    those same seeds is selection on the test set."""
    rs = [r for r in _rows(task) if r["seed"] == SEARCH_SEED]
    if not rs:
        return None
    b = max(rs, key=lambda r: r["score"])
    return (b["lr"], b["bs"])


def _ladder(t):
    rs = [r for r in _rows(t) if r["seed"] == SEARCH_SEED]
    out = []
    for bs in BATCHES:
        row = "  ".join(f"{r['lr']:g}:{r['score']:.4f}"
                        for r in sorted(rs, key=lambda x: x["lr"]) if r["bs"] == bs)
        if row:
            out.append(f"  bs={bs:<3} {row}")
    return rs, out


def read(task=None):
    import statistics
    n = len(BATCHES) * len(LRS)

    # --- the SWEEP, on the selection task ---
    rs, rows = _ladder(SELECTION_TASK)
    print(f"\n### SWEEP on {SELECTION_TASK.upper()} (selection task, seed {SEARCH_SEED})"
          f"  metric={METRIC[SELECTION_TASK]}   {len(rs)}/{n}")
    for r in rows:
        print(r)
    prox = best_config(SELECTION_TASK)
    if prox is None:
        print("  ⛔ no sweep results yet -- the proxy is UNDEFINED and no final cell can run")
        return
    if len(rs) < n:
        print(f"  ⛔ INCOMPLETE ({len(rs)}/{n}) -- this winner is provisional")
    print(f"  ⭐ PROXY carried to every column: lr={prox[0]:g} bs={prox[1]}")

    # --- transfer evidence: any OTHER task that happens to have a full ladder ---
    for t in TASKS:
        rs_t, rows_t = _ladder(t)
        if len(rs_t) < n:
            continue
        w = best_config(t)
        agree = "AGREES with the proxy" if w == prox else f"DISAGREES -- its own optimum is lr={w[0]:g} bs={w[1]}"
        print(f"\n### transfer check -- {t} was ALSO swept ({len(rs_t)}/{n}); it {agree}")
        for r in rows_t:
            print(r)
        at_proxy = [r["score"] for r in rs_t if (r["lr"], r["bs"]) == prox]
        at_own = [r["score"] for r in rs_t if (r["lr"], r["bs"]) == w]
        if at_proxy and at_own:
            print(f"  ⚠ cost of carrying the proxy here: {at_proxy[0]:.4f} vs {at_own[0]:.4f} "
                  f"= {at_proxy[0]-at_own[0]:+.4f}  (n=1, seed {SEARCH_SEED} -- not an estimate of much)")

    # --- the reported table ---
    print(f"\n### BASELINE COLUMNS at the proxy (median over {len(SEEDS)} seeds)")
    for t in TASKS:
        fin = [r["score"] for r in _rows(t) if (r["lr"], r["bs"]) == prox]
        if not fin:
            print(f"  {t:6} --")
            continue
        flag = "" if len(fin) == len(SEEDS) else f"   ⛔ {len(fin)}/{len(SEEDS)} seeds -- INCOMPLETE, do not quote"
        print(f"  {t:6} {statistics.median(fin):.4f}   n={len(fin)}  "
              f"(min {min(fin):.4f} max {max(fin):.4f}){flag}")


# --- selftest ---------------------------------------------------------------
def selftest():
    ok, bad = [], []

    def ck(c, m):
        (ok if c else bad).append(m)

    for lr in PUBLISHED_LRS:
        ck(lr in LRS, f"the PUBLISHED rung {lr:g} is inside the swept ladder")
    ck(max(LRS) > max(PUBLISHED_LRS), "the ladder extends ABOVE the published grid "
                                      "(measured: the optimum is at 1e-4)")
    ck(BATCHES == [16, 32], "the recipe's own two batch rungs, unchanged")
    ck(WEIGHT_DECAY == 0.1, "recipe: weight decay 0.1")
    ck(EPOCHS == 30,
       "⛔ 30 epochs, matching [R.310]'s arms -- NOT the recipe's 10. [measured] the "
       "budget alone is worth +0.33 F1 on CB, so a 10-epoch baseline beside 30-epoch "
       "arms measures the budget, not the method")
    # ⛔⛔ THE EPOCH BUDGET MUST BE PART OF THE CELL IDENTITY.
    ck(f"-ep{EPOCHS}-" in label("cb", 1e-5, 32, 42),
       "⛔ CONTROL: the cell id carries the epoch budget, so changing it cannot make a "
       "new run SKIP old cells via their `done` markers")
    ck(label("cb", 1e-5, 32, 42) != f"cb-base-lr1e-05-bs32-seed42",
       "⛔ CONTROL: and the id differs from the pre-correction 10-epoch form")
    ck(all(r.get("epochs") == EPOCHS for t in TASKS for r in _rows(t)),
       "⛔ CONTROL: the reader returns ONLY cells at the current epoch budget -- the "
       "10-epoch cells still on disk must never enter this table")

    for t in TASKS:
        for b in BATCHES:
            s = cell_args(t, 1e-4, b)
            ck("--mixed_precision fp16" in s, f"{t}/bs{b}: real AMP, not a cast")
            ck("--dtype float32" in s, f"{t}/bs{b}: fp32 master weights")
            ck("--classifier_lr" not in s, f"{t}/bs{b}: NO separate head lr (full FT is one group)")
            ck("adapter" not in s, f"{t}/bs{b}: no adapter flag survives")
            ck("--optimizer adamw " in s + " ", f"{t}/bs{b}: the plain full-FT optimizer")
            ck(f"--max_length {MAX_LENGTH}" in s, f"{t}/bs{b}: max_length pinned, not defaulted")
            w = warmup(t, b)
            ck(abs(w / steps_per_cell(t, b) - WARMUP_FRAC) < 0.02,
               f"{t}/bs{b}: warmup is {WARMUP_FRAC:g} of THIS cell's steps ({w}/{steps_per_cell(t,b)})")
    # ⛔ THE METRIC MUST BE THE SAME AUTHORITY [R.310] USES, per task.
    import r310_read as _R
    for t in TASKS + [SELECTION_TASK]:
        ck(METRIC[t] == _R.metric_of(t),
           f"{t}: metric {METRIC[t]!r} matches r310_read/_METRIC_FOR_TASK")
    ck(METRIC["cb"] == "f1",
       "⛔ CONTROL: CB is F1, not accuracy -- the hand-typed map had this wrong and "
       "compared an accuracy against the arms' F1")
    ck(METRIC["stsb"] == "pearson" and METRIC["cola"] == "matthews_correlation",
       "⛔ CONTROL: the regression/correlation tasks keep their own metrics")

    # ⭐ ONE SWEEP, CARRIED. The search must exist for the SELECTION TASK and for
    #   nothing else -- that is the whole protocol, and the earlier per-task version
    #   silently violated it.
    ck(SELECTION_TASK not in TASKS,
       f"the selection task ({SELECTION_TASK}) is NOT one of the reported columns, so "
       f"every reported column is out-of-sample")
    srch = cells(None, "search")
    ck(len(srch) == len(LRS) * len(BATCHES),
       f"the sweep is {len(LRS)}x{len(BATCHES)} = {len(srch)} cells, on ONE task")
    ck({c[0].split('-base-')[0] for c in srch} == {SELECTION_TASK},
       f"⛔ CONTROL: every swept cell is on {SELECTION_TASK}, none on a reported column")
    for t in TASKS:
        ck(cells(t, "search") == [], f"⛔ CONTROL: asking to sweep {t} yields ZERO cells")
    labs = [c[0] for c in srch]
    ck(len(labs) == len(set(labs)), f"{len(labs)} search labels, all distinct")

    # the final stage: the proxy, unchanged, on every reported column at every seed
    fin = cells(None, "final", winner=(1e-4, 32))
    ck(len(fin) == len(TASKS) * len(SEEDS),
       f"final is {len(TASKS)} columns x {len(SEEDS)} seeds = {len(fin)} cells")
    ck({c[0].split('-base-')[0] for c in fin} == set(TASKS),
       "final covers exactly the reported columns")
    ck(all("lr0.0001-bs32" in c[0] for c in fin),
       "⛔ CONTROL: EVERY final cell carries the SAME (lr, batch) -- that is what "
       "'carried as a proxy' means, and a per-task winner here would be a silent sweep")
    ck(len({c[0] for c in fin}) == len(fin), "final labels are distinct")
    ck(cells(None, "final", winner=None) == [] or best_config(SELECTION_TASK) is not None,
       "⛔ CONTROL: with no sweep on disk the final stage emits ZERO cells")

    for l in ok:
        print(f"  ✅ {l}")
    for l in bad:
        print(f"  ⛔ {l}")
    print(f"\n{len(ok)} passed, {len(bad)} failed")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--generate")
    ap.add_argument("--stage", default="search", choices=["search", "final"])
    ap.add_argument("--tasks", action="store_true")
    ap.add_argument("--selection-task", action="store_true")
    ap.add_argument("--proxy", action="store_true")
    ap.add_argument("--read", action="store_true")
    ap.add_argument("--task")
    ap.add_argument("--cost", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    if a.tasks:
        print(" ".join(TASKS)); return
    if a.selection_task:
        print(SELECTION_TASK); return
    if a.proxy:
        w = best_config(SELECTION_TASK)
        if w is None:
            sys.exit(f"no sweep results for {SELECTION_TASK} -- the proxy is UNDEFINED")
        print(f"{w[0]:g} {w[1]}"); return
    if a.read:
        read(a.task); return
    if a.cost:
        # [measured, CB fp16-AMP] 0.144 s/step. ⚠ AN ESTIMATE ACROSS TASKS: per-step
        # time depends on sequence length and padding is dynamic. Size a wall from a
        # canary, never from this.
        sps = 0.144
        tot = 0.0
        for t in TASKS:
            for b in BATCHES:
                tot += steps_per_cell(t, b) * sps
        print(f"search stage (1 seed): {tot/3600:.1f} GPU-h  [ESTIMATE from CB's s/step]")
        return
    if a.generate:
        t = a.generate
        os.makedirs(STATE, exist_ok=True)
        cs = cells(t, a.stage)
        with open(os.path.join(STATE, f"jobs_{t}.tsv"), "w") as f:
            for lab, sd, args_ in cs:
                f.write(f"{lab}\t{sd}\t{args_}\n")
        print(f"[fp16base] {t} stage={a.stage}: {len(cs)} cells")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
