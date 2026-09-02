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
EPOCHS      = 10
WARMUP_FRAC = 0.06
WEIGHT_DECAY = 0.1
MAX_LENGTH  = 128
SEEDS       = [42, 43, 44, 45, 46]
SEARCH_SEED = 42

# the published rungs, kept verbatim, plus the extension the measurement forced
PUBLISHED_LRS = [1e-5, 2e-5, 3e-5]
LRS           = [1e-5, 2e-5, 3e-5, 5e-5, 1e-4, 2e-4, 3e-4]
BATCHES       = [16, 32]           # the recipe's own two rungs

TASKS = ["cb", "mrpc", "stsb", "cola", "boolq", "sst2"]   # stream A, per [user]
# stream B's six, for the fir planner to import (fir has no dev-box state dir)
TASKS_FIR = ["rte", "mrpc", "stsb", "cola", "sst2", "qnli"]


def sizes():
    with open(SIZES) as f:
        return json.load(f)


def steps_per_cell(task, batch):
    n = sizes()[task]["train"]
    return -(-n // batch) * EPOCHS          # ceil-div, then epochs


def warmup(task, batch):
    return max(1, round(WARMUP_FRAC * steps_per_cell(task, batch)))


def label(task, lr, batch, seed):
    return f"{task}-base-lr{lr:g}-bs{batch}-seed{seed}"


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
    """search: the full ladder at ONE seed.  final: the winning config at the other
    four seeds (seed 42 is already on disk from the search -- never re-run)."""
    out = []
    if stage == "search":
        for batch in BATCHES:
            for lr in LRS:
                out.append((label(task, lr, batch, SEARCH_SEED), SEARCH_SEED,
                            cell_args(task, lr, batch)))
    else:
        if winner is None:
            winner = best_config(task)
        if winner is None:
            return []
        lr, batch = winner
        for seed in SEEDS:
            if seed == SEARCH_SEED:
                continue
            out.append((label(task, lr, batch, seed), seed, cell_args(task, lr, batch)))
    return out


# --- reading ----------------------------------------------------------------
METRIC = {"cb": "accuracy", "mrpc": "f1", "stsb": "pearson",
          "cola": "matthews_correlation", "boolq": "accuracy", "sst2": "accuracy"}


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
        lr = float(stem.split("-lr")[1].split("-bs")[0])
        bs = int(stem.split("-bs")[1].split("-seed")[0])
        sd = int(stem.split("-seed")[1])
        out.append(dict(lr=lr, bs=bs, seed=sd, score=float(v), label=stem))
    return out


def best_config(task):
    """⛔ Selection uses the SEARCH SEED ONLY. Picking on all seeds and then reporting
    those same seeds is selection on the test set."""
    rs = [r for r in _rows(task) if r["seed"] == SEARCH_SEED]
    if not rs:
        return None
    b = max(rs, key=lambda r: r["score"])
    return (b["lr"], b["bs"])


def read(task=None):
    import statistics
    for t in ([task] if task else TASKS):
        rs = _rows(t)
        srch = [r for r in rs if r["seed"] == SEARCH_SEED]
        print(f"\n### {t}  metric={METRIC[t]}  search {len(srch)}/{len(BATCHES)*len(LRS)}")
        if srch:
            for bs in BATCHES:
                row = "  ".join(f"{r['lr']:g}:{r['score']:.4f}"
                                for r in sorted(srch, key=lambda x: x["lr"]) if r["bs"] == bs)
                print(f"  bs={bs:<3} {row}")
            w = best_config(t)
            print(f"  ⭐ winner (seed {SEARCH_SEED} only): lr={w[0]:g} bs={w[1]}")
            fin = [r["score"] for r in rs if (r["lr"], r["bs"]) == w]
            if len(fin) >= 2:
                flag = "" if len(fin) == len(SEEDS) else f"  ⛔ {len(fin)}/{len(SEEDS)} seeds -- INCOMPLETE"
                print(f"  median over {len(fin)} seeds: {statistics.median(fin):.4f}"
                      f"  (min {min(fin):.4f} max {max(fin):.4f}){flag}")


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
    ck(EPOCHS == 10 and WEIGHT_DECAY == 0.1, "recipe: 10 epochs, weight decay 0.1")

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
    # one label per (task, lr, batch, seed) -- a collision silently overwrites a CSV
    labs = [c[0] for t in TASKS for c in cells(t, "search")]
    ck(len(labs) == len(set(labs)), f"{len(labs)} search labels, all distinct")
    ck(len(labs) == len(TASKS) * len(LRS) * len(BATCHES),
       f"search grid is {len(TASKS)}x{len(LRS)}x{len(BATCHES)} = {len(labs)} cells")
    # the final stage must never re-run the search seed
    fin = cells(TASKS[0], "final", winner=(1e-4, 32))
    ck(all(int(c[1]) != SEARCH_SEED for c in fin), "final stage skips the search seed (already on disk)")
    ck(len(fin) == len(SEEDS) - 1, f"final stage adds {len(SEEDS)-1} seeds")

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
    ap.add_argument("--read", action="store_true")
    ap.add_argument("--task")
    ap.add_argument("--cost", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())
    if a.tasks:
        print(" ".join(TASKS)); return
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
