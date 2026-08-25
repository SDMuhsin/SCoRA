#!/usr/bin/env python
"""[R.310] MULTI-TASK CAMERA-READY PLANNER.

WHAT THIS RUNS
  The `[R.305]`/`[R.306]` arm set, at its SELECTED hyperparameters, on seven new
  tasks: MRPC, STS-B, CoLA, SST-2, MNLI, BoolQ, CB.  9 arms x 5 seeds x 7 tasks
  = 315 confirmation cells, plus a 4-cell CB spot-check.  RTE is NOT re-run --
  it is already measured and is imported by the reader as the eighth column.

⛔ THE PROXY THIS RESTS ON, AND WHAT IS AND IS NOT FREE
  `llmdocs/baseline_hp_search_results.md` §0 is the authority.  Two clauses bite:

  (1) *"Carry P = lr*atom, not the raw learning rate"* -- the atom depends on
      MODEL WIDTH and on the arm's scale parameter.  Here the model is still
      `roberta-base` (d=768) and the budget is still k=256, so every atom is
      unchanged and **the raw `lr` transfers verbatim**.  This is the one
      condition under which copying `--learning_rate` is legitimate, and it
      stops being true the moment the model or `k` moves.  `selftest` asserts
      the arm strings came from the run record and were not retyped.

  (2) *"The epoch budget must NOT be carried over silently"* -- 30 epochs and
      140 warmup steps were chosen for a 2,490-example task.  THE USER'S
      DECISION (2026-08-23) IS 30 EPOCHS ON EVERY TASK, taken with the cost
      table in hand (MNLI alone is ~997 h of the ~1,238 h total).  That is
      recorded here, not re-argued.

⭐ WHAT IS NOT FREE, AND THE ONE THING THIS FILE DERIVES
  Holding `--num_train_epochs 30` does NOT hold `--num_warmup_steps 140`, because
  warmup is ABSOLUTE while the step count is not.  At 140 steps flat, warmup
  would be **58% of CB's whole run** (240 steps) and **0.04% of MNLI's** (368,160)
  -- and this repo measures warmup as worth +0.0036..+0.0450 `[R.67/R.68/R.78]`,
  so that is not a rounding error, it is a different protocol per task.  So the
  one setting derived here is the warmup RATIO:

      warmup_steps(task) = round( (140 / 2340) * steps(task) )

  i.e. RTE's own 5.983%, preserved.  Every other flag is copied unchanged.
  ⚠️ This is a DESIGN CHOICE, declared: it keeps the *shape* of the schedule
  fixed while the user's decision fixes the *length*.  Holding 140 absolute was
  the alternative and it is rejected for the CB/MNLI reason above.

⭐ THE REPORTED NUMBER IS THE MAX OVER THE 30 EPOCHS (user, 2026-08-24).
  That is already what `train_glue.py` records -- `best_metric_dict` tracks the
  per-epoch argmax of the task's OWN primary metric (`train_glue.py:2394`), so
  the new columns are produced by the identical code path as `[R.305]`'s RTE
  column and are comparable to it.  `[R.310]` adds a `best_epoch` column so the
  claim is checkable: an arm whose argmax sits at epoch 29 was still improving,
  and its number is a LOWER BOUND, not a converged one.

⛔ TRAPS THIS FILE IS BUILT AROUND
  T1 [R.303] `_upsert_result`'s key omits `seed`.  ONE CSV PER (task, arm, seed),
     with the seed in the filename.  `scripts/r304_upsert_gate.py` enforces it.
  T2 [R.306 §6.6] re-pointing a module global re-points ONLY that global.  This
     file NEVER mutates `r305_plan.D`; it reads the two prior runs through
     explicit paths and snapshots what it needs at import.
  T3 CONTEXT §4.4: `--adapter_target_modules` and `--dtype` must be passed
     EXPLICITLY.  Both are in COMMON and `selftest` asserts they survive.
  T4 CONTEXT §4.4: STS-B is REGRESSION (metric `pearson`) -- majority-class
     reasoning does not apply to it and the reader must not invent a floor.
  T5 The primary metric DIFFERS per task (`cola->matthews_correlation`,
     `stsb->pearson`, `mrpc/cb->f1`).  Reading the `accuracy` column, as
     `r305_plan.load()` does, would silently report the WRONG number on 4 of 8
     columns.  `r310_read.py` reads `_METRIC_FOR_TASK` from `train_glue.py`.

ORDER OF EXECUTION -- CHEAPEST TASK FIRST.  At ~1,238 h this grid cannot be
treated as atomic; the ordering below means five of seven tasks are complete
within ~3 days and every intermediate stopping point is a usable table.
"""
import argparse, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

D = os.path.join(ROOT, "scratchpad", "phaseR", "r310")
SIZES_PATH = os.path.join(D, "dataset_sizes.json")

R305_D = os.path.join(ROOT, "scratchpad", "phaseR", "r305")
R306_D = os.path.join(ROOT, "scratchpad", "phaseR", "r306")

CONFIRM_SEEDS = [42, 43, 44, 45, 46]     # identical to [R.305]/[R.306] stage D
SCREEN_SEED   = 41                       # spot-checks only; NEVER quotable

EPOCHS       = 30                        # USER DECISION 2026-08-23
BATCH        = 32
RTE_STEPS    = 2340                      # 30 * ceil(2490/32), the tuned cell
RTE_WARMUP   = 140
WARMUP_RATIO = RTE_WARMUP / RTE_STEPS    # 0.0598290598...

# task -> the order it runs in.  Cost-ascending; see the module docstring.
#
# ⭐ USER DECISION 2026-08-24: **MNLI is REPLACED BY QNLI.**  At 30 epochs MNLI is
# 368,160 steps = 546 h = 78% of the entire remaining grid.  QNLI was chosen from
# a measured candidate table as the only replacement that is (a) already wired,
# (b) verified non-degenerate -- FourierFT reaches 0.8422 on 4.8% of the data at
# 1/10 the epochs, against a 0.5054 majority -- and (c) still covers the regime
# MNLI was in the table FOR: 98,197 steps, 42x the budget the hyperparameters
# were tuned at, which is the strongest surviving test of the proxy.
# ⛔ WHAT THE SWAP COSTS, recorded so it is never quietly forgotten: the table no
# longer has a large-scale 3-CLASS task (only CB, at 250 train examples), and it
# no longer has the MNLI row.  ANLI R1 would have kept 3 classes for 24 h but is
# ADVERSARIALLY CONSTRUCTED against BERT-family models and would very likely have
# produced nine indistinguishable values at its 0.334 chance floor.
TASKS = ["cb", "mrpc", "stsb", "cola", "boolq", "sst2", "qnli"]

# The nine rows of the [R.305]/[R.306] table, in table order.  `src` says which
# run's manifest the SELECTED arg string is read out of.  ⛔ Nothing here is a
# hyperparameter: the flags come from `selected_args()` below.
ARMS = [
    ("fftm",     "FourierFT (merged)",                    "r305"),
    ("fftstock", "FourierFT (stock PEFT)",                "r305"),
    ("loca",     "LoCA",                                  "r305"),
    ("qwha",     "QWHA",                                  "r305"),
    ("wave1",    "WaveFT mu=1 (as published)",            "r305"),
    ("wave2",    "WaveFT mu=2 (this repo's rank fix)",    "r305"),
    ("lyra",     "LYRA",                                  "r305"),
    ("scora",    "SCoRA (ours) -- a-priori scaling",      "r305"),
    ("scora2",   "SCoRA (ours) -- scaling swept",         "r306"),
]
ARM_NAMES = [a for a, _, _ in ARMS]

# ⛔ BOTH SCoRA rows always ship together ([R.306]).  A planner that emitted one
# without the other would make the "we did not tune ours harder after it lost"
# claim unfalsifiable downstream.  `selftest` asserts it.
SCORA_ROWS = ("scora", "scora2")


# ============================================================================
# dataset sizes -- MEASURED, never typed (PROCESS: "never print a statistic as
# a literal, compute it").  Written by `--measure`, which loads the real
# datasets; every downstream number here is a function of this file.
# ============================================================================
def sizes():
    if not os.path.exists(SIZES_PATH):
        raise SystemExit(
            f"[r310] {SIZES_PATH} missing -- run `--measure` first.  Step counts "
            "and warmup are DERIVED from measured dataset sizes, never typed.")
    with open(SIZES_PATH) as f:
        return json.load(f)


def measure():
    """Load every task and record train/eval sizes and the eval label histogram."""
    import collections
    from datasets import load_dataset
    out = {}
    for t in ["rte", "mrpc", "stsb", "cola", "sst2", "mnli", "qnli"]:
        d = load_dataset("glue", t)
        ev = d["validation_matched" if t == "mnli" else "validation"]
        out[t] = {"train": len(d["train"]), "eval": len(ev)}
        out[t]["label_counts"] = (None if t == "stsb" else
                                  dict(sorted(collections.Counter(ev["label"]).items())))
    for t in ["boolq", "cb"]:
        d = load_dataset("super_glue", t)
        ev = d["validation"]
        out[t] = {"train": len(d["train"]), "eval": len(ev),
                  "label_counts": dict(sorted(collections.Counter(ev["label"]).items()))}
    for t, v in out.items():
        lc = v["label_counts"]
        v["majority"] = None if lc is None else max(lc.values()) / sum(lc.values())
    os.makedirs(D, exist_ok=True)
    with open(SIZES_PATH, "w") as f:
        json.dump({k: {kk: (dict((str(a), b) for a, b in vv.items())
                            if kk == "label_counts" and vv else vv)
                       for kk, vv in v.items()} for k, v in out.items()},
                  f, indent=1, sort_keys=True)
    print(f"[r310] measured {len(out)} tasks -> {SIZES_PATH}")
    return out


def steps_per_epoch(task, S=None):
    S = S or sizes()
    return math.ceil(S[task]["train"] / BATCH)


def total_steps(task, S=None):
    return EPOCHS * steps_per_epoch(task, S)


def warmup_for(task, S=None):
    """RTE's warmup RATIO, applied to this task's step count.  See docstring."""
    return int(round(WARMUP_RATIO * total_steps(task, S)))


def common(task, S=None):
    return ("--model_name_or_path roberta-base"
            f" --task_name {task}"
            " --dtype float32"
            " --adapter_target_modules query,value"
            f" --per_device_train_batch_size {BATCH}"
            f" --num_train_epochs {EPOCHS}"
            f" --num_warmup_steps {warmup_for(task, S)}")


# ============================================================================
# the selected hyperparameters -- READ OUT OF THE RUN RECORD
# ============================================================================
def _manifest(path):
    p = os.path.join(path, "manifest.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def selected_args(arm=None):
    """{arm: args} -- the EXACT flag string whose 5-seed mean is the reported
    `[R.305]`/`[R.306]` number for that arm.

    ⛔ Derived, never typed.  `r305_read.tuned(rec)` returns the winning
    CANDIDATE, keyed by the SELECTION-cell label it came from; the args that
    actually ran are on the stage-D manifest entries whose `from` is that label.
    Reading the selection cell's own args instead is wrong for `fftstock`, whose
    candidate is imported from `fftm`'s plane but which RUNS a different
    optimiser (`adamw-fourierft`) -- a hand-copy would have silently shipped
    eight identical FourierFT columns.
    """
    import r305_read as R5
    import r306_read as R6r
    conf5, data5 = R6r.r305_confirmed()
    data6 = R6r.r306_data()[0]
    mans = {"r305": _manifest(R305_D), "r306": _manifest(R306_D)}
    out = {}
    for a, _title, src in ARMS:
        if arm and a != arm:
            continue
        rec = data6.get("scora2") if src == "r306" else data5.get(a)
        t = R5.tuned(rec) if rec else None
        if not t:
            continue
        want = t[0]
        cand = sorted(lab for lab, m in mans[src].items()
                      if m.get("arm") == a and m.get("stage") == "D"
                      and m.get("from") == want)
        if not cand:
            continue
        argset = {mans[src][c]["args"] for c in cand}
        assert len(argset) == 1, f"{a}: stage-D cells disagree on args: {argset}"
        out[a] = argset.pop()
    return out


def rte_reference():
    """{arm: (mean, sd, per_seed)} -- the already-measured RTE column."""
    import r305_read as R5
    import r306_read as R6r
    _c, data5 = R6r.r305_confirmed()
    data6 = R6r.r306_data()[0]
    out = {}
    for a, _t, src in ARMS:
        rec = data6.get("scora2") if src == "r306" else data5.get(a)
        t = R5.tuned(rec) if rec else None
        if t:
            out[a] = (t[1], t[2], t[4])
    return out


# ============================================================================
# manifest + job emission
# ============================================================================
def manifest_path():
    """⛔ [R.306 §6.6]: a FUNCTION, not an import-time constant off `D`.  The
    trap this repo already hit was a derived path frozen at import while the
    directory global moved underneath it, writing 25 foreign cells into another
    run's manifest."""
    return os.path.join(D, "manifest.json")


def read_manifest():
    p = manifest_path()
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def write_manifest(m):
    os.makedirs(D, exist_ok=True)
    p = manifest_path()
    with open(p + ".tmp", "w") as f:
        json.dump(m, f, indent=1, sort_keys=True)
    os.replace(p + ".tmp", p)


def label_for(task, arm, seed):
    return f"{task}-{arm}-seed{seed}"


def plan_task(task, manifest, sel=None, S=None):
    """The (label, seed, args) list for one task.  Idempotent: a label already
    in the manifest is never re-emitted -- `generate()` APPENDS to the job file
    and the orchestrator re-runs every task on resume, so a stage that re-emits
    its own cells doubles the job list on each relaunch (measured, [R.305])."""
    sel = selected_args() if sel is None else sel
    S = S or sizes()
    jobs = []
    for arm in ARM_NAMES:
        if arm not in sel:
            continue
        for seed in CONFIRM_SEEDS:
            lab = label_for(task, arm, seed)
            if lab in manifest:
                continue
            args = f"{common(task, S)} {sel[arm]}"
            manifest[lab] = {"task": task, "arm": arm, "stage": "D",
                             "seed": seed, "args": args}
            jobs.append((lab, seed, args))
    jobs += plan_spot(task, manifest, sel, S)
    return jobs


# ---- the one spot-check, and why it is small -------------------------------
# CONTEXT §4.4: "CB and MNLI are 3-class => the classification head is a
# different size, so the --classifier_lr 5e-3 derivation should be re-checked".
# The head is dense(768->768) + out_proj(768->L): 590,592 + 768L + L.  L=2 gives
# 592,130 and L=3 gives 592,899 -- a 0.13% change, so the DERIVATION barely
# moves.  That argument is cheap to state and cheap to check on CB (0.04 h/cell)
# and ruinous to check on MNLI (66 h/cell), so it is checked on CB only and the
# MNLI case is left explicitly unchecked rather than silently assumed.
# ⛔ n=1 at the SELECTION seed.  PROCESS §5: never quotable as a result.
SPOT_TASKS = ("cb",)
SPOT_ARMS  = ("fftm", "scora")
# ⛔ "5e-3" IS THE CONTROL and it is not optional.  Without a cell at the SHIPPED
# value on the SAME seed, the alternatives can only be compared against the
# 5-seed confirmation mean, which runs on seeds 42-46 -- a different seed set.
# That comparison mixes a single draw with a mean over five others and cannot
# distinguish "the 3-class head moved the optimum" from "seed 41 is high".
# A spot-check without its own control is PROCESS §1's "control that cannot fail".
SPOT_CLF   = ("1e-3", "5e-3", "1e-2")


def plan_spot(task, manifest, sel=None, S=None):
    if task not in SPOT_TASKS:
        return []
    sel = selected_args() if sel is None else sel
    S = S or sizes()
    jobs = []
    for arm in SPOT_ARMS:
        if arm not in sel:
            continue
        for clf in SPOT_CLF:
            lab = f"{task}-{arm}-spot-clf{clf}"
            if lab in manifest:
                continue
            base = " ".join(_drop_flag(sel[arm].split(), "--classifier_lr"))
            args = f"{common(task, S)} {base} --classifier_lr {clf}"
            manifest[lab] = {"task": task, "arm": arm, "stage": "spot",
                             "seed": SCREEN_SEED, "args": args}
            jobs.append((lab, SCREEN_SEED, args))
    return jobs


# ============================================================================
# ⛔⛔ NO PER-TASK HYPERPARAMETER SWEEPS.  USER DECISION 2026-08-24.
#
# `[R.310]`'s design is: run every task at `[R.305]`/`[R.306]`'s RTE-SELECTED
# hyperparameters, as a declared proxy.  Per-task re-tuning is NOT affordable --
# a 3-rung ladder for 9 arms on 7 tasks is 189 extra cells, and a real re-tune is
# the whole `[R.305]` search (293 cells) times seven.  **The limitation is
# acknowledged in the writeup instead of being bought out.**
#
# A `psweep` stage briefly existed here and was REMOVED.  It ran once, on STS-B
# (12 cells + 4 extension cells, ~1.5 GPU-hours), before the decision above; that
# data is kept as an incidental observation for the limitations section and is
# read by `r310_read.psweep()`.  It is EXCLUDED from every results table, like
# every non-`D` stage.  Nothing here can emit such a cell again, and `selftest`
# asserts it.
#
# ⛔ WHAT THE RTE SEARCH DOES AND DOES NOT ESTABLISH -- the distinction that
# matters for the writeup, and the reason a sweep here was never the fix:
#   * ESTABLISHED: each arm's lr is optimal FOR RTE.  All 8 ladders were
#     BRACKETED -- 5 to 10 rungs each, every selected value strictly interior,
#     `[R.305]` stage B's edge-extension rule having already fired where needed.
#     The RTE search is complete; nothing about it is in question.
#   * NOT ESTABLISHED, AND NEVER CLAIMED: that the RTE-optimal lr is optimal on
#     any other task.  `baseline_hp_search_results.md` §0.3/§6 said so in advance:
#     "the optimum moves with the step budget ... treat the P values as a centre
#     for a short re-sweep, not as a final answer."
# So every column here is a MATCHED-PROTOCOL comparison -- every arm gets the
# identical, preregistered treatment -- and is not a claim about any arm's best
# achievable score on that task.  That is the limitation to state plainly.


def _drop_flag(toks, flag):
    out, i = [], 0
    while i < len(toks):
        if toks[i] == flag:
            i += 2
            continue
        out.append(toks[i])
        i += 1
    return out


def generate(task):
    os.makedirs(os.path.join(D, "csv"), exist_ok=True)
    man = read_manifest()
    jobs = plan_task(task, man)
    write_manifest(man)
    path = os.path.join(D, f"jobs_{task}.tsv")
    with open(path, "a") as f:
        for lab, seed, args in jobs:
            f.write(f"{lab}\t{seed}\t{args}\n")
    print(f"[r310] task {task}: emitted {len(jobs)} new cells -> {path}")
    return len(jobs)


# ============================================================================
# ⭐ MEASURED cost model, [R.310] 2026-08-24, from 141 completed cells.
#
# ⛔ THIS REPLACES A RETRACTED MODEL, TWICE OVER.
#  (1) `[R.305]`'s flat **0.65 s/step with no startup term** was a projection made
#      on RTE and is 1.8x too pessimistic here.
#  (2) An intermediate two-point fit `a*steps + b*eval_examples` was ALSO wrong:
#      exactly determined, zero residual, no way to check itself.  The third task
#      falsified it -- refitting on three drives the eval coefficient NEGATIVE
#      (-0.0057 s/eval-example), which is impossible.  It had absorbed CB's fixed
#      startup into a fictitious evaluation term, which under-weighted the
#      per-step term and made the step-heavy tasks look far cheaper than they are.
#
# The surviving model is fixed startup + per-step; EVALUATION IS NOT A MATERIAL
# TERM.  Three tasks spanning 22x in step count agree on the per-step rate to
# within 8%, which is the check the two-point fit could not do.
# ⭐ THE PER-STEP RATE DEPENDS ON INPUT SHAPE, by 3.2x.  Batches are padded to the
# longest member (`--pad_to_max_length` is NOT set), so a single-sentence task
# runs far shorter sequences than a sentence-PAIR task at the same batch size:
#   CoLA (single)  0.112 s/step   |   STS-B / MRPC (pair)  0.328-0.356 s/step
# Projecting SST-2 off the pair rate over-costs it ~3x.  The shape is not a
# judgement call -- `train_glue.task_to_keys[t][1] is None` decides it.
STARTUP_S    = 68.0           # process start, model load, tokenisation
S_PER_STEP   = {"single": (0.1068, 0.1189),    # CoLA, min/max over its 5 cells
                "pair":   (0.3277, 0.3557)}    # STS-B, MRPC
SINGLE_SENTENCE = {"cola", "sst2"}   # second key is None in `task_to_keys`


def projected_cell_s(task, S=None, rate=None):
    """(low, high) seconds for one cell.  A RANGE, because the per-step rate
    depends on sequence length and only sentence-pair tasks are measured."""
    st = total_steps(task, S)
    shape = "single" if task in SINGLE_SENTENCE else "pair"
    lo, hi = S_PER_STEP[shape] if rate is None else (rate, rate)
    return STARTUP_S + lo * st, STARTUP_S + hi * st


def measured_cell_s(task):
    """Mean seconds/cell for CONFIRMATION cells of `task`, from the driver log.
    Returns None if the task has not run.  ⛔ Measured beats projected always."""
    import re
    path = os.path.join(D, "run.log")
    if not os.path.exists(path):
        return None
    v = [int(m) for m in re.findall(rf"OK   {task}-[a-z0-9.]+-seed\d+  (\d+)s",
                                    open(path).read())]
    return (sum(v) / len(v), len(v)) if v else None


def cost_table():
    S = sizes()
    W = 3
    lo_t = hi_t = 0.0
    print(f"{'task':7s} {'train':>7s} {'steps':>7s} {'warmup':>7s} {'cells':>6s} "
          f"{'h/cell':>14s} {'task wall_h':>14s}  source")
    for t in TASKS:
        st, wu = total_steps(t, S), warmup_for(t, S)
        n = len(ARM_NAMES) * len(CONFIRM_SEEDS) + (len(SPOT_ARMS) * len(SPOT_CLF)
                                                   if t in SPOT_TASKS else 0)
        m = measured_cell_s(t)
        if m:
            lo = hi = m[0]
            src = f"MEASURED n={m[1]}"
        else:
            lo, hi = projected_cell_s(t, S)
            src = "projected"
        lo_t += lo * n / W / 3600
        hi_t += hi * n / W / 3600
        hc = f"{lo/3600:.2f}" if lo == hi else f"{lo/3600:.2f}-{hi/3600:.2f}"
        tw = f"{lo*n/W/3600:.1f}" if lo == hi else f"{lo*n/W/3600:.1f}-{hi*n/W/3600:.1f}"
        print(f"{t:7s} {S[t]['train']:7d} {st:7d} {wu:7d} {n:6d} {hc:>14s} {tw:>14s}  {src}")
    print(f"\nWHOLE GRID {lo_t:.0f}-{hi_t:.0f} h wall at {W} workers "
          f"= {lo_t/24:.1f}-{hi_t/24:.1f} days")
    rem_lo = rem_hi = 0.0
    for t in TASKS:
        if measured_cell_s(t):
            continue
        n = len(ARM_NAMES) * len(CONFIRM_SEEDS)
        lo, hi = projected_cell_s(t, S)
        rem_lo += lo * n / W / 3600
        rem_hi += hi * n / W / 3600
    print(f"REMAINING  {rem_lo:.0f}-{rem_hi:.0f} h = {rem_lo/24:.1f}-{rem_hi/24:.1f} days")
    print("⚠️  ranges are the measured per-step spread for the task's SHAPE "
          "(single-sentence\n    vs sentence-pair).  ⛔ BoolQ is the one task whose "
          "passages are much longer than\n    any measured pair task, so its "
          "projection is the most likely to come in HIGH.")


# ============================================================================
# SELFTEST -- PROCESS §1: controls as CODE, each able to FAIL.
# ============================================================================
def selftest():
    n = [0]

    def ck(cond, msg):
        n[0] += 1
        if not cond:
            print(f"FAIL: {msg}")
            selftest.failed = getattr(selftest, "failed", 0) + 1
    selftest.failed = 0

    S = sizes()

    # ---- 1. the derived warmup, and the defect it exists to prevent --------
    ck(abs(WARMUP_RATIO - 140 / 2340) < 1e-12, "warmup ratio is RTE's own")
    ck(warmup_for("cb", S) == int(round(WARMUP_RATIO * total_steps("cb", S))),
       "cb warmup is derived from its own step count")
    # CONTROL: the rejected alternative must be VISIBLY different, or the
    # derivation is decorative.  140 flat would be 58% of CB and 0.04% of MNLI.
    ck(warmup_for("cb", S) < 140 / 4, "flat-140 would over-warm CB (control)")
    ck(warmup_for("mnli", S) > 140 * 100, "flat-140 would under-warm MNLI (control)")
    for t in TASKS:
        r = warmup_for(t, S) / total_steps(t, S)
        ck(abs(r - WARMUP_RATIO) < 0.01, f"{t}: warmup ratio preserved ({r:.4f})")

    # ---- 2. step counts are DERIVED from measured sizes, not typed ---------
    ck(steps_per_epoch("rte", S) * EPOCHS == RTE_STEPS,
       "the measured RTE size reproduces the tuned cell's 2340 steps")
    # CONTROL: a mutated size must move the answer.
    mut = {k: dict(v) for k, v in S.items()}
    mut["cola"]["train"] = S["cola"]["train"] * 2
    ck(total_steps("cola", mut) == EPOCHS * math.ceil(2 * S["cola"]["train"] / BATCH),
       "step count tracks dataset size (control)")
    ck(total_steps("cola", mut) > total_steps("cola", S),
       "doubling the training set lengthens the run (control)")

    # ---- 3. the arm set, and the one rule that must not be relaxed ---------
    sel = selected_args()
    ck(len(sel) == len(ARM_NAMES), f"all {len(ARM_NAMES)} arms resolved, got {len(sel)}")
    ck(all(a in sel for a in SCORA_ROWS),
       "⛔ BOTH SCoRA rows present -- [R.306] forbids shipping one alone")
    # ⛔ fftstock is the arm a hand-copy gets wrong: its stage-D args differ from
    # the selection cell it was chosen from.  If these ever match, the reader has
    # silently duplicated the FourierFT column.
    ck(sel.get("fftstock") != sel.get("fftm"),
       "fftstock runs a DIFFERENT optimiser from fftm (the hand-copy trap)")
    ck("adamw-fourierft " in sel.get("fftstock", "") + " ",
       "fftstock is stock PEFT FourierFT")
    ck("adamw-fourierftmerged" in sel.get("fftm", ""), "fftm is the merged path")
    ck("--slr_scaling" in sel.get("scora2", ""), "scora2 carries the swept scale")
    ck("--slr_scaling" not in sel.get("scora", ""), "scora keeps the a-priori scale")

    # ---- 4. the args actually built for a cell ----------------------------
    for t in ("cb", "mnli"):
        a = f"{common(t, S)} {sel['fftm']}"
        ck("--dtype float32" in a, f"{t}: CONTEXT §4.4 --dtype explicit")
        ck("--adapter_target_modules query,value" in a,
           f"{t}: CONTEXT §4.4 --adapter_target_modules explicit")
        ck(f"--task_name {t} " in a + " ", f"{t}: task set")
        ck(f"--num_train_epochs {EPOCHS} " in a + " ", f"{t}: 30 epochs (user decision)")
        ck(a.count("--learning_rate") == 1, f"{t}: exactly one learning rate")
        ck(a.count("--num_warmup_steps") == 1, f"{t}: exactly one warmup setting")

    # the transferred lr is only legitimate because width and k are unchanged
    ck("roberta-base" in common("cb", S), "still roberta-base => atoms unchanged => lr transfers")
    ck("_k 256" in sel["fftm"] or "_k 256" in sel["qwha"], "still k=256")

    # ---- 5. planning: one CSV per (task, arm, seed), idempotent ------------
    man = {}
    j1 = plan_task("cola", man, sel, S)
    ck(len(j1) == len(ARM_NAMES) * len(CONFIRM_SEEDS),
       f"cola emits 9x5 cells, got {len(j1)}")
    labs = [l for l, _, _ in j1]
    ck(len(set(labs)) == len(labs), "labels unique")
    ck(all(f"seed{s}" in l for l, s, _ in j1), "⛔ [R.303] seed is IN the filename")
    j2 = plan_task("cola", man, sel, S)
    ck(j2 == [], "re-planning emits nothing (resume is idempotent)")
    # CONTROL: a fresh manifest must emit again, or the guard above is vacuous.
    ck(len(plan_task("cola", {}, sel, S)) == len(j1), "empty manifest re-emits (control)")

    # cross-task labels never collide
    m2 = {}
    all_labs = []
    for t in TASKS:
        all_labs += [l for l, _, _ in plan_task(t, m2, sel, S)]
    ck(len(set(all_labs)) == len(all_labs), "no label collides across tasks")
    exp = len(TASKS) * len(ARM_NAMES) * len(CONFIRM_SEEDS) + len(SPOT_ARMS) * len(SPOT_CLF)
    ck(len(all_labs) == exp, f"total cells {len(all_labs)} == {exp}")

    # every planned cell's args carry ITS OWN task and warmup, not a neighbour's
    for lab, m in m2.items():
        t = m["task"]
        ck(f"--task_name {t} " in m["args"] + " ", f"{lab}: own task")
        ck(f"--num_warmup_steps {warmup_for(t, S)}" in m["args"], f"{lab}: own warmup")

    # ---- 6. the spot-check block ------------------------------------------
    sp = [l for l in m2 if "-spot-" in l]
    ck(len(sp) == len(SPOT_ARMS) * len(SPOT_CLF), "spot block sized")
    ck(all(m2[l]["task"] in SPOT_TASKS for l in sp), "spot cells only on cheap tasks")
    ck(all(m2[l]["seed"] == SCREEN_SEED for l in sp),
       "spot cells run at the SELECTION seed (n=1, never quotable)")
    ck("5e-3" in SPOT_CLF, "⛔ the spot block carries its OWN same-seed control")
    for arm in SPOT_ARMS:
        ck(any(l.endswith("spot-clf5e-3") and m2[l]["arm"] == arm for l in sp),
           f"{arm}: the control cell is planned, not just the alternatives")
    for l in sp:
        ck(m2[l]["args"].count("--classifier_lr") == 1,
           f"{l}: the base --classifier_lr was REPLACED, not appended")

    # ---- 6b. ⛔ the planner CANNOT emit a hyperparameter-sweep cell ---------
    # USER DECISION 2026-08-24: no per-task sweeps.  These assertions are the
    # enforcement -- a future edit that reintroduces one fails the gate suite.
    ck(not hasattr(sys.modules[__name__], "plan_psweep"),
       "⛔ plan_psweep is REMOVED -- no per-task hyperparameter sweeps")
    ck(not hasattr(sys.modules[__name__], "plan_psweep_extend"),
       "⛔ plan_psweep_extend is REMOVED")
    mall = {}
    for t in TASKS:
        plan_task(t, mall, sel, S)
    stages = {v["stage"] for v in mall.values()}
    ck(stages <= {"D", "spot"},
       f"the planner emits only confirmation and spot cells, got {sorted(stages)}")
    ck("psweep" not in stages, "⛔ no psweep cell can be planned")
    # every emitted cell carries the arm's CARRIED lr, unmodified
    for lab, m in mall.items():
        if m["stage"] != "D":
            continue
        got = m["args"].split()[m["args"].split().index("--learning_rate") + 1]
        want = sel[m["arm"]].split()[sel[m["arm"]].split().index("--learning_rate") + 1]
        ck(float(got) == float(want),
           f"{lab}: runs the RTE-selected lr verbatim, unswept")

    # ---- 7. RTE is imported, never re-run ---------------------------------

    # ---- 7. RTE is imported, never re-run ---------------------------------
    ck("rte" not in TASKS, "RTE is not re-run -- it is already measured")
    ref = rte_reference()
    ck(len(ref) == len(ARM_NAMES), "the RTE reference column resolves for every arm")
    ck(abs(ref["fftm"][0] - 0.7906) < 5e-4,
       f"RTE reference reproduces [R.305]'s fftm 0.7906, got {ref['fftm'][0]:.4f}")
    ck(all(len(v[2]) == len(CONFIRM_SEEDS) for v in ref.values()),
       "the RTE reference is 5-seed on every arm")

    # ---- 7b. the cost model is shape-aware, and that matters by 3x ---------
    ck(projected_cell_s("sst2", S)[1] < projected_cell_s("qnli", S)[0],
       "SST-2 (single-sentence, 63k steps) is projected CHEAPER than QNLI "
       "(pair, 98k steps) -- the shape term, not just the step count")
    # CONTROL: if the shape term were dropped, that ordering would REVERSE.
    flat = [STARTUP_S + S_PER_STEP["pair"][1] * total_steps(t, S) for t in ("sst2", "qnli")]
    ck(flat[0] / projected_cell_s("sst2", S)[1] > 2.5,
       "ignoring shape would over-cost SST-2 by >2.5x (control)")
    ck(SINGLE_SENTENCE <= set(TASKS), "the single-sentence set is a subset of TASKS")

    # ---- 8. we did not silently drop a task the user asked for ------------
    # The original ask was seven tasks including MNLI.  MNLI was replaced by QNLI
    # by an EXPLICIT user decision (2026-08-24), recorded at the TASKS list above.
    # The assertion therefore checks the ORIGINAL ask minus exactly that one
    # documented substitution -- so any OTHER task going missing still fails here.
    asked = {"mrpc", "stsb", "cola", "sst2", "mnli", "boolq", "cb"}
    substituted = {"mnli": "qnli"}
    ck(set(TASKS) == (asked - set(substituted)) | set(substituted.values()),
       f"the requested tasks, with only the recorded substitution: {sorted(set(TASKS))}")
    ck(len(TASKS) == len(asked), "the task COUNT is unchanged by the substitution")
    ck("mnli" not in TASKS and "qnli" in TASKS,
       "⭐ MNLI->QNLI substitution is in force (USER DECISION 2026-08-24)")

    print(f"[r310] selftest: {n[0] - selftest.failed} passed, {selftest.failed} failed")
    if selftest.failed:
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--generate", choices=TASKS)
    ap.add_argument("--cost", action="store_true")
    ap.add_argument("--show", action="store_true", help="print the selected arm strings")
    ap.add_argument("--tasks", action="store_true",
                    help="print the task run order (the orchestrator reads this, so the "
                         "shell cannot drop a task the planner is asserting is present)")
    a = ap.parse_args()
    if a.tasks:
        print(" ".join(TASKS))
    elif a.measure:
        measure()
    elif a.selftest:
        selftest()
    elif a.generate:
        generate(a.generate)
    elif a.cost:
        cost_table()
    elif a.show:
        sel = selected_args()
        for arm, title, src in ARMS:
            print(f"\n### {arm}  [{src}]  {title}\n    {sel.get(arm)}")
    else:
        ap.print_help()
