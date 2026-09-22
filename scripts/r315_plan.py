#!/usr/bin/env python
"""[R.315] THE PLANNER -- the ablations re-measured on a HEAD-FREE objective.

⛔ It plans and it scores.  It launches nothing and it trains nothing.

THE DEFECT IT ADDRESSES.  Every ablation the letter reports (`[R.311-abl]`, `[R.312]`,
`[R.313]`, `[R.314]`) ran through `roberta-base` sequence classification, where a
**592,130-parameter head** trains beside a **6,144-parameter** adapter.  A knob of the
adapter can therefore look inert because the head absorbed the task, and no ablation in
that setting attributes to SCoRA's parameterisation.  On `google/gemma-2b` +
`q_proj,o_proj` + a causal-LM objective the trainable parameters are **9,216 and all of
them are the adapter** (`src/train_clm._assert_adapter_only`, which fails closed).

WHAT IS PLANNED HERE
  * **Half A -- the (r,s) ladder** at fixed budget, the direct port of `[R.311-abl]`.
  * **Half B -- the nine comparator arms** at the FROZEN gemma-2b proxy HPs.
  * **Stage 0 -- the equal-effort P spot-check** that licenses using those HPs on an
    objective they were not selected on.

⭐ THE HP TABLE IS IMPORTED, NOT RETYPED.  `fir_final_plan.SELECTED` is the authority
   (`GEMMA_HP_PROXY.md` is its human record).  Retyping it here is exactly the drift
   `scripts/fir_arms.py` exists to prevent.

Usage:
    env/bin/python scripts/r315_plan.py --show
    env/bin/python scripts/r315_plan.py --cmd stage0        # or halfA / halfB
    env/bin/python scripts/r315_plan.py --selftest
"""
import argparse
import hashlib
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src"))

import fir_final_plan as FFP           # noqa: E402  -- the frozen gemma-2b HP authority
import fir_arms as FA                  # noqa: E402  -- the per-arm scale/target flags

# --------------------------------------------------------------------------- #
#                            the frozen protocol                              #
# --------------------------------------------------------------------------- #
MODEL = "HuggingFaceTB/SmolLM2-135M"    # [USER DECISION, 2026-09-21]
TARGETS = "q_proj,o_proj"               # 60 nn.Linear modules, 576x576
DATASET = ("wikitext", "wikitext-2-raw-v1")
BLOCK = 512
MICRO_BS = 4
ACCUM = 4                      # total batch 16 blocks = 8,192 tokens
SEEDS = [42, 43, 44, 45, 46]   # the banked convention; the 5/5 gate needs all five
WARMUP_RATIO = 0.06            # the standard transformers/PEFT warmup fraction
WEIGHT_DECAY = 0.01            # FourierFT's own published E2E value (GPT-2 medium)
DTYPE = "float32"              # MASTER weights and optimizer state stay fp32
# ⛔ TF32 IS OFF, AND THE 2.02x THAT ARGUED FOR IT IS RETRACTED.  [measured] a standalone
#   probe showed 2.02x; the SAME toggle on the real training path gives 1.00x, twice.
TF32 = False
MIXED_PRECISION = "no"         # [USER DECISION] exact fp32; autocast-bf16 was declined
EPOCHS = 3                     # the canonical HF wikitext-2 causal-LM recipe
SCREEN_EPOCHS = 3              # the lr screen runs the FULL protocol, at ONE seed

# --------------------------------------------------------------------------- #
#  ⭐⭐ ONE COMMON RECIPE FOR EVERY ARM AND EVERY RUNG  [USER DECISION 2026-09-21] #
# --------------------------------------------------------------------------- #
# `[user]`: *"Find recommended hyperparameter recipes from online (it doesn\'t matter
# too much as long as every config of ablation is matched at common hyperparams)."*
# ⇒ NO per-arm learning-rate tuning happens anywhere in this study.  One lr, one
#   schedule, one batch, one epoch count, for all nine arms and all four ladder rungs.
#
# ⛔ BUT A COMMON RAW `lr` IS NOT A COMMON STEP.  The per-parameter atom
#   ||d dW/d theta||_F differs across these methods by up to 43x `[CONTEXT 2]`, so a
#   shared `lr` would hand some arms a 43x bigger step and call it fair.  The repo\'s own
#   rule is to carry `P` = `lr`*atom.  Both halves of the instruction are satisfied by
#   holding `P` common: ONE `lr` for everybody, and each arm\'s scale set A PRIORI so
#   that every arm\'s atom is the SAME.
#   ⭐ That is `CARRY_FORWARD 4.4`\'s atom-norm rule and `PROCESS 5` test 4 in one move:
#     "each method may use its own documented normalisation, but the constant must be
#     derived a priori from the transform\'s norm -- anything found by sweeping is
#     disqualified."  Nothing below was swept.
#
# THE REFERENCE ATOM is FourierFT\'s own, 0.138106793200498 `[CARRY_FORWARD 4.2]` -- i.e.
# `scaling` = 150 at 768x768, which is PEFT\'s DOCUMENTED DEFAULT for FourierFT and the
# middle of the 100-150 band its docs recommend.  At SmolLM2-135M\'s 576x576 the same
# atom is `scaling` = 112.5, still inside that documented band.
REF_ATOM = 0.138106793200498

# THE COMMON LEARNING RATE, carried from a PUBLISHED recipe in the `P` coordinate:
#   FourierFT\'s paper, GPT-2 Medium on E2E -- its own generative-LM setting:
#       lr = 2e-2, scaling alpha = 300, modules 1024x1024
#   => atom = 300/sqrt(2*1024*1024) = 0.207193,  P = 2e-2 * 0.207193 = 4.1439e-3
#   => at REF_ATOM that same P is  lr = 4.1439e-3 / 0.138107 = 0.030005
# ⚠ Carrying the raw 2e-2 instead would have handed every arm a 1.5x SMALLER step than
#   the published recipe, purely because our modules are narrower -- the exact mistake
#   `memory:hp-transfer-proxy` exists to prevent.
# ⭐ Independent corroboration: the gemma-2b search selected 0.0451 for `scora` at this
#   same atom -- a `P` within 1.5x of the published one, found by a completely
#   separate route.
# ⭐⭐ `--slr_init_norm unit`, and it is REQUIRED by the common-recipe design.
#   SCoRA\'s a-priori rule sets `scaling` = REF_ATOM/sqrt(t) so that the per-parameter
#   atom `scaling*||alpha_j||` equals REF_ATOM -- but under the shipped `raw` init,
#   ||alpha_j|| ~ sqrt(t) only IN EXPECTATION, with per-row sd 1/sqrt(2t): 6.2% at
#   t=128 and 17.7% at t=16.  So under `raw` the ladder\'s rungs differ in how
#   HETEROGENEOUS their per-row steps are, and that heterogeneity grows as `s` shrinks
#   -- a difference between rungs that has nothing to do with the (r,s) parameterisation
#   being ablated.
#   ⚠ [measured] the MEAN atom is NOT the problem: across the four rungs it is stable to
#     0.5% under `raw` (0.12888-0.12959).  An earlier reading of "up to 26% drift" came
#     from `fir_backbone_port.atom_of`, which probes ONE row and therefore reports that
#     per-draw sd as if it were a shift.  The mean is fine; the spread is the issue.
#   ⇒ `unit` rescales each row to ||alpha_j|| = sqrt(t) EXACTLY, so [measured] every rung
#     and both SCoRA arms sit at atom = 0.138107 = REF_ATOM to 6 decimals -- the same
#     step as all seven other arms.  It costs ZERO parameters and ZERO flops, keeps
#     dW = 0 at init, and keeps the step-0 gradient FIRST order (`[R.174]`, gates 6/6).
#   ⛔ IT IS A DEVIATION FROM THE LETTER\'S SHIPPED ARM, which is `raw`.  Declared here,
#     and it means no [R.315] SCoRA number may be compared against a banked `raw` one.
#     This study is self-contained -- new backbone, new objective, every arm re-run --
#     so nothing inside it is cross-compared; but the write-up must say which arm it
#     ablated.
SLR_INIT_NORM = "unit"

# ⭐ [MEASURED 2026-09-22, Stage 0, 6/6 cells] the screen's argmin was 2.0x, at the
#   ladder EDGE and 2.167% better than the anchor -- far outside the 0.2% tie band.
#   The prereg's edge rule therefore fires and ONE extension rung is owed at 4.0x.
STAGE0_EDGE = 2.0

COMMON_LR = 0.030
LR_SCREEN = [0.5, 1.0, 2.0]    # multiplies the COMMON lr; never a per-arm one
# ⛔ The prereg's EDGE RULE: an edge winner gets ONE extension rung in its direction, the
#   rule is re-applied ONCE, and if the extension wins again the recipe is reported as
#   UNBRACKETED and used anyway.  Frozen here so the extension cannot be chosen after
#   seeing which way the screen went.
LR_SCREEN_EXT = {0.5: 0.25, 2.0: 4.0}
SCREEN_ARMS = ["scora", "fftm"]

# A-PRIORI per-arm scale at 576x576, derived so that EVERY arm\'s atom == REF_ATOM.
# ⛔ MEASURED, NOT TYPED: each value is `frozen_gemma_scale * REF_ATOM / atom_measured`,
#   the atom measured by `fir_backbone_port.atom_of` at 576x576, every arm confirmed
#   EXACTLY LINEAR in its scale flag -- so the rescale is exact, not a fit.
#   Reproduce with `--verify-scales`.
DERIVED_SCALE = {
    "fftm": 112.5, "fftstock": 112.5, "wave1": 112.5, "wave2": 112.5,
    "qwha": 79.5495, "loca": 0.138106793200498, "lyra": 0.138106793200498,
    # ⛔ scora takes NO scale flag.  Its a-priori rule is REF_ATOM/sqrt(t), which is
    #   INDEPENDENT OF WIDTH, so it needs no porting at all (fir_arms: "DO NOT ADD ONE").
    "scora": None,
    # ⭐ scora2 is the SWEPT-scale row, and what [R.306] selected was the RATIO: its
    #   gemma value 0.0244141 is exactly 2.0000x scora\'s derived gamma = 0.0122074.
    #   gamma does not move with width, so the ratio carries unchanged and scora2 stays
    #   a genuinely different arm instead of collapsing onto scora.
    "scora2": 0.0244141,
}

# Half A -- (r, s) at a FIXED 9,216 parameters.  params/module = r*(s+t) = 2*r*s = 256.
# ⛔ The budget is held EXACTLY, not approximately: `selftest` asserts 2*r*s == 256 for
#   every rung, because `[R.312]` is the study that showed budget, not r or s, is the
#   dominant axis -- a rung off-budget would be measuring that instead.
LADDER = [(1, 128), (2, 64), (4, 32), (8, 16)]

# Half B -- the nine arms, in the letter's table order.
ARMS = FA.ARM_ORDER

# Stage 0 -- the P ladder.  P = lr * atom is the method-independent coordinate
# (`CONTEXT.md` 2); at FIXED backbone, targets and k the atom does not move, so
# multiplying the frozen lr by these factors moves P by exactly them.
P_RUNGS = [0.5, 1.0, 2.0]


# --------------------------------------------------------------------------- #
#          ⭐⭐ THE FREEZE.  llmdocs/ IS GITIGNORED, SO A COMMIT FREEZES NOTHING #
# --------------------------------------------------------------------------- #
# `PROCESS.md 4`: "git diff is not evidence when a file is untracked -- and llmdocs/
# and scratchpad/ are both gitignored here."  Every prereg in this repo lives in
# llmdocs, so "hash-frozen at <commit>" has always meant the PLANNER was committed,
# never the document.  That leaves the one thing `PROCESS.md 1.1` actually forbids --
# editing a rule after the first number lands -- undetectable.
#
# ⇒ The sha256 prefix of each frozen document is recorded HERE, in a TRACKED file,
#   before its stage ran.  `--selftest` recomputes them, so editing a frozen document
#   turns the whole gate suite RED instead of passing silently.
#   ⚠ A hash is not a timestamp: it proves the text has not changed since the freeze,
#     not when the freeze happened.  The commit that introduced each hash is the date,
#     and the cells' own mtimes are the independent check.
PREREG_SHA256 = {
    # frozen 13:55 UTC, while the pilot ran and before ANY epoch number existed
    "R315_pilot_reading_rule.md": "157a0e55040eb66a",
    # the pilot scored against that rule; R3 fired
    "R315_pilot_verdict.md":      "9fad2ceb30758a34",
    # frozen before the first Stage-0 cell (driver started 19:01 UTC)
    "R315_stage0_prereg.md":      "bd27f7d1e2db599d",
    # v1: frozen before ANY Half-A/Half-B cell existed.  SUPERSEDED by v2 but NOT
    # edited -- its hash stays checked, so what it said before the amendment is provable.
    "R315_prereg.md":             "6f7e36d97aaa2cce",
    # v2: the SmolLM2-135M backbone + the one-common-recipe regime, both [USER DECISION
    # 2026-09-21], frozen before any cell of THIS plan exists
    "R315_prereg_v2.md":          "fa44c71d1cdb4e1b",
}


def prereg_hashes():
    """(name -> (recorded, actual)) for every frozen document."""
    import hashlib
    out = {}
    for name, want in PREREG_SHA256.items():
        path = os.path.join(ROOT, "llmdocs", name)
        if not os.path.exists(path):
            out[name] = (want, None)
            continue
        with open(path, "rb") as fh:
            out[name] = (want, hashlib.sha256(fh.read()).hexdigest()[:16])
    return out


def _f(x):
    """A flag value, never in scientific notation (argparse takes it, humans read it)."""
    return f"{x:.10g}"


def common_flags(epochs, seeds, name, cell_dir="scratchpad/clm"):
    f = ["--model_name_or_path", MODEL, "--dtype", DTYPE,
         "--mixed_precision", MIXED_PRECISION]
    if TF32:
        f.append("--tf32")
    f += ["--adapter_target_modules", TARGETS,
          "--dataset_name", DATASET[0], "--dataset_config_name", DATASET[1],
          "--block_size", str(BLOCK),
          "--per_device_train_batch_size", str(MICRO_BS),
          "--gradient_accumulation_steps", str(ACCUM),
          "--num_train_epochs", str(epochs),
          "--warmup_ratio", _f(WARMUP_RATIO),
          "--weight_decay", _f(WEIGHT_DECAY),
          "--seeds", ",".join(str(s) for s in seeds),
          "--name", name, "--cell_dir", cell_dir, "--log_every", "150"]
    return f


def arm_flags(arm, lr_mult=1.0, rank=None, s=None):
    """The adapter flags for one arm under the COMMON recipe.

    ⛔ THE ONLY THINGS THAT EVER DIFFER BETWEEN ARMS: the method, its budget flag name,
       and its A-PRIORI scale (set so every arm\'s atom equals REF_ATOM).  The learning
       rate, weight decay, schedule, batch, epochs and seeds are IDENTICAL for all nine
       arms and all four ladder rungs -- that is what makes this a matched comparison
       rather than nine separately tuned ones.

    `rank`/`s` override SCoRA\'s budget split for Half A\'s ladder; `lr_mult` is used ONLY
    by the lr screen, and it moves the COMMON lr, never one arm\'s.
    """
    if arm not in DERIVED_SCALE:
        raise SystemExit(f"FAIL CLOSED: {arm!r} is not one of the nine frozen arms.")
    method = {"fftm": "fourierftmerged", "fftstock": "fourierft", "loca": "loca",
              "qwha": "qwha", "wave1": "haar", "wave2": "haar", "lyra": "spectral",
              "scora": "slr", "scora2": "slr"}[arm]
    f = ["--optimizer", f"adamw-{method}",
         "--learning_rate", _f(COMMON_LR * lr_mult)]
    # k = 256 per module everywhere; only the flag NAME differs between methods.
    if method == "fourierftmerged":
        f += ["--fourierftmerged_k", "256", "--fourierftmerged_seed", "777"]
    elif method == "fourierft":
        f += ["--fourierft_n_frequency", "256", "--fourierft_random_loc_seed", "777"]
    elif method == "loca":
        f += ["--loca_k", "256", "--loca_seed", "777"]
    elif method == "qwha":
        f += ["--qwha_k", "256", "--qwha_seed", "777", "--qwha_init_weights", "0"]
    elif method == "haar":
        f += ["--haar_k", "256", "--haar_seed", "777", "--haar_init_std", "0.0",
              "--haar_mu", "2" if arm == "wave2" else "1"]
    elif method == "spectral":
        f += ["--spectral_p", "16", "--spectral_q", "16", "--spectral_dropout", "0.0",
              "--spectral_d_initial", "0.07", "--spectral_freq_mode", "geometric"]
    elif method == "slr":
        r_, s_ = (rank if rank is not None else 1), (s if s is not None else 128)
        f += ["--slr_rank", str(r_), "--slr_s", str(s_),
              "--slr_init", "zero", "--slr_seed", "777",
              "--slr_init_norm", SLR_INIT_NORM]
    sf = FA.ARM_SCALE_FLAG[arm]
    scale = DERIVED_SCALE[arm]
    if scale is None:
        if sf:
            raise SystemExit(f"FAIL CLOSED: {arm} has scale flag {sf} but no value.")
    else:
        if not sf:
            raise SystemExit(f"FAIL CLOSED: {arm} has a scale but no flag to set.")
        f += [sf, _f(scale)]
    return f


def screen_verdict(rows, extended=False):
    """Apply Stage 0's FROZEN selection rule to the screen's rows.  Mechanical, so the
    reader cannot drift from `R315_prereg_v2.md` 2.

    `rows` = {(arm, mult): ppl}.  Returns a dict with the selection and its reasons.

    The rule, verbatim from the prereg:
      * winner = the rung minimising the SUM of the two screen arms' perplexities;
      * ties within 0.2% of the ANCHOR (1x) go to the anchor;
      * an EDGE winner (0.5x or 2x) triggers one extension, and if the extension wins
        again the recipe is reported as UNBRACKETED;
      * the selected value is ONE number applied to all nine arms and all four rungs.
    """
    mults = sorted({m for _a, m in rows})
    # ⛔ `extended` is a CLAIM that the extension rung was run, so it is checked against
    #   the FROZEN multiplier rather than against whatever rows happen to be present --
    #   otherwise a caller could pass extended=True on the original six cells and get a
    #   verdict that silently drops the edge flag.
    want = list(LR_SCREEN) + ([LR_SCREEN_EXT[STAGE0_EDGE]] if extended else [])
    missing = [(a, m) for a in SCREEN_ARMS for m in want if (a, m) not in rows]
    if missing:
        return {"complete": False, "missing": missing,
                "verdict": f"INCOMPLETE -- {len(missing)} screen cell(s) absent; "
                           f"PROCESS 1.4 forbids selecting on a partial screen"}
    totals = {m: sum(rows[(a, m)] for a in SCREEN_ARMS) for m in mults}
    best = min(totals, key=lambda m: totals[m])
    anchor_total = totals[1.0]
    tie = abs(totals[best] - anchor_total) / anchor_total <= 0.002
    selected = 1.0 if tie else best
    at_edge = selected in (min(mults), max(mults)) and selected != 1.0
    return {
        "extended": extended,
        "complete": True, "totals": totals, "argmin": best,
        "selected_mult": selected, "selected_lr": COMMON_LR * selected,
        "tie_with_anchor": tie,
        "rel_gap_to_anchor": (totals[best] - anchor_total) / anchor_total,
        "at_edge": at_edge,
        "verdict": (
            f"SELECTED lr = {COMMON_LR * selected:.6g} ({selected}x the published anchor)"
            + ("; the argmin was within 0.2% of the anchor, so the TIE RULE kept the "
               "anchor" if tie else "")
            + ("" if not at_edge else
               ("; ⛔⛔ THE EXTENSION WON AGAIN -- the recipe is UNBRACKETED. Per the "
                "prereg it is USED, and every row must carry that label"
                if extended else
                "; ⛔ WINNER IS AT A LADDER EDGE -- the prereg requires one extension "
                "rung in that direction before this is used"))),
    }


def verify_scales():
    """Recompute DERIVED_SCALE from the measured atoms.  Needs torch; not in --selftest."""
    import fir_backbone_port as BP
    m = n = 576
    print(f"target atom = {REF_ATOM:.12g}")
    bad = 0
    for arm in ARMS:
        flags = FA.parse_flags(" ".join(arm_flags(arm)))
        atom, linear, _ = BP.atom_of(arm, flags, m, n)
        note = ""
        if not linear:
            note = "  ⛔ NOT linear in its scale -- the rescale rule does not apply"
            bad += 1
        elif arm == "scora":
            # raw randn init: the atom matches REF_ATOM only IN EXPECTATION, sd 1/sqrt(2t)
            note = (f"  (a priori, no flag; {abs(atom / REF_ATOM - 1) * 100:.1f}% above "
                    f"REF_ATOM = 1.6x the 6.2% per-draw sd of --slr_init_norm raw. The "
                    f"rule holds IN EXPECTATION, not per draw -- declared, not matched)")
        elif arm == "scora2":
            note = f"  (the SWEPT row: 2.0x scora\'s gamma by design, so atom is 2x)"
        elif abs(atom / REF_ATOM - 1) > 0.01:
            note = "  ⛔ atom is off REF_ATOM by more than 1%"
            bad += 1
        print(f"  {arm:10s} atom {atom:.6f}  ratio {atom / REF_ATOM:6.3f}{note}")
    return 1 if bad else 0


# --------------------------------------------------------------------------- #
#                                  the cells                                  #
# --------------------------------------------------------------------------- #
def cells(stage):
    """Every cell of a stage, in a deterministic order.  One entry per CELL (an arm at
    a configuration); the five seeds run inside one invocation, sequentially."""
    out = []
    if stage == "stage0":
        # ⛔ This screens the ONE COMMON lr, not nine per-arm ones.  It runs on two arms
        #   so that a value which is good for one construction and terrible for another
        #   is visible; the SELECTION is a single number applied to all nine.
        for arm in SCREEN_ARMS:
            for p in LR_SCREEN:
                out.append({"stage": "stage0", "arm": arm, "p": p,
                            "epochs": SCREEN_EPOCHS, "seeds": [SEEDS[0]]})
    elif stage == "stage0ext":
        # ⛔ Only reachable when the screen's winner was an edge; the multiplier is the
        #   FROZEN one for that edge, never a value chosen after seeing the screen.
        ext = LR_SCREEN_EXT[STAGE0_EDGE]
        for arm in SCREEN_ARMS:
            out.append({"stage": "stage0ext", "arm": arm, "p": ext,
                        "epochs": SCREEN_EPOCHS, "seeds": [SEEDS[0]]})
    elif stage == "halfA":
        for (r, s) in LADDER:
            out.append({"stage": "halfA", "arm": "scora", "r": r, "s": s,
                        "epochs": EPOCHS, "seeds": list(SEEDS)})
    elif stage == "halfB":
        for arm in ARMS:
            out.append({"stage": "halfB", "arm": arm,
                        "epochs": EPOCHS, "seeds": list(SEEDS)})
    else:
        raise SystemExit(f"FAIL CLOSED: unknown stage {stage!r}")
    return out


# ⛔ THE PLAN VERSION IS PART OF EVERY CELL ID, and it is not decoration.
#   [incident, 2026-09-21] the backbone changed from gemma-2b to SmolLM2-135M by user
#   decision AFTER three Stage-0 cells had run.  Their ids -- `r315-s0-fftm-p1` and so
#   on -- are ids the NEW plan also generates, so the resumable driver would have seen
#   their `.done` markers and SKIPPED three cells, silently seeding the study with
#   gemma-2b results under SmolLM2 row labels.  Nothing would have crashed.
#   ⇒ Bump PLAN_VERSION whenever a change makes existing cells non-reusable.  Old
#     markers then simply do not match, and the old rows keep a name that says what
#     they were.
PLAN_VERSION = "v2"


def cell_id(c):
    if c["stage"] in ("stage0", "stage0ext"):
        return f"r315{PLAN_VERSION}-s0-{c['arm']}-p{_f(c['p']).replace('.', 'p')}"
    if c["stage"] == "halfA":
        return f"r315{PLAN_VERSION}-A-scora-r{c['r']}s{c['s']}"
    return f"r315{PLAN_VERSION}-B-{c['arm']}"


def cell_cmd(c):
    name = cell_id(c)
    f = ["env/bin/python", "-u", "src/train_clm.py"]
    f += common_flags(c["epochs"], c["seeds"], name)
    if c["stage"] == "halfA":
        f += arm_flags("scora", rank=c["r"], s=c["s"])
    elif c["stage"] in ("stage0", "stage0ext"):
        f += arm_flags(c["arm"], lr_mult=c["p"])
    else:
        f += arm_flags(c["arm"])
    return f


def n_seed_runs(stage):
    return sum(len(c["seeds"]) for c in cells(stage))


def digest():
    """Checksum of the whole plan, so the prereg can hash-freeze it."""
    blob = json.dumps([" ".join(cell_cmd(c)) for st in ("stage0", "halfA", "halfB")
                       for c in cells(st)], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def show():
    for st in ("stage0", "stage0ext", "halfA", "halfB"):
        cs = cells(st)
        print(f"\n=== {st}: {len(cs)} cells, {n_seed_runs(st)} seed-runs ===")
        for c in cs:
            print("  " + cell_id(c))
    print(f"\nplan digest: {digest()}")
    print(f"total seed-runs: {sum(n_seed_runs(s) for s in ('stage0', 'halfA', 'halfB'))}")


# --------------------------------------------------------------------------- #
#                                  selftest                                   #
# --------------------------------------------------------------------------- #
def selftest():
    passed = failed = 0

    def check(nm, cond, detail=""):
        nonlocal passed, failed
        if cond:
            passed += 1; print(f"  ✅ {nm}")
        else:
            failed += 1; print(f"  ⛔ {nm} {detail}")

    # G1 ⭐ the ladder holds the budget EXACTLY -- [R.312] showed budget is the
    # dominant axis, so an off-budget rung measures that instead of (r, s).
    check("G1 every rung is exactly 256 params/module (2*r*s)",
          all(2 * r * s == 256 for r, s in LADDER),
          detail=str([(r, s, 2 * r * s) for r, s in LADDER]))
    check("G1b the ladder spans r = 1..8 with no repeats",
          sorted(r for r, _ in LADDER) == [1, 2, 4, 8])

    # G2 ⭐⭐ ONE COMMON RECIPE: every arm and every rung carries the SAME lr.
    lrs = {}
    for st in ("halfA", "halfB"):
        for c in cells(st):
            cmd = cell_cmd(c)
            lrs[cell_id(c)] = cmd[cmd.index("--learning_rate") + 1]
    check("G2 every study cell carries the SAME learning rate (no per-arm tuning "
          "exists anywhere in this study)", len(set(lrs.values())) == 1,
          detail=str(sorted(set(lrs.values()))))
    check("G2b and it is the published-anchored COMMON_LR",
          set(lrs.values()) == {_f(COMMON_LR)}, str(set(lrs.values())))
    for knob in ("--weight_decay", "--num_train_epochs", "--warmup_ratio",
                 "--per_device_train_batch_size", "--block_size"):
        vals = set()
        for st in ("halfA", "halfB"):
            for c in cells(st):
                cmd = cell_cmd(c)
                vals.add(cmd[cmd.index(knob) + 1])
        check(f"G2c {knob} is common to every study cell", len(vals) == 1, str(vals))

    # G3 ⛔ scora must NOT acquire a scale flag (fir_arms: "DO NOT ADD ONE").
    check("G3 scora has no --slr_scaling (its scale is derived a priori, and its rule "
          "REF_ATOM/sqrt(t) is width-independent so nothing was ported)",
          "--slr_scaling" not in arm_flags("scora") and DERIVED_SCALE["scora"] is None)
    check("G3d ⭐ every SLR cell carries --slr_init_norm unit, so each rung's per-row "
          "step is EXACTLY REF_ATOM instead of only in expectation",
          all("--slr_init_norm" in arm_flags(a) and
              arm_flags(a)[arm_flags(a).index("--slr_init_norm") + 1] == "unit"
              for a in ("scora", "scora2")))
    check("G3e and the ladder rungs carry it too (the rungs are where the per-row "
          "spread would otherwise differ, 6.2% at t=128 vs 17.7% at t=16)",
          all("--slr_init_norm unit" in " ".join(arm_flags("scora", rank=r, s=s_))
              for r, s_ in LADDER))
    check("G3b scora2 DOES carry one (both rows always ship together)",
          "--slr_scaling" in arm_flags("scora2"))
    check("G3c ⭐ scora2 is exactly 2x scora's derived gamma -- the RATIO [R.306] "
          "selected, carried unchanged because gamma does not move with width",
          abs(DERIVED_SCALE["scora2"] / (REF_ATOM / math.sqrt(128)) - 2.0) < 1e-4,
          f"ratio = {DERIVED_SCALE['scora2'] / (REF_ATOM / math.sqrt(128)):.6f}")

    # G4 the screen moves the COMMON lr and nothing else, and it is not per-arm.
    base = float(arm_flags("fftm")[arm_flags("fftm").index("--learning_rate") + 1])
    hi = arm_flags("fftm", lr_mult=2.0)
    check("G4 a screen rung of 2.0 doubles the COMMON lr and changes nothing else",
          abs(float(hi[hi.index("--learning_rate") + 1]) - 2 * base) < 1e-9
          and [x for x in hi if x != _f(2 * base)]
              == [x for x in arm_flags("fftm") if x != _f(base)])
    check("G4b the screen selects ONE value for all nine arms, so it runs on a "
          "SUBSET of arms by design", set(SCREEN_ARMS) < set(ARMS))
    check("G4c the screen ladder brackets the published anchor (0.5x and 2x)",
          min(LR_SCREEN) < 1.0 < max(LR_SCREEN) and 1.0 in LR_SCREEN)

    # G4d ⭐ the atom-matching rule is what makes ONE lr fair.  The recorded scales are
    # checked for internal consistency here; --verify-scales re-measures them with torch.
    check("G4d every arm has a declared a-priori scale (or is declared as needing none)",
          set(DERIVED_SCALE) == set(ARMS), str(set(DERIVED_SCALE) ^ set(ARMS)))
    check("G4e the FourierFT-family scale is REF_ATOM's own alpha at 576x576 "
          "(= 112.5, inside PEFT's documented 100-150 band)",
          abs(DERIVED_SCALE["fftm"] - REF_ATOM * math.sqrt(2 * 576 * 576)) < 1e-6,
          f"{DERIVED_SCALE['fftm']} vs {REF_ATOM * math.sqrt(2 * 576 * 576):.4f}")

    # G5 Half A only moves (r, s); every other flag is scora's.
    a1 = cell_cmd({"stage": "halfA", "arm": "scora", "r": 1, "s": 128,
                   "epochs": EPOCHS, "seeds": SEEDS})
    a2 = cell_cmd({"stage": "halfA", "arm": "scora", "r": 4, "s": 32,
                   "epochs": EPOCHS, "seeds": SEEDS})
    diff = [(x, y) for x, y in zip(a1, a2) if x != y]
    check("G5 two rungs differ ONLY in --name, --slr_rank and --slr_s",
          len(a1) == len(a2) and len(diff) == 3, detail=str(diff))

    # G6 the run really is head-free by construction: the targets are decoder names
    # and no classifier flag is planned.
    cmd = " ".join(cell_cmd(cells("halfB")[0]))
    check("G6 no --classifier_lr is planned (it is inert on a causal LM and would "
          "read as a protocol that ran)", "--classifier_lr" not in cmd)
    check("G6b the targets are the decoder pair", f"--adapter_target_modules {TARGETS}" in cmd)

    # G7 dtype/tf32 are explicit on every cell -- train_clm fails closed without dtype.
    check("G7 every cell passes --dtype and an explicit --mixed_precision",
          all("--dtype" in cell_cmd(c) and "--mixed_precision" in cell_cmd(c)
              for st in ("stage0", "halfA", "halfB") for c in cells(st)))
    check("G7b the compute precision is the SAME on every cell of the study "
          "(a per-arm precision would disqualify the comparison)",
          len({cell_cmd(c)[cell_cmd(c).index("--mixed_precision") + 1]
               for st in ("stage0", "halfA", "halfB") for c in cells(st)}) == 1)
    check("G7c master weights stay fp32 (--dtype float32), never a half-precision CAST",
          DTYPE == "float32")
    check("G7d the total batch is 16 blocks regardless of the micro-batch layout",
          MICRO_BS * ACCUM == 16)

    # G8 five seeds on every STUDY cell (the 5/5 gate cannot be run on four).
    check("G8 halfA and halfB cells carry all five seeds",
          all(c["seeds"] == SEEDS for st in ("halfA", "halfB") for c in cells(st)))
    check("G8b stage 0 is 1 seed and is a SCREEN, never a verdict",
          all(len(c["seeds"]) == 1 for c in cells("stage0")))

    # G12 ⭐ the screen's selection rule, exercised on fixtures BEFORE the screen is read
    # (PROCESS.md 6: test the reader before the spend).
    def _rows(sc, ff):
        return {("scora", m): sc[i] for i, m in enumerate(LR_SCREEN)} | \
               {("fftm", m): ff[i] for i, m in enumerate(LR_SCREEN)}
    v = screen_verdict(_rows([20.0, 17.0, 19.0], [21.0, 18.0, 20.0]))
    check("G12 a clear interior winner selects the anchor",
          v["selected_mult"] == 1.0 and not v["at_edge"], v["verdict"])
    v = screen_verdict(_rows([17.0, 20.0, 21.0], [18.0, 21.0, 22.0]))
    check("G12b a 0.5x winner is selected AND flagged as an edge needing an extension",
          v["selected_mult"] == 0.5 and v["at_edge"], v["verdict"])
    v = screen_verdict(_rows([20.0, 17.00, 16.99], [21.0, 18.00, 17.99]))
    check("G12c ⭐ a winner within 0.2% of the anchor loses to the TIE RULE -- a one-seed "
          "screen cannot resolve less", v["selected_mult"] == 1.0 and v["tie_with_anchor"],
          v["verdict"])
    v = screen_verdict(_rows([20.0, 17.0, 15.0], [21.0, 18.0, 16.0]))
    check("G12d a 2x winner outside the tie band IS taken, and flagged as an edge",
          v["selected_mult"] == 2.0 and v["at_edge"], v["verdict"])
    # G12g ⭐ the EXTENSION path: once run, an extension that wins again is reported as
    # UNBRACKETED and used -- the prereg allows exactly one extension, not a search.
    ext = {("scora", m): v for m, v in zip([0.5, 1.0, 2.0, 4.0], [20.0, 17.0, 15.0, 13.0])}
    ext |= {("fftm", m): v for m, v in zip([0.5, 1.0, 2.0, 4.0], [21.0, 18.0, 16.0, 14.0])}
    v = screen_verdict(ext, extended=True)
    check("G12g an extension that wins again is UNBRACKETED and still selected",
          v["selected_mult"] == 4.0 and v["at_edge"] and "UNBRACKETED" in v["verdict"],
          v["verdict"])
    ext2 = dict(ext); ext2[("scora", 4.0)] = 30.0; ext2[("fftm", 4.0)] = 30.0
    v = screen_verdict(ext2, extended=True)
    check("G12h an extension that loses brackets the ladder and hands back the 2x rung",
          v["selected_mult"] == 2.0 and not v["at_edge"], v["verdict"])
    v = screen_verdict({k: x for k, x in ext.items() if k[1] != 4.0}, extended=True)
    check("G12i ⛔ claiming `extended` without the extension cells selects NOTHING",
          not v["complete"])
    check("G12j the extension multiplier is FROZEN per edge, not chosen after the fact",
          LR_SCREEN_EXT == {0.5: 0.25, 2.0: 4.0})

    partial = {("scora", m): 17.0 for m in LR_SCREEN}
    v = screen_verdict(partial)
    check("G12e ⛔ a partial screen selects NOTHING (PROCESS.md 1.4)",
          not v["complete"] and "INCOMPLETE" in v["verdict"])
    # the two arms are summed, so an arm that hates a rung can veto it
    v = screen_verdict(_rows([17.0, 17.5, 17.6], [30.0, 18.0, 18.1]))
    check("G12f ⭐ the rule is the SUM, so a rung one arm cannot train at is not chosen "
          "just because the other arm likes it", v["selected_mult"] == 1.0, v["verdict"])

    # G9c ⭐ the version tag is in every id, so a superseded cell can never be reused
    # by the resumable driver (the gemma-2b/SmolLM2 collision, 2026-09-21).
    check("G9c every cell id carries the plan version",
          all(PLAN_VERSION in cell_id(c)
              for st in ("stage0", "halfA", "halfB") for c in cells(st)))
    check("G9d and the superseded v1 ids are NOT regenerated",
          not any(cell_id(c) in ("r315-s0-fftm-p1", "r315-B-scora", "r315-A-scora-r1s128")
                  for st in ("stage0", "halfA", "halfB") for c in cells(st)))

    # G9 cell ids are unique and stable.
    ids = [cell_id(c) for st in ("stage0", "halfA", "halfB") for c in cells(st)]
    check("G9 every cell id is unique", len(ids) == len(set(ids)))
    check("G9b the digest is stable across calls", digest() == digest())

    # G11 ⭐⭐ the frozen documents are UNCHANGED since their stage was launched.
    for name, (want, got) in prereg_hashes().items():
        if got is None:
            check(f"G11 {name} exists", False, "MISSING -- a frozen prereg cannot vanish")
        else:
            check(f"G11 {name} unchanged since freeze ({want})", want == got,
                  f"recorded {want}, actual {got} -- a frozen rule was EDITED "
                  f"(PROCESS.md 1.1), or the hash was not updated at freeze time")

    # G10 an unknown arm or stage fails closed rather than planning something plausible.
    try:
        arm_flags("nope"); ok = False
    except SystemExit:
        ok = True
    check("G10 an unknown arm fails closed", ok)
    try:
        cells("halfC"); ok = False
    except SystemExit:
        ok = True
    check("G10b an unknown stage fails closed", ok)

    print(f"\nselftest: {passed} passed, {failed} failed")
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--cmd", choices=["stage0", "stage0ext", "halfA", "halfB"])
    ap.add_argument("--verify-scales", dest="verify_scales", action="store_true",
                    help="re-measure every arm's atom at 576x576 (needs torch)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.verify_scales:
        return verify_scales()
    if a.cmd:
        for c in cells(a.cmd):
            print(" ".join(cell_cmd(c)))
        return 0
    show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
