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
MODEL = "google/gemma-2b"
TARGETS = "q_proj,o_proj"
DATASET = ("wikitext", "wikitext-2-raw-v1")
BLOCK = 512
MICRO_BS = 4
ACCUM = 4                      # total batch 16 blocks = 8,192 tokens
SEEDS = [42, 43, 44, 45, 46]   # the banked convention; the 5/5 gate needs all five
WARMUP_RATIO = 0.05983         # [R.310]'s own RTE-derived ratio, carried as a ratio
WEIGHT_DECAY = 0.01
DTYPE = "float32"
TF32 = True                    # [measured] 2.02x, and it moves the eval ppl by <1e-6 rel

# ⭐ Set by the pilot, which is why the pilot ran first.  Both are CALIBRATION
# decisions (PROCESS.md 1.1 permits calibration to inform design, labelled as such).
EPOCHS = 3
SCREEN_EPOCHS = 1              # Stage 0 only

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


def _f(x):
    """A flag value, never in scientific notation (argparse takes it, humans read it)."""
    return f"{x:.10g}"


def common_flags(epochs, seeds, name, cell_dir="scratchpad/clm"):
    f = ["--model_name_or_path", MODEL, "--dtype", DTYPE]
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
    """The adapter flags for one arm, from the FROZEN table -- never retyped.

    `rank`/`s` override SCoRA's budget split for Half A's ladder; `lr_mult` moves P
    for Stage 0.  ⛔ No other knob may be passed: a per-arm constant found by sweeping
    disqualifies a fairness claim (PROCESS.md 5 test 4), and the scale flags below are
    the ones the gemma-2b search already selected, not new ones.
    """
    if arm not in FFP.SELECTED:
        raise SystemExit(f"FAIL CLOSED: {arm!r} is not one of the nine frozen arms.")
    sel = FFP.SELECTED[arm]
    method = {"fftm": "fourierftmerged", "fftstock": "fourierft", "loca": "loca",
              "qwha": "qwha", "wave1": "haar", "wave2": "haar", "lyra": "spectral",
              "scora": "slr", "scora2": "slr"}[arm]
    f = ["--optimizer", f"adamw-{method}",
         "--learning_rate", _f(sel["lr"] * lr_mult)]
    # k = 256 everywhere; the per-arm budget flag names differ, nothing else does.
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
              "--slr_init", "zero", "--slr_seed", "777"]
    sf = FA.ARM_SCALE_FLAG[arm]
    if sel["scaling"] is None:
        # ⛔ scora derives its scale from --slr_s a priori.  Adding one would convert
        #   the a-priori arm into a tuned one (fir_arms: "DO NOT ADD ONE").
        if sf:
            raise SystemExit(f"FAIL CLOSED: {arm} has scale flag {sf} but no frozen value.")
    else:
        if not sf:
            raise SystemExit(f"FAIL CLOSED: {arm} has a frozen scale but no flag to set.")
        f += [sf, _f(sel["scaling"])]
    for k, v in (sel.get("extra") or {}).items():
        f += [{"freq_exponent": "--spectral_freq_exponent"}[k], _f(v)]
    return f


# --------------------------------------------------------------------------- #
#                                  the cells                                  #
# --------------------------------------------------------------------------- #
def cells(stage):
    """Every cell of a stage, in a deterministic order.  One entry per CELL (an arm at
    a configuration); the five seeds run inside one invocation, sequentially."""
    out = []
    if stage == "stage0":
        for arm in ARMS:
            for p in P_RUNGS:
                out.append({"stage": "stage0", "arm": arm, "p": p,
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


def cell_id(c):
    if c["stage"] == "stage0":
        return f"r315-s0-{c['arm']}-p{_f(c['p']).replace('.', 'p')}"
    if c["stage"] == "halfA":
        return f"r315-A-scora-r{c['r']}s{c['s']}"
    return f"r315-B-{c['arm']}"


def cell_cmd(c):
    name = cell_id(c)
    f = ["env/bin/python", "-u", "src/train_clm.py"]
    f += common_flags(c["epochs"], c["seeds"], name)
    if c["stage"] == "halfA":
        f += arm_flags("scora", rank=c["r"], s=c["s"])
    elif c["stage"] == "stage0":
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
    for st in ("stage0", "halfA", "halfB"):
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

    # G2 the HP table is the frozen one, not a copy.
    check("G2 the nine arms come from fir_final_plan.SELECTED",
          set(ARMS) == set(FFP.SELECTED), detail=str(set(ARMS) ^ set(FFP.SELECTED)))
    for arm in ARMS:
        f = arm_flags(arm)
        lr = float(f[f.index("--learning_rate") + 1])
        if abs(lr - FFP.SELECTED[arm]["lr"]) > 1e-12:
            check(f"G2b {arm} carries its frozen lr", False, f"{lr} vs {FFP.SELECTED[arm]['lr']}")
            break
    else:
        check("G2b every arm carries its own frozen lr verbatim", True)

    # G3 ⛔ scora must NOT acquire a scale flag (fir_arms: "DO NOT ADD ONE").
    check("G3 scora has no --slr_scaling (its scale is derived a priori)",
          "--slr_scaling" not in arm_flags("scora"))
    check("G3b scora2 DOES carry one (both rows always ship together)",
          "--slr_scaling" in arm_flags("scora2"))

    # G4 the P ladder moves the learning rate by exactly its factor.
    base = float(arm_flags("fftm")[arm_flags("fftm").index("--learning_rate") + 1])
    hi = float(arm_flags("fftm", lr_mult=2.0)[arm_flags("fftm", 2.0).index("--learning_rate") + 1])
    check("G4 a P rung of 2.0 doubles the lr and nothing else",
          abs(hi - 2 * base) < 1e-9
          and [x for x in arm_flags("fftm", 2.0) if x != _f(2 * base)]
              == [x for x in arm_flags("fftm") if x != _f(base)])

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
    check("G7 every cell passes --dtype and --tf32",
          all("--dtype" in cell_cmd(c) and "--tf32" in cell_cmd(c)
              for st in ("stage0", "halfA", "halfB") for c in cells(st)))

    # G8 five seeds on every STUDY cell (the 5/5 gate cannot be run on four).
    check("G8 halfA and halfB cells carry all five seeds",
          all(c["seeds"] == SEEDS for st in ("halfA", "halfB") for c in cells(st)))
    check("G8b stage 0 is 1 seed and is a SCREEN, never a verdict",
          all(len(c["seeds"]) == 1 for c in cells("stage0")))

    # G9 cell ids are unique and stable.
    ids = [cell_id(c) for st in ("stage0", "halfA", "halfB") for c in cells(st)]
    check("G9 every cell id is unique", len(ids) == len(set(ids)))
    check("G9b the digest is stable across calls", digest() == digest())

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
    ap.add_argument("--cmd", choices=["stage0", "halfA", "halfB"])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd:
        for c in cells(a.cmd):
            print(" ".join(cell_cmd(c)))
        return 0
    show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
