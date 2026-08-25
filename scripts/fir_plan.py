#!/usr/bin/env python
"""[fir] THE PLANNER — build the exact `src/train_glue.py` command for one cell.

Every fir stage (03 preflight, 04 pilot, 05 sweep) gets its command strings from
HERE, so there is exactly ONE place that knows how a RoBERTa-tuned arm becomes a
gemma-2b arm.  A second site would be [FIR_SETUP G6] ("one honouring site, or two
that are tested against each other").

⛔⛔ TWO PROTOCOL CHOICES HAVE NO DEFAULT AND THE PLANNER REFUSES WITHOUT THEM.
    They are the user's, not the planner's, and a default would smuggle one in:

  --targets {q_v|q_o}
      q_v  q_proj,v_proj -- NAME-matched to RoBERTa's query,value.  ⚠ gemma-2b is
           MQA (1 KV head), so v_proj is 256x2048 while q_proj is 2048x2048:
           [measured] init perturbation spread 10.67x, max 0.0564.
      q_o  q_proj,o_proj -- SHAPE-matched: both 2048x2048, like RoBERTa's two
           square 768x768 modules.  [measured] spread 1.74x, max 0.0082 -- tighter
           than RoBERTa's own 2.67x.  This is [phase-m2]'s measured repair for a
           GQA/MQA backbone.
      ⚠ BOTH give 36 modules and an identical per-arm parameter budget, so the
        choice is about init stability, not budget.

  --port-mode {derived|asis}
      derived  use the a-priori scale* and lr* from sbatch/fir/port_<model>.json:
               scale* matches RoBERTa's MEDIAN init perturbation, lr* then holds
               P = lr*atom [hp-transfer-proxy].  Nothing is swept.
      asis     carry the RoBERTa constants unchanged.  ⛔ This is NOT the safe
               option: the atom falls as 1/sqrt(2mn), so at d=2048 `asis` runs a
               ~2.7x SMALLER effective step than the tuned RoBERTa point.

⚠⚠ WHAT THIS PLANNER CANNOT FIX, AND DOES NOT PRETEND TO:
  * `--classifier_lr 5e-3` was selected against RoBERTa's 592,130-param head.
    gemma-2b's head is `score` = 2048 x num_labels = 4,096 params -- 145x smaller,
    and the adapter (9,216) is now the MAJORITY of what trains instead of 1%.
    [R.115 4] says this must be RE-DERIVED, not assumed.  It is carried unchanged
    here and REPORTED as an open deviation; there is no a-priori rule for it.
  * SCoRA's atom is STOCHASTIC (rel sd 5.79% measured, [R.174]), so its lr* is a
    random variable.  --show prints the band.
  * 30 epochs is a cost question on a 2.5B model and must come from the PILOT's
    measured wall-clock, never from RoBERTa's schedule.

Usage:
    env/bin/python scripts/fir_plan.py --selftest
    env/bin/python scripts/fir_plan.py --targets q_o --port-mode derived --show
    env/bin/python scripts/fir_plan.py --targets q_o --port-mode derived \
        --task rte --arm scora --seed 42 --epochs 30 --cmd
"""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import fir_arms as FA                                                  # noqa: E402

MODEL = os.environ.get("FIR_MODEL", "google/gemma-2b")
SIZES_PATH = os.path.join(ROOT, "sbatch", "fir", "dataset_sizes.json")

TARGET_SETS = {"q_v": "q_proj,v_proj", "q_o": "q_proj,o_proj"}

# ---- protocol constants CARRIED from [R.305]/[R.310], with their provenance ----
BATCH = 32                     # [R.305]'s --per_device_train_batch_size
DTYPE = "float32"              # ⚠ [phase-m] trap: the argparse default is bf16 on a
                               #   bf16-capable box and is MACHINE-DEPENDENT.  Every
                               #   number in this lineage is float32.  Pass it ALWAYS.
RTE_WARMUP, RTE_STEPS = 140, 2340
WARMUP_RATIO = RTE_WARMUP / RTE_STEPS      # 0.0598290598... -- RTE's own


def sizes():
    if not os.path.exists(SIZES_PATH):
        raise SystemExit(f"FAIL CLOSED: {SIZES_PATH} absent (measured dataset sizes)")
    with open(SIZES_PATH) as f:
        return json.load(f)


def total_steps(task, epochs, S=None):
    S = S or sizes()
    if task not in S:
        raise SystemExit(f"FAIL CLOSED: no measured size for task {task!r}")
    import math
    return math.ceil(S[task]["train"] / BATCH) * epochs


def warmup_for(task, epochs, S=None):
    """RTE's warmup RATIO applied to THIS task's step count.

    ⛔ Holding the epoch count does NOT hold the warmup: warmup is ABSOLUTE while
    the step count is not.  A flat 140 would be 58% of CB's run and 0.04% of MNLI's.
    """
    return int(round(WARMUP_RATIO * total_steps(task, epochs, S)))


def port(model=MODEL):
    import fir_backbone_port as B
    return B.load_port(model)


def arm_args(arm, targets, port_mode, P=None, PT=None):
    """The arm's flag string, ported.  Returns (argstr, notes[])."""
    if targets not in TARGET_SETS:
        raise SystemExit(f"FAIL CLOSED: --targets must be one of {sorted(TARGET_SETS)}")
    if port_mode not in ("derived", "asis"):
        raise SystemExit("FAIL CLOSED: --port-mode must be 'derived' or 'asis'")
    P = P or FA.load()
    flags = FA.parse_flags(P["args"][arm])
    notes = []
    mods = TARGET_SETS[targets]

    # 1. every target-module flag moves.  ⛔ A RoBERTa module name on a decoder
    #    matches NOTHING: the adapter attaches to zero modules, the classifier head
    #    trains alone, and the row still looks entirely plausible.
    tf = FA.ARM_TARGET_FLAG[arm]
    if tf:
        if tf not in flags:
            raise SystemExit(f"FAIL CLOSED: {arm} has no {tf} -- ARM_TARGET_FLAG is stale")
        flags[tf] = mods

    # 2. scale + lr
    if port_mode == "derived":
        PT = PT or port()
        prof = PT["targets"][targets]["arms"][arm]
        sf = FA.ARM_SCALE_FLAG[arm]
        ds, dl = prof.get("derived_scale"), prof.get("derived_lr")
        if sf and ds is not None:
            flags[sf] = f"{float(ds):.6g}"
            notes.append(f"{sf} {P['args'][arm].split(sf)[1].split()[0]} -> {flags[sf]}")
        elif sf:
            # ⭐ derived_scale is None EXACTLY when the arm inits to dW == 0
            #    (loca, wave1, wave2, scora2 -- [measured] rel@init = 0.0000).
            #    There is then NO init perturbation to match, so the scale-matching
            #    rule has nothing to say and the scale stays put; the entire port is
            #    carried by lr through P = lr*atom, which is well defined regardless.
            #    ⛔ This is a DERIVATION, not a fallback: inventing a scale for a
            #       zero-init arm would be a knob nobody asked for.
            if (prof.get("rel_median") or 0.0) > 0:
                raise SystemExit(
                    f"FAIL CLOSED: {arm} has a NON-ZERO init perturbation "
                    f"({prof['rel_median']:.6g}) but the port table has no "
                    f"derived_scale for it -- the emit is stale or inconsistent.")
            notes.append(f"{sf} unchanged: {arm} inits to dW == 0, so there is no "
                         f"init perturbation to match; lr carries the whole port")
        if dl is None:
            raise SystemExit(f"FAIL CLOSED: no derived_lr for {arm}/{targets}")
        old_lr = flags["--learning_rate"]
        flags["--learning_rate"] = f"{float(dl):.6g}"
        notes.append(f"--learning_rate {old_lr} -> {flags['--learning_rate']}")
        sd = prof.get("atom_rel_sd") or 0.0
        if sd > 1e-6:
            notes.append(f"⚠ {arm} atom is STOCHASTIC (rel sd {sd*100:.2f}%) "
                         f"=> lr* carries the same band [R.174]")
    else:
        notes.append("⚠ port-mode=asis: RoBERTa constants carried UNCHANGED. The atom "
                     "falls as 1/sqrt(2mn), so the effective step is NOT the tuned one.")

    order = list(FA.parse_flags(P["args"][arm]).keys())
    return " ".join(f"{k} {flags[k]}".strip() for k in order), notes


def cell_cmd(arm, task, seed, targets, port_mode, epochs, model=MODEL, P=None, PT=None):
    """The FULL command for one cell, as a list of tokens."""
    a, _notes = arm_args(arm, targets, port_mode, P=P, PT=PT)
    common = (f"--model_name_or_path {model}"
              f" --task_name {task}"
              f" --dtype {DTYPE}"
              # ⚠ the GENERIC override, passed EXPLICITLY on every cell.  It wins over
              #   every per-arm flag, and `fftstock` (stock PEFT) has no per-arm flag
              #   at all -- without this it would target RoBERTa's defaults and adapt
              #   nothing.  [phase-m] trap 1.
              f" --adapter_target_modules {TARGET_SETS[targets]}"
              f" --per_device_train_batch_size {BATCH}"
              f" --num_train_epochs {epochs}"
              f" --num_warmup_steps {warmup_for(task, epochs)}")
    name = f"{task}-{arm}-{targets}-seed{seed}"
    return ["src/train_glue.py"] + common.split() + a.split() + ["--name", name]


def cell_env(arm, task, seed, targets, run_root):
    """Per-cell environment.  ⛔ ONE CSV PER CELL.

    `train_glue._upsert_result`'s key OMITS `seed` (scripts/r304_upsert_gate.py
    enforces this), so two seeds pointed at one CSV COLLAPSE INTO ONE ROW and the
    loss is silent.  The seed therefore lives in the FILENAME.
    Also: --seed is IGNORED by train_glue (argparse help says so); the seed comes
    from GLUE_SEEDS.  Passing --seed and expecting it to act would be [G3]
    ("recorded but never delivered") a seventh time.
    """
    return {
        "GLUE_SEEDS": str(seed),
        "GLUE_RESULTS_FILE": os.path.join(run_root, "csv",
                                          f"{task}-{arm}-{targets}-seed{seed}.csv"),
    }


# ---------------------------------------------------------------------------
def selftest():
    ok, bad = [], []

    def ck(c, l):
        (ok if c else bad).append(l)

    P = FA.load()
    S = sizes()

    # --- fail closed on the two protocol choices
    for t, pm in (("nope", "derived"), ("q_o", "nope")):
        try:
            arm_args("scora", t, pm, P=P); ck(False, f"CONTROL: refuses targets={t} mode={pm}")
        except SystemExit:
            ck(True, f"CONTROL: refuses targets={t} mode={pm}")

    # --- warmup is DERIVED, and a flat 140 would be visibly wrong
    ck(abs(WARMUP_RATIO - 140 / 2340) < 1e-12, "warmup ratio is RTE's own")
    for task in ("cb", "qnli"):
        ck(warmup_for(task, 30, S) == int(round(WARMUP_RATIO * total_steps(task, 30, S))),
           f"{task} warmup derived from its own step count")
    ck(warmup_for("cb", 30, S) < 140, "CONTROL: flat-140 would over-warm CB")
    ck(warmup_for("qnli", 30, S) > 140 * 10, "CONTROL: flat-140 would barely warm QNLI")

    # --- ⭐ THE ONE THAT MATTERS: no RoBERTa module name survives the port
    for targets in ("q_v", "q_o"):
        for arm in FA.ARM_ORDER:
            cmd = " ".join(cell_cmd(arm, "rte", 42, targets, "derived", 30, P=P))
            ck("query" not in cmd and "value" not in cmd,
               f"{arm}/{targets}: no RoBERTa module name survives")
            ck(f"--adapter_target_modules {TARGET_SETS[targets]}" in cmd,
               f"{arm}/{targets}: generic override present")
    # and the control: the UNPORTED string DOES contain them (so the check can fail)
    ck("query,value" in P["args"]["scora"], "CONTROL: the source string does name query,value")

    # --- derived actually MOVES lr, and asis does not
    d, _ = arm_args("fftm", "q_o", "derived", P=P)
    a, _ = arm_args("fftm", "q_o", "asis", P=P)
    ck(d != a, "CONTROL: derived and asis differ")
    ck("--learning_rate 0.5" in a, "asis keeps the RoBERTa lr")
    ck("--learning_rate 0.5" not in d, "derived moves the lr")

    # --- scora must NOT acquire a scale flag; scora2 must keep its own
    ds, _ = arm_args("scora", "q_o", "derived", P=P)
    ck("--slr_scaling" not in ds, "scora acquires NO --slr_scaling (its scale is a priori)")
    ds2, _ = arm_args("scora2", "q_o", "derived", P=P)
    ck("--slr_scaling" in ds2, "scora2 keeps its explicit --slr_scaling")

    # --- fftstock has no per-arm target flag but must still be targeted
    cmd = " ".join(cell_cmd("fftstock", "rte", 42, "q_o", "derived", 30, P=P))
    ck("--adapter_target_modules q_proj,o_proj" in cmd,
       "fftstock is targeted via the GENERIC override (it has no per-arm flag)")

    # --- dtype is always explicit (the machine-dependent bf16 default trap)
    ck(all(f"--dtype {DTYPE}" in " ".join(cell_cmd(a_, "rte", 42, "q_o", "derived", 30, P=P))
           for a_ in FA.ARM_ORDER), "every cell passes --dtype explicitly")

    # --- one CSV per cell, seed in the FILENAME
    e1 = cell_env("scora", "rte", 42, "q_o", "/run")
    e2 = cell_env("scora", "rte", 43, "q_o", "/run")
    ck(e1["GLUE_RESULTS_FILE"] != e2["GLUE_RESULTS_FILE"],
       "CONTROL: two seeds get DIFFERENT CSVs (the upsert key omits seed)")
    ck(e1["GLUE_SEEDS"] == "42", "seed is delivered via GLUE_SEEDS, not --seed")
    ck(not any(t == "--seed" for t in cell_cmd("scora", "rte", 42, "q_o", "derived", 30, P=P)),
       "CONTROL: --seed is NOT passed (train_glue ignores it)")

    # --- the emitted command parses back to the flags we think we set
    toks = cell_cmd("qwha", "cola", 44, "q_o", "derived", 30, P=P)
    ck(toks[0] == "src/train_glue.py", "command starts with the runner")
    ck("--qwha_target_modules" in toks and toks[toks.index("--qwha_target_modules") + 1]
       == "q_proj,o_proj", "per-arm target flag rewritten too")

    for l in ok:
        print(f"  ✅ {l}")
    for l in bad:
        print(f"  ⛔ {l}")
    print(f"selftest: {len(ok)} passed, {len(bad)} failed")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", choices=sorted(TARGET_SETS))
    ap.add_argument("--port-mode", dest="port_mode", choices=["derived", "asis"])
    ap.add_argument("--task"); ap.add_argument("--arm"); ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--show", action="store_true", help="print the ported arm table")
    ap.add_argument("--cmd", action="store_true", help="print ONE cell's command")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.targets or not a.port_mode:
        raise SystemExit("FAIL CLOSED: --targets and --port-mode are REQUIRED "
                         "(they are protocol choices; a default would smuggle one in)")
    P, PT = FA.load(), port(a.model)
    if a.show:
        print(f"model {a.model}  targets {a.targets} = {TARGET_SETS[a.targets]}  "
              f"port-mode {a.port_mode}")
        prof = PT["targets"][a.targets]
        print(f"{prof['n_modules']} modules, shapes {prof['shapes']}")
        for arm in FA.ARM_ORDER:
            s_, notes = arm_args(arm, a.targets, a.port_mode, P=P, PT=PT)
            print(f"\n  {arm}\n    {s_}")
            for n in notes:
                print(f"      {n}")
        return
    if a.cmd:
        for f in ("task", "arm", "seed", "epochs"):
            if getattr(a, f) is None:
                raise SystemExit(f"FAIL CLOSED: --{f} required with --cmd")
        env = cell_env(a.arm, a.task, a.seed, a.targets, "$FIR_RUN_ROOT")
        print(" ".join(f"{k}={v}" for k, v in env.items()) + " python "
              + " ".join(cell_cmd(a.arm, a.task, a.seed, a.targets, a.port_mode,
                                  a.epochs, model=a.model, P=P, PT=PT)))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
