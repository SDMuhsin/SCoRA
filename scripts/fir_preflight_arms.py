#!/usr/bin/env python
"""[fir] Run EVERY arm as a real short cell and check the RECEIPTS.

⛔ WHY A RECEIPT AND NOT A FLAG.  A RoBERTa module name on a decoder matches
NOTHING.  The adapter then attaches to ZERO modules, the classification head
trains alone, the run exits 0, the metric is plausible, and the row is wrong.
No flag check can see that -- only the runner's own report of what it built can.
[FIR_SETUP Law 2: verify the receipt, never the flag.]

⭐ AND IT ENUMERATES ACROSS EVERY ARM.  [FIR_SETUP G1] cost a whole sweep: a
shared control was missing on 4 of 13 arms and the tell was three unrelated
methods reporting numbers IDENTICAL TO THE BYTE.  So this checks the arms against
EACH OTHER, not each against a hardcoded expectation.

DECLARED, MEASURED EXCEPTION -- LoCA.  [measured] LoCA trains 768 parameters per
module (256 coefficients + a 2x256 LOCATION tensor) where every other arm trains
256.  That is the method, not a defect: LoCA's whole claim is learned locations.
It is asserted EXPLICITLY at 3x rather than waived, so if it ever stops being 3x
the check fires.

Usage (also the LOCAL smoke -- runs anywhere, CPU included):
    env/bin/python scripts/fir_preflight_arms.py --targets q_o --port-mode derived \
        --task rte --steps 8 --run-root /tmp/pf
    env/bin/python scripts/fir_preflight_arms.py --selftest
"""
import argparse, json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import fir_arms as FA                                                  # noqa: E402
import fir_plan as FP                                                  # noqa: E402

# LoCA's location tensor is (2, k) per module on top of its k coefficients.
LOCATION_MULTIPLIER = {"loca": 3}

RE_TRAINABLE = re.compile(r"trainable params:\s*([\d,]+)\s*\|\|\s*all params:\s*([\d,]+)")
# every custom arm logs "adapted N modules"; the PEFT/extra_repr arms log "N modules;"
RE_MODULES = re.compile(r"adapted\s+(\d+)\s+modules|:\s*(\d+)\s+modules;")
RE_ADAPTER_PARAMS = re.compile(r"adapter params=([\d,]+)")
RE_TARGETS = re.compile(r"target modules:\s*(\[[^\]]*\])")


def parse_receipts(text):
    out = {}
    m = RE_TRAINABLE.search(text)
    if m:
        out["trainable"] = int(m.group(1).replace(",", ""))
        out["all_params"] = int(m.group(2).replace(",", ""))
    mm = RE_MODULES.search(text)
    if mm:
        out["modules"] = int(mm.group(1) or mm.group(2))
    ap = RE_ADAPTER_PARAMS.search(text)
    if ap:
        out["adapter_params"] = int(ap.group(1).replace(",", ""))
    tg = RE_TARGETS.search(text)
    if tg:
        out["target_modules"] = tg.group(1)
    return out


def run_arm(arm, task, targets, port_mode, steps, run_root, epochs=1, seed=41,
            python=None, timeout=3600):
    python = python or sys.executable
    cmd = FP.cell_cmd(arm, task, seed, targets, port_mode, epochs)
    # ⚠ a PREFLIGHT is not a result: cap the data so a cell is seconds, and send the
    #   CSV somewhere a real table can never be reached from.
    batch = FP.BATCH
    cmd = cmd + ["--max_train_samples", str(steps * batch),
                 "--max_eval_samples", str(min(256, steps * batch))]
    # the warmup the planner derived is for the FULL run; at 8 steps it would be
    # the whole thing. Override explicitly so the cell is a build+step check.
    if "--num_warmup_steps" in cmd:
        cmd[cmd.index("--num_warmup_steps") + 1] = "0"
    env = dict(os.environ)
    env.update(FP.cell_env(arm, task, seed, targets, run_root))
    os.makedirs(os.path.join(run_root, "csv"), exist_ok=True)
    # ⚠ APPEND, never assign (numpy comes from the scipy-stack module on Alliance)
    env["PYTHONPATH"] = os.path.join(ROOT, "src") + os.pathsep + env.get("PYTHONPATH", "")
    p = subprocess.run([python] + cmd, cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or ""), cmd


# ⛔ DECLARED, MEASURED EXCEPTION 2 -- THE STOCK-PEFT ARM DUPLICATES THE HEAD.
#   `fftstock` is the only arm built by `get_peft_model(...)`, and with
#   task_type=SEQ_CLS PEFT sets modules_to_save=['classifier', 'score']: it keeps
#   the ORIGINAL head as a frozen copy beside the trainable one.
#     [measured, dev box, roberta-base + peft 0.13.2]
#         peft_all - base_all - adapter = 592,130 = exactly one head, and the extra
#         tensors are named `...classifier.original_module.*`
#     [measured, fir, gemma-2b + peft 0.18.1]
#         fftstock all_params exceeds every other arm's by exactly 4,096 = the gemma
#         head (2048 x 2 labels)
#   So its frozen total legitimately carries ONE EXTRA HEAD. ⛔ This is asserted, not
#   waived: the excess must be EXACTLY one head, so if PEFT ever saves something
#   else, or another arm starts duplicating, the check still fires.
#   ⚠ The trainable copy is the one the optimiser sees, and `--classifier_lr`'s group
#     rule is a SUBSTRING match on 'classifier'/'score' over requires_grad params
#     (train_glue.py:2058), so the copy lands in the classifier group and the frozen
#     original is skipped. Checked, because a name change there would be silent.
HEAD_DUPLICATING_ARMS = {"fftstock"}

# Arms whose runner prints no "adapted N modules" line. Their module count is
# DERIVED below and must match the arms that do print one -- see _derive_modules.
#   fftstock: PEFT's print_trainable_parameters() reports no module count.
#   lyra    : the spectral wrapper logs no count (the one-line fix is DEFERRED
#             while [R.310] is live on train_glue.py -- llmdocs/CONTEXT.md 1.3).
NO_MODULE_LOG_ARMS = {"fftstock", "lyra"}


def check(results, targets):
    """Cross-arm consistency. Returns (ok, lines, bad)."""
    lines, bad = [], []
    n_mod = {a: r["receipts"].get("modules") for a, r in results.items()}
    train = {a: r["receipts"].get("trainable") for a, r in results.items()}

    for a, r in results.items():
        if r["rc"] != 0:
            bad.append(f"{a}: exited {r['rc']}")
        if n_mod.get(a) == 0:
            bad.append(f"{a}: attached to ZERO modules (target names matched nothing)")
        if not train.get(a):
            bad.append(f"{a}: no trainable-params receipt")

    direct = {a: v for a, v in n_mod.items() if v}
    if not direct:
        bad.append("NO arm reported a module count -- nothing to bootstrap the derived "
                   "counts from; attachment is UNPROVEN for every arm")
        return (not bad), lines, bad
    if len(set(direct.values())) > 1:
        bad.append(f"arms DISAGREE on module count: {direct} -- at a matched protocol "
                   f"every arm must adapt the same modules")
        return (not bad), lines, bad
    nm = next(iter(direct.values()))

    # ⭐ THE HEAD IS INFERRED ONLY FROM ARMS THAT REPORTED THEIR OWN COUNT, so the
    #   derivation below is not circular: head <- direct arms, count <- head.
    heads = {a: train[a] - nm * 256 * LOCATION_MULTIPLIER.get(a, 1)
             for a in direct if train.get(a)}
    if len(set(heads.values())) != 1:
        bad.append(f"implied HEAD size differs across arms: {heads} -- one arm's "
                   f"adapter budget is not modules*256*multiplier")
        return (not bad), lines, bad
    h = next(iter(heads.values()))
    if h <= 0:
        bad.append(f"implied head {h} <= 0 -- the budget model is wrong")
        return (not bad), lines, bad

    # ⛔ AN ARM WITH NO MODULE-COUNT LINE IS NOT UNCHECKED, AND IT IS NOT WAIVED.
    #   Two of the nine runners print no count (see NO_MODULE_LOG_ARMS). The
    #   receipt that matters is "did the adapter attach", and trainable-minus-head
    #   answers it exactly: a run that matched NOTHING trains the head alone, so it
    #   would land on 0 modules here, not on 36. The derived count must be a whole
    #   number AND equal to what every other arm reported.
    derived = {}
    for a, t in train.items():
        if a in direct or not t:
            continue
        mult = LOCATION_MULTIPLIER.get(a, 1)
        rem = t - h
        q, r_ = divmod(rem, 256 * mult)
        if r_ != 0 or q <= 0:
            bad.append(f"{a}: no module-count receipt, and the derived count is not a whole "
                       f"number: (trainable {t:,} - head {h:,}) / {256*mult} = {rem/(256*mult):.4f} "
                       f"-- cannot prove the adapter attached")
            continue
        derived[a] = q
        if q != nm:
            bad.append(f"{a}: derived module count {q} != {nm} reported by every other arm "
                       f"-- this arm did not adapt the same modules")
    if derived:
        lines.append(f"  all arms adapted {nm} modules  ({targets})   "
                     f"[derived for {', '.join(sorted(derived))}: (trainable - head)/budget]")
    else:
        lines.append(f"  all arms adapted {nm} modules  ({targets})")

    lines.append(f"  implied head = {h:,} params "
                 f"(gemma `score` = hidden x num_labels; RoBERTa's was 592,130)")
    for a, t in sorted(train.items()):
        if not t:
            continue
        mult = LOCATION_MULTIPLIER.get(a, 1)
        tag = f"  (x{mult}: {a} also trains LOCATIONS)" if mult > 1 else ""
        src = "" if a in direct else "  [count derived]"
        lines.append(f"    {a:9s} trainable {t:>9,}  = adapter {nm*256*mult:>7,}"
                     f" + head {h:,}{tag}{src}")

    # ⛔ THE INVARIANT IS THE FROZEN BACKBONE, NOT THE TOTAL.
    #    `all params` INCLUDES the adapter's own parameters, and the adapters are not
    #    the same size: [measured, gemma-2b/q_o] loca reports 2,506,204,160 against
    #    every other arm's 2,506,185,728 -- a difference of exactly 18,432 = 36 x 512,
    #    its LOCATION tensors.  An earlier version compared the totals and fired
    #    "different backbones?" on a perfectly healthy pair.
    #    ...and then the CORRECTED version fired too, on fftstock, for the second
    #    reason above: PEFT keeps a frozen copy of the head. Both exceptions are
    #    subtracted EXPLICITLY, so the comparison is between quantities that really
    #    are the same thing on every arm.
    frozen, adj = {}, {}
    for a, r in results.items():
        ap_, tr_ = r["receipts"].get("all_params"), r["receipts"].get("trainable")
        if ap_ and tr_:
            frozen[a] = ap_ - tr_
            adj[a] = frozen[a] - (h if a in HEAD_DUPLICATING_ARMS else 0)
    fs = set(adj.values())
    if len(fs) > 1:
        bad.append(f"arms disagree on the FROZEN BACKBONE size {adj} (raw {frozen}) -- a "
                   f"different backbone, or an arm that froze/unfroze something the others "
                   f"did not. Head-duplicating arms (one extra frozen head, PEFT "
                   f"modules_to_save): {sorted(HEAD_DUPLICATING_ARMS)}")
    elif fs:
        base_frozen = fs.pop()
        lines.append(f"  frozen backbone identical across arms: {base_frozen:,} params")
        for a in sorted(HEAD_DUPLICATING_ARMS & set(frozen)):
            lines.append(f"    {a}: + one frozen head ({h:,}) kept by PEFT modules_to_save "
                         f"-- raw frozen {frozen[a]:,} [declared, measured]")
    return (not bad), lines, bad


def selftest():
    ok, bad = [], []

    def ck(c, l):
        (ok if c else bad).append(l)

    # --- the parser, against the REAL log lines this repo emits
    real = ("INFO - __main__ - SLR target modules: ['q_proj', 'o_proj']\n"
            "INFO - __main__ - SLR: adapted 36 modules, rank=1, s=t=128, adapter params=9,216\n"
            "trainable params: 13,312 || all params: 2,506,185,728 || trainable%: 0.0005\n")
    r = parse_receipts(real)
    ck(r["modules"] == 36, "parses 'adapted 36 modules'")
    ck(r["trainable"] == 13312, "parses trainable params with commas")
    ck(r["all_params"] == 2506185728, "parses all params")
    ck(r["adapter_params"] == 9216, "parses adapter params")
    ck("q_proj" in r["target_modules"], "parses the target-module list")
    # the OTHER log shape (extra_repr arms)
    r2 = parse_receipts("INFO - QWHA adapter: 36 modules; k=256; scaling=147\n"
                        "trainable params: 13,312 || all params: 2,506,185,728 || x\n")
    ck(r2["modules"] == 36, "parses the 'N modules;' log shape too")
    ck(parse_receipts("nothing here") == {}, "CONTROL: parses nothing out of nothing")

    # --- the cross-arm check on a fixture that IS the real fir run ------------
    # ⭐ REGRESSION FIXTURE, copied from preflight_56905037.out (gemma-2b, q_o,
    #   H100, peft 0.18.1). Two arms print no module count and one keeps a second
    #   frozen head; this shape MUST pass, because it is what a healthy run looks
    #   like, and the first two versions of this check failed it.
    good = {a: {"rc": 0, "receipts": {"modules": 36, "trainable": 13312,
                                      "all_params": 2506185728}}
            for a in FA.ARM_ORDER}
    good["loca"]["receipts"]["trainable"] = 36 * 256 * 3 + 4096
    # ⚠ loca's `all params` is LARGER because its location tensors are part of the
    #   model (36 x 512). This must PASS.
    good["loca"]["receipts"]["all_params"] = 2506185728 + 36 * 512
    for a in NO_MODULE_LOG_ARMS:
        good[a]["receipts"].pop("modules")
    # ⚠ fftstock: PEFT modules_to_save keeps the original head, so all_params is
    #   one head LARGER at the same trainable count. Measured: 2,506,189,824.
    good["fftstock"]["receipts"]["all_params"] = 2506185728 + 4096

    okg, lines, badg = check(good, "q_o")
    ck(okg, f"the REAL 9-arm fir result passes ({badg})")
    ck(any("head = 4,096" in l for l in lines), "infers the 4,096-param gemma head")
    ck(any("frozen backbone identical" in l for l in lines), "reports the frozen-backbone invariant")
    ck(any("count derived" in l for l in lines), "marks the two derived counts as derived")
    ck(any("one frozen head" in l for l in lines), "declares fftstock's duplicated head")

    for label, mutate in (
            ("zero modules", lambda d: d["scora"]["receipts"].__setitem__("modules", 0)),
            ("module disagreement", lambda d: d["qwha"]["receipts"].__setitem__("modules", 24)),
            ("a nonzero exit", lambda d: d["lyra"].__setitem__("rc", 1)),
            ("a budget that is not modules*256",
             lambda d: d["fftm"]["receipts"].__setitem__("trainable", 13312 + 7)),
            ("a different frozen backbone",
             lambda d: d["wave2"]["receipts"].__setitem__("all_params", 12345)),
            # --- the branches added 2026-08-26; each must be able to fail -------
            ("a DERIVED count that is not a whole number",
             lambda d: d["lyra"]["receipts"].__setitem__("trainable", 13312 + 7)),
            ("a DERIVED count that disagrees with the others (adapter attached to half)",
             lambda d: d["lyra"]["receipts"].__setitem__("trainable", 18 * 256 + 4096)),
            ("an arm with NO count whose adapter matched NOTHING (head trains alone)",
             lambda d: d["fftstock"]["receipts"].__setitem__("trainable", 4096)),
            ("fftstock's duplicated head being the WRONG size (not exactly one head)",
             lambda d: d["fftstock"]["receipts"].__setitem__("all_params", 2506185728 + 8192)),
            ("a NON-duplicating arm growing an extra head",
             lambda d: d["qwha"]["receipts"].__setitem__("all_params", 2506185728 + 4096)),
            ("EVERY arm losing its module count (nothing to bootstrap from)",
             lambda d: [r["receipts"].pop("modules", None) for r in d.values()]),
    ):
        import copy
        d = copy.deepcopy(good)
        mutate(d)
        o, _l, b = check(d, "q_o")
        ck(not o, f"CONTROL: check FIRES on {label}")

    for l in ok:
        print(f"  ✅ {l}")
    for l in bad:
        print(f"  ⛔ {l}")
    print(f"selftest: {len(ok)} passed, {len(bad)} failed")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=False)
    ap.add_argument("--port-mode", dest="port_mode", required=False)
    ap.add_argument("--task", default="rte")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--run-root", default=None)
    ap.add_argument("--arms", default=None, help="comma list; default = all nine")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.targets or not a.port_mode or not a.run_root:
        raise SystemExit("FAIL CLOSED: --targets, --port-mode and --run-root are required")

    arms = [x.strip() for x in a.arms.split(",")] if a.arms else list(FA.ARM_ORDER)
    results, logdir = {}, os.path.join(a.run_root, "logs")
    os.makedirs(logdir, exist_ok=True)
    for arm in arms:
        print(f"\n--- {arm} ---", flush=True)
        try:
            rc, out, cmd = run_arm(arm, a.task, a.targets, a.port_mode, a.steps, a.run_root)
        except subprocess.TimeoutExpired:
            rc, out, cmd = 124, "TIMEOUT", []
        with open(os.path.join(logdir, f"{arm}.log"), "w") as f:
            f.write(" ".join(cmd) + "\n\n" + out)
        rec = parse_receipts(out)
        results[arm] = {"rc": rc, "receipts": rec}
        print(f"  rc={rc}  receipts={rec}")
        if rc != 0:
            # ⚠ report the REAL failure, not a traceback on top of it. The last thing
            #   in a log is what gets read first.
            tail = [l for l in out.strip().splitlines()
                    if l.strip() and not l.startswith(" ")][-6:]
            for l in tail:
                print(f"    | {l[:200]}")

    print("\n" + "=" * 78)
    okc, lines, bad = check(results, a.targets)
    for l in lines:
        print(l)
    if bad:
        print("\n⛔ RECEIPT CHECK FAILED:")
        for b in bad:
            print(f"    {b}")
        print(f"\n  logs: {logdir}")
        sys.exit(1)
    print("\n✅ ALL ARMS BUILT, ATTACHED AND STEPPED — receipts consistent across arms")
    print(f"  logs: {logdir}")


if __name__ == "__main__":
    main()
