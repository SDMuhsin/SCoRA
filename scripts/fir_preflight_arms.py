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


def check(results, targets):
    """Cross-arm consistency. Returns (ok, lines)."""
    lines, bad = [], []
    n_mod = {a: r["receipts"].get("modules") for a, r in results.items()}
    train = {a: r["receipts"].get("trainable") for a, r in results.items()}
    allp = {a: r["receipts"].get("all_params") for a, r in results.items()}

    for a, r in results.items():
        if r["rc"] != 0:
            bad.append(f"{a}: exited {r['rc']}")
        if not n_mod.get(a):
            bad.append(f"{a}: NO module-count receipt -- cannot prove the adapter attached")
        elif n_mod[a] == 0:
            bad.append(f"{a}: attached to ZERO modules (target names matched nothing)")
        if not train.get(a):
            bad.append(f"{a}: no trainable-params receipt")

    mods = {v for v in n_mod.values() if v}
    if len(mods) > 1:
        bad.append(f"arms DISAGREE on module count: {n_mod} -- at a matched protocol "
                   f"every arm must adapt the same modules")
    elif mods:
        lines.append(f"  all arms adapted {mods.pop()} modules  ({targets})")

    # head size, inferred from the arm with the SMALLEST budget, then cross-checked
    base = {a: t for a, t in train.items() if t}
    if base:
        # expected adapter budget per arm = modules * 256 * multiplier
        nm = next(iter({v for v in n_mod.values() if v}), None)
        if nm:
            heads = {}
            for a, t in base.items():
                mult = LOCATION_MULTIPLIER.get(a, 1)
                heads[a] = t - nm * 256 * mult
            hs = set(heads.values())
            if len(hs) != 1:
                bad.append(f"implied HEAD size differs across arms: {heads} -- one arm's "
                           f"adapter budget is not modules*256*multiplier")
            else:
                h = hs.pop()
                lines.append(f"  implied head = {h:,} params "
                             f"(gemma `score` = hidden x num_labels; RoBERTa's was 592,130)")
                if h <= 0:
                    bad.append(f"implied head {h} <= 0 -- the budget model is wrong")
                for a, t in sorted(base.items()):
                    mult = LOCATION_MULTIPLIER.get(a, 1)
                    tag = f"  (x{mult}: {a} also trains LOCATIONS)" if mult > 1 else ""
                    lines.append(f"    {a:9s} trainable {t:>9,}  = adapter {nm*256*mult:>7,}"
                                 f" + head {h:,}{tag}")
    # ⛔ THE INVARIANT IS THE FROZEN BACKBONE, NOT THE TOTAL.
    #    `all params` INCLUDES the adapter's own parameters, and the adapters are not
    #    the same size: [measured, gemma-2b/q_o] loca reports 2,506,204,160 against
    #    every other arm's 2,506,185,728 -- a difference of exactly 18,432 = 36 x 512,
    #    its LOCATION tensors.  An earlier version of this check compared the totals
    #    and fired "different backbones?" on a perfectly healthy pair.  A gate that
    #    cries wolf trains you to ignore it.
    #    `all_params - trainable` is the frozen backbone and IS constant:
    #    2,506,172,416 for both, exactly.
    frozen = {}
    for a, r in results.items():
        ap_, tr_ = r["receipts"].get("all_params"), r["receipts"].get("trainable")
        if ap_ and tr_:
            frozen[a] = ap_ - tr_
    fs = set(frozen.values())
    if len(fs) > 1:
        bad.append(f"arms disagree on the FROZEN BACKBONE size {frozen} -- a different "
                   f"backbone, or an arm that froze/unfroze something the others did not")
    elif fs:
        lines.append(f"  frozen backbone identical across arms: {fs.pop():,} params")
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

    # --- the cross-arm check FIRES on each failure it exists to catch
    good = {a: {"rc": 0, "receipts": {"modules": 36, "trainable": 13312,
                                      "all_params": 2506185728}}
            for a in FA.ARM_ORDER}
    good["loca"]["receipts"]["trainable"] = 36 * 256 * 3 + 4096
    # ⚠ REGRESSION FIXTURE, from the real gemma-2b run: loca's `all params` is LARGER
    #   because its location tensors are part of the model.  This must PASS.
    good["loca"]["receipts"]["all_params"] = 2506185728 + 36 * 512
    okg, lines, badg = check(good, "q_o")
    ck(okg, f"a healthy 9-arm result passes ({badg})")
    ck(any("head = 4,096" in l for l in lines), "infers the 4,096-param gemma head")
    ck(any("frozen backbone identical" in l for l in lines), "reports the frozen-backbone invariant")
    ck(True if okg else False,
       "REGRESSION: loca's LARGER all_params (its locations) does NOT read as a different backbone")

    for label, mutate in (
            ("zero modules", lambda d: d["scora"]["receipts"].__setitem__("modules", 0)),
            ("module disagreement", lambda d: d["qwha"]["receipts"].__setitem__("modules", 24)),
            ("a nonzero exit", lambda d: d["lyra"].__setitem__("rc", 1)),
            ("a missing receipt", lambda d: d["wave1"]["receipts"].pop("modules")),
            ("a budget that is not modules*256",
             lambda d: d["fftm"]["receipts"].__setitem__("trainable", 13312 + 7)),
            ("a different frozen backbone",
             lambda d: d["wave2"]["receipts"].__setitem__("all_params", 12345)),
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
