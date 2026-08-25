#!/usr/bin/env python
"""The nine [R.305]/[R.306] arms, FROZEN into a file that survives the trip to fir.

⛔ THE PROBLEM THIS SOLVES.  `scripts/r310_plan.selected_args()` derives each arm's
exact flag string from the R.305/R.306 run manifests -- which live under
`scratchpad/phaseR/`, and `.gitignore` excludes `scratchpad`.  So on fir, after a
`git pull`, `selected_args()` returns `{}` SILENTLY (it returns {} for a missing
manifest by design).  A planner built on it would emit nine arms with no flags and
every cell would run the argparse DEFAULTS -- a completely plausible-looking sweep
of the wrong experiment.  That is FIR_SETUP's G-class defect: nothing crashes, the
sweep completes, the table is wrong.

⇒ On the DEV BOX (where the manifests exist) `--freeze` writes
  `sbatch/fir/arms_r305r306.json`, which IS tracked by git and therefore travels.
  Everywhere else `load()` reads that file and FAILS CLOSED if it is absent,
  malformed, or does not match its own recorded checksum.

⛔ FAIL CLOSED, NEVER OPEN (FIR_SETUP Law 3): every accessor here raises rather
   than returning a default.  An empty arm table must never be mistaken for a
   valid one.

Usage:
    env/bin/python scripts/fir_arms.py --freeze     # dev box only; rewrites the JSON
    env/bin/python scripts/fir_arms.py              # print what is frozen
    env/bin/python scripts/fir_arms.py --selftest
"""
import argparse, hashlib, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = os.path.join(ROOT, "sbatch", "fir", "arms_r305r306.json")

# The nine rows in table order, with the flag that carries each arm's SCALE.
#
# ⛔ THE SCALE FLAG IS NOT COSMETIC.  Porting to a new backbone changes the module
#    width, and every arm's per-parameter ATOM norm is a function of (m, n):
#      FourierFT / WaveFT   atom = s / sqrt(2mn)
#      QWHA                 atom = s / sqrt(mn)
#      SCoRA                atom = scaling * sqrt(t)     <- s-dependent, NOT width
#    so the RoBERTa-tuned raw `lr` is valid ONLY at d=768 (memory: hp-transfer-proxy,
#    "carry P = lr*atom, not the raw lr").  This column tells the porting instrument
#    which knob to move.
#
# ⚠ `scora` deliberately has NO scale flag: its selected args omit --slr_scaling, so
#   the adapter DERIVES its own scale from --slr_s.  That derivation already adapts
#   to width.  Adding a scale flag for it would be a SECOND knob and would silently
#   convert the a-priori arm into a tuned one.  `scora2` DOES carry an explicit
#   --slr_scaling because [R.306] swept it; both rows always ship together.
ARM_SCALE_FLAG = {
    "fftm":     "--fourierftmerged_scaling",
    "fftstock": "--fourierft_scaling",
    "loca":     "--loca_scale",
    "qwha":     "--qwha_scaling",
    "wave1":    "--haar_fourierft_scaling",
    "wave2":    "--haar_fourierft_scaling",
    "lyra":     "--spectral_scaling",
    "scora":    None,               # derives its scale from --slr_s. DO NOT ADD ONE.
    "scora2":   "--slr_scaling",
}

# The per-arm flag that names the adapter's OWN target modules.  train_glue.py has a
# generic `--adapter_target_modules` override that wins over all of these, but the
# arms set both in [R.305]/[R.310] and a port that moved only one would leave the
# other pointing at RoBERTa names -- which resolve to NOTHING on a decoder, i.e. an
# adapter attached to zero modules that still trains (the classifier head alone) and
# still writes a plausible row.
ARM_TARGET_FLAG = {
    "fftm":     "--fourierftmerged_target_modules",
    "fftstock": None,               # stock PEFT: target modules come from the PEFT config
    "loca":     "--loca_target_modules",
    "qwha":     "--qwha_target_modules",
    "wave1":    "--haar_target_modules",
    "wave2":    "--haar_target_modules",
    "lyra":     "--spectral_target_modules",
    "scora":    "--slr_target_modules",
    "scora2":   "--slr_target_modules",
}

ARM_ORDER = ["fftm", "fftstock", "loca", "qwha", "wave1", "wave2", "lyra", "scora", "scora2"]
SCORA_ROWS = ("scora", "scora2")


def _digest(args_by_arm):
    """Checksum over the arm table itself, so a hand-edit of the JSON is detected."""
    blob = json.dumps(args_by_arm, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def parse_flags(argstr):
    """'--a 1 --b x,y' -> {'--a': '1', '--b': 'x,y'}.  Value-less flags map to ''."""
    toks = argstr.split()
    out, i = {}, 0
    while i < len(toks):
        t = toks[i]
        if not t.startswith("--"):
            raise ValueError(f"expected a flag, got {t!r} in {argstr!r}")
        if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
            out[t] = toks[i + 1]; i += 2
        else:
            out[t] = ""; i += 1
    return out


def freeze():
    """DEV BOX ONLY.  Re-derive from the run manifests and rewrite the JSON."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import r310_plan as P
    sel = P.selected_args()
    if len(sel) != len(ARM_ORDER):
        raise SystemExit(
            f"REFUSING to freeze: selected_args() gave {len(sel)} arms {sorted(sel)}, "
            f"expected {len(ARM_ORDER)}.  The R.305/R.306 manifests under scratchpad/ "
            f"are missing or partial -- freeze only on the box that HAS them.")
    ref = P.rte_reference()
    payload = {
        "_README": "GENERATED by scripts/fir_arms.py --freeze on the dev box. "
                   "Do not hand-edit: the digest below is checked on load.",
        "source": "[R.305] stage-D manifests + [R.306] for scora2",
        "arm_order": ARM_ORDER,
        "args": {a: sel[a] for a in ARM_ORDER},
        "rte_reference_mean": {a: ref[a][0] for a in ARM_ORDER if a in ref},
        "digest": _digest({a: sel[a] for a in ARM_ORDER}),
    }
    os.makedirs(os.path.dirname(FROZEN), exist_ok=True)
    with open(FROZEN, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")
    return payload


def load():
    """Read the frozen table.  RAISES on anything unexpected -- never returns {}."""
    if not os.path.exists(FROZEN):
        raise SystemExit(
            f"FAIL CLOSED: {os.path.relpath(FROZEN, ROOT)} is absent.\n"
            f"  On the DEV BOX:  env/bin/python scripts/fir_arms.py --freeze\n"
            f"  Then COMMIT it -- scratchpad/ is gitignored and does not travel to fir.")
    with open(FROZEN) as f:
        p = json.load(f)
    for key in ("arm_order", "args", "digest"):
        if key not in p:
            raise SystemExit(f"FAIL CLOSED: {FROZEN} has no {key!r}")
    if p["arm_order"] != ARM_ORDER:
        raise SystemExit(f"FAIL CLOSED: arm order drifted: {p['arm_order']} != {ARM_ORDER}")
    if sorted(p["args"]) != sorted(ARM_ORDER):
        raise SystemExit(f"FAIL CLOSED: arm set drifted: {sorted(p['args'])}")
    got = _digest(p["args"])
    if got != p["digest"]:
        raise SystemExit(
            f"FAIL CLOSED: {os.path.relpath(FROZEN, ROOT)} was hand-edited.\n"
            f"  digest recorded {p['digest'][:16]}...  computed {got[:16]}...\n"
            f"  Re-freeze on the dev box rather than editing the JSON.")
    # ⛔ [R.306]: both SCoRA rows always ship together, or the "we did not tune ours
    #    harder after it lost" claim becomes unfalsifiable downstream.
    for a in SCORA_ROWS:
        if a not in p["args"]:
            raise SystemExit(f"FAIL CLOSED: {a} missing -- both SCoRA rows always ship together")
    # ⛔ fftstock is the arm a hand-copy gets wrong: it RUNS a different optimiser
    #    from fftm.  Eight identical FourierFT columns is a silent, plausible table.
    if p["args"]["fftstock"] == p["args"]["fftm"]:
        raise SystemExit("FAIL CLOSED: fftstock == fftm -- the hand-copy trap")
    if "adamw-fourierft " not in p["args"]["fftstock"] + " ":
        raise SystemExit("FAIL CLOSED: fftstock is not the stock PEFT optimiser")
    if "adamw-fourierftmerged" not in p["args"]["fftm"]:
        raise SystemExit("FAIL CLOSED: fftm is not the merged path")
    # every declared scale flag must actually be present in that arm's string
    for a, flag in ARM_SCALE_FLAG.items():
        if flag is None:
            if a == "scora" and "--slr_scaling" in p["args"][a]:
                raise SystemExit(
                    "FAIL CLOSED: `scora` carries an explicit --slr_scaling.  Its whole "
                    "point is the A-PRIORI scale; an explicit one makes it a second "
                    "tuned row and destroys the [R.306] contrast.")
            continue
        if flag not in p["args"][a]:
            raise SystemExit(f"FAIL CLOSED: {a} has no {flag} -- ARM_SCALE_FLAG is stale")
    return p


def selftest():
    global FROZEN
    ok, bad = [], []

    def ck(cond, label):
        (ok if cond else bad).append(label)

    # --- parse_flags
    f = parse_flags("--a 1 --b x,y --flagonly --c 2")
    ck(f == {"--a": "1", "--b": "x,y", "--flagonly": "", "--c": "2"}, "parse_flags roundtrip")
    try:
        parse_flags("garbage --a 1"); ck(False, "parse_flags rejects a non-flag head")
    except ValueError:
        ck(True, "parse_flags rejects a non-flag head")

    # --- digest is sensitive (a control that CAN fail)
    a1 = {"x": "--lr 1"}
    a2 = {"x": "--lr 2"}
    ck(_digest(a1) != _digest(a2), "digest changes when an arg changes")
    ck(_digest(a1) == _digest(dict(a1)), "digest is stable for equal tables")

    # --- the frozen file, if present, loads and passes every closed gate
    if os.path.exists(FROZEN):
        p = load()
        ck(len(p["args"]) == 9, "frozen table has 9 arms")
        ck(all(p["args"][a].strip() for a in ARM_ORDER), "no arm has an empty arg string")
        ck("--optimizer" in p["args"]["scora"], "scora carries an optimizer")
        ck("--slr_scaling" not in p["args"]["scora"], "scora has NO explicit scaling")
        ck("--slr_scaling" in p["args"]["scora2"], "scora2 DOES carry an explicit scaling")
        # every arm names query,value somewhere -- the thing a decoder port must move
        ck(all("query,value" in p["args"][a] for a in ARM_ORDER if ARM_TARGET_FLAG[a]),
           "every targeted arm names RoBERTa modules (so the port has something to move)")

        # ⭐ A CONTROL THAT CAN FAIL: corrupt the digest in a temp copy and prove
        #    load() REFUSES it.  A guard whose failure mode is silent needs a test
        #    that can fail (FIR_SETUP 7.2.1).
        import shutil, tempfile
        keep = FROZEN
        try:
            d = tempfile.mkdtemp()
            tmp = os.path.join(d, "arms.json")
            shutil.copy(keep, tmp)
            bad_payload = json.load(open(tmp))
            bad_payload["args"]["fftm"] = bad_payload["args"]["fftm"] + " --learning_rate 999"
            json.dump(bad_payload, open(tmp, "w"))
            FROZEN = tmp
            try:
                load(); ck(False, "CONTROL: load() refuses a hand-edited table")
            except SystemExit:
                ck(True, "CONTROL: load() refuses a hand-edited table")
            # and the fftstock==fftm trap
            bad2 = json.load(open(keep))
            bad2["args"]["fftstock"] = bad2["args"]["fftm"]
            bad2["digest"] = _digest(bad2["args"])
            json.dump(bad2, open(tmp, "w"))
            try:
                load(); ck(False, "CONTROL: load() refuses fftstock == fftm")
            except SystemExit:
                ck(True, "CONTROL: load() refuses fftstock == fftm")
        finally:
            FROZEN = keep
    else:
        ok.append("frozen table absent -- freeze gates skipped (dev box: --freeze)")

    for lbl in ok:
        print(f"  ✅ {lbl}")
    for lbl in bad:
        print(f"  ⛔ {lbl}")
    print(f"selftest: {len(ok)} passed, {len(bad)} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true", help="dev box only: re-derive and rewrite the JSON")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.freeze:
        p = freeze()
        print(f"froze {len(p['args'])} arms -> {os.path.relpath(FROZEN, ROOT)}")
        print(f"digest {p['digest']}")
    else:
        p = load()
        print(f"# {os.path.relpath(FROZEN, ROOT)}  digest {p['digest'][:16]}...")
    for a_ in ARM_ORDER:
        print(f"  {a_:9s} :: {p['args'][a_]}")
