#!/usr/bin/env python
"""[fir] Run ONE hyperparameter-grid cell and prove it actually adapted.

⛔ WHY THIS EXISTS INSTEAD OF THE SHELL CALLING train_glue DIRECTLY.
   A RoBERTa module name on a decoder matches NOTHING: the adapter attaches to
   zero modules, the head trains alone, the run exits 0, and MRPC's F1 lands
   somewhere entirely plausible.  Stage D of the preflight checks that ACROSS
   arms; a sweep cell is alone, so it must check ITSELF.  Every cell therefore
   parses its own log for the receipts and FAILS if the adapter did not attach --
   a cell that trained nothing must never write a `done` marker.

Usage:
    env/bin/python scripts/fir_hp_run_cell.py --cell <id> --run-root <dir>
    env/bin/python scripts/fir_hp_run_cell.py --selftest
"""
import argparse, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import fir_hp_plan as H                                                # noqa: E402
import fir_preflight_arms as PA                                        # noqa: E402

# the per-module budget every arm in a sweep grid trains (n_frequency / k = 256)
K = 256


def verify_receipts(text, cell):
    """(ok, note).  ⛔ Fails CLOSED: no receipt is a failure, not a pass.

    `fftstock` (stock PEFT) prints no module count, so the count is derived the
    same way stage D derives it -- (trainable - head) / K -- and the head is
    gemma's `score`, 4,096.  Both numbers are MEASURED on this backbone
    (preflight_56905037/56922969) and asserted, not assumed."""
    # ⛔ THE BUDGET MODEL IS PER-ARM, AND (trainable - head)/256 IS ONLY RIGHT FOR
    #   ARMS WITH MULTIPLIER 1. LoCA trains 256 coefficients AND a 2x256 location
    #   tensor per module (x3, [measured] 31,744 on gemma) -- it is not in any grid
    #   today, but "not today" is not a check. Fail closed rather than divide by the
    #   wrong budget and report a plausible module count.
    if not cell or not cell.get("arm"):
        return False, "no cell given -- the budget model is PER-ARM and cannot be guessed"
    mult = PA.LOCATION_MULTIPLIER.get(cell["arm"], 1)
    if mult != 1:
        return False, (f"{cell['arm']} trains {mult}x the coefficient budget "
                       f"(it also trains LOCATIONS); this cell-local derivation "
                       f"assumes multiplier 1 -- teach it the arm before sweeping it")
    r = PA.parse_receipts(text)
    tr = r.get("trainable")
    if not tr:
        return False, "NO trainable-params receipt -- cannot prove anything ran"
    head = 4096                       # gemma-2b `score` = hidden 2048 x 2 labels
    adapter = tr - head
    if adapter <= 0:
        return False, (f"trainable {tr:,} <= head {head:,} -- the ADAPTER ATTACHED TO "
                       f"NOTHING and only the classifier trained")
    if adapter % K:
        return False, (f"adapter params {adapter:,} is not a multiple of k={K} -- "
                       f"the budget model is wrong for this cell")
    n_mod = adapter // K
    if r.get("modules") and r["modules"] != n_mod:
        return False, (f"logged module count {r['modules']} != derived {n_mod} "
                       f"from trainable {tr:,}")
    if n_mod != 36:
        return False, (f"attached to {n_mod} modules, not the 36 every preflight arm "
                       f"reported -- this cell did not adapt the same model")
    return True, f"{n_mod} modules, adapter {adapter:,} + head {head:,} = {tr:,} trainable"


def run(cell_id, run_root, python=None, timeout=None):
    c = H.parse_cell_id(cell_id)
    cmd = H.cell_cmd(c)
    env = dict(os.environ)
    env.update(H.cell_env(c, run_root))
    env["PYTHONPATH"] = os.path.join(ROOT, "src") + os.pathsep + env.get("PYTHONPATH", "")
    os.makedirs(os.path.join(run_root, "csv"), exist_ok=True)
    os.makedirs(os.path.join(run_root, "logs"), exist_ok=True)
    logp = os.path.join(run_root, "logs", cell_id + ".log")
    print("cmd: " + " ".join(cmd), flush=True)
    p = subprocess.run([python or sys.executable] + cmd, cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=timeout)
    out = (p.stdout or "") + (p.stderr or "")
    with open(logp, "w") as f:
        f.write(" ".join(cmd) + "\n\n" + out)
    if p.returncode != 0:
        print(f"⛔ train_glue exited {p.returncode}. Last lines:")
        for l in [x for x in out.strip().splitlines() if x.strip()][-15:]:
            print(f"  | {l[:200]}")
        return p.returncode
    ok, note = verify_receipts(out, c)
    print(("✅ receipts: " if ok else "⛔ RECEIPT CHECK FAILED: ") + note)
    if not ok:
        print("   the cell exited 0 but did not train what it claims to. NOT marked done.")
        return 3
    csv = H.cell_env(c, run_root)["GLUE_RESULTS_FILE"]
    if not os.path.exists(csv):
        print(f"⛔ no results CSV at {csv} -- nothing to read later")
        return 4
    return 0


# ---------------------------------------------------------------------------
def selftest():
    ok, bad = [], []

    def ck(c, l):
        (ok if c else bad).append(l)

    real = ("INFO - __main__ - SLR target modules: ['q_proj', 'o_proj']\n"
            "trainable params: 13,312 || all params: 2,506,185,728 || trainable%: 0.0005\n")
    W1 = {"arm": "wave1"}                     # multiplier 1, like every grid arm
    good, note = verify_receipts(real, W1)
    ck(good, f"the REAL preflight receipt passes ({note})")

    # stock PEFT prints no module count -- it must still pass, by derivation
    ck(verify_receipts("trainable params: 13,312 || all params: 2,506,189,824 || x\n",
                       {"arm": "fftstock"})[0],
       "fftstock's count-free receipt passes by derivation")
    # ⭐ WaveFT: mu costs ZERO parameters, so mu=1 and mu=2 train the SAME 9,216 and
    #   the same derivation holds for both -- asserted, because it is the premise of
    #   letting the wave arms through this check at all.
    for a in ("wave1", "wave2"):
        ck(verify_receipts(real, {"arm": a})[0], f"{a}: the same 13,312 receipt passes "
                                                 f"(mu costs no parameters)")
    # ⛔ CONTROL: an arm with a LOCATION budget must be REFUSED, not divided wrongly
    okl, notel = verify_receipts("trainable params: 31,744 || all params: 2,506,204,160 || x\n",
                                 {"arm": "loca"})
    ck(not okl and "LOCATIONS" in notel,
       "CONTROL: a multiplier-3 arm (loca) is REFUSED by the cell-local derivation")
    ck(not verify_receipts(real, None)[0],
       "CONTROL: no cell at all is refused (the budget model is per-arm)")

    for label, text in (
        ("the adapter attaching to NOTHING (head trains alone)",
         "trainable params: 4,096 || all params: 2,506,172,416 || x\n"),
        ("no receipt at all", "the run said nothing\n"),
        ("a budget that is not a multiple of k",
         "trainable params: 13,313 || all params: 2,506,185,728 || x\n"),
        ("HALF the modules adapting",
         "trainable params: 8,704 || all params: 2,506,185,728 || x\n"),
        ("a logged count that contradicts the derived one",
         "SLR: adapted 18 modules\ntrainable params: 13,312 || all params: 2,5 || x\n"),
    ):
        # ⛔ PASS A REAL CELL. These controls used `None`, and the moment the
        #   per-arm budget guard landed, `None` was refused for the WRONG reason --
        #   every one of them would have passed vacuously while testing nothing.
        okc, notec = verify_receipts(text, W1)
        ck(not okc and "no cell given" not in notec, f"CONTROL: FIRES on {label}")

    ck(H.parse_cell_id(H.cell_id(H.cells()[0])) is not None, "a cell id round-trips")

    for l in ok:
        print(f"  ✅ {l}")
    for l in bad:
        print(f"  ⛔ {l}")
    print(f"selftest: {len(ok)} passed, {len(bad)} failed")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell")
    ap.add_argument("--run-root")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.cell or not a.run_root:
        raise SystemExit("FAIL CLOSED: --cell and --run-root are required")
    sys.exit(run(a.cell, a.run_root))


if __name__ == "__main__":
    main()
