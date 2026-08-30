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


# ⭐⭐ ONE RUNNER, TWO STAGES.  The SEARCH (`fir_hp_plan`) and the FINAL RUNS
#   (`fir_final_plan`) enumerate different cells, but a cell is executed and
#   VERIFIED identically: same command builder contract, same receipt check, same
#   fail-closed exits.  ⛔ Forking this file would fork `verify_receipts` -- the one
#   thing standing between "the adapter attached to nothing" and a plausible number
#   -- so the planner is a PARAMETER, not a copy.
PLANNERS = {"hp": "fir_hp_plan", "final": "fir_final_plan"}


def _planner(name):
    if name not in PLANNERS:
        raise SystemExit(f"FAIL CLOSED: --planner must be one of {sorted(PLANNERS)}")
    return __import__(PLANNERS[name])

# the per-module budget every arm in a sweep grid trains (n_frequency / k = 256)
K = 256


def verify_receipts(text, cell):
    """(ok, note).  ⛔ Fails CLOSED: no receipt is a failure, not a pass.

    `fftstock` (stock PEFT) prints no module count, so the count is derived the
    same way stage D derives it -- (trainable - head) / K -- and the head is
    gemma's `score`, 4,096.  Both numbers are MEASURED on this backbone
    (preflight_56905037/56922969) and asserted, not assumed."""
    # ⛔ THE BUDGET MODEL IS PER-ARM, AND (trainable - head)/256 IS ONLY RIGHT FOR
    #   ARMS WITH MULTIPLIER 1. LoCA trains 256 coefficients AND a 2x256 LOCATION
    #   tensor per module -- 3x, [measured] 31,744 trainable on gemma. Until the
    #   `loca` grid landed this function REFUSED any such arm; it now divides by the
    #   arm's own budget, from the SAME table stage D uses (fir_preflight_arms
    #   owns LOCATION_MULTIPLIER -- one site, not two that agree today).
    # ⚠ THE GUARD IS STILL CLOSED, in the direction that matters: an arm this table
    #   does not know defaults to 1, and a wrong divisor cannot produce 36 modules
    #   AND a whole number -- [measured] loca at multiplier 1 gives 108, which the
    #   n_mod check below rejects. A CONTROL in the selftest fires on exactly that.
    if not cell or not cell.get("arm"):
        return False, "no cell given -- the budget model is PER-ARM and cannot be guessed"
    mult = PA.LOCATION_MULTIPLIER.get(cell["arm"], 1)
    budget = K * mult
    r = PA.parse_receipts(text)
    tr = r.get("trainable")
    if not tr:
        return False, "NO trainable-params receipt -- cannot prove anything ran"
    head = head_params(cell.get("task"))
    adapter = tr - head
    if adapter <= 0:
        return False, (f"trainable {tr:,} <= head {head:,} -- the ADAPTER ATTACHED TO "
                       f"NOTHING and only the classifier trained")
    if adapter % budget:
        return False, (f"adapter params {adapter:,} is not a multiple of "
                       f"{budget:,} (k={K} x {mult} for {cell['arm']}) -- "
                       f"the budget model is wrong for this cell")
    n_mod = adapter // budget
    if r.get("modules") and r["modules"] != n_mod:
        return False, (f"logged module count {r['modules']} != derived {n_mod} "
                       f"from trainable {tr:,}")
    if n_mod != 36:
        return False, (f"attached to {n_mod} modules, not the 36 every preflight arm "
                       f"reported -- this cell did not adapt the same model")
    note = f"{n_mod} modules, adapter {adapter:,} + head {head:,} = {tr:,} trainable"
    if mult != 1:
        note += (f"  [{cell['arm']}: {K} coefficients + {(mult-1)*K} location params "
                 f"per module -- ⚠ NOT at parameter parity, and any table that "
                 f"places it beside the others must say so]")
    return True, note


def head_params(task, hidden=2048):
    """gemma-2b's `score` head: hidden x num_labels, no bias.  ⛔ DERIVED FROM THE
    TASK, never the constant 4,096.

    ⛔⛔ THE BUG THIS FIXES [2026-08-30]. It WAS the constant 4,096 -- correct for
      every 2-class task, and correct for the entire MRPC hyperparameter search, so
      658 cells never touched it. **STS-B IS A REGRESSION TASK: ONE output, so its
      head is 2,048.** The receipt check therefore subtracted 2,048 too much,
      derived 28 adapted modules instead of 36, and refused a cell that had trained
      perfectly (2700/2700 steps, pearson 0.8195) with exit=3.
    ⭐ AND NOTE HOW NARROWLY IT FAILED SAFE: 7,168 IS a multiple of 256, so the
      budget-modulo check passed and only the `n_mod != 36` check caught it. Had the
      arithmetic landed on 36, a WRONG head size would have passed SILENTLY. A
      derived value removes the coincidence.
    ⚠ `num_labels` comes from the MEASURED, COMMITTED dataset sizes -- a task with
      no label histogram is a regression task, which is exactly how `r310_read`
      decides that a metric has no majority-class floor."""
    if not task:
        raise SystemExit("FAIL CLOSED: the head size is PER TASK and cannot be guessed")
    import fir_plan as FP
    S = FP.sizes()
    if task not in S:
        raise SystemExit(f"FAIL CLOSED: no measured sizes for task {task!r}")
    lc = S[task].get("label_counts")
    return hidden * (1 if lc is None else len(lc))


# ⛔ A DISTINCT EXIT CODE FOR "THIS CELL IS NOT IN THIS GRID", because it is a
#   DIFFERENT EVENT from "the cell ran and failed" and the two must not be reported
#   as one. [2026-08-28] five canaries were submitted 40 s apart against ONE shared
#   cells.txt; four arrays read the last writer's list and looked up a `scora2` id
#   under a loca/qwha/lyra/scora pin. This guard refused all four -- ⭐ nothing wrong
#   was measured, which is the entire point of failing closed -- but the shell could
#   not tell the difference, so it left `started/` markers for cells that never ran.
RC_NOT_IN_GRID = 5


def run(cell_id, run_root, python=None, timeout=None, planner="hp"):
    H = _planner(planner)
    try:
        c = H.parse_cell_id(cell_id)
    except SystemExit as e:
        print(f"⛔ PLAN/GRID MISMATCH: {e}")
        _sel = getattr(H, "GRID_NAME", None) or getattr(H, "TASK_NAME", "?")
        print(f"   selected here: {_sel!r}  ({len(H.cells())} cells)")
        print( "   NOTHING RAN. The array read a plan file written for another grid;")
        print( "   re-submit this grid -- plan files are per-submission now.")
        return RC_NOT_IN_GRID
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
    W1 = {"arm": "wave1", "task": "mrpc"}     # multiplier 1, like every grid arm
    good, note = verify_receipts(real, W1)
    ck(good, f"the REAL preflight receipt passes ({note})")

    # stock PEFT prints no module count -- it must still pass, by derivation
    ck(verify_receipts("trainable params: 13,312 || all params: 2,506,189,824 || x\n",
                       {"arm": "fftstock", "task": "mrpc"})[0],
       "fftstock's count-free receipt passes by derivation")
    # ⭐ WaveFT: mu costs ZERO parameters, so mu=1 and mu=2 train the SAME 9,216 and
    #   the same derivation holds for both -- asserted, because it is the premise of
    #   letting the wave arms through this check at all.
    for a in ("wave1", "wave2"):
        ck(verify_receipts(real, {"arm": a, "task": "mrpc"})[0],
           f"{a}: the same 13,312 receipt passes "
                                                 f"(mu costs no parameters)")
    # ⭐ LoCA: [measured, gemma] 31,744 = 4,096 head + 36 x 256 x 3. It must PASS,
    #   and the note must SAY it is not at parameter parity -- [R.310] flags exactly
    #   this as the thing a joint table has to disclose.
    loca_receipt = "trainable params: 31,744 || all params: 2,506,204,160 || x\n"
    okl, notel = verify_receipts(loca_receipt, {"arm": "loca", "task": "mrpc"})
    ck(okl and "36 modules" in notel and "parity" in notel,
       f"loca's measured 31,744 receipt passes at multiplier 3 ({notel[:60]}...)")
    # ⛔ CONTROL THAT THE MULTIPLIER IS LOAD-BEARING: the SAME receipt read at
    #   multiplier 1 must be REFUSED. Without this, teaching the guard the arm would
    #   look identical to disabling it.
    ck(not verify_receipts(loca_receipt, {"arm": "wave1", "task": "mrpc"})[0],
       "CONTROL: loca's receipt is REFUSED when read at multiplier 1 (108 modules, "
       "not 36) -- the location budget is doing real work here")
    ck(not verify_receipts("trainable params: 13,312 || all params: 2 || x\n",
                           {"arm": "loca", "task": "mrpc"})[0],
       "CONTROL: a NON-loca receipt is refused when read at multiplier 3 "
       "(the guard fires in both directions)")
    ck(not verify_receipts(real, None)[0],
       "CONTROL: no cell at all is refused (the budget model is per-arm)")

    # ------------------------------------------------------------------
    # ⭐⭐ THE HEAD SIZE IS PER TASK. [2026-08-30] this was the constant 4,096 --
    #   right for every 2-class task and for all 658 MRPC search cells, and WRONG
    #   for STS-B, which is a REGRESSION task with ONE output. It refused a cell
    #   that had trained perfectly (2700/2700 steps, pearson 0.8195) with exit=3.
    # ------------------------------------------------------------------
    ck(head_params("mrpc") == 4096 and head_params("rte") == 4096
       and head_params("qnli") == 4096,
       "the 2-class tasks keep the measured 4,096-param head (658 search cells)")
    ck(head_params("stsb") == 2048,
       "⭐ STS-B is REGRESSION -- one output, so its head is 2,048, DERIVED from the "
       "measured label histogram being absent")
    # a real STS-B receipt: 36 x 256 adapter + a 2,048 head
    stsb_receipt = "trainable params: 11,264 || all params: 2,506,183,680 || x\n"
    ck(verify_receipts(stsb_receipt, {"arm": "wave1", "task": "stsb"})[0],
       "⭐ an STS-B receipt (11,264 = 36 x 256 + 2,048) now PASSES")
    # ⛔ CONTROLS, both directions -- the task must be LOAD-BEARING, or deriving it
    #   would be indistinguishable from having removed the check.
    ck(not verify_receipts(stsb_receipt, {"arm": "wave1", "task": "mrpc"})[0],
       "⛔ CONTROL: the SAME receipt read as a 2-class task is REFUSED (28 modules, "
       "not 36) -- that is exactly the failure this fixes")
    ck(not verify_receipts(real, {"arm": "wave1", "task": "stsb"})[0],
       "⛔ CONTROL: and an MRPC receipt read as STS-B is refused too (44 modules)")
    try:
        verify_receipts(real, {"arm": "wave1"})
        ck(False, "CONTROL: a cell with no task is refused")
    except SystemExit:
        ck(True, "CONTROL: a cell with NO task fails closed -- the head cannot be guessed")

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
    # ⛔ CONTROL: a cell id from ANOTHER grid must return RC_NOT_IN_GRID -- a code the
    #   shell can distinguish from "it ran and failed", so it can WITHDRAW the start
    #   marker. Not a crash, and not a plain 1.
    import tempfile as _tf
    _d = _tf.mkdtemp()
    ck(run("mrpc-nosucharm-q_o-lr9p9-sc1-clr1-seed42", _d) == RC_NOT_IN_GRID,
       f"CONTROL: a foreign cell id returns RC_NOT_IN_GRID ({RC_NOT_IN_GRID}), "
       f"distinguishably from a training failure")
    ck(not os.path.exists(os.path.join(_d, "logs")),
       "...and it wrote no log, because nothing ran")
    # ⭐ THE PLANNER IS A PARAMETER, SO BOTH VALUES MUST BE EXERCISED -- and each
    #   must REFUSE the other's cell ids, or the two stages could silently run each
    #   other's work out of a shared plan file (the 2026-08-28 failure, one level up).
    import fir_final_plan as _FP
    _fin = _FP.cell_id(_FP.cells(task="rte")[0])
    _hp = "mrpc-scora-q_o-lr0p0451498-clr0p002-seed42"
    ck(run(_fin, _d, planner="hp") == RC_NOT_IN_GRID,
       "CONTROL: a FINAL cell id is refused by the hp planner")
    ck(run(_hp, _d, planner="final") == RC_NOT_IN_GRID,
       "CONTROL: an hp cell id is refused by the final planner")
    try:
        _planner("nope"); ck(False, "CONTROL: an unknown planner is refused")
    except SystemExit:
        ck(True, "CONTROL: an unknown planner is refused")
    ck(_planner("final") is _FP and _planner("hp") is H,
       "...and the two names resolve to the two real planner modules")

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
    ap.add_argument("--planner", default="hp", choices=sorted(PLANNERS))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.cell or not a.run_root:
        raise SystemExit("FAIL CLOSED: --cell and --run-root are required")
    sys.exit(run(a.cell, a.run_root, planner=a.planner))


if __name__ == "__main__":
    main()
