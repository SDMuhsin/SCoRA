#!/usr/bin/env python
"""[fir] THE MRPC HYPERPARAMETER GRID for FourierFT on gemma-2b.  ONE STAGE.

USER DECISIONS 2026-08-26 -- do not re-litigate:
  * the FIRST fir job is a hyperparameter search, on MRPC, run in PARALLEL
  * ONE stage.  Broad and exhaustive over its own grid; NOT coarse-then-fine
  * cheap: ONE seed, FIVE epochs
  * `--classifier_lr` is IN the search (it had no a-priori port rule)
  * BOTH FourierFT arms: `fftm` (ours, proven bit-identical on fir) and
    `fftstock` (stock PEFT)
  * targets `q_o` (q_proj, o_proj -- shape-matched, 1.74x init spread)
  * grid shape: 5 lr x 4 scaling x 4 classifier_lr = 80 cells per arm, 160 total

⭐ NOTE WHAT THIS DISSOLVES.  `--port-mode derived|asis` was the third open
   protocol decision.  It only ever set `scaling` and `learning_rate` -- and this
   grid SWEEPS both.  The port table is now a PREDICTION to check against the
   measured optimum, not an input.  (The derived point for q_o is lr* 0.4697 at
   scale* 141.94; RoBERTa's tuned point was lr 0.5 at scaling 50.  Both are on the
   grid on purpose.)

⛔ WHAT A GRID CANNOT TELL YOU, stated before it is read:
   one seed.  A difference between two neighbouring cells at one seed is not a
   ranking; [R.273]'s null puts the seed-to-seed sigma on RoBERTa/RTE at a size
   that swallows most adjacent-cell gaps.  This grid LOCATES a region; confirming
   a point needs seeds, and that is a separate spend.

Usage:
    env/bin/python scripts/fir_hp_plan.py --selftest
    env/bin/python scripts/fir_hp_plan.py --list          # one cell id per line
    env/bin/python scripts/fir_hp_plan.py --cmd <cell-id> # the exact command
    env/bin/python scripts/fir_hp_plan.py --show
"""
import argparse, hashlib, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import fir_arms as FA                                                  # noqa: E402
import fir_plan as FP                                                  # noqa: E402

# ---------------------------------------------------------------------------
# THE GRIDS.  One place, committed, digested.  `g1` is kept because 160 measured
# cells refer to it; `g2` is the live one.  Select with FIR_HP_GRID (default g2)
# so the planner, the runner and the reader cannot disagree about which grid is
# in play -- three tools reading one env var, not a flag threaded through a shell.
# ---------------------------------------------------------------------------
TASK = "mrpc"
TARGETS = "q_o"
EPOCHS = 5
SEED = 42
ARMS = ["fftm", "fftstock"]

# ⭐ g1 -- RUN AND COMPLETE 2026-08-26. 160/160 cells, 0 failed, 16.3 GPU-h.
#   [measured] best F1 0.8945 at lr 1.5 / scaling 400 / clf_lr 5e-4 (fftm), but
#   ⛔ SCALING WAS AT THE GRID EDGE and strictly monotone across it:
#        sc  25 -> best F1 0.8352 / acc 0.7598      (RoBERTa's tuned 50: 0.8459)
#        sc  50 -> 0.8459 / 0.7794
#        sc 142 -> 0.8767 / 0.8186                  (the DERIVED scale* 141.94)
#        sc 400 -> 0.8945 / 0.8456
#   so the optimum lies OUTSIDE g1 and its best cell is not quotable as an optimum.
G1 = {
    "lrs": [0.05, 0.15, 0.5, 1.5, 4.0],
    "scalings": [25, 50, 142, 400],
    "clf_lrs": [5e-4, 2e-3, 5e-3, 2e-2],
}

# ⭐⭐ g2 -- THE DECISIVE GRID. Designed to need no successor [user, 2026-08-27:
#   "we can't keep having this back and forth and multiple stages"].
#
#   lr: FINER (ratio ~2.5, was ~3.2) and EXTENDED AT THE TOP to 15 -- 300x range.
#     ⚠ The bottom endpoint stays at 0.05 rather than going lower: g1 measured it
#       as the worst row (4 cells at the collapse floor). Extending downward would
#       buy known-dead cells.
#   scaling: EXTENDED 20x BEYOND g1's edge. This is where the edge actually was.
#     ⛔ AND THE TWO AXES ARE NOT INTERCHANGEABLE, which is why the extension is
#       NOT folded into the product lr*scaling: [measured, g1] at a MATCHED product
#       of 200, (lr 0.5, sc 400) scores 0.8823 while (lr 4, sc 50) scores 0.8354.
#       Large scaling at modest lr beats the reverse. AdamW's decoupled decay
#       shrinks the spectrum by lr*wd per step, so a large lr is TAXED in a way a
#       large scaling is not -- this repo's [R.0 5d] weight-decay trap, again.
#   classifier_lr: TRIMMED to the two that matter, which pays for the width above.
#     [measured, g1] 2e-2 is harmful (8 of the 9 at-floor cells use it); 2e-3 and
#     5e-3 differ by less than the seed noise; the axis is otherwise flat
#     (best F1 0.8823..0.8945 across all four values).
#   epochs stay at 5: [measured, g1] 0 of the top 30 cells peak at the LAST epoch
#     (18 at epoch 4, 11 at epoch 3), so the schedule is not the binding constraint.
G2 = {
    "lrs": [0.05, 0.15, 0.4, 1.0, 2.5, 6.0, 15.0],
    "scalings": [142, 400, 1100, 3000, 8000],
    "clf_lrs": [5e-4, 5e-3],
}

GRIDS = {"g1": G1, "g2": G2}
GRID_NAME = os.environ.get("FIR_HP_GRID", "g2")
if GRID_NAME not in GRIDS:
    raise SystemExit(f"FAIL CLOSED: FIR_HP_GRID={GRID_NAME!r} is not one of {sorted(GRIDS)}")
_G = GRIDS[GRID_NAME]
LRS, SCALINGS, CLF_LRS = _G["lrs"], _G["scalings"], _G["clf_lrs"]

# ⭐ g1 and g2 OVERLAP by construction (lr 0.05/0.15 x sc 142/400 x both clf_lrs x
#   2 arms = 16 cells). A cell id is a pure function of its knobs, so those cells
#   keep their g1 ids, their g1 CSVs and their g1 `done` markers -- the sweep
#   script skips them and they cost nothing. That is why the ids carry the VALUES
#   and not a grid name.


def _fmt(x):
    """A filename-safe, ROUND-TRIPPABLE number.  ⛔ Not str(): 0.05 and 5e-2 must
    not become two different cell ids for one cell."""
    s = f"{float(x):g}"
    return s.replace(".", "p").replace("-", "m").replace("+", "")


def _cells_of(grid, arms=None):
    """Enumerate a grid dict directly -- used by the selftest to compare g1 with g2
    without mutating module globals (which would make the test order-dependent)."""
    out = []
    for arm in (arms or ARMS):
        for lr in grid["lrs"]:
            for sc in grid["scalings"]:
                for clr in grid["clf_lrs"]:
                    out.append({"arm": arm, "task": TASK, "targets": TARGETS,
                                "seed": SEED, "epochs": EPOCHS,
                                "lr": lr, "scaling": sc, "classifier_lr": clr})
    return out


def cells(arms=None):
    """Every cell, in a DETERMINISTIC order -- the array index is this order, so
    it must not depend on a set, a dict iteration or the filesystem."""
    out = []
    for arm in (arms or ARMS):
        for lr in LRS:
            for sc in SCALINGS:
                for clr in CLF_LRS:
                    out.append({"arm": arm, "task": TASK, "targets": TARGETS,
                                "seed": SEED, "epochs": EPOCHS,
                                "lr": lr, "scaling": sc, "classifier_lr": clr})
    return out


def cell_id(c):
    return (f"{c['task']}-{c['arm']}-{c['targets']}"
            f"-lr{_fmt(c['lr'])}-sc{_fmt(c['scaling'])}-clr{_fmt(c['classifier_lr'])}"
            f"-seed{c['seed']}")


def parse_cell_id(cid):
    """The inverse.  Used by the array task, so a typo cannot silently run a
    DIFFERENT cell than the one whose name the CSV will carry."""
    for c in cells():
        if cell_id(c) == cid:
            return c
    raise SystemExit(f"FAIL CLOSED: {cid!r} is not a cell in this grid")


def digest():
    return hashlib.sha1(json.dumps(
        [cell_id(c) for c in cells()], sort_keys=True).encode()).hexdigest()[:12]


def _set_flag(tokens, flag, value):
    """Replace a flag's value IN PLACE, failing closed if the flag is absent.

    ⛔ Appending instead would leave the ORIGINAL value earlier in the command.
      argparse takes the last one, so it would work -- until something greps the
      command for the value it ran, and finds two.  [FIR_SETUP G3]"""
    if flag not in tokens:
        raise SystemExit(f"FAIL CLOSED: {flag} not in the cell command -- the arm's "
                         f"frozen flag string changed and this grid is stale")
    tokens = list(tokens)
    tokens[tokens.index(flag) + 1] = f"{value:g}" if isinstance(value, float) else str(value)
    return tokens


def cell_cmd(c, model=None):
    """The full `src/train_glue.py` command for one grid cell.

    ⭐ It is built by the SAME planner every other fir stage uses (fir_plan.cell_cmd)
      and then the three swept knobs are overridden, so the module-name port, the
      dtype, the batch size and the derived warmup cannot drift away from the rest
      of the port.  port_mode='derived' is passed for form only: both values it
      sets are overridden below."""
    kw = {"model": model} if model else {}
    cmd = FP.cell_cmd(c["arm"], c["task"], c["seed"], c["targets"], "derived",
                      c["epochs"], **kw)
    cmd = _set_flag(cmd, "--learning_rate", c["lr"])
    cmd = _set_flag(cmd, "--classifier_lr", c["classifier_lr"])
    sf = FA.ARM_SCALE_FLAG[c["arm"]]
    if not sf:
        raise SystemExit(f"FAIL CLOSED: {c['arm']} has no scale flag to sweep")
    cmd = _set_flag(cmd, sf, c["scaling"])
    # the cell NAME must carry every swept knob: the results row is keyed on it.
    cmd[cmd.index("--name") + 1] = cell_id(c)
    return cmd


def cell_env(c, run_root):
    """⛔ ONE CSV PER CELL (train_glue._upsert_result's key omits seed; two cells
    on one CSV collapse into one row, silently -- scripts/r304_upsert_gate.py)."""
    e = FP.cell_env(c["arm"], c["task"], c["seed"], c["targets"], run_root)
    e["GLUE_RESULTS_FILE"] = os.path.join(run_root, "csv", cell_id(c) + ".csv")
    return e


def steps_per_cell():
    return FP.total_steps(TASK, EPOCHS)


# ---------------------------------------------------------------------------
def selftest():
    ok, bad = [], []

    def ck(c, l):
        (ok if c else bad).append(l)

    cs = cells()
    n_expect = len(ARMS) * len(LRS) * len(SCALINGS) * len(CLF_LRS)
    ck(len(cs) == n_expect,
       f"grid {GRID_NAME} is {len(ARMS)}x{len(LRS)}x{len(SCALINGS)}x{len(CLF_LRS)} = {len(cs)} cells")
    ck({"g1": 160, "g2": 140}[GRID_NAME] == len(cs), f"{GRID_NAME} has its declared cell count")
    # ⛔ THE GRIDS MUST NOT SILENTLY BECOME THE SAME GRID, and g2 exists only because
    #   g1's optimum sat on its scaling edge -- so assert the extension is real.
    ck(max(G2["scalings"]) >= 10 * max(G1["scalings"]),
       f"g2 extends scaling >=10x past g1's edge ({max(G1['scalings'])} -> {max(G2['scalings'])})")
    ck(max(G2["lrs"]) > max(G1["lrs"]) and len(G2["lrs"]) > len(G1["lrs"]),
       "g2's lr axis is both WIDER at the top and FINER than g1's")
    ck(min(G2["lrs"]) == min(G1["lrs"]),
       "g2 keeps g1's bottom lr endpoint (measured worst; going lower buys dead cells)")
    ck(set(G2["clf_lrs"]) < set(G1["clf_lrs"]),
       "g2's classifier_lr values are a strict SUBSET of g1's (all already measured)")
    # the overlap is what makes the re-run cheap: those cells keep their g1 ids
    _g1 = {cell_id(c) for c in _cells_of(G1)}
    _g2 = {cell_id(c) for c in _cells_of(G2)}
    ck(len(_g1 & _g2) == 2 * 2 * 2 * len(ARMS),
       f"g1 and g2 share exactly {len(_g1 & _g2)} cells, which resume for free")
    ids = [cell_id(c) for c in cs]
    ck(len(set(ids)) == len(ids), "every cell id is unique")
    ck(cells() == cells(), "cell order is deterministic (array index is stable)")
    ck(all(parse_cell_id(i) is not None for i in ids[:5]), "cell ids round-trip")
    try:
        parse_cell_id("mrpc-fftm-q_o-lr9p9-sc1-clr1-seed42")
        ck(False, "CONTROL: an unknown cell id is refused")
    except SystemExit:
        ck(True, "CONTROL: an unknown cell id is refused")

    # --- the reference points must be reachable, or the search cannot speak to them
    ck(5e-3 in CLF_LRS, "the carried classifier_lr 5e-3 is on the grid")
    PT = FP.port()
    dl = PT["targets"][TARGETS]["arms"]["fftm"]["derived_lr"]
    ds = PT["targets"][TARGETS]["arms"]["fftm"]["derived_scale"]
    ck(min(LRS) < dl < max(LRS), f"the derived lr* {dl:.4g} is INSIDE the swept lr range")
    if GRID_NAME == "g1":
        ck(0.5 in LRS and 50 in SCALINGS, "g1 carries RoBERTa's tuned point exactly")
        ck(min(SCALINGS) < ds < max(SCALINGS), f"the derived scale* {ds:.4g} is INSIDE g1")
    else:
        # g2 deliberately starts AT the derived scale and climbs: everything below it
        # is measured and worse, so spending cells there again would buy nothing.
        ck(abs(min(SCALINGS) - round(ds)) <= 1,
           f"g2 starts at the derived scale* ({ds:.4g}) and extends upward only")
        ck(min(LRS) < 1.5 < max(LRS), "g1's best lr (1.5) is BRACKETED by g2's finer axis")
        ck(400 in SCALINGS, "g1's best scaling (400) is retained as an anchor")

    # --- the command really carries the swept values, for BOTH arms
    for arm in ARMS:
        c = dict(cells([arm])[7])
        cmd = cell_cmd(c)
        s = " ".join(cmd)
        ck(f"--learning_rate {c['lr']:g}" in s, f"{arm}: lr reaches the command")
        ck(f"--classifier_lr {c['classifier_lr']:g}" in s, f"{arm}: classifier_lr reaches it")
        ck(f"{FA.ARM_SCALE_FLAG[arm]} {c['scaling']}" in s, f"{arm}: scaling reaches it")
        ck(s.count("--learning_rate") == 1, f"{arm}: exactly ONE --learning_rate")
        ck(s.count("--classifier_lr") == 1, f"{arm}: exactly ONE --classifier_lr")
        ck(s.count(FA.ARM_SCALE_FLAG[arm]) == 1, f"{arm}: exactly ONE scale flag")
        ck("query" not in s and "value" not in s, f"{arm}: no RoBERTa module name survives")
        ck(f"--adapter_target_modules {FP.TARGET_SETS[TARGETS]}" in s,
           f"{arm}: the generic target override is present")
        ck(f"--num_train_epochs {EPOCHS}" in s, f"{arm}: 5 epochs")
        ck("--dtype float32" in s, f"{arm}: float32 (the bf16 default is machine-dependent)")
        ck(cell_id(c) in s, f"{arm}: the cell id is the run name")

    # --- CONTROL: overriding a flag that is not there must FAIL, not append
    try:
        _set_flag(["--a", "1"], "--nope", 1); ck(False, "CONTROL: _set_flag fails closed")
    except SystemExit:
        ck(True, "CONTROL: _set_flag fails closed")

    # --- one CSV per cell, and the name carries every knob
    seen = {}
    for c in cs:
        f = cell_env(c, "/tmp/x")["GLUE_RESULTS_FILE"]
        ck(f not in seen, "no two cells share a CSV") if f in seen else None
        seen[f] = 1
    ck(len(seen) == len(cs), f"{len(cs)} cells -> {len(seen)} distinct CSVs")

    # --- warmup is derived from MRPC's own step count, not RTE's absolute 140
    w = FP.warmup_for(TASK, EPOCHS)
    ck(w == int(round(FP.WARMUP_RATIO * steps_per_cell())), "warmup derived for mrpc/5ep")
    ck(w < 140, f"CONTROL: RTE's flat 140 would over-warm this run (derived {w})")

    for l in ok:
        print(f"  ✅ {l}")
    for l in bad:
        print(f"  ⛔ {l}")
    print(f"selftest: {len(ok)} passed, {len(bad)} failed")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--cmd", metavar="CELL_ID")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--arms", default=None)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    arms = [x.strip() for x in a.arms.split(",")] if a.arms else None
    if a.list:
        for c in cells(arms):
            print(cell_id(c))
        return
    if a.cmd:
        print(" ".join(cell_cmd(parse_cell_id(a.cmd))))
        return
    if a.show:
        cs = cells(arms)
        print(f"GRID {GRID_NAME}  (set FIR_HP_GRID to switch; g1 is the completed 160-cell grid)")
        print(f"grid digest {digest()}  |  {len(cs)} cells  |  task {TASK}  "
              f"targets {TARGETS}  epochs {EPOCHS}  seed {SEED}")
        print(f"  arms          : {', '.join(arms or ARMS)}")
        print(f"  learning_rate : {LRS}")
        print(f"  scaling       : {SCALINGS}")
        print(f"  classifier_lr : {CLF_LRS}")
        print(f"  steps per cell: {steps_per_cell()}  "
              f"(mrpc train 3,668 / batch {FP.BATCH} x {EPOCHS} epochs)")
        print(f"  warmup        : {FP.warmup_for(TASK, EPOCHS)} steps (RTE's ratio, MRPC's steps)")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
