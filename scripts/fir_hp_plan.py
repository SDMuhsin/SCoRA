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
import re, argparse, hashlib, json, os, sys

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
FFT_ARMS = ["fftm", "fftstock"]
WAVE_ARMS = ["wave1", "wave2"]

# ⭐ g1 -- RUN AND COMPLETE 2026-08-26. 160/160 cells, 0 failed, 16.3 GPU-h.
#   [measured] best F1 0.8945 at lr 1.5 / scaling 400 / clf_lr 5e-4 (fftm), but
#   ⛔ SCALING WAS AT THE GRID EDGE and strictly monotone across it:
#        sc  25 -> best F1 0.8352 / acc 0.7598      (RoBERTa's tuned 50: 0.8459)
#        sc  50 -> 0.8459 / 0.7794
#        sc 142 -> 0.8767 / 0.8186                  (the DERIVED scale* 141.94)
#        sc 400 -> 0.8945 / 0.8456
#   so the optimum lies OUTSIDE g1 and its best cell is not quotable as an optimum.
G1 = {
    "arms": FFT_ARMS, "coord": "lr",
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
    "arms": FFT_ARMS, "coord": "lr",
    "lrs": [0.05, 0.15, 0.4, 1.0, 2.5, 6.0, 15.0],
    "scalings": [142, 400, 1100, 3000, 8000],
    "clf_lrs": [5e-4, 5e-3],
}

# ---------------------------------------------------------------------------
# ⭐⭐ w1 -- THE WaveFT GRID.  A DIFFERENT COORDINATE SYSTEM, on purpose.
# ---------------------------------------------------------------------------
# g1/g2 swept RAW lr.  This grid sweeps `P = lr * atom` -- the EFFECTIVE STEP on
# dW -- and DERIVES lr from it, because that is the coordinate every prior WaveFT
# search in this repo was built in and the only one whose numbers transfer across
# a width change [hp-transfer-proxy; baseline_hp_search_results.md ss0(1)].
#
#   atom(s) = s / sqrt(2mn)   for FourierFT AND WaveFT alike, INDEPENDENT of mu
#             [R.267, measured]  =>  lr = P / atom(s), and (P, s) is a shear of (lr, s).
#
# ⭐ WHY THE SHEAR IS WORTH IT.  [g1, measured] lr and scaling are NOT
#   interchangeable: at a matched product, large-scaling/small-lr beat the reverse.
#   In (lr, s) that fact has to be read off anti-diagonals; in (P, s) it is a ROW.
#   The nuisance axis and the axis with the peak are separated.
#
# THE THREE ANCHORS, all ON the grid and all derived, never typed:
#   * P/P_ref = 1   is RoBERTa's own tuned WaveFT step, carried across the width
#     change.  [R.305] selected P = 0.0828641 for BOTH mu, and at s = 75 it
#     reproduces the port table's derived_lr* = 3.2 EXACTLY (asserted in selftest).
#   * P/P_ref = 6   is the PREDICTION.  [g1+g2, measured] FourierFT's gemma optimum
#     sits at 6.000x its own RoBERTa-tuned P (0.1381 vs 0.0230).  If that inflation
#     is a property of the BACKBONE rather than of FourierFT, WaveFT's optimum is
#     here.  It is rung 4 of 6 -- dead centre, so the prediction can FAIL VISIBLY.
#   * P/P_ref = 0.5 is the floor anchor: below RoBERTa's own optimum, where the
#     measured evidence says the step is too small.
#
# ⛔ WHY THE LADDER REACHES 38x AND NOT 16x.  Two independent measurements say
#   WaveFT wants MORE step than any a-priori rule predicts, and that a WaveFT
#   optimum has run off the top of a ladder in this repo before:
#     * [R.271/R.280] on RTE the mu=1 AND mu=2 screening argmax sat at the TOP of
#       BOTH ladders -- reported as a LOWER BOUND, and the bracketing extension was
#       never run.  That is exactly the outcome this grid must not repeat.
#     * [R.271] WaveFT's PUBLISHED point is 16.6x off in P from its screening
#       argmax, and scores 0.5993/0.5921 here -- near collapse.
#   The ratio is ~2.5 (g2's lr ratio, which resolved a clean ridge), so 6 rungs
#   span 76x, with two rungs of margin above the prediction.
#
# ⛔ WHY THE SCALING AXIS IS COARSE (ratio 4) AND WIDE (64x).  At fixed P, s and lr
#   trade off exactly EXCEPT through AdamW's decoupled decay, which shrinks theta by
#   lr*wd per step and so taxes the small-s/large-lr side [R.211, g1 measured].
#   The expected s response is therefore MONOTONE-AND-SATURATING, not peaked: it
#   needs RANGE to find where it saturates, not resolution to locate a peak.
#   ⭐ And unlike FourierFT, WaveFT CANNOT be capped from above by init damage --
#     `--haar_init_std 0.0` means dW == 0 at init at EVERY scaling, so g2's
#     sc-8000 collapse (a 34% relative perturbation of the frozen weights before a
#     single step) has no analogue here.  The ceiling, if any, is optimisation.
#   75 is wave1's own RoBERTa-tuned scaling; wave2's 150 is bracketed by 75/300.
#
# ⛔ NOT SWEPT, and each for a reason that is not cost:
#   * `--haar_mu` -- FIXED A PRIORI at 1 (published) and 2 (this repo's rank fix);
#     train_glue.py:484 says DO NOT SWEEP.  The two values ARE the two arms.
#   * `--haar_init_std 0.0` -- the published method's own init.  Sweeping it would
#     make this a different method, not a tuned one.
#   * `--haar_k 256` -- budget parity is the premise of the whole comparison.
#   * `--haar_scaling` -- ABLATION ONLY (train_glue.py:487): it OVERRIDES the
#     a-priori atom-matching rule.  The swept knob is `--haar_fourierft_scaling`.
#   * epochs -- [g1, measured] 0 of the top 30 cells peaked at the last epoch.
#   * classifier_lr -- kept at g2's two survivors. [g1, measured] the axis is flat
#     across 40x (best F1 0.8823..0.8945) except that 2e-2 is harmful (8 of 9
#     at-floor cells).  The head is the SAME 4,096-param `score` layer for every
#     arm, so that measurement is arm-independent; only its interaction is not,
#     and two values price that at 2x rather than 4x.
W1 = {
    "arms": WAVE_ARMS, "coord": "p",
    "p_mults": [0.5, 1.0, 2.5, 6.0, 15.0, 38.0],
    "scalings": [75, 300, 1200, 4800],
    "clf_lrs": [5e-4, 5e-3],
}

GRIDS = {"g1": G1, "g2": G2, "w1": W1}
GRID_NAME = os.environ.get("FIR_HP_GRID", "g2")
if GRID_NAME not in GRIDS:
    raise SystemExit(f"FAIL CLOSED: FIR_HP_GRID={GRID_NAME!r} is not one of {sorted(GRIDS)}")
_G = GRIDS[GRID_NAME]
ARMS = _G["arms"]
COORD = _G["coord"]
SCALINGS, CLF_LRS = _G["scalings"], _G["clf_lrs"]
LRS = _G.get("lrs", [])            # [] on a P-parameterised grid: lr is DERIVED there
P_MULTS = _G.get("p_mults", [])    # [] on an lr-parameterised grid

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


# ---------------------------------------------------------------------------
# ⭐ THE P COORDINATE.  Every number below is READ FROM THE PORT TABLE, which was
#   emitted from the model itself and carries a digest.  Nothing here is typed:
#   a hand-copied atom is exactly the silent error this repo keeps paying for.
# ---------------------------------------------------------------------------
_PT_CACHE = {}


def _pt(PT=None):
    """The port table, read ONCE.  cells() derives an lr per cell and is called on
    every planner invocation; re-parsing the JSON 96 times is pure waste."""
    if PT is not None:
        return PT
    if "pt" not in _PT_CACHE:
        _PT_CACHE["pt"] = FP.port()
    return _PT_CACHE["pt"]


def atom_per_scale(targets=TARGETS, arm=None, PT=None):
    """atom / scale at the TARGET width, i.e. 1/sqrt(2mn).

    ⭐ FourierFT and WaveFT share atom = s/sqrt(2mn) [R.267], and for WaveFT it is
      INDEPENDENT OF mu -- which is why one (P, s) grid serves both arms and their
      rows are directly comparable.  Asserted across arms, not assumed."""
    PT = _pt(PT)
    prof = PT["targets"][targets]["arms"]
    arms = [arm] if arm else list(ARMS)
    vals = []
    for a in arms:
        pr = prof[a]
        if not pr.get("scale"):
            raise SystemExit(f"FAIL CLOSED: {a} has no scale in the port table -- "
                             f"P cannot be defined for an arm with no scale knob")
        vals.append(pr["atom_median"] / pr["scale"])
    if max(vals) - min(vals) > 1e-12 * max(vals):
        raise SystemExit(f"FAIL CLOSED: arms {arms} do NOT share atom/scale at "
                         f"{targets} ({vals}) -- one (P, scaling) grid cannot serve them")
    return vals[0]


def p_ref(arm=None, PT=None):
    """The REFERENCE effective step: the P that [R.305] selected on roberta-base/RTE.

    ⭐ Read as lr*atom from the REFERENCE half of the port table, so it is the same
      quantity `baseline_hp_search_results.md` tells you to carry.  wave1 and wave2
      selected DIFFERENT (lr, scale) pairs with IDENTICAL P -- asserted here, because
      if that ever stopped being true the two arms would need two grids."""
    PT = _pt(PT)
    ref = PT["reference"]["__ref__"]["arms"]
    arms = [arm] if arm else list(ARMS)
    vals = [ref[a]["lr"] * ref[a]["atom_median"] for a in arms]
    if max(vals) - min(vals) > 1e-12 * max(vals):
        raise SystemExit(f"FAIL CLOSED: arms {arms} have DIFFERENT reference P {vals} -- "
                         f"a shared P ladder would mean a different thing per arm")
    return vals[0]


def lr_for(p_mult, scaling, PT=None):
    """lr = P / atom(scaling), the whole point of the coordinate."""
    return p_mult * p_ref(PT=PT) / (scaling * atom_per_scale(PT=PT))


def axes():
    """The grid's TESTABLE axes as (label, cell key, values).

    ⛔ The reader's edge report used to name lr/scaling/classifier_lr literally.
      On a P-parameterised grid `lr` is not an axis at all -- it takes 24 distinct
      values, one per (P, scaling) pair -- so an edge test on it would be
      meaningless in exactly the way this repo's checks keep failing.  The grid
      declares its own axes; the reader asks."""
    if COORD == "p":
        first = ("P/P_ref", "p_mult", P_MULTS)
    else:
        first = ("lr", "lr", LRS)
    return [first, ("scaling", "scaling", SCALINGS),
            ("classifier_lr", "classifier_lr", CLF_LRS)]


def canary_indices(ids=None):
    """One CENTRAL cell per arm, as indices into cells().

    ⛔ DERIVED FROM THE GRID, NEVER HARDCODED -- the first version named `lr 0.5 /
      scaling 142` literally and would have died at submit time the day the grid was
      replaced.  ⚠ And CENTRAL on every axis: the version after that took the MAX
      scaling, so on a grid whose top scaling collapses the model the canary would
      be a dead cell -- fine for wall-clock, useless as a smoke test."""
    ids = ids or [cell_id(c) for c in cells()]
    def mid(v):
        return sorted(set(v))[(len(set(v)) - 1) // 2]
    out = []
    for arm in ARMS:
        want = [c for c in cells([arm])
                if c["scaling"] == mid(SCALINGS) and c["classifier_lr"] == mid(CLF_LRS)
                and (c["p_mult"] if COORD == "p" else c["lr"])
                    == mid(P_MULTS if COORD == "p" else LRS)]
        if len(want) != 1:
            raise SystemExit(f"FAIL CLOSED: {len(want)} central cells for {arm}, expected 1")
        out.append(ids.index(cell_id(want[0])))
    return out


def _cells_of(grid, arms=None, PT=None):
    """Enumerate a grid dict directly -- used by the selftest to compare grids
    without mutating module globals (which would make the test order-dependent).

    ⭐ The OUTER loop is the first axis in both coordinate systems, so the cell
      ORDER (and therefore every Slurm array index) is built the same way whether
      lr is swept or derived."""
    out = []
    first = grid.get("p_mults") if grid["coord"] == "p" else grid["lrs"]
    for arm in (arms or grid["arms"]):
        for v in first:
            for sc in grid["scalings"]:
                for clr in grid["clf_lrs"]:
                    c = {"arm": arm, "task": TASK, "targets": TARGETS,
                         "seed": SEED, "epochs": EPOCHS,
                         "scaling": sc, "classifier_lr": clr}
                    if grid["coord"] == "p":
                        c["p_mult"] = v
                        c["lr"] = lr_for(v, sc, PT=PT)
                    else:
                        c["lr"] = v
                    out.append(c)
    return out


def cells(arms=None, PT=None):
    """Every cell of the SELECTED grid, in a DETERMINISTIC order -- the array index
    is this order, so it must not depend on a set, a dict iteration or the
    filesystem."""
    return _cells_of(_G, arms, PT=PT)


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
    _first = P_MULTS if COORD == "p" else LRS
    n_expect = len(ARMS) * len(_first) * len(SCALINGS) * len(CLF_LRS)
    ck(len(cs) == n_expect,
       f"grid {GRID_NAME} is {len(ARMS)}x{len(_first)}x{len(SCALINGS)}x{len(CLF_LRS)} = {len(cs)} cells")
    ck({"g1": 160, "g2": 140, "w1": 96}[GRID_NAME] == len(cs),
       f"{GRID_NAME} has its declared cell count")
    ck(len(axes()) == 3 and all(len(a[2]) > 0 for a in axes()),
       f"{GRID_NAME} declares 3 non-empty axes: {[a[0] for a in axes()]}")
    ck(all(a[1] in cs[0] for a in axes()),
       "every declared axis key exists on a cell (the reader indexes cells by it)")
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
    ck(len(_g1 & _g2) == 2 * 2 * 2 * len(G1["arms"]),
       f"g1 and g2 share exactly {len(_g1 & _g2)} cells, which resume for free")
    ids = [cell_id(c) for c in cs]
    ck(len(set(ids)) == len(ids), "every cell id is unique")
    # --- the canary: one cell per arm, CENTRAL on every axis, derived from the grid
    ci = canary_indices(ids)
    ck(len(ci) == len(ARMS) and len(set(ci)) == len(ci),
       f"the canary is {len(ARMS)} distinct cells, one per arm")
    ck([parse_cell_id(ids[i])["arm"] for i in ci] == list(ARMS),
       "...one per ARM, in arm order (the stock-PEFT / second code path is covered)")
    for i in ci:
        cc = parse_cell_id(ids[i])
        for name, key, axis in axes():
            vs = sorted(set(axis))
            ck(len(vs) < 3 or cc[key] not in (vs[0], vs[-1]),
               f"CONTROL: the canary is not at an extreme of {name} "
               f"(a dead corner measures wall-clock but smokes nothing)")
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
    if COORD == "lr":
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
    else:
        # ------------------------------------------------------------------
        # w1: the P coordinate.  ⛔ EVERY ANCHOR IS DERIVED FROM THE PORT TABLE.
        #   A grid whose anchors are typed numbers is a grid that silently stops
        #   pointing at the thing it claims to point at.
        # ------------------------------------------------------------------
        ref = PT["reference"]["__ref__"]["arms"]
        tgt = PT["targets"][TARGETS]["arms"]
        ck(abs(ref["wave1"]["lr"] * ref["wave1"]["atom_median"]
               - ref["wave2"]["lr"] * ref["wave2"]["atom_median"]) < 1e-12,
           "wave1 and wave2 selected the SAME reference P -- one ladder serves both")
        ck(abs(tgt["wave1"]["atom_median"] / tgt["wave1"]["scale"]
               - tgt["wave2"]["atom_median"] / tgt["wave2"]["scale"]) < 1e-15,
           "the two arms share atom/scale at the target width (atom is mu-INDEPENDENT)")
        ck(1.0 in P_MULTS, "P/P_ref = 1 (RoBERTa's own tuned step) is ON the ladder")
        ck(6.0 in P_MULTS,
           "P/P_ref = 6 (FourierFT's MEASURED gemma inflation) is ON the ladder")
        ck(min(P_MULTS) < 6.0 < max(P_MULTS),
           "...and it is INTERIOR, so the prediction can be falsified by this grid")
        # ⭐ the coordinate itself is checked against the port's own derived lr*:
        #   at the reference scale, P/P_ref = 1 MUST reproduce derived_lr exactly.
        for a in ARMS:
            sc = tgt[a]["scale"]
            ck(abs(lr_for(1.0, sc) - tgt[a]["derived_lr"]) < 1e-9,
               f"{a}: P/P_ref=1 at scale {sc:g} reproduces the port's lr* "
               f"{tgt[a]['derived_lr']:.6g}")
        ck(int(ref["wave1"]["scale"]) in SCALINGS,
           "wave1's own RoBERTa-tuned scaling (75) is on the scaling axis")
        ck(min(SCALINGS) < ref["wave2"]["scale"] < max(SCALINGS),
           "wave2's RoBERTa-tuned scaling (150) is BRACKETED by the axis")
        # ⛔ THE FAILURE THIS GRID EXISTS TO NOT REPEAT: [R.271]/[R.280] left BOTH
        #   WaveFT arms at the TOP of BOTH RTE ladders, as lower bounds. Require
        #   real margin above the prediction, not one token rung.
        ck(max(P_MULTS) / 6.0 >= 5.0,
           f"the P ladder runs >=5x PAST the prediction (to {max(P_MULTS):g}x) -- "
           f"[R.271] ran off the top of its ladder and was never bracketed")
        ck(max(SCALINGS) / max(ref[a]["scale"] for a in ARMS) >= 30,
           "the scaling axis runs >=30x past the RoBERTa-tuned scale")
        ck(min(P_MULTS) < 1.0, "there is a rung BELOW RoBERTa's step (the floor anchor)")
        # ⛔ a P grid must not silently become an lr grid
        ck(all("p_mult" in c for c in cs), "every w1 cell carries its P multiplier")
        ck(len({c["lr"] for c in cs}) > len(P_MULTS),
           "CONTROL: lr is DERIVED per (P, scaling), not a swept axis")
        # ⛔ NOT-SWEPT knobs must be absent from the id and constant in the command
        one = " ".join(cell_cmd(cs[0]))
        ck("--haar_mu" in one and "--haar_init_std 0.0" in one,
           "mu and the zero init reach the command (fixed a priori, never swept)")
        ck(" --haar_scaling " not in one,
           "CONTROL: --haar_scaling (ABLATION ONLY) is NOT set -- the swept knob is "
           "--haar_fourierft_scaling")
        def _mu(c):
            t = cell_cmd(c)
            return t[t.index("--haar_mu") + 1]
        ck({_mu(c) for c in cells(["wave1"])} == {"1"},
           "wave1 is mu=1 in EVERY cell (the published method)")
        ck({_mu(c) for c in cells(["wave2"])} == {"2"},
           "wave2 is mu=2 in EVERY cell (this repo's rank fix)")

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



def _selftest_every_grid():
    """⭐ RUN THE CHECKS FOR EVERY GRID, NOT JUST THE SELECTED ONE.

    ⛔ Module globals are bound to FIR_HP_GRID at IMPORT, so one process can only
      ever check one grid -- and `run_all_gates.py` sets no env var, so for two
      grids the suite was green while a *different* grid's checks had never run in
      it. That is this repo's Law 1 in miniature: a check must exercise what the
      job will actually run. Re-exec once per grid and aggregate.
    """
    import subprocess
    if os.environ.get("FIR_HP_GRID"):
        print(f"  ⚠ FIR_HP_GRID={os.environ['FIR_HP_GRID']} is set; checking ALL grids anyway.")
    tot_p = tot_f = 0
    for g in sorted(GRIDS):
        print(f"--- grid {g} " + "-" * 50)
        env = dict(os.environ, FIR_HP_GRID=g, FIR_HP_ONE_GRID="1")
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--selftest"],
                           capture_output=True, text=True, env=env,
                           cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        m = None
        for m in re.finditer(r"selftest:\s*(\d+) passed, (\d+) failed", r.stdout):
            pass
        if m is None:
            print(f"  ⛔ grid {g} produced no selftest line (rc={r.returncode})")
            tot_f += 1
            continue
        tot_p += int(m.group(1)); tot_f += int(m.group(2))
        if r.returncode != 0 and int(m.group(2)) == 0:
            tot_f += 1     # ⛔ fail closed: a crash after a green line is still a failure
    print("=" * 62)
    print(f"selftest: {tot_p} passed, {tot_f} failed  (all {len(GRIDS)} grids)")
    return 1 if tot_f else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--cmd", metavar="CELL_ID")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--arms", default=None)
    a = ap.parse_args()
    if a.selftest:
        # ⛔ EVERY GRID, ALWAYS -- deliberately IGNORING FIR_HP_GRID. The first
        #   version skipped the fan-out when that var was set, so an operator who
        #   had exported a grid for a submit got a suite that silently covered ONE
        #   grid instead of three, with a smaller green total and no warning. A
        #   check must not cover LESS because of an unrelated environment variable.
        if not os.environ.get("FIR_HP_ONE_GRID"):
            sys.exit(_selftest_every_grid())
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
        print(f"GRID {GRID_NAME}  (set FIR_HP_GRID to switch; "
              f"known: {', '.join(sorted(GRIDS))})")
        print(f"grid digest {digest()}  |  {len(cs)} cells  |  task {TASK}  "
              f"targets {TARGETS}  epochs {EPOCHS}  seed {SEED}")
        print(f"  arms          : {', '.join(arms or ARMS)}")
        if COORD == "p":
            # ⭐ Print the DERIVED lr for every cell of the plane. The swept knob is
            #   P; lr is what actually reaches the command line, and a reader who
            #   cannot see it cannot sanity-check the corners.
            pr, aps = p_ref(), atom_per_scale()
            print(f"  coordinate    : P = lr*atom  (atom = scaling/{1/aps:.4f} at {TARGETS})")
            print(f"  P/P_ref       : {P_MULTS}      P_ref = {pr:.7f}  "
                  f"[R.305]'s selected step, IDENTICAL for both mu")
            print(f"  scaling       : {SCALINGS}")
            print(f"  classifier_lr : {CLF_LRS}")
            print(f"  derived lr    :  {'P/P_ref':>9s}" +
                  "".join(f"{('sc'+str(x)):>10s}" for x in SCALINGS))
            for m in P_MULTS:
                mark = "  <- RoBERTa" if m == 1.0 else ("  <- prediction" if m == 6.0 else "")
                print(f"                 {m:>9g}" +
                      "".join(f"{lr_for(m, x):>10g}" for x in SCALINGS) + mark)
        else:
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
