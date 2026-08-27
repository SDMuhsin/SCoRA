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

# ---------------------------------------------------------------------------
# ⭐⭐ w2 -- THE BUDGET-EQUALISATION GRID.  [user, 2026-08-27: "increase budget so
#   they're equal"]
# ---------------------------------------------------------------------------
# ⛔ THE PROBLEM IT FIXES, STATED AS A NUMBER.  [measured] FourierFT was searched
#   over 142 DISTINCT cells per arm on this cell (g1's 80 + g2's 70, 8 shared);
#   WaveFT over 48.  A 2.96x search advantage, and it runs in FourierFT's favour.
#   `[Dodge et al., EMNLP 2019 §6]` is explicit that the direction matters: "if a
#   model with a small budget outperforms a model with a large budget, increasing
#   the small budget will not change this conclusion.  However, if a model with a
#   large budget outperforms a model with a small budget, the difference might be
#   due to the model or the budget (or both)."  Ours is the SECOND case, so the
#   0.8904-vs-0.8873 gap cannot be attributed at all until the budgets match.
#
# ⚠ AND EQUAL COUNTS ARE STILL NOT SUFFICIENT -- the same section says "fixing the
#   same number of hyperparameter trials for both models does not imply a fair
#   comparison", because the spaces differ and past human effort is unmeasurable.
#   w1's bounds were themselves chosen USING g1/g2's results, which is borrowed
#   effort in WaveFT's favour and cannot be netted off.  This grid removes the one
#   asymmetry that IS countable; it does not make the comparison clean.
#
# ⭐ WHERE THE CELLS GO, AND WHY NOT SOMEWHERE EASIER.  The honest way to spend an
#   equalising budget is the way the other family's was spent, not wherever it
#   most helps.  FourierFT's 142 = a broad plane x FOUR classifier_lr values (g1)
#   + a finer, wider plane x two (g2).  WaveFT's 48 has only ever had TWO
#   classifier_lr values, so part of the gap is a knob axis it never received.
#   w2 therefore restores g1's full four-value classifier_lr axis and puts it on a
#   plane that INTERLEAVES w1's, doubling the resolution of both swept axes:
#     P/P_ref  w1 {0.5, 1, 2.5, 6, 15, 38} + w2 {0.3, 0.7, 1.6, 4, 10, 24}
#              => a union ladder of ratio ~1.55 spanning 0.3 - 38 (127x)
#     scaling  w1 {75, 300, 1200, 4800}    + w2 {150, 600, 2400, 9600}
#              => a union ladder of EXACTLY ratio 2 spanning 75 - 9600 (128x)
#   ⛔ The two P axes are DISJOINT by construction, so every w2 cell is new and the
#     budget really does rise by 96/arm rather than resuming w1 markers for free.
#   ⚠ I am not pretending this buys a better optimum. At one seed, cells 1.55x
#     apart in P differ by less than the seed noise `[R.273]`, so most of what a
#     denser search buys is SELECTION INFLATION -- which is exactly the thing
#     FourierFT's extra 94 cells bought it, and exactly what equalising removes.
#
# 6 x 4 x 4 = 96 new cells per arm => 144/arm total, against FourierFT's 142.
# ⚠ It overshoots by 2. That is the CONSERVATIVE direction for the standing
#   result: if FourierFT still leads while now holding the SMALLER budget, Dodge's
#   asymmetry applies and the conclusion is safe; the reverse would have been
#   unattributable. A selftest asserts the wave family ends >= the fft family.
W2 = {
    "arms": WAVE_ARMS, "coord": "p",
    "p_mults": [0.3, 0.7, 1.6, 4.0, 10.0, 24.0],
    "scalings": [150, 600, 2400, 9600],
    "clf_lrs": [5e-4, 2e-3, 5e-3, 2e-2],          # g1's axis, restored in full
}

# ⭐ A READING VIEW, NOT A RUN TARGET. `wave` is the UNION of w1 and w2 -- the whole
#   144-cell-per-arm WaveFT search, which is the thing a budget-equalised claim is
#   about. ⛔ It is a union of two disjoint factorial BLOCKS, not one factorial, so
#   the reader tests edges per block and says so; a bare min/max over the union
#   axes would claim a bracketing the design does not have.
WAVE_ALL = {"arms": WAVE_ARMS, "coord": "p", "union": ["w1", "w2"]}

GRIDS = {"g1": G1, "g2": G2, "w1": W1, "w2": W2, "wave": WAVE_ALL}
GRID_NAME = os.environ.get("FIR_HP_GRID", "g2")
if GRID_NAME not in GRIDS:
    raise SystemExit(f"FAIL CLOSED: FIR_HP_GRID={GRID_NAME!r} is not one of {sorted(GRIDS)}")
_G = GRIDS[GRID_NAME]
ARMS = _G["arms"]
COORD = _G["coord"]
IS_UNION = "union" in _G
SCALINGS, CLF_LRS = _G.get("scalings", []), _G.get("clf_lrs", [])
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


def atom_per_scale(targets=TARGETS, arm=None, PT=None, arms=None):
    """atom / scale at the TARGET width, i.e. 1/sqrt(2mn).

    ⭐ FourierFT and WaveFT share atom = s/sqrt(2mn) [R.267], and for WaveFT it is
      INDEPENDENT OF mu -- which is why one (P, s) grid serves both arms and their
      rows are directly comparable.  Asserted across arms, not assumed."""
    PT = _pt(PT)
    prof = PT["targets"][targets]["arms"]
    arms = [arm] if arm else list(arms or ARMS)
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


def p_ref(arm=None, PT=None, arms=None):
    """The REFERENCE effective step: the P that [R.305] selected on roberta-base/RTE.

    ⭐ Read as lr*atom from the REFERENCE half of the port table, so it is the same
      quantity `baseline_hp_search_results.md` tells you to carry.  wave1 and wave2
      selected DIFFERENT (lr, scale) pairs with IDENTICAL P -- asserted here, because
      if that ever stopped being true the two arms would need two grids."""
    PT = _pt(PT)
    ref = PT["reference"]["__ref__"]["arms"]
    # ⛔ ARMS MUST BE PASSED WHEN ENUMERATING A GRID THAT IS NOT THE SELECTED ONE.
    #   P_ref is a FAMILY quantity -- 0.0828641 for WaveFT, 0.0230178 for FourierFT.
    #   Falling back to the module ARMS would silently derive w1/w2's learning rates
    #   from FourierFT's reference step whenever FIR_HP_GRID happened to be g2.
    arms = [arm] if arm else list(arms or ARMS)
    vals = [ref[a]["lr"] * ref[a]["atom_median"] for a in arms]
    if max(vals) - min(vals) > 1e-12 * max(vals):
        raise SystemExit(f"FAIL CLOSED: arms {arms} have DIFFERENT reference P {vals} -- "
                         f"a shared P ladder would mean a different thing per arm")
    return vals[0]


def lr_for(p_mult, scaling, PT=None, arms=None):
    """lr = P / atom(scaling), the whole point of the coordinate."""
    return (p_mult * p_ref(PT=PT, arms=arms)
            / (scaling * atom_per_scale(PT=PT, arms=arms)))


def axes_of(grid):
    """A GRID DICT's testable axes.  ⛔ Takes the dict, not the globals: a union
    view has to ask its members, and the equalisation gate has to enumerate every
    grid while a different one is selected."""
    if "union" in grid:
        raise SystemExit("FAIL CLOSED: a union view has no single axis set -- "
                         "ask its member grids (member_grids())")
    first = (("P/P_ref", "p_mult", grid["p_mults"]) if grid["coord"] == "p"
             else ("lr", "lr", grid["lrs"]))
    return [first, ("scaling", "scaling", grid["scalings"]),
            ("classifier_lr", "classifier_lr", grid["clf_lrs"])]


def member_grids():
    """[(name, grid dict)] -- itself for a plain grid, its members for a union."""
    if IS_UNION:
        return [(n, GRIDS[n]) for n in _G["union"]]
    return [(GRID_NAME, _G)]


def axes():
    """The SELECTED grid's testable axes as (label, cell key, values).

    ⛔ The reader's edge report used to name lr/scaling/classifier_lr literally.
      On a P-parameterised grid `lr` is not an axis at all -- it takes 24 distinct
      values, one per (P, scaling) pair -- so an edge test on it would be
      meaningless in exactly the way this repo's checks keep failing.  The grid
      declares its own axes; the reader asks."""
    return axes_of(_G)


def canary_indices(ids=None):
    """One CENTRAL cell per arm, as indices into cells().

    ⛔ DERIVED FROM THE GRID, NEVER HARDCODED -- the first version named `lr 0.5 /
      scaling 142` literally and would have died at submit time the day the grid was
      replaced.  ⚠ And CENTRAL on every axis: the version after that took the MAX
      scaling, so on a grid whose top scaling collapses the model the canary would
      be a dead cell -- fine for wall-clock, useless as a smoke test."""
    if IS_UNION:
        raise SystemExit("FAIL CLOSED: 'wave' is a READING VIEW, not a run target -- "
                         "canary and submit against w1 or w2")
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
    if "union" in grid:
        # ⛔ DEDUPE BY CELL ID, DETERMINISTICALLY. Members are disjoint by design
        #   (asserted in the selftest), but a union that silently double-counted a
        #   shared cell would inflate the very budget number this view exists to
        #   report.
        out, seen = [], set()
        for name in grid["union"]:
            for c in _cells_of(GRIDS[name], arms, PT=PT):
                i = cell_id(c)
                if i not in seen:
                    seen.add(i); out.append(c)
        return out
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
                        c["lr"] = lr_for(v, sc, PT=PT, arms=grid["arms"])
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

def budget_per_arm():
    """{arm: number of DISTINCT cells that arm has been / will be searched over,
    across EVERY grid this planner knows}.

    ⭐ THE FAIRNESS CLAIM, MADE COMPUTABLE. "Both families got the same tuning
      effort" is the kind of sentence that rots silently the moment a grid is
      added or trimmed. It is checked here instead of asserted in prose.
    ⛔ Union VIEWS are skipped: they re-enumerate their members, so counting them
      would double nothing but would make the number depend on how many views
      happen to exist."""
    out = {}
    for name, g in GRIDS.items():
        if "union" in g:
            continue
        for c in _cells_of(g):
            out.setdefault(c["arm"], set()).add(cell_id(c))
    return {a: len(v) for a, v in out.items()}


def selftest():
    ok, bad = [], []

    def ck(c, l):
        (ok if c else bad).append(l)

    cs = cells()
    _first = P_MULTS if COORD == "p" else LRS
    if IS_UNION:
        n_expect = sum(len(_cells_of(g)) for _n, g in member_grids())
        ck(len(cs) == n_expect,
           f"union {GRID_NAME} is the sum of its blocks = {len(cs)} cells")
    else:
        n_expect = len(ARMS) * len(_first) * len(SCALINGS) * len(CLF_LRS)
        ck(len(cs) == n_expect,
           f"grid {GRID_NAME} is {len(ARMS)}x{len(_first)}x{len(SCALINGS)}"
           f"x{len(CLF_LRS)} = {len(cs)} cells")
    ck({"g1": 160, "g2": 140, "w1": 96, "w2": 192, "wave": 288}[GRID_NAME] == len(cs),
       f"{GRID_NAME} has its declared cell count")
    if not IS_UNION:
        ck(len(axes()) == 3 and all(len(a[2]) > 0 for a in axes()),
           f"{GRID_NAME} declares 3 non-empty axes: {[a[0] for a in axes()]}")
        ck(all(a[1] in cs[0] for a in axes()),
           "every declared axis key exists on a cell (the reader indexes cells by it)")
    else:
        # ⛔ A UNION IS A READING VIEW. Prove it cannot be mistaken for a run target,
        #   in both directions: it refuses a canary, and its members do not.
        try:
            canary_indices(); ck(False, "CONTROL: a union refuses to pick a canary")
        except SystemExit:
            ck(True, "CONTROL: a union refuses to pick a canary (it is not a run target)")
        ck(all(len(_cells_of(g)) > 0 for _n, g in member_grids()),
           "...and every member block is itself enumerable")
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

    # ------------------------------------------------------------------
    # ⭐⭐ THE BUDGET-EQUALISATION GATE. Runs under EVERY grid, because it is a
    #   statement about the whole planner, not about the one that is selected.
    #   `[Dodge et al., EMNLP 2019 §6]`: an unequal search budget makes a
    #   large-budget win unattributable. This asserts the asymmetry is gone.
    # ------------------------------------------------------------------
    B = budget_per_arm()
    fft = [B[a] for a in FFT_ARMS]
    wav = [B[a] for a in WAVE_ARMS]
    ck(len(set(fft)) == 1 and len(set(wav)) == 1,
       f"each family's arms are searched equally ({FFT_ARMS}={fft}, {WAVE_ARMS}={wav})")
    ck(min(wav) >= max(fft),
       f"⭐ WaveFT's budget {min(wav)}/arm is >= FourierFT's {max(fft)}/arm "
       f"-- the countable asymmetry is removed (and overshooting is the "
       f"CONSERVATIVE direction for the standing result)")
    ck(min(wav) <= max(fft) * 1.1,
       f"...and it does not OVERSHOOT materially ({min(wav)} vs {max(fft)}, "
       f"{min(wav)/max(fft):.3f}x) -- a budget advantage is the same defect mirrored")
    # ⛔ w2 must be all-new cells, or the budget does not actually rise.
    _w1 = {cell_id(c) for c in _cells_of(W1)}
    _w2 = {cell_id(c) for c in _cells_of(W2)}
    ck(not (_w1 & _w2),
       f"w1 and w2 are DISJOINT -- all {len(_w2)} w2 cells are new budget, none resume")
    ck(not (set(W1["p_mults"]) & set(W2["p_mults"])),
       "...enforced on the P axis itself, not just on whole cells")
    ck(set(W1["clf_lrs"]) < set(W2["clf_lrs"]) == set(G1["clf_lrs"]),
       "w2 restores g1's FULL four-value classifier_lr axis, which w1 never had")
    _u = sorted({c["scaling"] for c in _cells_of(WAVE_ALL)})
    _r = {round(_u[i+1] / _u[i], 6) for i in range(len(_u) - 1)}
    ck(_r == {2.0}, f"the union scaling ladder is uniform ratio 2 ({_u})")
    ck(len(_cells_of(WAVE_ALL)) == len(_w1) + len(_w2),
       "the union view enumerates every member cell exactly once")
    try:
        axes_of(WAVE_ALL); ck(False, "CONTROL: a union refuses a single axis set")
    except SystemExit:
        ck(True, "CONTROL: a union refuses a single axis set (it is two blocks)")
    # --- the canary: one cell per arm, CENTRAL on every axis, derived from the grid
    #   (⛔ a union view has none, by design -- that control fires above)
    ci = [] if IS_UNION else canary_indices(ids)
    if IS_UNION:
        ck(True, "the union view has no canary (checked above); skipping canary asserts")
    ck(IS_UNION or (len(ci) == len(ARMS) and len(set(ci)) == len(ci)),
       f"the canary is {len(ARMS)} distinct cells, one per arm")
    ck(IS_UNION or [parse_cell_id(ids[i])["arm"] for i in ci] == list(ARMS),
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
    ck(5e-3 in (CLF_LRS or {c["classifier_lr"] for c in cs}),
       "the carried classifier_lr 5e-3 is on the grid")
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
        # ⛔ THE ANCHORS ARE A PROPERTY OF THE WHOLE WaveFT SEARCH, NOT OF ONE BLOCK.
        #   w2 is an INTERLEAVE and deliberately shares no value with w1, so asserting
        #   "P=1 is on the ladder" against w2 alone fails for the very reason w2 is
        #   correct. Check the union, whichever wave grid is selected.
        UP = sorted({c["p_mult"] for c in _cells_of(WAVE_ALL)})
        US = sorted({c["scaling"] for c in _cells_of(WAVE_ALL)})
        ck(1.0 in UP, "P/P_ref = 1 (RoBERTa's own tuned step) is ON the WaveFT ladder")
        ck(6.0 in UP,
           "P/P_ref = 6 (FourierFT's MEASURED gemma inflation) is ON the ladder")
        ck(min(UP) < 6.0 < max(UP),
           "...and it is INTERIOR, so the prediction can be falsified by this search")
        # ⭐ the coordinate itself is checked against the port's own derived lr*:
        #   at the reference scale, P/P_ref = 1 MUST reproduce derived_lr exactly.
        for a in ARMS:
            sc = tgt[a]["scale"]
            ck(abs(lr_for(1.0, sc) - tgt[a]["derived_lr"]) < 1e-9,
               f"{a}: P/P_ref=1 at scale {sc:g} reproduces the port's lr* "
               f"{tgt[a]['derived_lr']:.6g}")
        ck(int(ref["wave1"]["scale"]) in US,
           "wave1's own RoBERTa-tuned scaling (75) is on the scaling ladder")
        ck(int(ref["wave2"]["scale"]) in US,
           "wave2's own RoBERTa-tuned scaling (150) is too (w2 added it)")
        # ⛔ THE FAILURE THIS GRID EXISTS TO NOT REPEAT: [R.271]/[R.280] left BOTH
        #   WaveFT arms at the TOP of BOTH RTE ladders, as lower bounds. Require
        #   real margin above the prediction, not one token rung.
        ck(max(UP) / 6.0 >= 5.0,
           f"the P ladder runs >=5x PAST the prediction (to {max(UP):g}x) -- "
           f"[R.271] ran off the top of its ladder and was never bracketed")
        ck(max(US) / max(ref[a]["scale"] for a in ARMS) >= 30,
           "the scaling ladder runs >=30x past the RoBERTa-tuned scale")
        ck(min(UP) < 1.0, "there is a rung BELOW RoBERTa's step (the floor anchor)")
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
        if IS_UNION:
            print(f"  ⚠ UNION VIEW of {', '.join(_G['union'])} -- a reading view, NOT a run "
                  f"target. Two disjoint factorial BLOCKS, not one factorial.")
            for n, g in member_grids():
                print(f"    {n}: " + "  ".join(f"{lab}={v}" for lab, _k, v in axes_of(g))
                      + f"   ({len(_cells_of(g))} cells)")
            for lab, key, _ in axes_of(GRIDS[_G["union"][0]]):
                u = sorted({c[key] for c in cs})
                r = [u[i+1]/u[i] for i in range(len(u)-1)]
                print(f"  union {lab:14s}: {u}"
                      + (f"   ratio {min(r):.2f}-{max(r):.2f}, span {u[-1]/u[0]:.0f}x" if r else ""))
            per = len(cs) // max(1, len(arms or ARMS))
            print(f"  ⭐ budget: {len(cs)} cells = {per} per arm")
            print(f"  steps per cell: {steps_per_cell()}  "
                  f"(mrpc train 3,668 / batch {FP.BATCH} x {EPOCHS} epochs)")
            print(f"  warmup        : {FP.warmup_for(TASK, EPOCHS)} steps (RTE's ratio, MRPC's steps)")
            return
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
