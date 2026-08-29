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

# ---------------------------------------------------------------------------
# ⭐⭐ THE FIVE REMAINING ARMS.  [user, 2026-08-28: "set up the hyperparameter
#   search ranges for the remaining frequency-domain LoRA methods ... the deciding
#   criteria should be that the optimal paper-reported values should fall in range
#   and that the search will be defensible in a camera-ready paper."]
# ---------------------------------------------------------------------------
# ⛔ THE ADMISSIBILITY RULE, WHICH IS THE USER'S STATED CRITERION AND `[R.258]`'s:
#   an arm's PUBLISHED operating point must be INSIDE the swept ladders, or the
#   search can only find the best point in a box the author never used.  Every
#   ladder below is sized by that rule FIRST and by cost second.  The published
#   points, from `[N.1 §0.1]` and re-checked against `src/train_glue.py`'s own
#   argparse help (which cites the papers):
#
#     LoCA   alpha = 1.0, coefficient lr 5e-4..1e-2 per task (primary 5e-3),
#            location lr 1e-4                    [train_glue.py:502,505]
#     LYRA   gamma = 1, lr 2e-2, freq_exponent 3.0            [R.258 §1]
#     QWHA   ⛔ NO published RoBERTa/GLUE point exists -- the paper tunes on a
#            QUANTISED LLaMA                     [train_glue.py:494; R.236 §3.5]
#     SCoRA  ours; there is nothing published to be inside of.
#
# ⭐ WHERE THE LADDER WIDTHS COME FROM -- and this is BORROWED EFFORT, declared.
#   `[Dodge et al., EMNLP 2019 §6]` says the bounds a searcher chooses carry
#   unmeasurable prior human effort.  Ours are not guesses: they are read off the
#   282 FourierFT and 288 WaveFT cells ALREADY MEASURED on this exact cell
#   (mrpc / q_o / gemma-2b / 5 epochs / seed 42).  Marginals over those grids:
#
#     P/P_ref     0.3   0.7   1.0   1.6   2.5   4     6    10    15    24    38
#     wave best  .8652 .8881 .8873 .8904 .8873 .8896 .8808 .8881 .8711 .8662 .8503
#     wave floor  0/32  0/32  0/16  0/32  0/16  6/32  2/16  8/32  6/16 11/32  5/16
#
#     scaling      75   150   300   600  1200  2400  4800  9600
#     wave floor 0/24  0/48  0/24  0/48  3/24 11/48 10/24 14/48
#
#   ⇒ [measured] the LIVE window on this backbone is P/P_ref ~ 0.3..10 (FourierFT's
#     own optimum sits at 6, WaveFT's at 1.6) and scale ~ 1..16x the arm's own
#     RoBERTa-tuned scale.  Every ladder below spans AT LEAST that window with a
#     rung of margin on each side, and is widened further wherever a published
#     point falls outside it (LoCA reaches DOWN to 0.025x for exactly that reason).
#   ⚠ This is prior effort spent in the new arms' favour and it CANNOT be netted
#     off against anything.  Say it in the paper; do not present these bounds as
#     a priori.
#
# ⛔ WHAT IS NOT SWEPT, each for a reason that is not cost:
#   * `--loca_location_lr`      published 1e-4; BOTH [R.305] OFAT probes of it lost.
#   * `--loca_learn_location_iter` UNSET = the paper's own 10%-of-steps rule -- ⭐ the
#       one setting here that AUTO-ADAPTS to a new task; [R.305]'s fixed-468 probe
#       lost 0.0253.  Pinning it would be porting a RoBERTa/RTE step count.
#   * `--qwha_init_weights 0`   the authors' default (and FourierFT's init).
#   * `--spectral_p/q 16`       THE BUDGET (p*q = 256/module).  Sweeping it would
#       break budget parity, which is the premise of the whole comparison [R.233:
#       "LYRA names a family, not a budget" -- p=q=128 is a 64x arm].
#   * `--spectral_d_initial 0.07`, `--spectral_freq_mode geometric`  the banked arm.
#   * `--slr_rank 1`, `--slr_s 128`, `--slr_init zero`   r*(s+t) = 256 = the budget.
#   * `k` = 256 everywhere, 5 epochs, MRPC, q_o, seed 42 -- as for g1/g2/w1/w2.

# ⭐ THE THREE LADDER SHAPES, named so the selftest can assert they are shared.
#   A "standard" P ladder: ratio 2.5, 6 rungs, geometric about P/P_ref = 1, i.e.
#   0.16 .. 15.6 (98x).  It brackets the measured 0.3..10 window with a rung of
#   margin at each end and puts RoBERTa's own carried step ON the ladder.
P_LADDER_25 = [0.16, 0.4, 1.0, 2.5, 6.25, 15.625]

# ---------------------------------------------------------------------------
# LoCA.  ⛔ THE ONE ARM WHOSE PUBLISHED POINT FORCES A WIDER, COARSER LADDER.
# ---------------------------------------------------------------------------
# atom == alpha exactly [measured, port table: atom_median 0.2500002 at alpha 0.25],
# and it is WIDTH-INDEPENDENT, so the published (alpha 1, lr 5e-4..1e-2) is a
# published P of 5e-4..1e-2 = 0.033x..0.667x of P_ref (0.015).  That is BELOW the
# 0.3x floor of the measured window, so the ladder must reach down to it -- and it
# must still reach the 6x-10x top where FourierFT's own gemma optimum sits.
# ⇒ 480x of span in 6 rungs ⇒ ratio ~3.5.
# ⚠ SAY THIS PLAINLY RATHER THAN HIDING IT: LoCA's P axis is the COARSEST of the
#   five (3.5 vs 2.5), and the resolution was spent to buy the width its own
#   published range demands.  [R.259]: report per-axis resolution, not cell count.
# ⭐ At alpha = 1 (the published alpha) the P/P_ref = 0.3 rung IS lr 0.0045 -- the
#   published lr 5e-3 to within 10%.  The published cell is not merely bracketed,
#   it is very nearly RUN.  Asserted in the selftest.
# alpha ladder: ratio 2, 0.125..4 = 0.5x..16x the RoBERTa-tuned 0.25, which is the
#   measured live scale window; published 1.0 and tuned 0.25 are both INTERIOR.
#   ⭐ LoCA inits to dW == 0 [measured, rel_median 0.0], so -- like WaveFT and
#     unlike FourierFT/QWHA -- there is no init-damage ceiling on alpha; the only
#     asymmetry along a matched-P row is AdamW's decay tax [R.211], which is
#     monotone-and-saturating and wants RANGE, not resolution.
L1 = {
    "arms": ["loca"], "coord": "p",
    "p_mults": [0.025, 0.08, 0.3, 1.0, 3.5, 12.0],
    "scalings": [0.125, 0.25, 0.5, 1.0, 2.0, 4.0],
    "clf_lrs": [5e-4, 2e-3, 5e-3, 2e-2],
    "scale_label": "loca_scale (alpha)",
}

# ---------------------------------------------------------------------------
# QWHA.  No published point exists, so the ladders are anchored on the PORT.
# ---------------------------------------------------------------------------
# atom = s/sqrt(mn) (NOT sqrt(2mn) -- QWHA's layer divides by sqrt(out_features)),
# so the scale ladder is DERIVED from the port's own scale* = 147.31 rather than
# typed.  The RoBERTa-tuned 53.033 lands at 0.36x, interior between the 0.25x and
# 0.5x rungs.
# ⛔ QWHA IS AN INIT-PERTURBING ARM (`--qwha_init_weights 0` = randn spectrum,
#   [measured] rel_median 0.00892 at s=53.033), so unlike LoCA/WaveFT it CAN be
#   capped from above by init damage before optimisation gets a say.  [g2, measured]
#   FourierFT's own rel/collapse curve on this backbone: rel 0.048 was its optimum,
#   rel 0.13 still produced its 4th-best cell (3/28 at floor), rel 0.36 was half
#   dead (11/28).  The 8x top rung sits at rel 0.198 -- degrading but demonstrably
#   still alive at FourierFT's calibration, which is what an upper BRACKET has to
#   be.  Going further would buy known-dead cells, as g2's sc-8000 row did.
Q1 = {
    "arms": ["qwha"], "coord": "p",
    "p_mults": list(P_LADDER_25),
    "scale_mults": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0], "scale_base": "derived",
    "clf_lrs": [5e-4, 2e-3, 5e-3, 2e-2],
}

# ---------------------------------------------------------------------------
# ⭐ LYRA.  THE ONLY ARM WITH A THIRD METHOD KNOB -- and it gets the BIGGEST budget.
# ---------------------------------------------------------------------------
# atom == gamma (`--spectral_scaling`), width-independent, so published (gamma 1,
# lr 2e-2) is a published P of 0.02 = 1.667x P_ref -- comfortably interior to the
# standard ladder.  gamma's ladder is ratio 4 over 64x so that OUR tuned 0.1 and
# the PUBLISHED 1.0 -- 10x apart -- are both interior.
#
# ⭐⭐ `--spectral_freq_exponent` IS SWEPT HERE, AND IT HAS NEVER BEEN SWEPT AT THE
#   WARMED PROTOCOL IN THIS REPO.  [R.233 §3]: [P.26] swept it UNWARMED and got
#   0.6787..0.6968 across {1,2,3,4,5} on RTE, but [R.160] bars comparing warmed and
#   unwarmed runs, so "is the banked exponent 2.0 LYRA's own optimum" is UNKNOWN --
#   and the PUBLISHED value is 3.0, not 2.0.  A baseline benchmarked at a value its
#   own authors did not use is exactly what `PROCESS §5 test 5` bars.  {1,2,3,5}
#   puts both 2.0 and the published 3.0 strictly interior.
#
# ⇒ 6 x 4 x 4 x 2 = 192 cells, the LARGEST budget of any arm here (FourierFT 142,
#   WaveFT 146).  ⭐ THAT ASYMMETRY IS DELIBERATE AND IT IS THE CONSERVATIVE
#   DIRECTION: LYRA is a BASELINE, so over-searching it can only make our own claim
#   harder, and `[Dodge §6]`'s asymmetry means a loss under a LARGER budget is
#   attributable while a loss under a smaller one is not.  [R.305 §LYRA] already
#   flagged it as "the most structural knobs ... the most room to be under-tuned by
#   a fixed-budget search".
# ⚠ THE PRICE, STATED: to afford the 4th axis, LYRA's classifier_lr axis is g2's
#   TWO survivors, not g1's four.  [g1, measured] that axis is flat across 40x
#   (best F1 0.8823..0.8945 over all four values) except that 2e-2 is harmful (8 of
#   its 9 at-floor cells), and the head is the SAME 4,096-param `score` layer for
#   every arm -- so this is the cheapest axis in the design to spend, and LYRA still
#   ends with 33% MORE cells than any other arm.
Y1 = {
    "arms": ["lyra"], "coord": "p",
    "p_mults": list(P_LADDER_25),
    "scalings": [0.05, 0.2, 0.8, 3.2],
    "clf_lrs": [5e-4, 5e-3],
    "extra": {"key": "freq_exponent", "flag": "--spectral_freq_exponent",
              "label": "spectral_freq_exponent", "id": "ex", "values": [1.0, 2.0, 3.0, 5.0]},
    "scale_label": "spectral_scaling (gamma)",
}

# ---------------------------------------------------------------------------
# ⭐⭐ SCoRA (ours) -- ONE magnitude axis, ON PURPOSE, and 36 cells is not
#   under-tuning.
# ---------------------------------------------------------------------------
# `scora`'s scale is DERIVED a priori from --slr_s (fir_arms: "DO NOT ADD ONE");
# setting it by hand is precisely what makes the arm `scora2`.  So this grid has NO
# scale axis, and lr = P / atom directly.
# ⭐ REPORT PER-AXIS RESOLUTION, NOT CELL COUNT [R.259]: 9 rungs at ratio 2 over
#   256x is the FINEST P axis of any arm in this comparison (the others are 2.5 and
#   3.5).  [R.305] made exactly this argument ("SCoRA -- one axis, on purpose").
# ⭐ AND THE DIRECTION IS THE SAFE ONE.  36 < 140 means OUR arm holds the SMALLEST
#   budget of the nine.  `[Dodge §6]`: "if a model with a small budget outperforms a
#   model with a large budget, increasing the small budget will not change this
#   conclusion."  A SCoRA win is therefore attributable; a SCoRA loss is the one
#   thing this design cannot rule out, and that is the correct way round for a
#   method's authors to be wrong.
S1 = {
    "arms": ["scora"], "coord": "p", "no_scale": True,
    "p_mults": [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    "clf_lrs": [5e-4, 2e-3, 5e-3, 2e-2],
}

# ---------------------------------------------------------------------------
# SCoRA-2 (ours) -- the swept-scaling row.  ⛔ 140 cells, CAPPED BELOW FourierFT's.
# ---------------------------------------------------------------------------
# ⛔ 7 x 5 x 4 = 140, deliberately NOT 6 x 6 x 4 = 144.  144 would put OUR arm 1.4%
#   ABOVE the FourierFT comparator's 142, and a budget advantage to the arm making
#   the claim is the same defect as w1's deficit, mirrored.  A selftest asserts
#   max(ours) <= min(comparator).
# The scaling ladder is DERIVED as multiples of [R.306]'s own swept optimum
# (0.00610352), ratio 2 over 16x, that value interior.
S2 = {
    "arms": ["scora2"], "coord": "p",
    "p_mults": [0.064, 0.16, 0.4, 1.0, 2.5, 6.25, 15.625],
    "scale_mults": [0.25, 0.5, 1.0, 2.0, 4.0], "scale_base": "ref",
    "clf_lrs": [5e-4, 2e-3, 5e-3, 2e-2],
}

# ---------------------------------------------------------------------------
# ⚠⚠ wref -- WaveFT AT ITS OWN PUBLISHED POINT.  4 cells.  THE ADMISSIBILITY GAP
#   IN A SEARCH WE HAVE ALREADY RUN, closed the same way [R.305] closed it on
#   roberta-base: 2 explicit REF cells per arm, not a widened grid.
# ---------------------------------------------------------------------------
# `[R.258]` (its core finding stands): WaveFT's published point (lambda = 25,
# lr = 1e-4) is outside w1/w2 in BOTH directions -- 1500x in lr, 91x in atom, 16.6x
# in the product -- and they are OPPOSITE directions, which is why it survives the
# eye test.  Every OTHER arm in this comparison now has its published point inside
# its ladders; WaveFT would be the only one that does not.
# ⭐ WHY REF CELLS AND NOT A WIDER LADDER: the published point is 33x below the
#   bottom of the measured live window, so a ladder reaching it would spend ~1/3 of
#   WaveFT's budget on cells the measured marginals say are dead. The criterion is
#   "the published point was RUN", and 4 cells buy exactly that.
# ⛔ BOTH NUMBERS ARE DERIVED, NOT TYPED INTO A LADDER: lambda multiplies an
#   orthonormal IDWT (`[R.258 §2]`, and that section's stated assumption about the
#   authors' normalisation is INHERITED here), so the published atom IS 25 and the
#   scaling that reproduces it at this width is 25/aps.
# ⚠ These 2 cells/arm COUNT toward WaveFT's budget in budget_per_arm(), taking it
#   to 146 vs FourierFT's 142.  Counting a fixed published point as search budget
#   OVERSTATES WaveFT's advantage, which is the fail-closed direction.
PUBLISHED_WAVEFT = {"atom": 25.0, "lr": 1e-4}     # [N.1 §0.1, arXiv 2505.12532 §5.1]

# ---------------------------------------------------------------------------
# ⛔⛔ EVERY ARM'S PUBLISHED OPERATING POINT, IN ONE PLACE, AS A GATE.
# ---------------------------------------------------------------------------
# `[user, 2026-08-28]`: "the optimal paper-reported values should fall in range".
# `[R.258]` is the reason it has to be MACHINE-CHECKED rather than eyeballed: it
# found WaveFT's published point outside the swept box in BOTH axes at once, in
# OPPOSITE directions -- which is exactly the configuration that survives a human
# glance at two ladders. So the criterion lives here, as data, and the selftest
# asserts it for whichever arms the selected grid carries.
#
#   "P"        the published EFFECTIVE STEP lr*atom, in the same absolute units as
#              p_ref().  A range where the authors report a per-task range.
#   "axes"     published values on the grid's OWN other axes, by cell key.
#   "ref_grid" ⚠ set ONLY where the published point is too far outside the live
#              window to sweep, and is instead RUN as explicit REF cells.  A named
#              escape hatch, checked to actually contain the arm -- not a waiver.
#
# ⛔ QWHA has NO entry, and that is a FACT about QWHA, not an omission: its paper
#    tunes on a QUANTISED LLaMA and publishes no RoBERTa/GLUE point
#    [train_glue.py:494; R.236 §3.5]. Its ladders are anchored on the port instead.
PUBLISHED = {
    # LoCA: alpha = 1, coefficient lr 5e-4..1e-2 per task  [train_glue.py:502,505;
    # R.258 §1 quotes the primary point as lr 5e-3].  atom == alpha, so P == lr.
    "loca": {"P": [("lr range", 5e-4, 1e-2)], "axes": {"scaling": 1.0}},
    # LYRA: gamma = 1, lr 2e-2, freq_exponent 3.0  [R.258 §1].  atom == gamma.
    "lyra": {"P": [("lr 2e-2 at gamma 1", 2e-2, 2e-2)],
             "axes": {"scaling": 1.0, "freq_exponent": 3.0}},
    # WaveFT: lambda = 25, lr 1e-4 -- P = 2.5e-3, which is 33x BELOW the bottom of
    # the measured live window.  ⚠ NOT on w1/w2's ladders and deliberately not put
    # there; the `wref` block runs the point itself.  [R.258]
    "wave1": {"P": [("lambda 25 x lr 1e-4", 2.5e-3, 2.5e-3)], "ref_grid": "wref"},
    "wave2": {"P": [("lambda 25 x lr 1e-4", 2.5e-3, 2.5e-3)], "ref_grid": "wref"},
}
WREF = {"arms": WAVE_ARMS, "coord": "p", "published_point": "waveft",
        "clf_lrs": [5e-4, 5e-3]}

# ⭐ A READING VIEW, NOT A RUN TARGET. `wave` is the UNION of w1 and w2 -- the whole
#   144-cell-per-arm WaveFT search, which is the thing a budget-equalised claim is
#   about. ⛔ It is a union of two disjoint factorial BLOCKS, not one factorial, so
#   the reader tests edges per block and says so; a bare min/max over the union
#   axes would claim a bracketing the design does not have.
WAVE_ALL = {"arms": WAVE_ARMS, "coord": "p", "union": ["w1", "w2"]}

# ---------------------------------------------------------------------------
# ⭐⭐ THE EDGE PROBES.  Four 2-cell RAYS off the four grids whose best cell landed
#   ON a ladder edge (`FIR_GEMMA_PORT.md` §20.3).  A probe is NOT a widened grid.
# ---------------------------------------------------------------------------
# `[user, 2026-08-29]`: "extend the search ranges for the 4 cases that need it".
#
# ⛔⛔ WHY A RAY AND NOT A WIDER FACTORIAL -- read this before enlarging one.
#   1. THE QUESTION AN EDGE POSES IS ONE-DIMENSIONAL. The reader flags an edge to
#      say "the metric may still be RISING when the ladder stops". Two cells off
#      the end of that one axis, at the winning cell's own other coordinates,
#      answer exactly that. A full extra factorial block answers it too -- for
#      28-56x the GPU -- and at ONE SEED the extra cells buy mostly SELECTION
#      INFLATION, which is the differential `w2` was built to remove (§3.3), not
#      discovery.
#   2. ⛔ THE BUDGET GATE IS DIRECTIONAL AND `scora2` HAS EXACTLY 2 CELLS OF
#      HEADROOM. `max(ours) <= comparator = 142` and scora2 already holds 140. A
#      3rd cell for OUR arm would have to be paid for by extending the FourierFT
#      comparator too (2 arms x N cells), so "just widen it" is not a small change:
#      it is ~8.6 GPU-h for one rung. The ray fits under the gate as it stands.
#   3. ⭐ AND THE MARGINALS SAY MOST OF THESE EDGES ARE NOT REAL. `[measured, §20]`
#      max F1 along each flagged axis:
#        loca   alpha  0.125->4 : .8897 .8801 .8863 .8792 .8908 .8908   FLAT
#        lyra   exp    1->5     : .8661 .8660 .8696 .8784                FLAT
#        scora2 scale  16x span : .8950 .8935 .8840 .8847 .8955          FLAT
#        qwha   P      0.16->15.6: .8308 .8398 .8576 .8780 .8828 .8845   ⭐ RISING
#      Three of the four axes carry NO trend -- the flag fired because the
#      single-seed argmax happened to land on the top rung of an INERT axis. Only
#      QWHA's P axis is monotone to its edge. Spending a factorial block to widen
#      an inert axis would be measuring noise more precisely.
#   ⇒ A PROBE THAT RISES IS A REASON TO BUILD A REAL BLOCK. A probe that does not
#     rise CLOSES the edge warning for ~0.1 GPU-h. Either way the ray comes first.
#
# ⚠ WHAT A PROBE CANNOT DO, stated before it is read: it cannot RELOCATE an
#   optimum. It holds the other axes at the winner's values, so it explores a line,
#   not a region, and (like everything else here) at ONE SEED. It answers "does the
#   metric keep rising past the edge", and nothing else.
#
# ⛔ THE ANCHOR IS TYPED, AND IT MUST BE. It is a MEASURED result -- the best cell
#   of a finished grid -- and results live in gitignored `logs/`, which does not
#   travel to fir (CONTEXT §4.1). A planner that read them would enumerate an EMPTY
#   probe there, silently, exactly as `r310_plan.selected_args()` would (§3.6). So
#   the coordinates are constants here, with their provenance, and the selftest
#   asserts each one RESOLVES TO A CELL OF THE BASE GRID -- which is the property
#   that would actually break if a ladder were edited.
EDGE_PROBES = {
    # best loca cell, F1 0.8908 -- alpha is at the TOP rung of {0.125..4}
    "locax": {"base": "loca", "arms": ["loca"],
              "anchor": {"p_mult": 3.5, "scaling": 4.0, "classifier_lr": 2e-3},
              "steps": [("scaling", [8.0, 16.0])]},
    # best qwha cell, F1 0.8845 -- P is at the TOP rung of P_LADDER_25.
    #   ⭐ the one probe whose axis is measurably alive; the scale axis peaked
    #   INTERIOR (294.6) and needs nothing.
    "qwhax": {"base": "qwha", "arms": ["qwha"],
              "anchor": {"p_mult": 15.625, "scale_mult": 2.0, "classifier_lr": 5e-3},
              "steps": [("p_mult", [39.0625, 97.65625])]},
    # best lyra cell, F1 0.8784 -- the exponent is at the TOP of {1,2,3,5}
    # ⭐ [measured, CPU] the exponent SATURATES, and 13 is near its asymptote: at
    #   d=2048/k=16 the geometric index set is {0,1,2,3,4,5,10,23,...} at ex=5,
    #   {0..8,17,40,...} at ex=8 and {0..10,18,56,...} at ex=13 -- i.e. it converges
    #   on this codebase's OWN `freq_mode="contiguous"`. So this probe asks "is the
    #   flagged edge just LYRA wanting contiguous low frequencies?", and going past
    #   ~13 would buy almost no new index set. ⭐ k is PRESERVED at every exponent
    #   (the collision loop increments rather than drops), so budget parity holds.
    "lyrax": {"base": "lyra", "arms": ["lyra"],
              "anchor": {"p_mult": 2.5, "scaling": 0.8, "classifier_lr": 5e-3,
                         "freq_exponent": 5.0},
              "steps": [("freq_exponent", [8.0, 13.0])]},
    # ⛔ best scora2 cell, F1 0.8955 -- TWO edges at once (scaling at the top rung,
    #   classifier_lr at the bottom), and exactly 2 cells of budget headroom. So
    #   this probe spends ONE cell on EACH edge rather than two on either.
    "scora2x": {"base": "scora2", "arms": ["scora2"],
                "anchor": {"p_mult": 6.25, "scale_mult": 4.0, "classifier_lr": 5e-4},
                "steps": [("scale_mult", [8.0]), ("classifier_lr", [1e-4])]},
}
# ⭐ Materialised into grid dicts so every downstream tool (`--list`, the runner,
#   the reader, the shell gate) sees them as ordinary selectable grids.
PROBE_GRIDS = {n: {"arms": p["arms"], "coord": "p", "probe": n}
               for n, p in EDGE_PROBES.items()}

GRIDS = {"g1": G1, "g2": G2, "w1": W1, "w2": W2, "wave": WAVE_ALL,
         "loca": L1, "qwha": Q1, "lyra": Y1, "scora": S1, "scora2": S2, "wref": WREF,
         **PROBE_GRIDS}
# ⛔ THE ARMS WHOSE RESULT WE ARE CLAIMING. The budget gate is DIRECTIONAL -- an
#   arm we own must never hold a LARGER search budget than the comparator -- so it
#   needs to know which arms are ours. `[R.306]`: both SCoRA rows always ship
#   together, so both are here.
OUR_ARMS = list(FA.SCORA_ROWS)
GRID_NAME = os.environ.get("FIR_HP_GRID", "g2")
if GRID_NAME not in GRIDS:
    raise SystemExit(f"FAIL CLOSED: FIR_HP_GRID={GRID_NAME!r} is not one of {sorted(GRIDS)}")
_G = GRIDS[GRID_NAME]
ARMS = _G["arms"]
COORD = _G["coord"]
IS_UNION = "union" in _G
# ⛔ NO_SCALE IS A GRID KIND, NOT A MISSING KEY. `scora` derives its scale from
#   --slr_s a priori (fir_arms: "DO NOT ADD ONE"), so its grid has no scale AXIS --
#   which is a different thing from a grid that forgot to declare one. Everything
#   downstream (cell ids, axes(), cell_cmd, the canary picker, the reader's edge
#   test) branches on this flag rather than on `len(scalings) == 0`.
NO_SCALE = bool(_G.get("no_scale"))
EXTRA = _G.get("extra")            # None except LYRA's freq_exponent (a 4th axis)
# ⛔ A PROBE IS A GRID KIND TOO, for the same reason NO_SCALE is one: everything
#   downstream (axes, the canary picker, the reader, the shell gate) must branch
#   on a DECLARED kind, never on the shape of a ladder it happens to find.
PROBE = _G.get("probe")            # the EDGE_PROBES key, or None
# ⛔ THE 4th-AXIS KEYS ARE MODULE-LEVEL, NOT READ OFF THE SELECTED GRID. cell_id()
#   and cell_cmd() must mean the same thing for a cell no matter which grid happens
#   to be selected -- budget_per_arm() enumerates every grid at once, and an id that
#   depended on FIR_HP_GRID would make the fairness count depend on an env var.
EXTRA_KEYS = {"freq_exponent": "--spectral_freq_exponent"}
CLF_LRS = _G.get("clf_lrs", [])
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


def _anchor_scale(prof):
    """The scale at which the port table's own `derived_lr` was computed.

    ⛔ It is `derived_scale` when the arm perturbs the weights at init (the
      scale-matching rule had something to say) and `scale` when it does not
      (loca / wave / scora2 init to dW == 0, so derived_scale is None BY
      DERIVATION -- fir_plan.py:127).  Asking for the wrong one silently checks the
      P coordinate against a point the port never claimed."""
    return prof["derived_scale"] if prof.get("derived_scale") is not None else prof["scale"]


def resolve_scalings(grid, PT=None):
    """A grid dict's scale ladder, with the DERIVED kinds expanded.

    ⭐ Three kinds, and the last two exist so no anchor is ever TYPED:
        `scalings`     an explicit ladder.  Used where the published value is an
                       absolute constant that must literally be on the ladder
                       (LoCA's alpha, LYRA's gamma).
        `scale_mults` + `scale_base='derived'`   multiples of the PORT's scale*
                       (QWHA: its anchor is a width-corrected quantity).
        `scale_mults` + `scale_base='ref'`       multiples of the arm's own
                       RoBERTa-tuned scale (SCoRA-2: its anchor is [R.306]'s
                       swept optimum, which no port rule moves).
    """
    if grid.get("probe"):
        # ⛔ A PROBE HAS NO LADDER OF ITS OWN. Its scale value (when it has one) is
        #   the ANCHOR's, resolved against the BASE grid -- returning a one-value
        #   "ladder" here would let an edge test run on a line.
        return []
    if grid.get("no_scale") or grid.get("union") or grid.get("published_point"):
        # ⛔ NONE of these has a scale LADDER, and each for a different reason: no
        #   scale axis at all; two blocks with two ladders; one fixed published
        #   point. Returning [] is right for all three -- inventing one is not.
        return []
    if "scalings" in grid:
        return list(grid["scalings"])
    mults, base = grid["scale_mults"], grid["scale_base"]
    PT = _pt(PT)
    arm = grid["arms"][0]
    prof = PT["targets"][TARGETS]["arms"][arm]
    if base == "derived":
        anchor = prof["derived_scale"]
        if anchor is None:
            raise SystemExit(f"FAIL CLOSED: {arm} has no derived_scale (it inits to "
                             f"dW == 0), so scale_base='derived' has no anchor")
    elif base == "ref":
        anchor = prof["scale"]
        if anchor is None:
            raise SystemExit(f"FAIL CLOSED: {arm} has no scale at all -- use no_scale")
    else:
        raise SystemExit(f"FAIL CLOSED: unknown scale_base {base!r}")
    return [anchor * m for m in mults]


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


def atom_fixed(PT=None, arms=None, targets=TARGETS):
    """The arm's atom at the TARGET width when it has NO scale knob.

    ⚠ `scora`'s atom is STOCHASTIC (rel sd 5.79% [R.174]) because its factor is
      drawn randn, so this median is a CENTRE, not a constant, and the derived lr
      carries the same 5.8% band.  That is a property of the arm, not of the grid;
      the alternative (`--slr_init_norm unit`) would be a different arm."""
    PT = _pt(PT)
    prof = PT["targets"][targets]["arms"]
    arms = list(arms or ARMS)
    vals = [prof[a]["atom_median"] for a in arms]
    if max(vals) - min(vals) > 1e-12 * max(vals):
        raise SystemExit(f"FAIL CLOSED: arms {arms} do not share an atom ({vals})")
    return vals[0]


def lr_for(p_mult, scaling, PT=None, arms=None, no_scale=False):
    """lr = P / atom, the whole point of the coordinate.

    ⛔ `no_scale` is NOT "scaling == 1". On a scale-less arm the atom is a fixed
      measured quantity, not `scaling * atom_per_scale`; atom_per_scale() itself
      FAILS CLOSED for such an arm, and it should."""
    P = p_mult * p_ref(PT=PT, arms=arms)
    if no_scale:
        return P / atom_fixed(PT=PT, arms=arms)
    return P / (scaling * atom_per_scale(PT=PT, arms=arms))


def scalings():
    """The SELECTED grid's scale ladder.

    ⛔ A FUNCTION, NOT AN IMPORT-TIME GLOBAL. Resolving a `scale_mults` ladder reads
      the port table; binding it at import would make merely IMPORTING this planner
      fail on a box without `port_<model>.json`, for every grid, including the four
      that do not need it."""
    return resolve_scalings(_G)


def axes_of(grid):
    """A GRID DICT's testable axes.  ⛔ Takes the dict, not the globals: a union
    view has to ask its members, and the equalisation gate has to enumerate every
    grid while a different one is selected."""
    if "union" in grid:
        raise SystemExit("FAIL CLOSED: a union view has no single axis set -- "
                         "ask its member grids (member_grids())")
    if grid.get("probe"):
        # ⛔ A PROBE IS A RAY, NOT A FACTORIAL. Handing back its 1-2 stepped values
        #   as an "axis" would let the reader's edge test fire on a line whose every
        #   cell is an endpoint by construction -- the <3-values case, but worse,
        #   because it would read as a bracketing claim the design never makes.
        raise SystemExit("FAIL CLOSED: an edge probe is a RAY off a base grid's "
                         "edge -- it has no interior; read it against its base")
    if grid.get("published_point"):
        # ⛔ A REF BLOCK HAS NO AXES AT ALL, and that is the point of it: it sits at
        #   ONE published point. Handing back a degenerate axis set would let the
        #   reader's edge test run and report "the optimum is at the edge" on a
        #   block whose whole design is that it has no interior.
        raise SystemExit("FAIL CLOSED: a REF block sweeps nothing -- it is one "
                         "published operating point, not a ladder")
    first = (("P/P_ref", "p_mult", grid["p_mults"]) if grid["coord"] == "p"
             else ("lr", "lr", grid["lrs"]))
    out = [first]
    # ⛔ A SCALE-LESS GRID DECLARES TWO AXES, and the reader must be told two rather
    #   than be handed an empty third. `scora` has one magnitude knob BY DESIGN.
    if not grid.get("no_scale"):
        out.append((grid.get("scale_label", "scaling"), "scaling", resolve_scalings(grid)))
    out.append(("classifier_lr", "classifier_lr", grid["clf_lrs"]))
    # ⭐ THE FOURTH AXIS EXISTS FOR EXACTLY ONE ARM. It is declared by the grid, so
    #   the reader's edge test, the canary picker and the budget gate all see it
    #   without any of them knowing the word "lyra".
    if grid.get("extra"):
        e = grid["extra"]
        out.append((e["label"], e["key"], e["values"]))
    return out


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
    if PROBE:
        # ⛔ A PROBE HAS NO CENTRAL CELL EITHER -- every one of its cells is, by
        #   construction, one step BEYOND an edge. Picking one and calling it a
        #   canary would smoke-test the most extreme cell in the block. It is 2
        #   cells; submit it whole.
        raise SystemExit("FAIL CLOSED: an edge probe has no central cell (every "
                         "cell is past an edge). Submit it whole -- it is 2 cells.")
    if _G.get("published_point"):
        # ⛔ A CANARY IS "ONE CENTRAL CELL PER ARM" AND A REF BLOCK HAS NO CENTRE.
        #   It is 4 cells at one fixed point; submit it whole. Refusing is not a
        #   limitation -- picking cell 0 and calling it a canary would be a lie
        #   about what was smoke-tested.
        raise SystemExit("FAIL CLOSED: a REF block has no central cell (it is ONE "
                         "published point). Submit it whole -- it is 4 cells.")
    ids = ids or [cell_id(c) for c in cells()]
    def mid(v):
        return sorted(set(v))[(len(set(v)) - 1) // 2]
    # ⛔ CENTRAL ON EVERY AXIS THE GRID DECLARES -- including a scale axis that is
    #   absent (scora) and a fourth one that exists for a single arm (lyra). The
    #   earlier version named `scaling` and three axes literally, so a grid with a
    #   different shape would either crash at submit time or silently pick a corner.
    ax = axes_of(_G)
    out = []
    for arm in ARMS:
        want = [c for c in cells([arm])
                if all(c[key] == mid(vals) for _lab, key, vals in ax)]
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
    if grid.get("published_point"):
        return _published_cells(grid, arms, PT=PT)
    if grid.get("probe"):
        return _probe_cells(grid, arms, PT=PT)
    out = []
    first = grid.get("p_mults") if grid["coord"] == "p" else grid["lrs"]
    ns = bool(grid.get("no_scale"))
    # ⭐ `[None]` for an absent scale axis and `[None]` for an absent 4th axis keep
    #   the loop nest -- and therefore the CELL ORDER, which IS the Slurm array
    #   index -- byte-identical for the four grids that have neither.
    scs = [None] if ns else resolve_scalings(grid, PT=PT)
    exs = grid["extra"]["values"] if grid.get("extra") else [None]
    for arm in (arms or grid["arms"]):
        for v in first:
            for sc in scs:
                for clr in grid["clf_lrs"]:
                    for ex in exs:
                        c = {"arm": arm, "task": TASK, "targets": TARGETS,
                             "seed": SEED, "epochs": EPOCHS,
                             "scaling": sc, "classifier_lr": clr}
                        if grid.get("extra"):
                            c[grid["extra"]["key"]] = ex
                        if grid["coord"] == "p":
                            c["p_mult"] = v
                            c["lr"] = lr_for(v, sc, PT=PT, arms=grid["arms"], no_scale=ns)
                        else:
                            c["lr"] = v
                        out.append(c)
    return out


def _base_ladder(base, key):
    """The BASE grid's own ladder for a probed axis key, in the units the probe
    steps in.  ⛔ `scale_mult` is answered with the MULTIPLIERS and `scaling` with
    the RESOLVED values: comparing a multiplier against a resolved ladder would
    silently call 8.0 'outside' a ladder that runs to 1178."""
    if key == "scale_mult":
        return list(base.get("scale_mults") or [])
    if key == "scaling":
        return list(resolve_scalings(base))
    if key == "p_mult":
        return list(base.get("p_mults") or [])
    if key == "classifier_lr":
        return list(base.get("clf_lrs") or [])
    if base.get("extra") and key == base["extra"]["key"]:
        return list(base["extra"]["values"])
    return None


def probe_anchor_cell(name, PT=None):
    """The BASE-grid cell an edge probe is anchored on, fully resolved.

    ⭐ Built through the SAME code path as a base-grid cell -- the anchor is stated
      in the base grid's own coordinates (`scale_mult` where the base derives its
      ladder, a literal `scaling` where the base types one) and the scale is
      resolved against the BASE, so a typed anchor cannot drift into a value the
      base grid never had. The selftest asserts the id it produces is a member of
      the base grid, which is the check that actually fires when a ladder moves."""
    pr = EDGE_PROBES[name]
    base = GRIDS[pr["base"]]
    ns = bool(base.get("no_scale"))
    a = dict(pr["anchor"])
    if "scale_mult" in a:
        if "scale_mults" not in base:
            raise SystemExit(f"FAIL CLOSED: probe {name} gives a scale_mult but base "
                             f"{pr['base']!r} types its ladder -- give `scaling`")
        anchors = resolve_scalings(base, PT=PT)
        m = a.pop("scale_mult")
        mults = base["scale_mults"]
        # ⛔ RESOLVED THROUGH THE BASE'S OWN LADDER where the multiplier is on it,
        #   and through its anchor otherwise (that is how a probe STEPS PAST the
        #   top rung). Never re-derived from a second copy of the anchor value.
        a["scaling"] = anchors[mults.index(m)] if m in mults else anchors[0] / mults[0] * m
    c = {"arm": pr["arms"][0], "task": TASK, "targets": TARGETS, "seed": SEED,
         "epochs": EPOCHS, "scaling": None if ns else a.get("scaling"),
         "classifier_lr": a["classifier_lr"], "p_mult": a["p_mult"]}
    if base.get("extra"):
        c[base["extra"]["key"]] = a.get(base["extra"]["key"])
    c["lr"] = lr_for(c["p_mult"], c["scaling"], PT=PT, arms=pr["arms"], no_scale=ns)
    return c


def _probe_cells(grid, arms=None, PT=None):
    """The probe's cells: the anchor, with ONE axis stepped past the base's edge.

    ⛔ ONE AXIS AT A TIME, never a cross product. Two cells that differ in two knobs
      cannot say which knob moved the metric, and the whole purpose of the block is
      to attribute a single edge."""
    name = grid["probe"]
    pr = EDGE_PROBES[name]
    base = GRIDS[pr["base"]]
    ns = bool(base.get("no_scale"))
    anchor = probe_anchor_cell(name, PT=PT)
    out = []
    for arm in (arms or grid["arms"]):
        for key, vals in pr["steps"]:
            for v in vals:
                c = dict(anchor, arm=arm)
                if key == "scale_mult":
                    mults = base["scale_mults"]
                    c["scaling"] = resolve_scalings(base, PT=PT)[0] / mults[0] * v
                else:
                    c[key] = v
                c["lr"] = lr_for(c["p_mult"], c["scaling"], PT=PT,
                                 arms=pr["arms"], no_scale=ns)
                out.append(c)
    return out


def _published_cells(grid, arms=None, PT=None):
    """The REF cells: one FIXED published operating point, crossed with clf_lr.

    ⛔ NOT A SEARCH. There is no ladder here and there must not be one -- the whole
      point is that these cells sit where the AUTHORS put them, so that "the
      published point was run" stops being a thing we have to argue about.
      ⭐ Both numbers are DERIVED from the published constants at THIS width, never
      typed as a scaling: the published lambda IS the atom, so the scaling that
      reproduces it is lambda / (atom-per-unit-scale)."""
    if grid["published_point"] != "waveft":
        raise SystemExit(f"FAIL CLOSED: unknown published point {grid['published_point']!r}")
    arms = list(arms or grid["arms"])
    aps = atom_per_scale(PT=PT, arms=grid["arms"])
    sc = PUBLISHED_WAVEFT["atom"] / aps
    lr = PUBLISHED_WAVEFT["lr"]
    p_mult = PUBLISHED_WAVEFT["atom"] * lr / p_ref(PT=PT, arms=grid["arms"])
    out = []
    for arm in arms:
        for clr in grid["clf_lrs"]:
            out.append({"arm": arm, "task": TASK, "targets": TARGETS, "seed": SEED,
                        "epochs": EPOCHS, "scaling": sc, "classifier_lr": clr,
                        "p_mult": p_mult, "lr": lr})
    return out


def cells(arms=None, PT=None):
    """Every cell of the SELECTED grid, in a DETERMINISTIC order -- the array index
    is this order, so it must not depend on a set, a dict iteration or the
    filesystem."""
    return _cells_of(_G, arms, PT=PT)


def cell_id(c):
    """⛔⛔ APPEND-ONLY, AND THE SELFTEST PINS THE OLD GRIDS' DIGESTS.
      572 CSVs, `done` markers and `fail` markers on fir are keyed by this string.
      A cell with no scale DROPS the `-sc` component (its lr is unique per rung
      anyway) and a cell with a 4th axis APPENDS `-ex`; neither can change an id
      that any existing grid emits, because no existing grid sets either key."""
    sc = "" if c.get("scaling") is None else f"-sc{_fmt(c['scaling'])}"
    ex = ""
    for k in sorted(EXTRA_KEYS):
        if c.get(k) is not None:
            ex += f"-ex{_fmt(c[k])}"
    return (f"{c['task']}-{c['arm']}-{c['targets']}"
            f"-lr{_fmt(c['lr'])}{sc}-clr{_fmt(c['classifier_lr'])}{ex}"
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
    if c.get("scaling") is None:
        # ⛔ AND IT MUST BE THE ARM THAT HAS NO SCALE, NOT THE GRID THAT FORGOT ONE.
        #   If an arm WITH a scale flag reached here, the flag would silently keep
        #   whatever value the port left on it -- a hidden, unswept constant inside
        #   a grid that claims to sweep everything it moves.
        if sf:
            raise SystemExit(f"FAIL CLOSED: {c['arm']} HAS a scale flag ({sf}) but this "
                             f"cell has no scaling -- a no_scale grid may only carry "
                             f"arms whose scale is derived a priori")
    else:
        if not sf:
            raise SystemExit(f"FAIL CLOSED: {c['arm']} has no scale flag to sweep")
        cmd = _set_flag(cmd, sf, c["scaling"])
    # ⭐ the 4th axis, for the one arm that has one
    for k, flag in sorted(EXTRA_KEYS.items()):
        if c.get(k) is not None:
            cmd = _set_flag(cmd, flag, c[k])
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
        if PROBE:
            _shape = [len(ARMS), sum(len(v) for _k, v in EDGE_PROBES[PROBE]["steps"])]
        elif _G.get("published_point"):
            _shape = [len(ARMS), len(CLF_LRS)]
        else:
            _shape = [len(ARMS), len(_first)] + ([] if NO_SCALE else [len(scalings())]) \
                     + [len(CLF_LRS)] + ([len(EXTRA["values"])] if EXTRA else [])
        n_expect = 1
        for _x in _shape:
            n_expect *= _x
        ck(len(cs) == n_expect,
           f"grid {GRID_NAME} is {'x'.join(str(x) for x in _shape)} = {len(cs)} cells")
    # ⛔ THE COUNT IS PINNED PER GRID, NOT COMPUTED. It is the number the fairness
    #   argument is made of; if a ladder is edited, this line must be edited too, in
    #   the same commit, deliberately.
    ck({"g1": 160, "g2": 140, "w1": 96, "w2": 192, "wave": 288,
        "loca": 144, "qwha": 144, "lyra": 192, "scora": 36, "scora2": 140,
        "wref": 4,
        # ⛔ 2 EACH, AND scora2x's 2 ARE A HARD CEILING, not a shape preference:
        #   140 + 2 == the comparator's 142 exactly (the directional gate below).
        "locax": 2, "qwhax": 2, "lyrax": 2, "scora2x": 2}[GRID_NAME] == len(cs),
       f"{GRID_NAME} has its declared cell count")
    if not IS_UNION:
        _nax = 2 if NO_SCALE else 3
        _nax += 1 if EXTRA else 0
        if _G.get("published_point"):
            # ⛔ AND PROVE IT REFUSES, in both directions: the REF block has no axes,
            #   the searched grids do.
            try:
                axes(); ck(False, "CONTROL: a REF block refuses an axis set")
            except SystemExit:
                ck(True, "CONTROL: a REF block refuses an axis set (it sweeps nothing)")
            _nax = None
        if PROBE:
            try:
                axes(); ck(False, "CONTROL: an edge probe refuses an axis set")
            except SystemExit:
                ck(True, "CONTROL: an edge probe refuses an axis set (it is a ray)")
            _nax = None
        if _nax is not None:
            ck(len(axes()) == _nax and all(len(a[2]) > 0 for a in axes()),
               f"{GRID_NAME} declares {_nax} non-empty axes: {[a[0] for a in axes()]}")
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
    # ⛔⛔ THE CELL ID IS A DATABASE KEY ON A CLUSTER WE CANNOT SSH TO.
    #   572 CSVs, `done` markers and `fail` markers under
    #   /scratch/.../runs/hpsweep are named by these strings. If cell_id() ever
    #   changes shape, every one of them becomes unreachable -- the sweep would
    #   silently re-run 28.8 + 30.2 GPU-hours of finished work and `--status` would
    #   report 0 done. The four completed grids' digests are PINNED here, so a
    #   change to the id format is a red suite on the dev box rather than a
    #   discovery on fir. ⚠ These four lines may NEVER be updated to match new
    #   output; if they fail, the ID FORMAT is the thing that is wrong.
    # ------------------------------------------------------------------
    FROZEN_DIGESTS = {"g1": "371130518338", "g2": "f3827b29e2f0",
                      "w1": "772ed48d94fe", "w2": "fceff68cd24b",
                      "wave": "095a832e45d8"}
    for _g, _d in sorted(FROZEN_DIGESTS.items()):
        # ⚠ computed EXACTLY as digest() does -- enumeration order, not sorted.
        #   json.dumps(sort_keys=True) does not sort a LIST, so a sorted() here
        #   would silently compute a different number from the one recorded.
        ck(hashlib.sha1(json.dumps(
            [cell_id(c) for c in _cells_of(GRIDS[_g])],
            sort_keys=True).encode()).hexdigest()[:12] == _d,
           f"⭐ {_g}'s cell ids are BYTE-IDENTICAL to the run that produced its "
           f"CSVs on fir (digest {_d})")

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

    # ------------------------------------------------------------------
    # ⭐⭐ THE SAME GATE, GENERALISED TO ALL NINE ARMS, AND MADE DIRECTIONAL.
    #   `[Dodge et al., EMNLP 2019 §6]`: a budget mismatch is only fatal in ONE
    #   direction -- "if a model with a large budget outperforms a model with a
    #   small budget, the difference might be due to the model or the budget (or
    #   both)", whereas a small-budget win survives any increase. So the rule is not
    #   "everything equal"; it is:
    #      OURS may never hold MORE budget than the comparator, and
    #      no BASELINE may hold LESS.
    #   Both directions are unattributable failures if violated, and they are
    #   opposite failures -- which is exactly why a single "budgets are equal"
    #   sentence in a paper is not a check.
    # ------------------------------------------------------------------
    COMP = min(B[a] for a in FFT_ARMS)          # the comparator every claim is against
    base = {a: n for a, n in B.items() if a not in OUR_ARMS}
    ours = {a: n for a, n in B.items() if a in OUR_ARMS}
    ck(max(ours.values()) <= COMP,
       f"⭐ OURS holds no more budget than the comparator "
       f"({ {a: n for a, n in sorted(ours.items())} } vs {COMP}) -- a small-budget "
       f"win is attributable [Dodge §6]; a large-budget win is not")
    ck(min(base.values()) >= COMP,
       f"⭐ every BASELINE holds at least the comparator's {COMP} "
       f"({ {a: n for a, n in sorted(base.items())} }) -- under-searching a baseline "
       f"is the same defect pointed the other way")
    # ⚠ AND AN OVERSHOOT NEEDS A REASON THAT IS NOT "we felt like it". The only
    #   admissible one is a genuinely LARGER search space: [R.259] equal cell counts
    #   across arms with DIFFERENT knob counts is not equal effort either. LYRA is
    #   the one arm with a 4th method axis; the exemption is keyed on that fact, not
    #   on its name, so it evaporates the day the axis does.
    def _naxes(a):
        # ⛔ PROBES AND REF BLOCKS ARE NOT LADDERS, so they cannot contribute an
        #   axis COUNT -- axes_of() fails closed on both, and a bare comprehension
        #   over GRIDS would take the exception rather than skip the grid.
        return max((len(axes_of(g)) for _n, g in GRIDS.items()
                    if not g.get("union") and not g.get("published_point")
                    and not g.get("probe") and a in g["arms"]), default=0)
    comp_ax = _naxes(FFT_ARMS[0])
    for a, n in sorted(B.items()):
        ck(n <= COMP * 1.1 or _naxes(a) > comp_ax,
           f"{a}: budget {n} is within 1.1x the comparator's {COMP}, OR it sweeps "
           f"more axes than the comparator ({_naxes(a)} vs {comp_ax})")
    ck(_naxes("lyra") == 4 and _naxes("scora") == 2,
       "CONTROL: the axis count is REAL -- lyra sweeps 4 axes and scora sweeps 2, "
       "so the exemption above can both fire and fail to fire")
    # ⭐ [R.259] / [R.305 §"SCoRA 5 -- one axis, on purpose"]: for a low-knob arm the
    #   defensible number is PER-AXIS RESOLUTION, not cell count. Assert scora's one
    #   magnitude axis is the FINEST of the nine, so "36 cells" can be reported with
    #   the sentence that makes it fair.
    def _ratio(a):
        ls = [sorted(g["p_mults"]) for _n, g in GRIDS.items()
              if g.get("coord") == "p" and not g.get("union")
              and not g.get("published_point") and not g.get("probe")
              and a in g["arms"]]
        rs = [l[i+1] / l[i] for l in ls for i in range(len(l) - 1)]
        return max(rs) if rs else None
    _sr = _ratio("scora")
    ck(_sr is not None and all(_sr <= (_ratio(a) or 9e9) + 1e-9
                               for a in B if _ratio(a) is not None),
       f"⭐ scora's single P axis is the FINEST of every arm's (ratio {_sr:g}) -- "
       f"the number that makes its 36 cells defensible is RESOLUTION, not count")
    # ------------------------------------------------------------------
    # ⭐⭐ THE EDGE PROBES. Global (they run under EVERY grid): a probe is a claim
    #   about the RELATIONSHIP between two grids, so checking it only when the probe
    #   happens to be selected would be the `--selftest`-skipped-its-fan-out defect
    #   again (§4.2).
    # ------------------------------------------------------------------
    for _pn, _pr in sorted(EDGE_PROBES.items()):
        _base = GRIDS[_pr["base"]]
        _bids = {cell_id(c) for c in _cells_of(_base)}
        _pids = {cell_id(c) for c in _cells_of(GRIDS[_pn])}
        # ⛔⛔ THE ONE CHECK THAT ACTUALLY FIRES WHEN A LADDER MOVES. The anchor is a
        #   TYPED measured result; if the base grid is ever edited, this is what
        #   turns "the probe points at a cell that was never run" from a silent
        #   wrong answer into a red suite.
        ck(cell_id(probe_anchor_cell(_pn)) in _bids,
           f"⭐ {_pn}: its anchor IS a cell of {_pr['base']} "
           f"({cell_id(probe_anchor_cell(_pn))})")
        ck(_pr["arms"] == _base["arms"],
           f"{_pn}: probes exactly the arms {_pr['base']} searched")
        ck(not (_pids & _bids),
           f"{_pn}: its {len(_pids)} cells are ALL NEW budget -- none duplicates "
           f"a cell {_pr['base']} already ran")
        ck(len(_pids) == sum(len(v) for _k, v in _pr["steps"]) * len(_pr["arms"]),
           f"{_pn}: one cell per stepped value, one axis at a time (no cross product)")
        # ⭐ AND EVERY STEP MUST LAND OUTSIDE THE BASE LADDER, or it is not a probe --
        #   it is a duplicate wearing a new grid name.
        for _k, _vs in _pr["steps"]:
            _lad = _base_ladder(_base, _k)
            for _v in _vs:
                ck(_lad is not None and (_v > max(_lad) or _v < min(_lad)),
                   f"⭐ {_pn}: {_k}={_v:g} is OUTSIDE {_pr['base']}'s ladder "
                   f"({min(_lad):g}..{max(_lad):g}) -- a probe steps PAST an edge")
        # ⛔ A CONTROL THAT FIRES: the anchor's OWN value on the probed axis is by
        #   construction INSIDE the ladder, so the test above must reject it.
        #   Without this, "outside" could be vacuously true for any number.
        for _k, _vs in _pr["steps"]:
            _lad = _base_ladder(_base, _k)
            _av = _pr["anchor"].get(_k)
            ck(_av is None or not (_av > max(_lad) or _av < min(_lad)),
               f"CONTROL: {_pn}: the ANCHOR's own {_k}={_av} is INSIDE the base "
               f"ladder, so the outside-test is a real constraint")
    # ⛔⛔ AND THE CEILING, MADE ARITHMETIC. `scora2x` exists at exactly 2 cells
    #   because 140 + 2 == the comparator's budget. Assert the headroom is GONE, so
    #   that a future 3rd cell fails here rather than in a paper.
    _B = budget_per_arm()
    _COMP = min(_B[a] for a in FFT_ARMS)
    ck(_B["scora2"] == _COMP,
       f"⭐ scora2's probe uses its LAST cell of headroom ({_B['scora2']} == the "
       f"comparator's {_COMP}) -- a wider probe for OUR arm would need the "
       f"comparator extended too [Dodge §6]")
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
    _no_canary = IS_UNION or bool(_G.get("published_point")) or bool(PROBE)
    if PROBE:
        try:
            canary_indices(); ck(False, "CONTROL: an edge probe refuses to pick a canary")
        except SystemExit:
            ck(True, "CONTROL: an edge probe refuses to pick a canary (no centre)")
    if _G.get("published_point"):
        try:
            canary_indices(); ck(False, "CONTROL: a REF block refuses to pick a canary")
        except SystemExit:
            ck(True, "CONTROL: a REF block refuses to pick a canary (it has no centre)")
    ci = [] if _no_canary else canary_indices(ids)
    if IS_UNION:
        ck(True, "the union view has no canary (checked above); skipping canary asserts")
    ck(_no_canary or (len(ci) == len(ARMS) and len(set(ci)) == len(ci)),
       f"the canary is {len(ARMS)} distinct cells, one per arm")
    ck(_no_canary or [parse_cell_id(ids[i])["arm"] for i in ci] == list(ARMS),
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
    # ⚠ A PROBE CARRIES ITS ANCHOR'S clf_lr, not a ladder: requiring 5e-3 here would
    #   demand a value the winning cell may not have had (locax's anchor is 2e-3).
    #   The BASE grid is where that anchor has to hold, and it is checked there.
    ck(bool(PROBE) or 5e-3 in (CLF_LRS or {c["classifier_lr"] for c in cs}),
       "the carried classifier_lr 5e-3 is on the grid (a probe inherits its anchor's)")
    PT = FP.port()
    if COORD == "lr":
        dl = PT["targets"][TARGETS]["arms"]["fftm"]["derived_lr"]
        ds = PT["targets"][TARGETS]["arms"]["fftm"]["derived_scale"]
        ck(min(LRS) < dl < max(LRS), f"the derived lr* {dl:.4g} is INSIDE the swept lr range")
        if GRID_NAME == "g1":
            ck(0.5 in LRS and 50 in scalings(), "g1 carries RoBERTa's tuned point exactly")
            ck(min(scalings()) < ds < max(scalings()),
               f"the derived scale* {ds:.4g} is INSIDE g1")
        else:
            # g2 deliberately starts AT the derived scale and climbs: everything below it
            # is measured and worse, so spending cells there again would buy nothing.
            ck(abs(min(scalings()) - round(ds)) <= 1,
               f"g2 starts at the derived scale* ({ds:.4g}) and extends upward only")
            ck(min(LRS) < 1.5 < max(LRS), "g1's best lr (1.5) is BRACKETED by g2's finer axis")
            ck(400 in scalings(), "g1's best scaling (400) is retained as an anchor")
    else:
        # ------------------------------------------------------------------
        # THE P COORDINATE.  ⛔ EVERY ANCHOR IS DERIVED FROM THE PORT TABLE.
        #   A grid whose anchors are typed numbers is a grid that silently stops
        #   pointing at the thing it claims to point at.
        # ------------------------------------------------------------------
        ref = PT["reference"]["__ref__"]["arms"]
        tgt = PT["targets"][TARGETS]["arms"]

        # --- checks EVERY P grid must pass, whatever its arm -----------------
        ck(all("p_mult" in c for c in cs), "every cell carries its P multiplier")
        # ⭐ THE COORDINATE ITSELF, CHECKED AGAINST THE PORT'S OWN derived_lr.
        #   At the port's anchor scale, P/P_ref = 1 must reproduce lr* -- that is
        #   what makes `P` the same quantity the port table talks about.
        #   ⚠ RELATIVE, at 1e-5, and not absolute at 1e-9: for the zero-init arms
        #     the reference and target atoms are two SEPARATE float32 measurements
        #     of a quantity that is analytically identical (loca's atom IS alpha),
        #     so they agree to ~1e-6 relative, not to machine epsilon.
        for a in ARMS:
            if _G.get("published_point"):
                continue                       # a REF block does not sit at P=1
            dl = tgt[a]["derived_lr"]
            got = (lr_for(1.0, None, no_scale=True) if NO_SCALE
                   else lr_for(1.0, _anchor_scale(tgt[a])))
            ck(abs(got - dl) <= 1e-5 * abs(dl),
               f"{a}: P/P_ref=1 reproduces the port's lr* {dl:.6g} (got {got:.6g})")
        if not NO_SCALE and not _G.get("published_point"):
            ck(len({c["lr"] for c in cs}) > len(P_MULTS),
               "CONTROL: lr is DERIVED per (P, scaling), not a swept axis")
        # ⭐ THE MEASURED LIVE WINDOW ON THIS BACKBONE, made a gate. [g1+g2+w1+w2,
        #   measured, 570 cells on this exact cell] FourierFT's optimum sits at
        #   P/P_ref 6 and WaveFT's at 1.6; below 0.3 and above ~10 the marginals
        #   fill up with collapse-floor cells. Any ladder that does not bracket
        #   0.3..10 either cannot see the optimum or is spending cells on dead ones.
        if P_MULTS and not _G.get("published_point"):
            # ⛔ THE UNION OF EVERY BLOCK THAT SEARCHES THIS ARM, not this block
            #   alone. w2 INTERLEAVES w1 and shares no rung with it by design, so a
            #   per-block test would fail for exactly the reason w2 is correct --
            #   the same trap the WaveFT anchors block already documents.
            UPa = sorted({c["p_mult"] for a in ARMS for _n, g in GRIDS.items()
                          if not g.get("union") and not g.get("published_point")
                          and not g.get("probe")
                          and a in g["arms"] for c in _cells_of(g, [a])})
            ck(min(UPa) <= 0.3 and max(UPa) >= 10.0,
               f"the P ladder searched for {'/'.join(ARMS)} brackets the MEASURED "
               f"live window 0.3-10x ({min(UPa):g}..{max(UPa):g})")
            ck(1.0 in UPa, "the carried roberta step P/P_ref = 1 is ON that ladder")

        # --- the PUBLISHED operating point, per arm -------------------------
        # ⛔⛔ THE USER'S STATED CRITERION FOR THIS WHOLE DESIGN, AND [R.258]'s:
        #   "the optimal paper-reported values should fall in range". Asserted for
        #   every arm that HAS a published point, in the coordinates the grid
        #   actually sweeps -- because [R.258]'s WaveFT miss survived the eye test
        #   precisely by being off in two axes in OPPOSITE directions.
        for a in ARMS:
            pub = PUBLISHED.get(a)
            # ⛔ A UNION VIEW AND A REF BLOCK HAVE NO SINGLE LADDER to be interior
            #   to; their members are checked under their own names.
            if not pub or _G.get("published_point") or not P_MULTS:
                continue
            pr_a = p_ref(arms=[a])
            rg = pub.get("ref_grid")
            covered = bool(rg) and a in {c["arm"] for c in _cells_of(GRIDS[rg])}
            if rg:
                ck(covered,
                   f"⚠ {a}: its published point is OUTSIDE these ladders and is "
                   f"covered by the REF grid {rg!r} instead -- asserted to actually "
                   f"contain {a}, because a named escape hatch that does not run is "
                   f"worse than no escape hatch")
            for lab, lo, hi in pub["P"]:
                for v in (lo, hi):
                    m = v / pr_a
                    ck(covered or min(P_MULTS) < m < max(P_MULTS),
                       f"⭐ {a}: published {lab} = P/P_ref {m:.4g} is INTERIOR to the "
                       f"P ladder ({min(P_MULTS):g}..{max(P_MULTS):g})"
                       + ("  [via REF cells]" if covered else ""))
            for flag, v in ({} if covered else pub.get("axes", {})).items():
                ax = {k: vals for _l, k, vals in axes()}
                if flag not in ax:
                    ck(False, f"{a}: published axis {flag!r} is not swept by this grid")
                    continue
                ck(min(ax[flag]) < v < max(ax[flag]),
                   f"⭐ {a}: published {flag} = {v:g} is INTERIOR to its ladder "
                   f"({min(ax[flag]):g}..{max(ax[flag]):g})")
            # ⛔ AND A CONTROL THAT CAN FIRE: a value the authors did NOT use, one
            #   decade outside, must be OUTSIDE -- otherwise "interior" is vacuous
            #   because the ladder is wide enough to contain anything.
            _lo = min(v for _l, lo, hi in pub["P"] for v in (lo, hi)) / pr_a
            ck(covered or not (min(P_MULTS) < _lo / 100 < max(P_MULTS)),
               f"CONTROL: {a}: a point 100x below the published one is OUTSIDE the "
               f"ladder (so 'interior' is a real constraint)")

    if COORD == "p" and set(ARMS) <= set(WAVE_ARMS) and not _G.get("published_point"):
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

    # ------------------------------------------------------------------
    # ⭐ THE REF BLOCK. Its whole job is that the PUBLISHED point was RUN, so the
    #   checks are about fidelity to the publication, not about a ladder.
    # ------------------------------------------------------------------
    if _G.get("published_point"):
        tgtw = FP.port()["targets"][TARGETS]["arms"]
        for a in ARMS:
            cw = [c for c in cells([a])][0]
            # the derived scaling must reproduce the PUBLISHED atom at THIS width
            atom = cw["scaling"] * tgtw[a]["atom_median"] / tgtw[a]["scale"]
            ck(abs(atom - PUBLISHED_WAVEFT["atom"]) < 1e-9 * PUBLISHED_WAVEFT["atom"],
               f"{a}: the derived scaling {cw['scaling']:.6g} reproduces the PUBLISHED "
               f"atom (lambda) {PUBLISHED_WAVEFT['atom']:g} at this width")
            ck(cw["lr"] == PUBLISHED_WAVEFT["lr"],
               f"{a}: the learning rate IS the published {PUBLISHED_WAVEFT['lr']:g}, "
               f"not a derived one")
        # ⛔ AND IT MUST BE OUTSIDE THE SEARCHED LADDERS, or it is not a REF cell --
        #   it is a duplicate. [R.258] is only a finding because the point is far out.
        UPw = {c["p_mult"] for c in _cells_of(WAVE_ALL)}
        pm = cells()[0]["p_mult"]
        ck(pm < min(UPw),
           f"CONTROL: the published P/P_ref {pm:.4g} is BELOW w1/w2's bottom rung "
           f"{min(UPw):g} -- if it were inside, these cells would be redundant")
        ck(not ({cell_id(c) for c in cs} & {cell_id(c) for c in _cells_of(WAVE_ALL)}),
           "no REF cell duplicates a searched cell (they are new budget, honestly counted)")
        ck(set(CLF_LRS) == set(W1["clf_lrs"]),
           "the REF cells use w1's classifier_lr pair, so the head is not a new variable")

    # --- the command really carries the swept values, for BOTH arms
    for arm in ARMS:
        # ⚠ cell 7 on a searched grid (deliberately not cell 0 -- a corner);
        #   a REF block has only 2 cells per arm, so take the last one it has.
        _sub = cells([arm])
        c = dict(_sub[7] if len(_sub) > 7 else _sub[-1])
        cmd = cell_cmd(c)
        s = " ".join(cmd)
        ck(f"--learning_rate {c['lr']:g}" in s, f"{arm}: lr reaches the command")
        ck(f"--classifier_lr {c['classifier_lr']:g}" in s, f"{arm}: classifier_lr reaches it")
        if c.get("scaling") is None:
            # ⛔ CONTROL, IN THE OTHER DIRECTION: a no-scale arm must acquire NO
            #   scale flag. `scora`'s whole contrast with `scora2` is that its scale
            #   is derived a priori (fir_arms: "DO NOT ADD ONE").
            ck(FA.ARM_SCALE_FLAG[arm] is None and "--slr_scaling" not in s,
               f"{arm}: NO scale flag appears (its scale is derived a priori)")
        else:
            # ⚠ format it the way _set_flag does (`:g`), or a derived scale like
            #   73.65268377433769 is looked up as its repr and never found.
            ck(f"{FA.ARM_SCALE_FLAG[arm]} {c['scaling']:g}" in s,
               f"{arm}: scaling reaches it")
        ck(s.count("--learning_rate") == 1, f"{arm}: exactly ONE --learning_rate")
        ck(s.count("--classifier_lr") == 1, f"{arm}: exactly ONE --classifier_lr")
        ck(FA.ARM_SCALE_FLAG[arm] is None
           or s.count(FA.ARM_SCALE_FLAG[arm]) == 1, f"{arm}: exactly ONE scale flag")
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
        if _G.get("published_point"):
            # ⛔ A REF BLOCK IS NOT A LADDER, so it must not be printed as one.
            c0 = cs[0]
            print(f"  ⚠ REF CELLS at {_G['published_point'].upper()}'s OWN PUBLISHED "
                  f"POINT -- a FIXED point, not a search. [R.258]")
            print(f"  published     : atom (lambda) {PUBLISHED_WAVEFT['atom']:g}   "
                  f"lr {PUBLISHED_WAVEFT['lr']:g}")
            print(f"  derived here  : {FA.ARM_SCALE_FLAG[c0['arm']]} {c0['scaling']:.6g}   "
                  f"--learning_rate {c0['lr']:g}")
            print(f"  P/P_ref       : {c0['p_mult']:.5g}  "
                  f"(P_ref = {p_ref():.7f}) -- {1/c0['p_mult']:.0f}x BELOW the "
                  f"searched ladder's bottom rung, which is why it is a REF cell")
            print(f"  classifier_lr : {CLF_LRS}")
        elif COORD == "p":
            # ⭐ Print the DERIVED lr for every cell of the plane. The swept knob is
            #   P; lr is what actually reaches the command line, and a reader who
            #   cannot see it cannot sanity-check the corners.
            pr = p_ref()
            scs = scalings()
            if NO_SCALE:
                print(f"  coordinate    : P = lr*atom  (atom = {atom_fixed():.6g} at "
                      f"{TARGETS}, FIXED -- this arm derives its scale a priori)")
            else:
                print(f"  coordinate    : P = lr*atom  "
                      f"(atom = scaling * {atom_per_scale():.6g} at {TARGETS})")
            print(f"  P/P_ref       : {P_MULTS}")
            print(f"                  P_ref = {pr:.7f} = the [R.305]/[R.306] "
                  f"roberta-base step, carried")
            if not NO_SCALE:
                lab = _G.get("scale_label", "scaling")
                print(f"  {lab:14s}: " + ", ".join(f"{x:g}" for x in scs))
            print(f"  classifier_lr : {CLF_LRS}")
            if EXTRA:
                print(f"  {EXTRA['label']:14s}: {EXTRA['values']}   "
                      f"⭐ a 4th axis: {EXTRA['flag']}")
            print()
            hdr = [f"{'sc ' + _fmt(x):>12s}" for x in scs] or [f"{'lr':>12s}"]
            print(f"  derived lr    :  {'P/P_ref':>9s}" + "".join(hdr))
            for m in P_MULTS:
                mark = "   <- P_ref (the carried roberta step)" if m == 1.0 else ""
                row = ([lr_for(m, x) for x in scs] if scs
                       else [lr_for(m, None, no_scale=True)])
                print(f"                 {m:>9g}" +
                      "".join(f"{v:>12.6g}" for v in row) + mark)
        else:
            print(f"  learning_rate : {LRS}")
            print(f"  scaling       : {scalings()}")
            print(f"  classifier_lr : {CLF_LRS}")
        print(f"  steps per cell: {steps_per_cell()}  "
              f"(mrpc train 3,668 / batch {FP.BATCH} x {EPOCHS} epochs)")
        print(f"  warmup        : {FP.warmup_for(TASK, EPOCHS)} steps (RTE's ratio, MRPC's steps)")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
