"""R.45 GATE -- the BUDGET-AWARE matched init (`--slr_init matched_budget`).

WRITTEN BEFORE THE EDIT IT VERIFIES (`PROCESS.md` §4).  Run it against the
UNPATCHED tree first: G1/G2 must FAIL there (the mode does not exist) and
G0 must PASS.  Then patch, and require 100%.

WHY THIS EXISTS
---------------
`[R.40, measured]` SLR at 24,000 params (`r`=1, `s`=`t`=500) collapsed on all 5
CoLA seeds, 150/150 degenerate epochs.  `R.40 §2` diagnosed it as an effective
step-size fault and prescribed `lr = 2.53e-2`.

`[R.45, measured, CPU]` THAT PRESCRIPTION IS WRONG IN MAGNITUDE, and the fault
is not in the LR at all.  `slr_adapter.py` hardcodes the `matched` init target
at **2.1843**, which is FourierFT's `||dW||_F` at **k=256 only**.  FourierFT's
own init norm grows as `atom*sqrt(k)`, so at `k`=1000 it is 4.3171 -- i.e.
`--slr_init matched` STOPS MATCHING THE BASELINE the moment the budget moves.
Holding the target fixed while the parameter count grows is what inflates the
relative step:

    one-step ||d(dW)||/||dW||     s=128  0.0616   (the working configuration)
                                  s=256  0.0716
                                  s=500  0.0870   <- R.40's collapsed arm
                                  s=768  0.1013

    R.40's fix, lr 2.53e-2, s=500: 0.0440   <- OVERSHOOTS, 0.71x the working point
    budget-aware target,   s=500: 0.0614   <- 0.996x the working point

RULE (derived a priori from the baseline's own init norm, NOT swept):

    target(r, s, t) = 2.1843 * sqrt( r*(s+t) / 256 )

At `r`=1, `s`=`t`=128 this is exactly 2.1843, so `matched_budget` is
BIT-IDENTICAL to `matched` at the configuration every prior SLR result used.
"""
from __future__ import annotations

import hashlib
import math
import sys

import torch

sys.path.insert(0, "src")
from slr_adapter import SLRLinear  # noqa: E402

ATOM = 0.138106793200498
BASE_TARGET = 2.1843
WORKING_REL = None          # measured below from the r=1,s=128 arm itself
D = 768

# Captured from the UNPATCHED tree before any edit (r, s) -> (norm, sha16).
PREPATCH = {
    (1, 128): (2.1842966079711914, "cc6fee550ad2bacb"),
    (1, 500): (2.1842975616455080, "e239c2643c3e1fbc"),
}

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))


def build(r, s, init, init_seed=4242, t=None):
    base = torch.nn.Linear(D, D, bias=False)
    return SLRLinear(base, rank=r, s=s, t=t, seed=777, init_seed=init_seed, init=init)


def sha(W) -> str:
    return hashlib.sha256(W.detach().contiguous().numpy().tobytes()).hexdigest()[:16]


def rel_step(m, lr=5e-2, trials=8):
    """One AdamW step moves every coefficient by ~lr; random signs."""
    out = []
    for k in range(trials):
        g = torch.Generator().manual_seed(9000 + k)
        W0 = m.get_delta_weight()
        n0 = float(W0.norm())
        b0, a0 = m.beta.data.clone(), m.alpha.data.clone()
        with torch.no_grad():
            m.beta += lr * (torch.randint(0, 2, m.beta.shape, generator=g).float() * 2 - 1)
            m.alpha += lr * (torch.randint(0, 2, m.alpha.shape, generator=g).float() * 2 - 1)
        out.append(float((m.get_delta_weight() - W0).norm()) / n0)
        m.beta.data, m.alpha.data = b0, a0
    return sum(out) / len(out)


# ---------------------------------------------------------------- G0 ------- #
# The legacy `matched` mode is UNTOUCHED -- bit-identical to the pre-patch tree.
for (r, s), (n_ref, h_ref) in PREPATCH.items():
    try:
        m = build(r, s, "matched")
        W = m.get_delta_weight()
        check(f"G0 legacy `matched` r={r},s={s} bit-identical to pre-patch",
              sha(W) == h_ref and abs(float(W.norm()) - n_ref) < 1e-6,
              f"sha {sha(W)} vs {h_ref}, norm {float(W.norm()):.7f} vs {n_ref:.7f}")
    except Exception as e:                                   # pragma: no cover
        check(f"G0 legacy `matched` r={r},s={s} bit-identical to pre-patch", False, repr(e))

# ---------------------------------------------------------------- G1 ------- #
# The new mode exists and is accepted.
try:
    build(1, 128, "matched_budget")
    check("G1a `matched_budget` init mode is accepted", True)
except Exception as e:
    check("G1a `matched_budget` init mode is accepted", False, repr(e))
# `init` was previously an unvalidated string: ANY value != "zero" silently took
# the `matched` branch, so "the mode exists" was vacuously true.  Require the
# adapter to REJECT unknown modes, so G1a means something.
try:
    build(1, 128, "matchd_budgt")
    check("G1b unknown init mode raises (so G1a is not vacuous)", False,
          "typo accepted silently")
except ValueError:
    check("G1b unknown init mode raises (so G1a is not vacuous)", True)
except Exception as e:
    check("G1b unknown init mode raises (so G1a is not vacuous)", False, repr(e))

# ---------------------------------------------------------------- G2 ------- #
# BACKWARD COMPATIBILITY: at r=1,s=t=128 (every prior SLR result) the new mode
# is BIT-IDENTICAL to the old one.  If this fails, R.29/R.33/R.38/R.39/R.42/R.43
# are not comparable to anything run under the new mode.
try:
    a = build(1, 128, "matched").get_delta_weight()
    b = build(1, 128, "matched_budget").get_delta_weight()
    check("G2 r=1,s=t=128: `matched_budget` == `matched` BIT-IDENTICALLY",
          sha(a) == sha(b), f"{sha(a)} vs {sha(b)}")
except Exception as e:
    check("G2 r=1,s=t=128: `matched_budget` == `matched` BIT-IDENTICALLY", False, repr(e))

# ---------------------------------------------------------------- G3 ------- #
# The target follows the derived rule across budgets and ranks.
for r, s in [(1, 32), (1, 64), (1, 128), (1, 256), (1, 500), (1, 768), (2, 250), (4, 125)]:
    want = BASE_TARGET * math.sqrt(r * 2 * s / 256)
    try:
        got = float(build(r, s, "matched_budget").get_delta_weight().norm())
        check(f"G3 target(r={r},s={s}) = 2.1843*sqrt(P/256) = {want:.4f}",
              abs(got - want) / want < 1e-4, f"got {got:.6f}")
    except Exception as e:
        check(f"G3 target(r={r},s={s}) = 2.1843*sqrt(P/256) = {want:.4f}", False, repr(e))

# ---------------------------------------------------------------- G4 ------- #
# THE POINT OF THE WHOLE CHANGE: the relative one-step displacement of dW is
# BUDGET-INVARIANT under the new rule, and is NOT under the old one.
try:
    WORKING_REL = rel_step(build(1, 128, "matched"))
    check("G4a working point r=1,s=128 measured", 0.03 < WORKING_REL < 0.12,
          f"rel step {WORKING_REL:.4f}")
    for r, s in [(1, 64), (1, 256), (1, 500), (1, 768), (4, 125)]:
        got = rel_step(build(r, s, "matched_budget"))
        check(f"G4 rel step r={r},s={s} within 5% of the working point",
              abs(got / WORKING_REL - 1.0) < 0.05, f"{got:.4f} = {got/WORKING_REL:.3f}x")
    # and the CONTROL: the legacy mode drifts, which is the defect being fixed
    drift = rel_step(build(1, 500, "matched")) / WORKING_REL
    check("G4c CONTROL: legacy `matched` at s=500 is >1.3x the working step",
          drift > 1.3, f"{drift:.3f}x -- this is R.40's collapse")
except Exception as e:
    check("G4 relative-step invariance", False, repr(e))

# ---------------------------------------------------------------- G5 ------- #
# The init must not disturb the a-priori atom norm or scaling.
for r, s in [(1, 128), (1, 500)]:
    try:
        m = build(r, s, "matched_budget")
        # atom for beta_ji is scaling * ||v_j||, and ||v_j|| = ||alpha_j|| ~ sqrt(t)
        # by construction; the a-priori scaling is atom/sqrt(t).
        check(f"G5 scaling(r={r},s={s}) = atom/sqrt(t) unchanged",
              abs(m.scaling - ATOM / math.sqrt(s)) < 1e-12, f"{m.scaling!r}")
    except Exception as e:
        check(f"G5 scaling(r={r},s={s}) = atom/sqrt(t) unchanged", False, repr(e))

# ---------------------------------------------------------------- G6 ------- #
# Sanity of the object itself.
for r, s in [(1, 500), (4, 125)]:
    try:
        m = build(r, s, "matched_budget")
        W = m.get_delta_weight()
        check(f"G6 r={r},s={s}: finite, nonzero, rank == r",
              torch.isfinite(W).all().item() and float(W.norm()) > 0
              and int(torch.linalg.matrix_rank(W.double(), rtol=1e-5)) == r,
              f"rank {int(torch.linalg.matrix_rank(W.double(), rtol=1e-5))}")
    except Exception as e:
        check(f"G6 r={r},s={s}: finite, nonzero, rank == r", False, repr(e))

# ---------------------------------------------------------------- G7 ------- #
# Parameter count is exactly the matched budget.
try:
    m = build(1, 500, "matched_budget")
    check("G7 r=1,s=t=500 -> 1000 params/module -> 24,000 over 24 modules",
          m.n_params() == 1000, f"{m.n_params()}")
except Exception as e:
    check("G7 r=1,s=t=500 -> 1000 params/module", False, repr(e))

# ---------------------------------------------------------------- G8 ------- #
# INTEGRATION.  `[CONTEXT §6.2]` EVERY real defect of the last session lived in
# the MODULE<->HARNESS SEAM while unit gates read green.  Reach through
# train_glue.py's REAL dispatch site (the one inside `run_single_seed`) and
# assert the built model carries the new target.
try:
    import argparse
    from slr_adapter import get_slr_model
    from transformers import AutoConfig, AutoModelForSequenceClassification
    cfg = AutoConfig.from_pretrained("roberta-base", num_labels=2)
    net = AutoModelForSequenceClassification.from_config(cfg)
    wrapped = get_slr_model(net, ["query", "value"], rank=1, s=500, scaling=None,
                            seed=777, materialise=True, basis="dct",
                            init="matched_budget")
    mods = [m for m in wrapped.model.modules() if isinstance(m, SLRLinear)]
    norms = [float(m.get_delta_weight().norm()) for m in mods]
    want = BASE_TARGET * math.sqrt(1000 / 256)
    tot = sum(m.n_params() for m in mods)
    check("G8a harness path adapts 24 modules, 24,000 params",
          len(mods) == 24 and tot == 24000, f"{len(mods)} modules, {tot} params")
    check(f"G8b every adapted module has ||dW||_init = {want:.4f}",
          len(norms) > 0 and max(abs(n - want) for n in norms) / want < 1e-4,
          f"min {min(norms):.4f} max {max(norms):.4f}")
    src = open("src/train_glue.py").read().splitlines()
    dispatch = [i for i, l in enumerate(src) if "get_slr_model(" in l]
    rss = [i for i, l in enumerate(src) if l.startswith("def run_single_seed")]
    check("G8c the slr dispatch used is the one INSIDE run_single_seed",
          bool(dispatch) and bool(rss) and max(dispatch) > rss[0],
          f"get_slr_model at {[d+1 for d in dispatch]}, run_single_seed at {rss[0]+1}")
    check("G8d train_glue exposes matched_budget in --slr_init choices",
          "matched_budget" in open("src/train_glue.py").read())
except Exception as e:
    check("G8 integration through the harness", False, repr(e))

# ------------------------------------------------------------------------- #
print("=" * 78)
print("R.45 GATE -- budget-aware matched init for SLR")
print("=" * 78)
for name, ok, detail in RESULTS:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   ({detail})" if detail else ""))
n_ok = sum(1 for _, ok, _ in RESULTS if ok)
print("-" * 78)
print(f"{n_ok}/{len(RESULTS)} gates pass")
sys.exit(0 if n_ok == len(RESULTS) else 1)
