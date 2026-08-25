#!/usr/bin/env python
"""
Q.1 gate suite for `--spectral_freq_mode random_subset` + `--spectral_freq_seed`.

Two duties:
  (A) REGRESSION -- prove the edit did not move the deployed LYRA path.  The
      reference hashes in `scratchpad/phaseQ/golden_pre_edit.json` were captured
      from the pre-edit file (PROCESS.md 4: verify an edit with an instrument
      written BEFORE the edit; `git diff` is not evidence here because llmdocs/
      and scratchpad/ are gitignored and this file is untracked).
  (B) CORRECTNESS -- the new mode is a one-knob change: same band, same count,
      orthonormal rows, hence atom norm identical a priori with spread zero
      (CARRY_FORWARD.md 4.4), and it is actually threaded to the layer (the
      positional-constructor trap that already bit once in this file).

Run:  env/bin/python src/verify_freq_subset.py
"""
import hashlib
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_adapter import (  # noqa: E402
    _dct_basis_at_indices,
    _generate_freq_indices,
    SpectralAdapterLinear,
    get_spectral_adapter_model,
)

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "scratchpad", "phaseQ", "golden_pre_edit.json")
D, Q, P = 768, 16, 16
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def _sha(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def _layer(**kw):
    torch.manual_seed(0)
    base = nn.Linear(D, D, bias=True)
    lay = SpectralAdapterLinear(base, p=P, q=Q, scaling=0.2, dropout=0.0,
                                d_initial=0.07, basis_seed=777, **kw)
    lay.eval()
    return lay


def main():
    print("Q.1 -- random_subset frequency mode, gate suite\n")
    gold = json.load(open(GOLDEN))

    # ---- (A) regression: the pre-existing paths are bit-identical -----------
    print("A. Regression against the pre-edit golden reference")
    for tag, kw in [("lyra_exp3", dict(freq_mode="geometric", freq_exponent=3.0, basis="dct")),
                    ("contig", dict(freq_mode="contiguous", freq_exponent=2.0, basis="dct")),
                    ("rand_orth", dict(freq_mode="geometric", freq_exponent=3.0, basis="random"))]:
        lay = _layer(**kw)
        torch.manual_seed(1)
        x = torch.randn(4, 7, D)
        with torch.no_grad():
            y = lay(x)
        g = gold["golden"][tag]
        check(f"{tag}: dct_in bit-identical", _sha(lay.dct_in) == g["dct_in_sha"])
        check(f"{tag}: dct_out bit-identical", _sha(lay.dct_out) == g["dct_out_sha"])
        check(f"{tag}: forward bit-identical", _sha(y) == g["y_sha"],
              f"y_sum={float(y.sum()):.6f} vs {g['y_sum']:.6f}")

    for key, expected in gold["freq_sets"].items():
        mode, exp = key.split(":")
        got = _generate_freq_indices(d=D, k=Q, mode=mode, exponent=float(exp))
        check(f"freq set unchanged: {key}", got == expected, f"{got[:5]}...")

    # ---- (B) the new mode ---------------------------------------------------
    print("\nB. random_subset properties")
    half = D // 2
    sets = {s: _generate_freq_indices(d=D, k=Q, mode="random_subset", freq_seed=s)
            for s in (101, 202, 303)}
    for s, f in sets.items():
        check(f"seed {s}: {Q} distinct indices", len(f) == Q and len(set(f)) == Q, str(f))
        check(f"seed {s}: inside the geometric band [0,{half}]",
              min(f) >= 0 and max(f) <= half)
        check(f"seed {s}: sorted ascending", f == sorted(f))
    check("distinct seeds give distinct sets",
          len({tuple(v) for v in sets.values()}) == 3)
    check("deterministic across calls",
          _generate_freq_indices(d=D, k=Q, mode="random_subset", freq_seed=101) == sets[101])

    # independence from global RNG state: the deployed set must not depend on
    # the training seed (PROCESS.md 2.7 -- measure the object that ships).
    torch.manual_seed(12345)
    a = _generate_freq_indices(d=D, k=Q, mode="random_subset", freq_seed=101)
    torch.manual_seed(999)
    b = _generate_freq_indices(d=D, k=Q, mode="random_subset", freq_seed=101)
    check("independent of global RNG state", a == b == sets[101])

    # ---- (C) the arms differ in EXACTLY one property ------------------------
    print("\nC. One-knob equivalence with the incumbent")
    inc = _layer(freq_mode="geometric", freq_exponent=3.0, basis="dct")
    arms = {"exp3.0": inc}
    for s in (101, 202, 303):
        arms[f"rand-{s}"] = _layer(freq_mode="random_subset", freq_exponent=3.0,
                                   basis="dct", freq_seed=s)
    for tag, lay in arms.items():
        for side, C in (("in", lay.dct_in), ("out", lay.dct_out)):
            err = (C @ C.T - torch.eye(C.shape[0])).abs().max().item()
            check(f"{tag}: C_{side} rows orthonormal", err < 1e-5, f"max|CCt-I|={err:.2e}")
        check(f"{tag}: trainable params = {P*Q}",
              sum(p.numel() for p in lay.parameters() if p.requires_grad) == P * Q)
        # atom norm = ||d(dW)/dS_ij||_F = scaling * ||c_out_i|| * ||c_in_j|| = scaling
        # Analytically the atom norm is `scaling` IDENTICALLY for every arm
        # (orthonormal rows => ||c_out_i (x) c_in_j||_F = 1), so the spread is
        # exactly zero in exact arithmetic.  The deployed buffers are float32,
        # where rounding leaves ~5e-5 RELATIVE spread; an absolute 1e-6 gate on
        # the sd sits below float32 epsilon here and is unsatisfiable.  Gated
        # both ways: fp32 to a float32-defensible relative tolerance, fp64 to
        # 1e-12 to show the exactness is analytic and not a tuned tolerance.
        atoms = torch.tensor([[(torch.outer(lay.dct_out[i], lay.dct_in[j]).norm()
                                * lay.scaling).item() for j in range(Q)] for i in range(P)],
                             dtype=torch.float64)
        rel = (atoms.std() / atoms.mean()).item()
        check(f"{tag}: atom norm == scaling, fp32 relative spread < 1e-4",
              abs(atoms.mean().item() - 0.2) < 1e-5 and rel < 1e-4,
              f"mean={atoms.mean():.9f} rel_sd={rel:.2e}")
        idx = getattr(lay, "freq_in_indices", list(range(Q)))
        C64 = _dct_basis_at_indices(D, idx, torch.float64)
        a64 = torch.tensor([[(torch.outer(C64[i], C64[j]).norm() * 0.2).item()
                             for j in range(Q)] for i in range(P)], dtype=torch.float64)
        check(f"{tag}: spread is exactly zero when the basis is built in fp64",
              (a64.std() / a64.mean()).item() < 1e-12,
              f"fp64 rel_sd={(a64.std()/a64.mean()).item():.2e}")
        r = torch.linalg.matrix_rank(lay.get_delta_weight()).item()
        check(f"{tag}: rank(dW) <= {Q}", r <= Q, f"rank={r}")

    check("random arms differ from the incumbent basis",
          all(_sha(arms[f"rand-{s}"].dct_in) != _sha(inc.dct_in) for s in (101, 202, 303)))

    # ---- (D) threading: model factory -> layer ------------------------------
    print("\nD. Threading through the model factory (the positional-call trap)")
    inner = nn.Sequential()
    inner.add_module("query", nn.Linear(D, D))
    inner.add_module("value", nn.Linear(D, D))
    wrapper = nn.Module()
    wrapper.body = inner
    wrapper.classifier = nn.Linear(D, 2)
    m = get_spectral_adapter_model(wrapper, ["query", "value"], p=P, q=Q, scaling=0.2,
                                   d_initial=0.07, freq_mode="random_subset",
                                   freq_exponent=3.0, freq_seed=202)
    layers = [mod for mod in m.modules() if isinstance(mod, SpectralAdapterLinear)]
    check("factory adapted both modules", len(layers) == 2, f"{len(layers)}")
    check("freq_seed reached every layer",
          all(getattr(l, "freq_seed", None) == 202 for l in layers))
    check("layer basis matches the generator's set",
          all(l.freq_in_indices == sets[202] for l in layers))

    # ---- (E) harness surface ------------------------------------------------
    print("\nE. Harness surface (argparse + results row)")
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_glue.py")).read()
    check("--spectral_freq_seed exists", '"--spectral_freq_seed"' in src)
    check("random_subset is an accepted choice", '"random_subset"' in src)
    check("freq_seed passed to the factory", "freq_seed=args.spectral_freq_seed" in src)
    for col in ("spectral_freq_seed", "spectral_basis", "spectral_basis_seed"):
        check(f"results row records {col} (PROCESS 1.5c)",
              src.count(f'"{col}"') >= 2, f"{src.count(chr(34)+col+chr(34))} refs")

    npass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{'='*60}\n{npass}/{len(RESULTS)} gates passed")
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
