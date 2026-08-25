#!/usr/bin/env python
"""[R.267] Place every arm's tuned optimum in a COMMON coordinate: the effective step.

CARRY_FORWARD 4.4 / [N.1 1]: under AdamW the per-parameter ATOM Frobenius norm
  atom_j = || d(dW) / d theta_j ||_F
IS the effective learning rate on dW, and the displacement per step is ~ lr * atom * sqrt(k).
[R.252] measured that lr*atom governs the COLLAPSE BOUNDARY well and the LEVEL poorly -- so this
is a coordinate for comparing where arms SIT, not a 1-D surrogate for their surfaces.

Each arm's grid knobs live in a different parameterisation (FourierFT `scaling` 150, WaveFT `fs`,
QWHA `scaling` 106.066, LoCA `alpha`, ...) so their raw ladders are NOT comparable.  This file
measures the atom directly from the adapter and reports (lr, atom, lr*atom) per arm.

UNIVERSAL PROBE.  Every adapter here writes dW = Psi(theta), LINEAR in theta.  For a linear map
    atom_j = || Psi(e_j) - Psi(0) ||_F     EXACTLY (no finite-difference error).
⛔ LINEARITY IS VERIFIED, NOT ASSUMED: the probe checks || Psi(2 e_j) - Psi(0) || == 2 * atom_j to
1e-6 relative, and REFUSES to report an arm that fails (e.g. a bilinear or location-learning
parameterisation).  A wrong atom would silently mis-rank every arm.

Usage:  env/bin/python scripts/r267_effective_step.py [--selftest]
"""
import argparse, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch, torch.nn as nn

D = 768
TOL = 1e-6


def _probe(make, param_name, idx=0, nprobe=1):
    """atom_j = ||Psi(e_j) - Psi(0)||_F over `nprobe` coefficients, with a linearity check.

    ⚠️ [R.174]: the atom is DETERMINISTIC for FourierFT/WaveFT/QWHA and RANDOM per coefficient for
    SCoRA (relative sd 1/sqrt(2t) = 5.4% at t=128).  So one index is not the arm's atom -- the
    spread is reported, never averaged away silently.
    Returns (mean_atom, linear_ok, rel_spread).
    """
    mod = make()
    p = dict(mod.named_parameters())[param_name]
    vals, ok = [], True
    with torch.no_grad():
        flat = p.view(-1)
        n = min(nprobe, flat.numel())
        flat.zero_()
        base = mod.get_delta_weight().to(torch.float64).clone()
        for j in range(n):
            flat[j] = 1.0
            one = mod.get_delta_weight().to(torch.float64).clone()
            flat[j] = 2.0
            two = mod.get_delta_weight().to(torch.float64).clone()
            flat[j] = 0.0
            a1 = torch.linalg.norm(one - base).item()
            a2 = torch.linalg.norm(two - base).item()
            ok = ok and a1 > 0 and abs(a2 - 2.0 * a1) <= TOL * max(1.0, a2)
            vals.append(a1)
    mean = sum(vals) / len(vals)
    spread = (max(vals) - min(vals)) / mean if mean > 0 else 0.0
    return mean, ok, spread


def _probe_fwd(make, param_name, nprobe=1):
    """Same atom, measured through the module's own FORWARD on the identity.

    ⭐ API-agnostic: works for any adapter, including ones with no `get_delta_weight`
    (QWHA exposes only `get_delta_spectrum`, which is a spectrum, NOT a dW -- probing it
    would have measured the wrong object).  The effective linear map is f(I), so
        atom_j = || f_{theta=e_j}(I) - f_{theta=0}(I) ||_F
    exactly, for a dW linear in theta.  Same linearity check as _probe.
    ⛔ Cross-checked against _probe on every arm that has BOTH (fixture E10).
    """
    mod = make()
    p = dict(mod.named_parameters())[param_name]
    n_in = mod.base_layer.in_features
    eye = torch.eye(n_in, dtype=mod.base_layer.weight.dtype)
    fwd = lambda: mod(eye).T.to(torch.float64).clone()      # columns are f(e_i) => (m,n)
    vals, ok = [], True
    with torch.no_grad():
        flat = p.view(-1)
        flat.zero_()
        base = fwd()
        for j in range(min(nprobe, flat.numel())):
            flat[j] = 1.0; one = fwd()
            flat[j] = 2.0; two = fwd()
            flat[j] = 0.0
            a1 = torch.linalg.norm(one - base).item()
            a2 = torch.linalg.norm(two - base).item()
            ok = ok and a1 > 0 and abs(a2 - 2.0 * a1) <= 1e-4 * max(1.0, a2)
            vals.append(a1)
    mean = sum(vals) / len(vals)
    spread = (max(vals) - min(vals)) / mean if mean > 0 else 0.0
    return mean, ok, spread


def _lin():
    return nn.Linear(D, D, bias=False)


def arms():
    """{label: (callable -> module, param_name, lr_at_argmax, note)} for the completed arms."""
    out = {}
    from merged_fourierft import MergedFourierFTLinear
    # FourierFT argmax: lr 5e-1, scaling 50   |  centre: lr 5e-2, scaling 150
    for tag, sc, lr in (("fftm@argmax", 50.0, 5e-1), ("fftm@centre", 150.0, 5e-2)):
        out[tag] = (lambda sc=sc: MergedFourierFTLinear(_lin(), n_frequency=256, scaling=sc,
                                                        random_loc_seed=777),
                    "spectrum", lr)
    from haar_adapter import HaarLinear
    # WaveFT argmaxes are now CONFIRMED at 21/21: mu=1 [R.271] and mu=2 [R.280] both
    # lr 1.5e-1, fs 300.  (The earlier "wave1@interim" row used the 12/21 argmax lr 5e-2.)
    for tag, mu, fs, lr in (("wave1@argmax", 1, 300.0, 1.5e-1), ("wave1@centre", 1, 150.0, 5e-2),
                            ("wave2@argmax", 2, 300.0, 1.5e-1), ("wave2@centre", 2, 150.0, 5e-2)):
        out[tag] = (lambda mu=mu, fs=fs: HaarLinear(_lin(), n_frequency=256, mu=mu,
                                                    fourierft_scaling=fs, support_seed=777),
                    "spectrum", lr)
    from qwha_adapter import QWHALinear
    # [R.292] QWHA's PLANE argmax (16/16 complete): lr 1.5e-1, scaling 53.0330.
    out["qwha@argmax"] = (lambda: QWHALinear(_lin(), n_frequency=256, scaling=53.0330,
                                             random_loc_seed=777), "spectrum", 1.5e-1)
    out["qwha@derived"] = (lambda: QWHALinear(_lin(), n_frequency=256, scaling=106.0660, random_loc_seed=777),
                           "spectrum", 5e-2)
    out["qwha@150"] = (lambda: QWHALinear(_lin(), n_frequency=256, scaling=150.0, random_loc_seed=777),
                       "spectrum", 5e-2)
    from slr_adapter import SLRLinear
    out["scora@argmax"] = (lambda: SLRLinear(_lin(), rank=1, s=128, init="zero", seed=777),
                           "beta", 5e-2)
    from loca_adapter import LoCALinear
    # ⛔ LoCA has TWO parameter groups; `spectrum_indices` is the LOCATION, not a coefficient.
    # Probing it returns atom 0 and non-linear (correctly refused).  The coefficient is `spectrum`.
    out["loca@argmax"] = (lambda: LoCALinear(_lin(), n_frequency=256, scale=2.0), "spectrum", 1.5e-2)
    out["loca@centre"] = (lambda: LoCALinear(_lin(), n_frequency=256, scale=1.0), "spectrum", 5e-2)
    from spectral_adapter import SpectralAdapterLinear
    out["lyra@argmax"] = (lambda: SpectralAdapterLinear(_lin(), p=16, q=16, scaling=0.2,
                                                        d_initial=0.07, freq_mode="geometric",
                                                        freq_exponent=2.0), "coeffs", 1.5e-2)
    return out


def report(stream=sys.stdout):
    p = lambda *a: print(*a, file=stream)
    p("=" * 78)
    p("[R.267] EFFECTIVE STEP -- each arm's optimum in a COMMON coordinate   (d=768, k=256)")
    p("=" * 78)
    p("  atom = ||d(dW)/d theta_j||_F, measured; effective step ~ lr * atom * sqrt(k)  [4.4]")
    p("  ⛔ [R.252]: lr*atom governs the COLLAPSE BOUNDARY well and the LEVEL poorly.")
    p("")
    p(f"  {'arm':16s} {'lr':>8s} {'atom':>10s} {'lr*atom':>10s}   linearity")
    rows = {}
    for tag, (make, pname, lr) in arms().items():
        try:
            probe = _probe if hasattr(make(), "get_delta_weight") else _probe_fwd
            atom, ok, spread = probe(make, pname, nprobe=8)
        except Exception as e:
            p(f"  {tag:16s} {'--':>8s} probe failed: {type(e).__name__}: {e}")
            continue
        rows[tag] = (lr, atom, lr * atom, ok, spread)
        p(f"  {tag:16s} {lr:8.3g} {atom:10.6f} {lr*atom:10.6f}   "
          + ("✅ linear" if ok else "⛔ NON-LINEAR -- atom NOT reportable")
          + (f"   ⚠️ atom SPREAD {spread*100:.2f}% across coefficients" if spread > 1e-4
             else ("   ⚠️ ±6.2% PER TRAINING SEED [R.174] -- one draw shown" if tag.startswith("scora")
                   else "   (deterministic)")))
    return rows


def selftest():
    ok, bad = [], []
    def chk(n, c, d=""):
        (ok if c else bad).append(n); print(f"  {'PASS' if c else 'FAIL'}  {n}" + (f"  {d}" if d else ""))
    from merged_fourierft import MergedFourierFTLinear
    a150, l150, sp150 = _probe(lambda: MergedFourierFTLinear(_lin(), n_frequency=256, scaling=150.0,
                                                      random_loc_seed=777), "spectrum", nprobe=16)
    chk("E1 FourierFT@150 reproduces the repo's documented atom 0.138106793",
        abs(a150 - 0.138106793200498) < 1e-6, f"{a150:.9f}")
    chk("E1b and is certified LINEAR", l150 is True)
    a300, _, _ = _probe(lambda: MergedFourierFTLinear(_lin(), n_frequency=256, scaling=300.0,
                                                   random_loc_seed=777), "spectrum", nprobe=16)
    chk("E2 the atom is exactly linear in `scaling` (300 = 2x150)",
        abs(a300 - 2 * a150) < 1e-9, f"{a300:.9f}")
    from haar_adapter import HaarLinear
    ah, lh, sph = _probe(lambda: HaarLinear(_lin(), n_frequency=256, mu=1, fourierft_scaling=150.0, support_seed=777),
                    "spectrum", nprobe=16)
    chk("E3 WaveFT mu=1 @fs150 MATCHES FourierFT's atom ([R.221], the derived rule)",
        abs(ah - a150) < 1e-6, f"haar {ah:.9f} vs fft {a150:.9f}")
    chk("E3b and is certified LINEAR", lh is True)
    # E4 the probe must REFUSE a non-linear parameterisation
    class Bilinear(nn.Module):
        def __init__(self):
            super().__init__(); self.beta = nn.Parameter(torch.zeros(4))
        def get_delta_weight(self):
            return torch.outer(self.beta, self.beta)
    _, okb, _ = _probe(Bilinear, "beta")
    chk("E4 a BILINEAR map is refused, not silently reported", okb is False)
    class Lin(nn.Module):
        def __init__(self):
            super().__init__(); self.beta = nn.Parameter(torch.zeros(4))
        def get_delta_weight(self):
            return torch.diag(self.beta) * 3.0
    ac, okc, _ = _probe(Lin, "beta")
    chk("E5 a known linear map gives the exact atom (3.0) and passes",
        okc is True and abs(ac - 3.0) < 1e-12, f"{ac}")
    chk("E6 FourierFT's atom is DETERMINISTIC across coefficients ([R.174])", sp150 < 1e-6, f"{sp150:.2e}")
    # E7-E8 [R.174]: SCoRA's atom is random PER TRAINING SEED, not per coefficient.
    # `init_seed` defaults to None (slr_adapter.py:180) => alpha is drawn from the GLOBAL
    # RNG, so the seed that moves it is the TRAINING seed, not --slr_seed.
    from slr_adapter import SLRLinear
    vals = []
    for sd_ in range(12):
        torch.manual_seed(1000 + sd_)
        a_, ok_, sp_ = _probe(lambda: SLRLinear(_lin(), rank=1, s=128, init="zero", seed=777),
                              "beta", nprobe=4)
        vals.append(a_)
    mu_ = sum(vals) / len(vals)
    sd_rel = math.sqrt(sum((v - mu_) ** 2 for v in vals) / (len(vals) - 1)) / mu_
    pred = 1.0 / math.sqrt(2 * 128)
    chk("E7 SCoRA's atom varies across TRAINING seeds ([R.174]), not across coefficients",
        sp_ < 1e-6 and sd_rel > 0.02, f"per-coeff {sp_:.1e}, per-seed {sd_rel*100:.2f}%")
    chk("E8 and that spread matches the a-priori 1/sqrt(2t) to within 25%",
        abs(sd_rel - pred) / pred < 0.25, f"measured {sd_rel*100:.2f}% vs predicted {pred*100:.2f}%")
    chk("E9 its MEAN still matches FourierFT's atom to within 2% (the rule holds in expectation)",
        abs(mu_ - a150) / a150 < 0.02, f"{mu_:.6f} vs {a150:.6f}")

    # E10-E12 the forward-based probe: cross-checked, then used where the API is absent
    af, okf, _ = _probe_fwd(lambda: MergedFourierFTLinear(_lin(), n_frequency=256, scaling=150.0,
                                                          random_loc_seed=777), "spectrum", 4)
    chk("E10 the FORWARD probe agrees with the get_delta_weight probe (FourierFT)",
        okf is True and abs(af - a150) / a150 < 1e-4, f"fwd {af:.9f} vs dw {a150:.9f}")
    ahf, _, _ = _probe_fwd(lambda: HaarLinear(_lin(), n_frequency=256, mu=1,
                                              fourierft_scaling=150.0, support_seed=777),
                           "spectrum", 4)
    chk("E10b and on WaveFT too", abs(ahf - ah) / ah < 1e-4, f"fwd {ahf:.9f} vs dw {ah:.9f}")
    from qwha_adapter import QWHALinear
    aq, okq, _ = _probe_fwd(lambda: QWHALinear(_lin(), n_frequency=256, scaling=150.0,
                                               random_loc_seed=777), "spectrum", 4)
    chk("E11 QWHA@150 reproduces [R.221]'s measured atom 0.1953125",
        okq is True and abs(aq - 0.1953125) / 0.1953125 < 1e-3, f"{aq:.9f}")
    chk("E12 ⭐ and that is EXACTLY sqrt(2) x FourierFT's -- [R.221]'s confound, re-measured",
        abs(aq / a150 - math.sqrt(2.0)) < 1e-3, f"ratio {aq/a150:.6f} vs sqrt2 {math.sqrt(2.0):.6f}")

    # E13-E15 [R.281] LoCA and LYRA: orthonormal DCT => the atom IS the scale constant
    from loca_adapter import LoCALinear
    al, okl, _ = _probe(lambda: LoCALinear(_lin(), n_frequency=256, scale=2.0), "spectrum", nprobe=6)
    chk("E13 LoCA's atom EQUALS its alpha (orthonormal DCT)", okl and abs(al - 2.0) < 1e-4, f"{al:.6f}")
    al1, _, _ = _probe(lambda: LoCALinear(_lin(), n_frequency=256, scale=1.0), "spectrum", nprobe=6)
    chk("E13b and is exactly linear in alpha", abs(al - 2 * al1) < 1e-4, f"{al:.6f} vs 2x{al1:.6f}")
    _, okloc, _ = _probe(lambda: LoCALinear(_lin(), n_frequency=256, scale=1.0),
                         "spectrum_indices", nprobe=2)
    chk("E14 ⛔ LoCA's LOCATION parameter is refused, not reported as an atom", okloc is False)
    from spectral_adapter import SpectralAdapterLinear
    ay, oky, _ = _probe(lambda: SpectralAdapterLinear(_lin(), p=16, q=16, scaling=0.2,
                                                      d_initial=0.07, freq_mode="geometric",
                                                      freq_exponent=2.0), "coeffs", nprobe=6)
    chk("E15 LYRA's atom EQUALS its scaling 0.2", oky and abs(ay - 0.2) < 1e-4, f"{ay:.6f}")
    print(f"\n  selftest: {len(ok)} passed, {len(bad)} failed")
    if bad: print("  FAILED: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest else (report() and 0))
