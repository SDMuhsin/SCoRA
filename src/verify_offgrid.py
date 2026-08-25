"""Gate suite for `src/offgrid_adapter.py`  (Phase R.1).

Run:  env/bin/python src/verify_offgrid.py

Every gate is written BEFORE any accuracy number exists (PROCESS 1.1, 6).
The load-bearing ones are:

  G5  FourierFT is EXACTLY this module at integer locations, phi=0.  That is
      what makes the frozen control the closest generic control rather than a
      different method (PROCESS 5 test 8).
  G6  the atom norm, MEASURED by autograd and checked against the a-priori
      closed form -- the J.6 bug (7.24x effective LR, invisible to every
      static probe) in its new costume, now with FOUR parameter kinds.
  G7  rank, with an EXPLICIT relative threshold.  torch.linalg.matrix_rank's
      default tolerance reported 766 for a rank-103 matrix twice in Phase Q.
  G10 analytic vs finite-difference d/du -- if off-grid gradients were wrong
      the whole construction would be a no-op that still trains.
"""

from __future__ import annotations

import math
import sys

import torch
from torch.overrides import TorchFunctionMode

sys.path.insert(0, "src")
from offgrid_adapter import GAMMA, OffGridLinear, wrap_unit  # noqa: E402

PASS, FAIL = 0, 0
SHAPES = [(768, 768), (768, 3072), (255, 129), (64, 64)]


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name}  {detail}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def mk(m, n, K=8, seed=0, train_locations=True, dtype=torch.float64,
       scaling=None, fourierft_scaling=150.0):
    base = torch.nn.Linear(n, m, bias=False).to(dtype)
    return OffGridLinear(base, n_atoms=K, seed=seed,
                         fourierft_scaling=fourierft_scaling,
                         scaling=scaling, train_locations=train_locations)


# --------------------------------------------------------------------------- #
def g1_factorisation_exact():
    print("\nG1  factored dW == the defining sum (fp64)")
    for (m, n) in SHAPES:
        L = mk(m, n, K=8)
        a = L.get_delta_weight(torch.float64)
        b = L.get_delta_weight_naive(torch.float64)
        rel = float((a - b).norm() / b.norm())
        check(f"G1 {m}x{n}", rel < 1e-12, f"rel={rel:.3e}")


def g2_wrap_is_gradient_transparent():
    print("\nG2  wrap_unit is exact and gradient-transparent")
    t = torch.linspace(-40.3, 51.7, 97, dtype=torch.float64, requires_grad=True)
    w = wrap_unit(t)
    check("G2 range", bool((w.abs() <= 0.5 + 1e-12).all()),
          f"max|w|={float(w.abs().max()):.4f}")
    check("G2 congruent", float((torch.cos(2 * math.pi * w)
                                 - torch.cos(2 * math.pi * t)).abs().max()) < 1e-11)
    w.sum().backward()
    check("G2 dwrap/dt == 1", float((t.grad - 1).abs().max()) == 0.0)


def g3_forward_matches_materialised():
    print("\nG3  delta_apply(x) == x @ dW^T")
    for (m, n) in SHAPES:
        L = mk(m, n, K=8)
        x = torch.randn(11, n, dtype=torch.float64)
        y = L.delta_apply(x)
        ref = x @ L.get_delta_weight(torch.float64).T
        rel = float((y - ref).norm() / ref.norm())
        check(f"G3 {m}x{n}", rel < 1e-12, f"rel={rel:.3e}")


class _ShapeSpy(TorchFunctionMode):
    def __init__(self):
        self.worst = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in (out if isinstance(out, (tuple, list)) else [out]):
            if isinstance(t, torch.Tensor):
                self.worst = max(self.worst, t.numel())
        return out


def g4_no_mn_materialisation():
    print("\nG4  no m x n tensor is allocated anywhere in the forward path")
    for (m, n) in [(2048, 2048), (768, 3072)]:
        L = mk(m, n, K=8, dtype=torch.float32)
        x = torch.randn(4, n)
        spy = _ShapeSpy()
        with spy:
            L.delta_apply(x)
        check(f"G4 {m}x{n}", spy.worst < m * n,
              f"largest tensor {spy.worst:,} < m*n {m * n:,}")


def g5_fourierft_is_the_integer_special_case():
    print("\nG5  FourierFT == this module at INTEGER locations with phi=0")
    for (m, n) in SHAPES:
        K = 8
        L = mk(m, n, K=K)
        g = torch.Generator().manual_seed(5)
        p = torch.randint(0, m, (K,), generator=g)
        q = torch.randint(0, n, (K,), generator=g)
        coef = torch.randn(K, generator=g, dtype=torch.float64)
        with torch.no_grad():
            L.c.copy_(coef)
            L.u.copy_(p.to(torch.float64) / GAMMA)
            L.v.copy_(q.to(torch.float64) / GAMMA)
            L.phi.zero_()
        mine = L.get_delta_weight(torch.float64)
        # peft/tuners/fourierft/layer.py verbatim
        S = torch.zeros(m, n, dtype=torch.float64)
        S.index_put_((p, q), coef, accumulate=True)
        ref = torch.fft.ifft2(S).real * L.fourierft_scaling
        rel = float((mine - ref).norm() / ref.norm())
        check(f"G5 {m}x{n}", rel < 1e-10, f"rel={rel:.3e}")


def g6_atom_norm():
    print("\nG6  atom Frobenius norm: MEASURED vs a-priori closed form")
    m = n = 768
    L = mk(m, n, K=16, seed=3)
    target = 150.0 / math.sqrt(2.0 * m * n)          # FourierFT's, exactly
    print(f"      FourierFT bar = {target:.15f}")
    for which in ("c", "u", "v", "phi"):
        meas, closed = [], []
        for j in range(6):
            meas.append(L.atom_frobenius(j, which))
            closed.append(L.atom_frobenius_closed_form(j, which))
        rel = max(abs(a - b) / b for a, b in zip(meas, closed))
        check(f"G6 {which} measured==closed", rel < 2e-3, f"max rel={rel:.3e}")
    # the amplitude atom must hit FourierFT's bar to the digit
    ac = [L.atom_frobenius(j, "c") for j in range(16)]
    relbar = max(abs(a - target) / target for a in ac)
    spread = (max(ac) - min(ac)) / target
    check("G6 c-atom == FourierFT bar", relbar < 2e-3,
          f"max rel dev={relbar:.3e}, spread={spread:.3e}")
    # and the OTHER three must be within a few % of it once |c| is divided out,
    # i.e. GAMMA equalises the effective LR across parameter kinds a priori.
    ratios = {}
    for which in ("u", "v", "phi"):
        r = [L.atom_frobenius(j, which) / (abs(float(L.c[j])) * target)
             for j in range(16)]
        ratios[which] = (min(r), max(r))
    ok = all(0.97 < lo and hi < 1.03 for lo, hi in ratios.values())
    check("G6 GAMMA equalises effective LR", ok,
          " ".join(f"{k}:[{lo:.4f},{hi:.4f}]" for k, (lo, hi) in ratios.items()))


def g7_rank():
    print("\nG7  rank(dW) == 2K exactly (EXPLICIT relative threshold, 1e-5)")
    for K in (4, 16, 64):
        L = mk(768, 768, K=K, seed=7)
        dW = L.get_delta_weight(torch.float64)
        sv = torch.linalg.svdvals(dW)
        r = int((sv > sv[0] * 1e-5).sum())
        cliff = float(sv[2 * K - 1] / sv[2 * K]) if 2 * K < len(sv) else float("inf")
        check(f"G7 K={K}", r == 2 * K, f"rank={r} expected={2 * K}, "
                                       f"singular cliff={cliff:.3e}")
        # the default tolerance is the Phase Q trap; show it disagrees
        rdef = int(torch.linalg.matrix_rank(dW))
        print(f"        (default matrix_rank tolerance would say {rdef})")


def g8_param_counts():
    print("\nG8  trainable scalars: 4K adaptive / K frozen")
    for K in (16, 64, 256):
        a = mk(768, 768, K=K, train_locations=True)
        f = mk(768, 768, K=K, train_locations=False)
        na = sum(p.numel() for n_, p in a.named_parameters()
                 if p.requires_grad and n_ in ("c", "u", "v", "phi"))
        nf = sum(p.numel() for n_, p in f.named_parameters()
                 if p.requires_grad and n_ in ("c", "u", "v", "phi"))
        check(f"G8 K={K}", na == 4 * K and nf == K, f"adaptive={na} frozen={nf}")


def g9_gradient_flow():
    print("\nG9  gradients reach the locations iff train_locations")
    for tl in (True, False):
        L = mk(768, 768, K=8, train_locations=tl)
        x = torch.randn(5, 768, dtype=torch.float64)
        L.delta_apply(x).pow(2).sum().backward()
        gu = L.u.grad
        got = (gu is not None) and float(gu.abs().max()) > 0
        check(f"G9 train_locations={tl}", got == tl,
              f"u.grad={'nonzero' if got else 'none'}; "
              f"c.grad={'nonzero' if float(L.c.grad.abs().max()) > 0 else 'ZERO'}")


def g10_analytic_vs_finite_difference():
    print("\nG10 analytic d/du == central finite difference (the off-grid check)")
    L = mk(768, 768, K=8, seed=11)
    x = torch.randn(3, 768, dtype=torch.float64)

    def loss():
        return L.delta_apply(x).pow(2).sum()

    loss().backward()
    # h = 1e-4.  MIS-SET FIRST TIME, recorded rather than quietly changed: at
    # h=1e-6 this gate read rel=1.4e-6 and FAILED its own 1e-6 bar.  Sweeping h
    # showed the error grows as h SHRINKS -- 7.6e-9 / 3.9e-8 / 3.8e-7 / 5.5e-6
    # at h = 1e-4 / 1e-5 / 1e-6 / 1e-7 -- i.e. 1/h roundoff, not truncation.
    # The analytic derivative was right; the probe was in the roundoff regime.
    for which in ("u", "v", "phi", "c"):
        par = getattr(L, which)
        ana = par.grad.clone()
        h, fd = 1e-4, torch.zeros_like(ana)
        for j in range(len(ana)):
            with torch.no_grad():
                par[j] += h
            lp = float(loss())
            with torch.no_grad():
                par[j] -= 2 * h
            lm = float(loss())
            with torch.no_grad():
                par[j] += h
            fd[j] = (lp - lm) / (2 * h)
        rel = float((ana - fd).norm() / fd.norm())
        check(f"G10 d/d{which}", rel < 1e-6, f"rel={rel:.3e}")


def g11_train_eval_identical():
    print("\nG11 deterministic; train() and eval() give identical output")
    L = mk(768, 768, K=8, dtype=torch.float32)
    x = torch.randn(7, 768)
    L.train(); a = L(x); a2 = L(x)
    L.eval(); b = L(x)
    check("G11 deterministic", float((a - a2).abs().max()) == 0.0)
    check("G11 train==eval", float((a - b).abs().max()) == 0.0)


def g12_report_pr_and_rank():
    print("\nG12 REPORTED (not a gate): PR/d^2 and rank vs the bars")
    print("      bars [CARRY_FORWARD 4.2]: FourierFT PR/d^2 0.33433, SparseFT 0.00060")
    for K in (64, 256):
        L = mk(768, 768, K=K, seed=13)
        dW = L.get_delta_weight(torch.float64)
        pr = float((dW.pow(2).sum() ** 2) / (768 ** 2 * dW.pow(4).sum()))
        sv = torch.linalg.svdvals(dW)
        sr = float(sv.pow(2).sum() / sv[0] ** 2)
        par = 4 * K
        print(f"      K={K:4d}  params/mod={par:4d}  rank<={2 * K:4d}  "
              f"stable_rank={sr:7.2f}  PR/d^2={pr:.5f}  "
              f"rank/param={2 * K / par:.3f}")


def g13_drift_instrument():
    print("\nG13 location_drift (P3's instrument) reads slots, aliasing-proof")
    L = mk(768, 768, K=8, seed=1)
    d0 = L.location_drift()
    check("G13 zero at init", all(v == 0.0 for v in d0.values()), str(d0))
    with torch.no_grad():
        L.u += 3.0 / GAMMA          # exactly 3 grid slots
        L.v += 0.25 / GAMMA
    d = L.location_drift()
    check("G13 reads slots", abs(d["u_median"] - 3.0) < 1e-9
          and abs(d["v_median"] - 0.25) < 1e-9,
          f"u={d['u_median']:.9f} v={d['v_median']:.9f}")
    with torch.no_grad():
        L.u += 768.0 / GAMMA        # a full period -- the SAME atom
    check("G13 aliasing-proof", abs(L.location_drift()["u_median"] - 3.0) < 1e-6,
          f"{L.location_drift()['u_median']:.6f} (a naive diff would say 771)")
    F = mk(768, 768, K=8, seed=1, train_locations=False)
    check("G13 frozen arm null", all(v == 0.0 for v in F.location_drift().values()))


def g15_weight_decay_must_not_move_the_frequencies():
    """R.0 5d: AdamW's DECOUPLED weight decay drags the stored u_tilde (which is
    large, ~1400) toward 0 = DC at 0.193 slots/step, 14x the gradient's 0.0138.
    Unfixed, the adaptive arm measures a wd-driven migration to low frequency
    rather than adaptivity.  Locations and phase must be in a wd=0 param group."""
    print("\nG15 weight decay must NOT drive the locations (the R.0 5d confound)")
    grad_rate = GAMMA * 5e-2 * 20          # 20 steps of pure gradient, upper bound
    out = {}
    for wd in (0.01, 0.0):
        torch.manual_seed(0)
        L = mk(768, 768, K=64, seed=101, dtype=torch.float32)
        opt = torch.optim.AdamW([L.c, L.u, L.v, L.phi], lr=5e-2, weight_decay=wd)
        x = torch.randn(8, 768)
        for _ in range(20):
            opt.zero_grad(); L.delta_apply(x).pow(2).sum().backward(); opt.step()
        out[wd] = L.location_drift()["u_median"]
    check("G15 wd=0.01 SWAMPS the gradient (the bug)", out[0.01] > 4 * grad_rate,
          f"{out[0.01]:.3f} slots vs gradient bound {grad_rate:.3f}")
    check("G15 wd=0.0 stays within the gradient bound (the fix)",
          out[0.0] <= grad_rate, f"{out[0.0]:.3f} <= {grad_rate:.3f}")
    print(f"        => train_glue.py MUST place u,v,phi in a weight_decay=0.0 group")


def g14_budget_on_the_real_model():
    """Opt-in (--with-model): the budget must land EXACTLY on Q.6's arms."""
    print("\nG14 budget on roberta-base, target query,value (needs the model)")
    import transformers
    from offgrid_adapter import get_offgrid_adapter_model
    want = {(64, True): (6144, 598274), (256, False): (6144, 598274),
            (64, False): (1536, 593666)}
    for (K, tl), (wad, wtr) in want.items():
        m = transformers.AutoModelForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=2, torch_dtype=torch.float32)
        mm = get_offgrid_adapter_model(m, ["query", "value"], n_atoms=K,
                                       seed=101, train_locations=tl)
        ad = mm.get_adapter_params()
        tr = sum(p.numel() for p in mm.parameters() if p.requires_grad)
        nm = len([x for x in mm.modules() if isinstance(x, OffGridLinear)])
        check(f"G14 K={K} train_loc={tl}", ad == wad and tr == wtr and nm == 24,
              f"{nm} modules, adapter={ad:,}, trainable={tr:,}")
        del m, mm


def main():
    if "--with-model" in sys.argv:
        g14_budget_on_the_real_model()
    for fn in (g1_factorisation_exact, g2_wrap_is_gradient_transparent,
               g3_forward_matches_materialised, g4_no_mn_materialisation,
               g5_fourierft_is_the_integer_special_case, g6_atom_norm,
               g7_rank, g8_param_counts, g9_gradient_flow,
               g10_analytic_vs_finite_difference, g11_train_eval_identical,
               g13_drift_instrument, g15_weight_decay_must_not_move_the_frequencies,
               g12_report_pr_and_rank):
        fn()
    print(f"\n==== {PASS} passed, {FAIL} failed ====")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
