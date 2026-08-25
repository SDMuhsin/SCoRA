#!/usr/bin/env python
"""Q.8 gate suite: the product-set support arm of merged FourierFT.

(A) REGRESSION -- the default (scattered) path is bit-identical to the golden
    reference captured BEFORE the edit (scratchpad/phaseQ/golden_merged_pre_edit.json).
(B) ONE KNOB -- product vs scattered must differ in EXACTLY the support geometry:
    same k, same transform, same scaling, same parameter count, same atom norm;
    only rank(dW) changes, and it must change to <= sqrt(k) exactly.
(C) The `materialise` threading defect found in Q.8 is fixed.

Run:  env/bin/python src/verify_product_support.py
"""
import hashlib
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merged_fourierft import (  # noqa: E402
    MergedFourierFTAdapterModel,
    MergedFourierFTLinear,
    peft_fourierft_indices,
    product_set_indices,
)

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "scratchpad", "phaseQ", "golden_merged_pre_edit.json")
R = []


def check(name, ok, detail=""):
    R.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def _sha(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def main():
    print("Q.8 -- product-set support, gate suite\n")
    gold = json.load(open(GOLDEN))

    print("A. Regression: the default scattered path is untouched")
    for k in (256, 1000):
        idx = peft_fourierft_indices(768, 768, k, 777)
        check(f"scattered indices k={k} bit-identical", _sha(idx) == gold[f"idx_k{k}"])
    torch.manual_seed(0)
    lay = MergedFourierFTLinear(nn.Linear(768, 768), n_frequency=256,
                                scaling=150.0, random_loc_seed=777)
    lay.eval()
    torch.manual_seed(1)
    x = torch.randn(2, 5, 768)
    with torch.no_grad():
        y = lay(x)
    check("default forward bit-identical", _sha(y) == gold["y_sha"],
          f"y_sum={float(y.sum()):.6f} vs {gold['y_sum']:.6f}")
    check("default support is 'scattered'", lay.support == "scattered")

    print("\nB. One knob: product vs scattered")
    K, D = 256, 768
    r = int(K ** 0.5)
    pidx = product_set_indices(D, D, K, 777)
    sidx = peft_fourierft_indices(D, D, K, 777)
    check(f"product has exactly k={K} coefficients", pidx.shape[1] == K, str(tuple(pidx.shape)))
    check(f"product uses exactly {r} distinct rows", len(set(pidx[0].tolist())) == r)
    check(f"product uses exactly {r} distinct cols", len(set(pidx[1].tolist())) == r)
    check("product locations are distinct", len({tuple(v) for v in pidx.t().tolist()}) == K)
    check("scattered uses ~k distinct rows (not a product set)",
          len(set(sidx[0].tolist())) > 4 * r, f"{len(set(sidx[0].tolist()))} rows")
    check("in range", int(pidx.max()) < D and int(pidx.min()) >= 0)
    check("deterministic", _sha(product_set_indices(D, D, K, 777)) == _sha(pidx))
    check("seed changes the set",
          _sha(product_set_indices(D, D, K, 778)) != _sha(pidx))
    try:
        product_set_indices(D, D, 255, 777)
        check("non-square k rejected", False)
    except ValueError:
        check("non-square k rejected", True)

    # rank: the whole point
    def dW(idx, seed=0):
        g = torch.Generator().manual_seed(seed)
        spec = torch.zeros(D, D, dtype=torch.float64)
        spec[idx[0], idx[1]] = torch.randn(K, generator=g, dtype=torch.float64)
        return torch.fft.ifft2(spec).real * 150.0

    rp = torch.linalg.matrix_rank(dW(pidx)).item()
    rs = torch.linalg.matrix_rank(dW(sidx)).item()
    # The cap is 2*sqrt(k), NOT sqrt(k): FourierFT takes `.real`, and
    # conj(ifft2(S)) = ifft2(S_flip) (CARRY_FORWARD.md 4.1), so the effective
    # support is supp U flip(supp) -- a union of TWO product sets.  My first
    # gate asserted <= sqrt(k) and was simply wrong about the construction.
    # LYRA's real DCT core has no conjugate partner, so it caps at sqrt(k)=16
    # while this arm caps at 32; the arms are therefore NOT rank-identical and
    # the findings must say so.
    check(f"product rank <= 2*sqrt(k) = {2*r} (flip-pair doubling)", rp <= 2 * r, f"rank={rp}")
    check("scattered rank is far higher", rs > 4 * r, f"rank={rs}")
    sp = lambda A: (lambda s: (s.pow(2).sum() / s[0] ** 2).item())(torch.linalg.svdvals(A))
    check("product stable rank << scattered", sp(dW(pidx)) * 4 < sp(dW(sidx)),
          f"{sp(dW(pidx)):.2f} vs {sp(dW(sidx)):.2f}")

    # matched budget + matched atom norm
    torch.manual_seed(0)
    lp = MergedFourierFTLinear(nn.Linear(D, D), n_frequency=K, scaling=150.0,
                               random_loc_seed=777, support="product")
    torch.manual_seed(0)
    ls = MergedFourierFTLinear(nn.Linear(D, D), n_frequency=K, scaling=150.0,
                               random_loc_seed=777, support="scattered")
    np_ = sum(p.numel() for p in lp.parameters() if p.requires_grad)
    ns_ = sum(p.numel() for p in ls.parameters() if p.requires_grad)
    check(f"identical trainable params ({K})", np_ == ns_ == K, f"{np_} vs {ns_}")

    def atoms(idx):
        # ||d(dW)/d theta_j||_F for each coefficient
        out = []
        for j in range(0, K, 37):
            spec = torch.zeros(D, D, dtype=torch.float64)
            spec[idx[0][j], idx[1][j]] = 1.0
            out.append((torch.fft.ifft2(spec).real * 150.0).norm().item())
        return torch.tensor(out)

    ap, as_ = atoms(pidx), atoms(sidx)
    check("atom norm identical between arms",
          abs(ap.mean() - as_.mean()) / as_.mean() < 1e-9,
          f"product {ap.mean():.9f} vs scattered {as_.mean():.9f}")
    # 1e-9 was below the achievable precision of the fft round-trip here; both
    # arms show the SAME 1.2e-07 residue, so it cancels in the comparison.
    check("atom norm spread ~0 within each arm (fft round-trip residue)",
          ap.std() / ap.mean() < 1e-6 and as_.std() / as_.mean() < 1e-6,
          f"rel sd {ap.std()/ap.mean():.2e} / {as_.std()/as_.mean():.2e}")

    print("\nC. Threading (the defect this phase found)")
    for mat, expect in (("ifft2", "ifft2"), ("lowrank", "lowrank"), ("batched", "ifft2")):
        m = nn.Module()
        m.query = nn.Linear(64, 64)
        m.classifier = nn.Linear(64, 2)
        w = MergedFourierFTAdapterModel(m, ["query"], n_frequency=16, scaling=150.0,
                                        materialise=mat)
        got = [l.materialise for l in m.modules() if isinstance(l, MergedFourierFTLinear)]
        check(f"materialise={mat} reaches the layer as {expect}", got == [expect], str(got))
        if mat == "batched":
            check("batched still builds the hub", w.hub is not None)
    m = nn.Module()
    m.query = nn.Linear(64, 64)
    m.classifier = nn.Linear(64, 2)
    MergedFourierFTAdapterModel(m, ["query"], n_frequency=16, scaling=150.0, support="product")
    got = [l.support for l in m.modules() if isinstance(l, MergedFourierFTLinear)]
    check("support reaches the layer", got == ["product"], str(got))

    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_glue.py")).read()
    check("--fourierftmerged_support exists", '"--fourierftmerged_support"' in src)
    check("support passed to the factory", "support=args.fourierftmerged_support" in src)
    check("results row records it (PROCESS 1.5c)", src.count('"fourierftmerged_support"') >= 2)

    print(f"\n{'='*60}\n{sum(R)}/{len(R)} gates passed")
    return 0 if all(R) else 1


if __name__ == "__main__":
    sys.exit(main())
