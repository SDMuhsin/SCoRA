"""Cheapest-decisive gate for ShrinkFT [R.75], run BEFORE its 10 GPU cells (PROCESS.md 2.3).

THE QUESTION.  ShrinkFT's whole claim is that the surviving coefficient support is chosen by
the INPUT.  If instead the same coefficients survive for (nearly) every token, then
shrink_l(C x) ~ M (C x) for a FIXED mask M -- i.e. the arm degenerates to a static masked
adapter, which is a per-frequency weighting, which [O.2] already BARS (gradient SNR of dW is
white in frequency, 1.03-1.05x flat).  In that branch R.75 is predicted null BEFORE the spend.

Measured on real RoBERTa activations at the adapted sites (query/value inputs), RTE.
Nothing is trained; this is a property of the frozen backbone's activations.
"""
import sys, math, argparse
sys.path.insert(0, "src")
import torch
from shrinkft_adapter import dct_matrix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="rte"); ap.add_argument("--q", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--n_batches", type=int, default=4); ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()

    from transformers import AutoTokenizer, AutoModel
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModel.from_pretrained("roberta-base").eval()
    keys = {"rte": ("sentence1", "sentence2"), "cola": ("sentence", None),
            "mrpc": ("sentence1", "sentence2")}[a.task]
    ds = load_dataset("glue", a.task)["train"]

    grabbed = []
    hooks = []
    def mk(name):
        def h(mod, inp, out): grabbed.append((name, inp[0].detach()))
        return h
    for n, m in model.named_modules():
        if n.endswith("attention.self.query") or n.endswith("attention.self.value"):
            hooks.append(m.register_forward_hook(mk(n)))

    rows = []
    for b in range(a.n_batches):
        sl = ds[b * a.bs:(b + 1) * a.bs]
        texts = (sl[keys[0]], sl[keys[1]]) if keys[1] else (sl[keys[0]],)
        enc = tok(*texts, truncation=True, max_length=128, padding=True, return_tensors="pt")
        grabbed.clear()
        with torch.no_grad():
            model(**enc)
        mask = enc["attention_mask"].bool().reshape(-1)
        for name, act in grabbed:
            d = act.shape[-1]
            X = act.reshape(-1, d)[mask]              # real tokens only
            # PROCESS.md 2.7: measure the distribution the SHIPPED OBJECT has.  The adapter
            # thresholds over the k SELECTED input frequencies (ShrinkFTLinear.Cn is (k,n)),
            # NOT over the full d-point spectrum.  An earlier version of this probe used the
            # full spectrum and is superseded.
            from shrinkft_adapter import scattered_support
            _, cols = scattered_support(d, d, a.k, 777)
            C = dct_matrix(d)[cols].to(X.dtype)        # (k, n) -- exactly the adapter's basis
            U = (X @ C.T).abs()                        # (tokens, k) coefficient magnitudes
            lam = torch.quantile(U, a.q, dim=-1, keepdim=True)
            S = (U > lam)                              # survival mask per token
            p = S.double().mean(0)                     # per-coefficient survival rate
            # how token-dependent is the support?
            frac_always = (p > 0.95).double().mean().item()
            frac_never  = (p < 0.05).double().mean().item()
            # Jaccard between random token pairs
            idx = torch.randperm(S.shape[0])[:64]
            A = S[idx[:32]].double(); B = S[idx[32:64]].double()
            inter = (A * B).sum(1); union = ((A + B) > 0).double().sum(1)
            jac = (inter / union.clamp(min=1)).mean().item()
            rows.append((name, S.shape[0], frac_always, frac_never, jac,
                         p.std().item()))
    for h in hooks: h.remove()

    import statistics as st
    fa = st.mean(r[2] for r in rows); fn = st.mean(r[3] for r in rows)
    jc = st.mean(r[4] for r in rows); sd = st.mean(r[5] for r in rows)
    ntok = rows[0][1]
    print("="*74)
    print(f"ShrinkFT support probe -- {a.task}, q={a.q}, {len(rows)} module-batches, "
          f"~{ntok} tokens/batch")
    print("="*74)
    print(f"  coefficients surviving in >95% of tokens : {fa*100:6.2f}%")
    print(f"  coefficients surviving in < 5% of tokens : {fn*100:6.2f}%")
    print(f"  => effectively FIXED (either way)        : {(fa+fn)*100:6.2f}%")
    print(f"  Jaccard(surviving set) between tokens    : {jc:6.3f}   (1.0 = identical support)")
    print(f"  sd of per-coefficient survival rate      : {sd:6.3f}   (0 = perfectly uniform)")
    print()
    # A random-token-independent baseline: if survival were i.i.d. per token with rate 1-q,
    # Jaccard would be r/(2-r) with r = 1-q.
    r = 1.0 - a.q; jac_null = r / (2 - r)
    print(f"  i.i.d. null (no token structure) Jaccard : {jac_null:6.3f}")
    print(f"  excess over null                         : {jc - jac_null:+6.3f}")
    print()
    if (fa + fn) > 0.80 or jc > 0.85:
        print("  ⛔ VERDICT: the surviving support is effectively FIXED across tokens.")
        print("     ShrinkFT degenerates to a static per-frequency MASK, which [O.2] bars.")
        print("     => R.75 is PREDICTED NULL. Do not spend the 10 cells on this q.")
    elif jc - jac_null < 0.05:
        print("  ⭐ VERDICT: the support is essentially AS TOKEN-DEPENDENT AS CHANCE.")
        print("     The selection carries real per-token variation (not a fixed mask), so the")
        print("     [O.2] bar does NOT apply -- but neither is there evident structure for it")
        print("     to exploit. R.75 remains an open empirical question; the spend is justified.")
    else:
        print("  ⭐⭐ VERDICT: support is token-dependent AND structured (Jaccard above the")
        print("     i.i.d. null). This is the branch where ShrinkFT has something to exploit.")
    print("\n⚠️ This probes the INPUT spectrum only. It bounds whether the MECHANISM can act;")
    print("   it says nothing about whether acting on it helps accuracy (PROCESS.md 4).")

if __name__ == "__main__":
    main()
