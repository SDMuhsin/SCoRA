"""
Verification harness for the GASA Conv2d adapter (`gasa_adapter.py`) and its
FourierFT / LoRA conv baselines.

Sections:
  (1) FORWARD CORRECTNESS
      - GASAConv2d.get_delta_weight() equals the Σ_r (u_r v_rᵀ)⊗(Φ_p s_r) formula
        computed independently (standard + depthwise), to machine precision.
      - the theory's factored subspace forward (project onto Φ_p first) equals the
        dense contraction (reproduces verify_gasa.py §G on the module's own Φ_p).
      - a zero-init GASA module has ΔW == 0 and forward == frozen conv.
      - FourierFTConv2d reproduces the reshaped-ifft2 semantics, and matches PEFT's
        FourierFTLayer.get_delta_weight() bit-for-bit on the [m,n] matrix.
  (2) PARAM COUNTS at matched budget (ResNet 64x64x3x3; ConvNeXt 768ch 7x7 depthwise).
  (3) RECONSTRUCTION on a REAL conv ΔW (short fine-tune diff, cached to scratchpad):
      (3a) matched spectral-budget capture (reproduces gasa_theory §C): bottom-p
           grid-Laplacian modes vs random-p DFT (FourierFT floor) vs sparse-p vs floor.
      (3b) matched TOTAL-budget full-module LS fit of each parameterization.

Run:
  cd /workspace/lora_research_signal; source env/bin/activate
  export PYTHONPATH=src HF_HOME=$PWD/data
  python src/verify_gasa_conv.py
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn

from gasa_adapter import (
    GASAConv2d, FourierFTConv2d, LoRAConv2d,
    grid_lowpass_basis, gasa_param_count, match_budget,
)

torch.backends.cuda.matmul.allow_tf32 = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE = os.environ.get("HF_HOME", "/workspace/lora_research_signal/data")
SCRATCH = "/workspace/lora_research_signal/scratchpad"


def line(c="="):
    print(c * 78)


# ===========================================================================
# (1) FORWARD CORRECTNESS
# ===========================================================================
def verify_forward():
    line()
    print("(1) FORWARD CORRECTNESS")
    line("-")
    torch.manual_seed(0)

    # --- 1a. standard GASA reconstruction vs the Σ_r (u v^T)⊗(Φ s) formula ---
    Cout, Cin, K, R, p = 5, 3, 4, 2, 6
    conv = nn.Conv2d(Cin, Cout, K)
    g = GASAConv2d(conv, rank=R, p=p, scaling=1.0)
    with torch.no_grad():
        g.u.copy_(torch.randn(R, Cout)); g.v.copy_(torch.randn(R, Cin))
        g.s.copy_(torch.randn(R, p))
    dW_mod = g.get_delta_weight().detach()
    Phi = g.Phi
    dW_ref = torch.zeros(Cout, Cin, K, K)
    for r in range(R):
        spatial = (Phi @ g.s[r]).reshape(K, K)          # Φ_p s_r
        dW_ref += torch.einsum("o,i,xy->oixy", g.u[r], g.v[r], spatial)
    res = (dW_mod - dW_ref).abs().max().item()
    print(f"  1a std GASA recon: max|module - Σ_r(u v^T)⊗(Φ s)| = {res:.2e}  "
          f"(P_GASA={R*(Cout+Cin+p)})")

    # --- 1b. factored subspace forward == dense contraction (theory §G) ---
    # patch/stride=K case: one output token per patch, grid contracted.
    x = torch.randn(Cin, K * K)                          # (C_in, g)
    dW_flat = dW_mod.reshape(Cout, Cin, K * K)
    dy_dense = torch.einsum("oix,ix->o", dW_flat, x)
    xG = x @ Phi                                         # project grid -> p coords
    dy_fac = torch.zeros(Cout)
    for r in range(R):
        z = g.v[r] @ xG                                  # contract input channels
        a = g.s[r] @ z                                   # spectral filter
        dy_fac += a * g.u[r]                             # expand output channels
    print(f"  1b factored subspace forward == dense: max|Δ| = "
          f"{(dy_dense - dy_fac).abs().max().item():.2e}")

    # --- 1c. depthwise GASA reconstruction ---
    C, Kd, pd = 8, 7, 5
    dwconv = nn.Conv2d(C, C, Kd, groups=C)
    gd = GASAConv2d(dwconv, p=pd, depthwise=True)
    with torch.no_grad():
        gd.S.copy_(torch.randn(C, pd))
    dWd = gd.get_delta_weight().detach()
    Phid = gd.Phi
    dWd_ref = torch.stack([(Phid @ gd.S[c]).reshape(Kd, Kd) for c in range(C)]).unsqueeze(1)
    print(f"  1c depthwise GASA recon: max|module - Φ_p s_c| = "
          f"{(dWd - dWd_ref).abs().max().item():.2e}  (params={C*pd}, shape={tuple(dWd.shape)})")

    # --- 1d. zero-init GASA => ΔW == 0 and output == frozen conv ---
    g0 = GASAConv2d(nn.Conv2d(Cin, Cout, K, padding=1), rank=R, p=p)
    xin = torch.randn(2, Cin, 12, 12)
    dW0 = g0.get_delta_weight().abs().max().item()
    out_adapt = g0(xin)
    out_base = g0.base_layer(xin)
    print(f"  1d zero-init GASA: max|ΔW|={dW0:.2e}  "
          f"max|adapter_out - frozen_conv_out|={(out_adapt - out_base).abs().max().item():.2e}")
    g0d = GASAConv2d(nn.Conv2d(C, C, Kd, groups=C, padding=3), depthwise=True, p=pd)
    xin_d = torch.randn(2, C, 12, 12)
    print(f"     zero-init depthwise: max|ΔW|={g0d.get_delta_weight().abs().max().item():.2e}  "
          f"max|out-base|={(g0d(xin_d) - g0d.base_layer(xin_d)).abs().max().item():.2e}")

    # --- 1e. FourierFT semantics: independent ifft2 + PEFT cross-check ---
    Cout2, Cin2, K2, nf = 6, 4, 3, 20
    fconv = nn.Conv2d(Cin2, Cout2, K2)
    ff = FourierFTConv2d(fconv, n_frequency=nf, scaling=0.7, random_loc_seed=777)
    dW_ff = ff.get_delta_weight().detach()
    # independent reference
    m, n = Cout2, Cin2 * K2 * K2
    dense = torch.zeros(m, n, dtype=ff.spectrum.dtype)
    idx = ff.indices
    dense[idx[0], idx[1]] = ff.spectrum.detach()
    dW_ref2 = (torch.fft.ifft2(dense).real * 0.7).reshape(Cout2, Cin2, K2, K2)
    print(f"  1e FourierFT recon: max|module - reshaped ifft2| = "
          f"{(dW_ff - dW_ref2).abs().max().item():.2e}")
    try:
        from peft.tuners.fourierft.layer import FourierFTLinear
        lin = nn.Linear(n, m, bias=False)
        pl = FourierFTLinear(lin, adapter_name="d", n_frequency=nf, scaling=0.7,
                             random_loc_seed=777, init_weights=False)
        with torch.no_grad():
            pl.fourierft_spectrum["d"].copy_(ff.spectrum)
        peft_dw = pl.get_delta_weight("d").detach()          # (m, n)
        idx_match = torch.equal(pl.indices["d"], ff.indices)
        dmax = (peft_dw - dW_ff.reshape(m, n)).abs().max().item()
        print(f"     vs PEFT FourierFTLayer: indices_identical={idx_match}  "
              f"max|Δ get_delta_weight| = {dmax:.2e}")
    except Exception as e:
        print(f"     (PEFT cross-check skipped: {type(e).__name__}: {e})")


# ===========================================================================
# (2) PARAM COUNTS at matched budget
# ===========================================================================
def _mk_conv(Cout, Cin, K, depthwise):
    return (nn.Conv2d(Cout, Cout, K, groups=Cout) if depthwise
            else nn.Conv2d(Cin, Cout, K))


def _instantiate_counts(Cout, Cin, K, rank, p, depthwise):
    """Instantiate the three modules and return their ACTUAL trainable numel."""
    b = match_budget(Cout, Cin, K, rank, p, depthwise)
    g = GASAConv2d(_mk_conv(Cout, Cin, K, depthwise), rank=rank, p=p, depthwise=depthwise)
    ff = FourierFTConv2d(_mk_conv(Cout, Cin, K, depthwise),
                         n_frequency=b["fourierft_n_frequency"])
    lo = LoRAConv2d(_mk_conv(Cout, Cin, K, depthwise), rank=b["lora_rank"])
    gc = sum(pp.numel() for pp in g.parameters() if pp.requires_grad)
    fc = sum(pp.numel() for pp in ff.parameters() if pp.requires_grad)
    lc = sum(pp.numel() for pp in lo.parameters() if pp.requires_grad)
    return b, gc, fc, lc


def verify_param_counts():
    line()
    print("(2) PARAM COUNTS at matched budget (actual module numel)")
    line("-")
    cases = [
        ("ResNet 64x64x3x3 (standard)", 64, 64, 3, 4, 9, False),
        ("ResNet 64x64x3x3 (standard)", 64, 64, 3, 4, 4, False),
        ("ConvNeXt 768ch 7x7 (depthwise)", 768, 1, 7, 1, 16, True),
        ("ConvNeXt 768ch 7x7 (depthwise)", 768, 1, 7, 1, 8, True),
    ]
    print(f"  {'case':<34} {'R':>2} {'p':>3} {'P_GASA':>7} {'GASA':>7} {'FourierFT':>9} "
          f"{'LoRA(r)':>8} {'LoRA#':>7}")
    for name, Cout, Cin, K, R, p, dw in cases:
        b, gc, fc, lc = _instantiate_counts(Cout, Cin, K, R, p, dw)
        assert gc == b["P_gasa"], (gc, b["P_gasa"])
        assert fc == b["fourierft_n_frequency"]
        assert lc == b["lora_params"]
        match = "==" if gc == fc else "!="
        print(f"  {name:<34} {R:>2} {p:>3} {b['P_gasa']:>7} {gc:>7} {fc:>9} "
              f"{b['lora_rank']:>8} {lc:>7}  GASA{match}FFT")
    print("  note: GASA n_freq matched EXACTLY to P_GASA. LoRA is rank-quantized:")
    print("        its rank-1 floor (m+n) can exceed a small P_GASA (standard conv,")
    print("        large n=C_in*K^2); depthwise (small n=K^2) matches closely.")


# ===========================================================================
# real ΔW generation (short fine-tune, cached)
# ===========================================================================
def _short_finetune_delta(model_name, target_conv, npz_path, steps=40, n_img=512, bs=32):
    if os.path.exists(npz_path):
        d = np.load(npz_path)
        return d["dW"], str(d["name"])
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    from datasets import load_dataset
    print(f"    [gen] short fine-tune of {model_name} to produce real ΔW "
          f"({steps} steps)...")
    proc = AutoImageProcessor.from_pretrained(model_name, cache_dir=CACHE)
    ds = load_dataset("cifar10", cache_dir=CACHE)["train"].select(range(n_img))
    model = AutoModelForImageClassification.from_pretrained(
        model_name, num_labels=10, ignore_mismatched_sizes=True, cache_dir=CACHE).to(DEVICE)
    # locate target conv
    conv = dict(model.named_modules())[target_conv]
    W0 = conv.weight.detach().clone()
    mean = torch.tensor(proc.image_mean).view(3, 1, 1)
    std = torch.tensor(proc.image_std).view(3, 1, 1)
    edge = proc.size.get("shortest_edge", 224)

    def batch_px(items):
        imgs = [it["img"].convert("RGB") for it in items]
        return proc(images=imgs, return_tensors="pt")["pixel_values"]

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    torch.manual_seed(0)
    step = 0
    while step < steps:
        for i in range(0, n_img, bs):
            items = [ds[j] for j in range(i, min(i + bs, n_img))]
            px = batch_px(items).to(DEVICE)
            labels = torch.tensor([it["label"] for it in items]).to(DEVICE)
            logits = model(pixel_values=px).logits
            loss = loss_fn(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % 10 == 0 or step == 1:
                print(f"      step {step}/{steps} loss={loss.item():.3f} "
                      f"dev={px.device}", flush=True)
            if step >= steps:
                break
    dW = (conv.weight.detach() - W0).cpu().numpy().astype(np.float64)
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez(npz_path, dW=dW, name=target_conv)
    del model
    torch.cuda.empty_cache()
    return dW, target_conv


# ===========================================================================
# (3a) matched SPECTRAL-budget capture (reproduces gasa_theory §C)
# ===========================================================================
def _spatial_maps(dW):
    """Reshape a conv ΔW [Cout,Cin,K,K] to (num_maps, K, K)."""
    Cout, Cin, K, _ = dW.shape
    return dW.reshape(-1, K, K), K


def gasa_lowpass_capture(maps, K, p):
    """Energy-weighted fraction of ΔW energy in the bottom-p L_G modes (= module Φ_p)."""
    Phi = grid_lowpass_basis(K, p, torch.float64).numpy()      # (K², p)
    M = maps.reshape(-1, K * K)                                 # (num_maps, K²)
    coeff = M @ Phi                                             # (num_maps, p)
    return float((coeff ** 2).sum() / (M ** 2).sum())


def fourierft_random_capture(maps, K, k, n_draws=300):
    """Expected energy in k random 2D-DFT freqs (FourierFT floor), energy-weighted."""
    rng = np.random.default_rng(0)
    num = den = 0.0
    for m in maps:
        F = np.fft.fft2(m)
        den += np.sum(np.abs(F) ** 2)
        acc = 0.0
        for _ in range(n_draws):
            idx = rng.choice(K * K, size=k, replace=False)
            r, c = np.unravel_index(idx, (K, K))
            acc += np.sum(np.abs(F[r, c]) ** 2)
        num += acc / n_draws
    return float(num / den)


def sparseft_capture(maps, k):
    num = den = 0.0
    for m in maps:
        den += np.sum(m ** 2)
        num += np.sum(np.sort(m.ravel() ** 2)[::-1][:k])
    return float(num / den)


def verify_reconstruction_spectral(tag, dW):
    maps, K = _spatial_maps(dW)
    N = K * K
    print(f"  [{tag}] {maps.shape[0]} spatial {K}x{K} maps, N={N}, "
          f"||ΔW||={np.linalg.norm(dW):.3e}")
    print(f"    {'k=p':>4} {'GASA(bottom-p)':>15} {'FourierFT(rand)':>16} "
          f"{'sparse-p':>9} {'floor':>7} {'GASA/FFT':>9} {'GASA/floor':>11}")
    ks = [p for p in [1, 2, 4, 6, 9, 16, 25, 36] if p <= N]
    for k in ks:
        gasa = gasa_lowpass_capture(maps, K, k)
        fft = fourierft_random_capture(maps, K, k)
        sp = sparseft_capture(maps, k)
        floor = k / N
        print(f"    {k:>4} {gasa*100:>14.1f}% {fft*100:>15.1f}% {sp*100:>8.1f}% "
              f"{floor*100:>6.1f}% {gasa/fft:>8.2f}x {gasa/floor:>10.2f}x")


# ===========================================================================
# (3b) matched TOTAL-budget full-module LS fit
# ===========================================================================
def _fit_module(module, target, iters=3000, lr=0.05):
    module.to(DEVICE)
    target = target.to(DEVICE)
    params = [p for p in module.parameters() if p.requires_grad]
    # break the symmetric zero-init so gradients flow immediately in the fit
    with torch.no_grad():
        for p in params:
            if p.abs().sum() == 0:
                p.add_(0.01 * torch.randn_like(p))
    opt = torch.optim.Adam(params, lr=lr)
    tnorm2 = (target ** 2).sum()
    best = float("inf")
    for it in range(iters):
        opt.zero_grad()
        resid = ((module.get_delta_weight() - target) ** 2).sum()
        resid.backward()
        opt.step()
        best = min(best, resid.item())
    return 1.0 - best / tnorm2.item()


def verify_reconstruction_module(tag, dW, rank, p, depthwise):
    Cout, Cin, K, _ = dW.shape
    b = match_budget(Cout, Cin, K, rank, p, depthwise)
    target = torch.tensor(dW, dtype=torch.float32)
    # base convs (weights are frozen & irrelevant to get_delta_weight)
    g = GASAConv2d(_mk_conv(Cout, Cin, K, depthwise), rank=rank, p=p, depthwise=depthwise)
    ff = FourierFTConv2d(_mk_conv(Cout, Cin, K, depthwise), n_frequency=b["fourierft_n_frequency"])
    lo = LoRAConv2d(_mk_conv(Cout, Cin, K, depthwise), rank=b["lora_rank"])
    cap_g = _fit_module(g, target)
    cap_f = _fit_module(ff, target)
    # LoRA closed-form optimum via SVD of the reshaped matrix
    M = target.reshape(Cout, -1)
    sv = torch.linalg.svdvals(M)
    cap_l_svd = float((sv[:b["lora_rank"]] ** 2).sum() / (sv ** 2).sum())
    cap_l = _fit_module(lo, target)
    print(f"  [{tag}] rank={rank} p={p} dw={depthwise} | P_GASA={b['P_gasa']} "
          f"ff_n_freq={b['fourierft_n_frequency']} lora_rank={b['lora_rank']} "
          f"(lora#={b['lora_params']})")
    print(f"    full-module LS capture:  GASA={cap_g*100:.1f}%  "
          f"FourierFT={cap_f*100:.1f}%  LoRA={cap_l*100:.1f}% (SVD-opt {cap_l_svd*100:.1f}%)  "
          f"| GASA/FFT={cap_g/max(cap_f,1e-9):.2f}x")


# ===========================================================================
def main():
    t0 = time.time()
    verify_forward()
    verify_param_counts()

    line()
    print("(3) RECONSTRUCTION on REAL conv ΔW")
    line("-")
    # ResNet-18 64x64x3x3 (standard) and ConvNeXt 96ch 7x7 (depthwise)
    dW_res, res_name = _short_finetune_delta(
        "microsoft/resnet-18",
        "resnet.encoder.stages.0.layers.0.layer.0.convolution",
        os.path.join(SCRATCH, "deltaW_resnet18.npz"))
    dW_cnx, cnx_name = _short_finetune_delta(
        "facebook/convnext-tiny-224",
        "convnext.encoder.stages.0.layers.0.dwconv",
        os.path.join(SCRATCH, "deltaW_convnext.npz"))
    print(f"  real ΔW: ResNet '{res_name}' shape={dW_res.shape}; "
          f"ConvNeXt '{cnx_name}' shape={dW_cnx.shape}")

    print()
    print("(3a) matched SPECTRAL-budget capture (theory §C reproduction)")
    verify_reconstruction_spectral("ResNet 64x64x3x3", dW_res)
    verify_reconstruction_spectral("ConvNeXt 96x1x7x7 depthwise", dW_cnx)

    print()
    print("(3b) matched TOTAL-budget full-module LS fit")
    verify_reconstruction_module("ResNet 64x64x3x3", dW_res, rank=4, p=4, depthwise=False)
    verify_reconstruction_module("ConvNeXt 96x1x7x7 dw", dW_cnx, rank=1, p=8, depthwise=True)

    line()
    print(f"ALL VERIFICATIONS COMPLETE in {time.time()-t0:.1f}s")
    line()


if __name__ == "__main__":
    main()
