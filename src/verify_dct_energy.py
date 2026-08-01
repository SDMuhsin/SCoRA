"""
Verify DCT energy concentration of fine-tuning weight updates.

Full fine-tunes BERT-base on CoLA for 3 epochs, computes ΔW for all Q+V
layers, and analyzes 2D DCT energy distribution. This determines whether
the proposed "Spectral Anatomy" figure will show dramatic concentration.

Outputs:
  results/dct_verification/summary.txt   — human-readable summary
  results/dct_verification/delta_w.npz   — raw ΔW matrices for all Q+V layers
  results/dct_verification/compaction.npz — compaction curve data for plotting
"""
import os
import sys
import json
import time

import numpy as np
import torch
from scipy.fft import dctn

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
for var in ['HF_HOME', 'HF_DATASETS_CACHE', 'TRANSFORMERS_CACHE', 'TORCH_HOME']:
    os.environ.setdefault(var, os.path.join(os.getcwd(), 'data'))

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

OUT_DIR = os.path.join('results', 'dct_verification')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compaction_curve_sorted(coeffs_flat, total_energy, n_points=200):
    """Cumulative energy when adding coefficients from largest to smallest."""
    sorted_sq = np.sort(coeffs_flat ** 2)[::-1]
    cumsum = np.cumsum(sorted_sq)
    # Sample at log-spaced points
    n = len(sorted_sq)
    indices = np.unique(np.geomspace(1, n, n_points).astype(int))
    indices = np.clip(indices, 1, n)
    return indices, cumsum[indices - 1] / total_energy


def compaction_curve_contiguous(dct_2d, total_energy, max_k=None):
    """Cumulative energy for contiguous [:k, :k] DCT blocks."""
    m, n = dct_2d.shape
    if max_k is None:
        max_k = min(m, n)
    ks = []
    fracs = []
    for k in range(1, max_k + 1):
        ks.append(k * k)  # number of coefficients
        fracs.append(np.sum(dct_2d[:k, :k] ** 2) / total_energy)
    return np.array(ks), np.array(fracs)


def compaction_curve_dft_random(delta_w, n_draws=50, n_points=20):
    """Cumulative energy for random DFT coefficient subsets."""
    dft = np.fft.fft2(delta_w)
    dft_energy = np.sum(np.abs(dft) ** 2)
    m, n = delta_w.shape
    total_coeffs = m * n

    # Budget values to evaluate
    budgets = np.unique(np.geomspace(4, total_coeffs, n_points).astype(int))
    budgets = np.clip(budgets, 4, total_coeffs)

    means = []
    stds = []
    for k in budgets:
        fracs = []
        for _ in range(n_draws):
            idx = np.random.choice(total_coeffs, size=k, replace=False)
            rows, cols = np.unravel_index(idx, (m, n))
            fracs.append(np.sum(np.abs(dft[rows, cols]) ** 2) / dft_energy)
        means.append(np.mean(fracs))
        stds.append(np.std(fracs))
    return budgets, np.array(means), np.array(stds)


def compaction_curve_svd(delta_w, total_energy):
    """Cumulative energy from SVD rank-r truncation."""
    U, S, Vt = np.linalg.svd(delta_w, full_matrices=False)
    m, n = delta_w.shape
    cumsum = np.cumsum(S ** 2)
    # LoRA params at rank r = r*(m+n), but express on the same x-axis (num coefficients)
    # For fair comparison: LoRA rank-r uses r*(m+n) params
    params = np.arange(1, len(S) + 1) * (m + n)
    return params, cumsum / total_energy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--analyze-only', action='store_true',
                        help='Skip training, load saved delta_w.npz')
    cli_args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    t0 = time.time()

    saved_path = os.path.join(OUT_DIR, 'delta_w.npz')

    if cli_args.analyze_only and os.path.exists(saved_path):
        print(f"\n--analyze-only: loading saved ΔW from {saved_path}")
        data = np.load(saved_path)
        layer_names = sorted(data.files)
        delta_ws = {name: data[name].astype(np.float64) for name in layer_names}
        print(f"Loaded {len(delta_ws)} ΔW matrices")
        return run_analysis(delta_ws, t0)

    # ------------------------------------------------------------------
    # 1. Load model, save initial Q+V weights
    # ------------------------------------------------------------------
    model_name = 'bert-base-uncased'
    print(f"\nLoading {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, cache_dir='./data'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./data')

    w0 = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if name.endswith('.query') or name.endswith('.value'):
                w0[name] = module.weight.detach().clone().cpu()

    print(f"Captured {len(w0)} Q+V weight matrices")

    # ------------------------------------------------------------------
    # 2. Prepare CoLA dataset
    # ------------------------------------------------------------------
    print("Loading CoLA dataset...")
    dataset = load_dataset('glue', 'cola', cache_dir='./data')

    def tokenize_fn(examples):
        return tokenizer(
            examples['sentence'], padding='max_length',
            truncation=True, max_length=128,
        )

    tokenized = dataset.map(tokenize_fn, batched=True)

    # ------------------------------------------------------------------
    # 3. Fine-tune for 3 epochs
    # ------------------------------------------------------------------
    print(f"\nFine-tuning {model_name} on CoLA (3 epochs, lr=2e-5)...")
    training_args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, 'ft_tmp'),
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=100,
        save_strategy='no',
        report_to='none',
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        tokenizer=tokenizer,
    )

    trainer.train()
    ft_time = time.time() - t0
    print(f"Fine-tuning complete in {ft_time:.0f}s")

    # Save ΔW immediately so we don't lose them if analysis crashes
    delta_ws_save = {}
    for name in sorted(w0.keys()):
        module = dict(model.named_modules())[name]
        w_after = module.weight.detach().cpu().numpy().astype(np.float64)
        w_before = w0[name].numpy().astype(np.float64)
        delta_ws_save[name.replace('.', '_')] = (w_after - w_before).astype(np.float32)
    np.savez_compressed(os.path.join(OUT_DIR, 'delta_w.npz'), **delta_ws_save)
    print(f"Saved {len(delta_ws_save)} ΔW matrices to {OUT_DIR}/delta_w.npz")

    # Free model from GPU
    del model, trainer
    torch.cuda.empty_cache()

    # Load saved data with original dotted names
    delta_ws = {k.replace('.', '_'): v.astype(np.float64)
                for k, v in delta_ws_save.items()}
    return run_analysis(delta_ws, t0)


def run_analysis(delta_ws, t0):
    """Analyze DCT energy concentration of saved ΔW matrices."""
    # ------------------------------------------------------------------
    # 4. Compute ΔW and 2D DCT analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DCT ENERGY CONCENTRATION ANALYSIS OF WEIGHT UPDATES")
    print("=" * 80)

    np.random.seed(42)
    layer_names = sorted(delta_ws.keys())

    # Storage
    dct_coeffs_all = {}
    spot_checks = []
    all_sorted_curves = []
    all_contig_curves = []
    all_dft_random_curves = []
    all_svd_curves = []

    for name in layer_names:
        delta_w = delta_ws[name].astype(np.float64)
        m, n = delta_w.shape

        # 2D DCT (orthonormal)
        dct_coeffs = dctn(delta_w, type=2, norm='ortho')
        total_energy = np.sum(dct_coeffs ** 2)

        dct_coeffs_all[name] = dct_coeffs.astype(np.float32)

        if total_energy < 1e-30:
            print(f"\n{name}: ||ΔW|| ≈ 0, skipping")
            continue

        # --- Spot checks ---
        sc = {'name': name, 'shape': (m, n), 'frobenius': float(np.sqrt(total_energy))}

        # DCT contiguous blocks
        for k in [4, 8, 16, 32, 64]:
            if k > min(m, n):
                continue
            frac = float(np.sum(dct_coeffs[:k, :k] ** 2) / total_energy)
            sc[f'dct_contig_{k}x{k}'] = frac

        # DFT random 256 (50 draws)
        dft = np.fft.fft2(delta_w)
        dft_energy = np.sum(np.abs(dft) ** 2)
        dft_fracs = []
        for _ in range(50):
            idx = np.random.choice(m * n, size=256, replace=False)
            rows, cols = np.unravel_index(idx, (m, n))
            dft_fracs.append(float(np.sum(np.abs(dft[rows, cols]) ** 2) / dft_energy))
        sc['dft_random_256_mean'] = float(np.mean(dft_fracs))
        sc['dft_random_256_std'] = float(np.std(dft_fracs))

        # DCT sorted (optimal) at 256
        flat_dct = dct_coeffs.flatten()
        sorted_sq = np.sort(flat_dct ** 2)[::-1]
        sc['dct_sorted_256'] = float(np.sum(sorted_sq[:256]) / total_energy)

        # SVD rank-1
        S_vals = np.linalg.svd(delta_w, compute_uv=False)
        sc['svd_rank1'] = float(S_vals[0] ** 2 / total_energy)

        spot_checks.append(sc)

        # --- Print per-layer ---
        print(f"\n{name} ({m}×{n}, ||ΔW||_F = {sc['frobenius']:.6f}):")
        print(f"  DCT contiguous blocks:")
        for k in [4, 8, 16, 32, 64]:
            key = f'dct_contig_{k}x{k}'
            if key in sc:
                n_coeffs = k * k
                print(f"    {k:3d}×{k:3d} ({n_coeffs:6d} coeffs = "
                      f"{n_coeffs / (m * n) * 100:.3f}%):  "
                      f"{sc[key] * 100:8.4f}% energy")
        print(f"  DFT random 256 (50 draws):  {sc['dft_random_256_mean'] * 100:8.4f}% "
              f"± {sc['dft_random_256_std'] * 100:.4f}%")
        print(f"  DCT sorted 256 (optimal):   {sc['dct_sorted_256'] * 100:8.4f}%")
        print(f"  SVD rank-1 (1536 params):    {sc['svd_rank1'] * 100:8.4f}%")

        # --- Full compaction curves (for this layer) ---
        s_k, s_frac = compaction_curve_sorted(flat_dct, total_energy, n_points=300)
        all_sorted_curves.append((s_k, s_frac))

        c_k, c_frac = compaction_curve_contiguous(dct_coeffs, total_energy, max_k=128)
        all_contig_curves.append((c_k, c_frac))

        dft_k, dft_mean, dft_std = compaction_curve_dft_random(delta_w, n_draws=50, n_points=30)
        all_dft_random_curves.append((dft_k, dft_mean, dft_std))

        svd_k, svd_frac = compaction_curve_svd(delta_w, total_energy)
        all_svd_curves.append((svd_k, svd_frac))

    # ------------------------------------------------------------------
    # 5. Aggregate and print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY (averaged across all 24 Q+V layers)")
    print("=" * 80)

    print(f"\n{'Method':<35s} {'Coeffs':>8s} {'% of 768²':>10s} {'Energy':>16s}")
    print("-" * 73)

    for k in [4, 8, 16, 32, 64]:
        key = f'dct_contig_{k}x{k}'
        vals = [sc[key] for sc in spot_checks if key in sc]
        if not vals:
            continue
        n_coeffs = k * k
        pct = n_coeffs / (768 * 768) * 100
        tag = "  ← Spectral p=16" if k == 16 else ""
        label = f"  DCT contiguous {k}×{k}"
        print(f"{label:<35s} {n_coeffs:>8d} {pct:>9.3f}% "
              f"{np.mean(vals) * 100:>8.3f}% ± {np.std(vals) * 100:.3f}%{tag}")

    dft_means = [sc['dft_random_256_mean'] for sc in spot_checks]
    print(f"{'  DFT random 256':<35s} {256:>8d} "
          f"{256 / (768 * 768) * 100:>9.3f}% "
          f"{np.mean(dft_means) * 100:>8.3f}% ± {np.std(dft_means) * 100:.3f}%"
          f"  ← FourierFT")

    dct_opt = [sc['dct_sorted_256'] for sc in spot_checks]
    print(f"{'  DCT sorted 256 (optimal)':<35s} {256:>8d} "
          f"{256 / (768 * 768) * 100:>9.3f}% "
          f"{np.mean(dct_opt) * 100:>8.3f}% ± {np.std(dct_opt) * 100:.3f}%")

    svd_vals = [sc['svd_rank1'] for sc in spot_checks]
    print(f"{'  SVD rank-1 (LoRA r=1)':<35s} {1536:>8d} "
          f"{1536 / (768 * 768) * 100:>9.3f}% "
          f"{np.mean(svd_vals) * 100:>8.3f}% ± {np.std(svd_vals) * 100:.3f}%")

    # ------------------------------------------------------------------
    # 6. Verdict
    # ------------------------------------------------------------------
    dct16_vals = [sc.get('dct_contig_16x16', 0) for sc in spot_checks]
    dct16_mean = np.mean(dct16_vals) * 100
    dft_random_mean = np.mean(dft_means) * 100
    dct_opt_mean = np.mean(dct_opt) * 100

    print(f"\n{'=' * 80}")
    print("VERDICT")
    print(f"{'=' * 80}")
    print(f"  At 256 coefficients (0.043% of 768²):")
    print(f"    DCT contiguous 16×16 (Spectral):  {dct16_mean:.4f}%")
    print(f"    DFT random 256 (FourierFT):       {dft_random_mean:.4f}%")
    print(f"    DCT sorted 256 (optimal):         {dct_opt_mean:.4f}%")
    print(f"    Ratio (Spectral / FourierFT):     {dct16_mean / max(dft_random_mean, 0.0001):.2f}x")
    print(f"    Spectral captures {dct16_mean / max(dct_opt_mean, 0.0001) * 100:.1f}% "
          f"of optimal DCT")

    if dct16_mean > dft_random_mean * 2:
        verdict = "STRONG concentration. Figure will be dramatic."
    elif dct16_mean > dft_random_mean * 1.3:
        verdict = "Good concentration. Figure will show clear advantage."
    elif dct16_mean > dft_random_mean:
        verdict = "Mild concentration. Advantage visible but modest."
    else:
        verdict = "No concentration advantage. Figure approach needs rethinking."
    print(f"\n  → {verdict}")
    print(f"{'=' * 80}")

    # ------------------------------------------------------------------
    # 7. Save data
    # ------------------------------------------------------------------
    # Raw ΔW matrices
    np.savez_compressed(
        os.path.join(OUT_DIR, 'delta_w.npz'),
        **{name.replace('.', '_'): arr for name, arr in delta_ws.items()}
    )
    print(f"\nSaved ΔW matrices to {OUT_DIR}/delta_w.npz")

    # DCT coefficients
    np.savez_compressed(
        os.path.join(OUT_DIR, 'dct_coeffs.npz'),
        **{name.replace('.', '_'): arr for name, arr in dct_coeffs_all.items()}
    )
    print(f"Saved DCT coefficients to {OUT_DIR}/dct_coeffs.npz")

    # Compaction curves (averaged)
    # Interpolate all curves to common x-axis for averaging
    common_k = np.unique(np.geomspace(1, 768 * 768, 500).astype(int))

    def interpolate_curves(curves_list, common_x):
        """Interpolate and average multiple (x, y) curves."""
        interp_ys = []
        for x, y in curves_list:
            interp_y = np.interp(common_x, x, y, left=0.0, right=1.0)
            interp_ys.append(interp_y)
        return np.mean(interp_ys, axis=0), np.std(interp_ys, axis=0)

    sorted_mean, sorted_std = interpolate_curves(all_sorted_curves, common_k)
    contig_mean, contig_std = interpolate_curves(all_contig_curves, common_k)

    # DFT random needs separate handling (has mean and std per point)
    dft_common_x = np.unique(np.geomspace(4, 768 * 768, 100).astype(int))
    dft_interp_means = []
    dft_interp_stds = []
    for k, mu, sd in all_dft_random_curves:
        dft_interp_means.append(np.interp(dft_common_x, k, mu, left=0.0, right=1.0))
        dft_interp_stds.append(np.interp(dft_common_x, k, sd, left=0.0, right=0.0))
    dft_avg_mean = np.mean(dft_interp_means, axis=0)
    dft_avg_std = np.mean(dft_interp_stds, axis=0)

    svd_mean_curve, svd_std_curve = interpolate_curves(all_svd_curves, common_k)

    np.savez_compressed(
        os.path.join(OUT_DIR, 'compaction.npz'),
        common_k=common_k,
        dct_sorted_mean=sorted_mean,
        dct_sorted_std=sorted_std,
        dct_contiguous_mean=contig_mean,
        dct_contiguous_std=contig_std,
        dft_common_k=dft_common_x,
        dft_random_mean=dft_avg_mean,
        dft_random_std=dft_avg_std,
        svd_k=common_k,
        svd_mean=svd_mean_curve,
        svd_std=svd_std_curve,
    )
    print(f"Saved compaction curves to {OUT_DIR}/compaction.npz")

    # JSON summary
    summary = {
        'model': 'bert-base-uncased',
        'task': 'cola',
        'epochs': 3,
        'n_layers': len(spot_checks),
        'dct_contig_16x16_mean': float(dct16_mean),
        'dft_random_256_mean': float(dft_random_mean),
        'dct_sorted_256_mean': float(dct_opt_mean),
        'ratio_spectral_over_fourierft': float(dct16_mean / max(dft_random_mean, 0.001)),
        'verdict': verdict,
        'per_layer': spot_checks,
    }
    with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {OUT_DIR}/summary.json")

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.0f}s")


if __name__ == '__main__':
    main()
