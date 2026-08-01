"""
Spectral Anatomy of PEFT Weight Updates.

Trains Spectral, FourierFT, and LoRA on BERT-base/CoLA, extracts the learned
ΔW for each method, and produces a signal-processing figure comparing their
frequency-domain structure.

Figure layout (2 rows × 4 columns):
  Row 1: 2D DCT log-power spectrum of ΔW for (a) Spectral, (b) FourierFT,
         (c) LoRA, plus (d) radial power spectral density line plot.
  Row 2: Spatial domain ΔW heatmaps for (e) Spectral, (f) FourierFT,
         (g) LoRA, plus (h) transform compaction curves.

Usage:
  python src/spectral_figure.py               # Full pipeline (train + plot)
  python src/spectral_figure.py --plot-only    # Reuse saved ΔW, just replot
"""
import os
import re
import sys
import time
import argparse

import numpy as np
from scipy.fft import dctn
try:  # heavy deps only needed for training; --plot-only runs without them
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
for var in ['HF_HOME', 'HF_DATASETS_CACHE', 'TRANSFORMERS_CACHE', 'TORCH_HOME']:
    os.environ.setdefault(var, os.path.join(os.getcwd(), 'data'))

# Our custom Spectral adapter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
try:  # heavy deps only needed for training; --plot-only runs without them
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from datasets import load_dataset
    from peft import FourierFTConfig, get_peft_model, TaskType
    from peft import LoraConfig as PeftLoraConfig
    from spectral_adapter import SpectralAdapterModel, SpectralAdapterLinear
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = 'bert-base-uncased'
TASK = 'cola'
OUT_DIR = os.path.join('results', 'spectral_figure')

# Method-specific configs
SPECTRAL_P = 16
SPECTRAL_Q = 16
FOURIERFT_N_FREQ = 256   # Match Spectral's 256 coefficients per layer
FOURIERFT_SCALING = 150.0
LORA_R = 8
LORA_ALPHA = 16
TARGET_MODULES = ['query', 'value']

# Training config
N_EPOCHS = 3
BATCH_SIZE = 32
LR = 2e-5
SEED = 42

# Representative layer for detailed panels
REP_LAYER = 'L06_query'


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def canonical_key(name):
    """Convert module path to canonical form like 'L06_query'."""
    match = re.search(r'layer[._](\d+)', name)
    if not match:
        return name.replace('.', '_')
    layer_num = int(match.group(1))
    if 'query' in name:
        return f'L{layer_num:02d}_query'
    elif 'value' in name:
        return f'L{layer_num:02d}_value'
    elif 'key' in name:
        return f'L{layer_num:02d}_key'
    else:
        return f'L{layer_num:02d}_other'


def get_tokenized_dataset(tokenizer):
    """Load and tokenize CoLA."""
    dataset = load_dataset('glue', TASK, cache_dir='./data')
    def tokenize_fn(examples):
        return tokenizer(
            examples['sentence'], padding='max_length',
            truncation=True, max_length=128,
        )
    tokenized = dataset.map(tokenize_fn, batched=True, load_from_cache_file=False)
    # Remove columns the Trainer doesn't need (avoids collator confusion)
    cols_to_remove = ['sentence', 'idx']
    for split in tokenized:
        tokenized[split] = tokenized[split].remove_columns(
            [c for c in cols_to_remove if c in tokenized[split].column_names]
        )
    print(f"  Tokenized dataset columns: {tokenized['train'].column_names}")
    return tokenized


def save_qv_weights(model):
    """Save query and value weights from a plain BERT model."""
    w0 = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if name.endswith('.query') or name.endswith('.value'):
                w0[name] = module.weight.detach().clone().cpu()
    return w0


def make_trainer(model, tokenizer, tokenized, method_name):
    """Build and return a HuggingFace Trainer."""
    training_args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, f'tmp_{method_name}'),
        num_train_epochs=N_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=100,
        save_strategy='no',
        report_to='none',
        seed=SEED,
        remove_unused_columns=False,
    )
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        tokenizer=tokenizer,
    )


# ---------------------------------------------------------------------------
# Training and ΔW extraction for each method
# ---------------------------------------------------------------------------

def train_spectral(tokenizer, tokenized):
    """Train Spectral adapter and extract ΔW."""
    print("\n" + "=" * 70)
    print(f"TRAINING: Spectral Adapter (p={SPECTRAL_P}, q={SPECTRAL_Q})")
    print("=" * 70)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, cache_dir='./data')

    model = SpectralAdapterModel(
        model, target_modules=TARGET_MODULES,
        p=SPECTRAL_P, q=SPECTRAL_Q, scaling=1.0,
    )
    model.print_trainable_parameters()

    trainer = make_trainer(model, tokenizer, tokenized, 'spectral')
    trainer.train()

    # Extract ΔW from each adapted layer
    delta_ws = {}
    model.eval()
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, SpectralAdapterLinear):
                dw = module.get_delta_weight().cpu().numpy().astype(np.float64)
                delta_ws[canonical_key(name)] = dw

    print(f"  Extracted ΔW for {len(delta_ws)} layers: {sorted(delta_ws.keys())}")

    del model, trainer
    torch.cuda.empty_cache()
    return delta_ws


def train_fourierft(tokenizer, tokenized):
    """Train FourierFT and extract ΔW."""
    print("\n" + "=" * 70)
    print(f"TRAINING: FourierFT (n_frequency={FOURIERFT_N_FREQ})")
    print("=" * 70)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, cache_dir='./data')

    # Save pre-training weights before peft wrapping changes names
    w0 = save_qv_weights(model)
    print(f"  Saved w0 for {len(w0)} layers")

    peft_config = FourierFTConfig(
        n_frequency=FOURIERFT_N_FREQ,
        target_modules=TARGET_MODULES,
        task_type=TaskType.SEQ_CLS,
        scaling=FOURIERFT_SCALING,
        random_loc_seed=777,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    trainer = make_trainer(model, tokenizer, tokenized, 'fourierft')
    trainer.train()

    # Merge adapter weights into base model
    model = model.merge_and_unload()

    # Extract ΔW by comparing to pre-training weights
    w_after = save_qv_weights(model)
    delta_ws = {}
    for name in w0:
        if name in w_after:
            dw = (w_after[name] - w0[name]).numpy().astype(np.float64)
            delta_ws[canonical_key(name)] = dw

    print(f"  Extracted ΔW for {len(delta_ws)} layers: {sorted(delta_ws.keys())}")

    del model, trainer
    torch.cuda.empty_cache()
    return delta_ws


def train_lora(tokenizer, tokenized):
    """Train LoRA and extract ΔW."""
    print("\n" + "=" * 70)
    print(f"TRAINING: LoRA (r={LORA_R})")
    print("=" * 70)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, cache_dir='./data')

    w0 = save_qv_weights(model)
    print(f"  Saved w0 for {len(w0)} layers")

    peft_config = PeftLoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    trainer = make_trainer(model, tokenizer, tokenized, 'lora')
    trainer.train()

    model = model.merge_and_unload()

    w_after = save_qv_weights(model)
    delta_ws = {}
    for name in w0:
        if name in w_after:
            dw = (w_after[name] - w0[name]).numpy().astype(np.float64)
            delta_ws[canonical_key(name)] = dw

    print(f"  Extracted ΔW for {len(delta_ws)} layers: {sorted(delta_ws.keys())}")

    del model, trainer
    torch.cuda.empty_cache()
    return delta_ws


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def radial_average_2d(spectrum_2d):
    """Compute radial average of a 2D power spectrum.

    Args:
        spectrum_2d: 2D array of power values (already squared).

    Returns:
        (freq_bins, avg_power): radial frequency bins and average power in each.
    """
    m, n = spectrum_2d.shape
    fy = np.arange(m, dtype=np.float64) / m
    fx = np.arange(n, dtype=np.float64) / n
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX ** 2 + FY ** 2)

    max_r = np.sqrt(0.5 ** 2 + 0.5 ** 2)
    n_bins = min(m, n) // 2
    bins = np.linspace(0, max_r, n_bins + 1)

    freqs = []
    powers = []
    for i in range(n_bins):
        mask = (R >= bins[i]) & (R < bins[i + 1])
        if mask.any():
            freqs.append((bins[i] + bins[i + 1]) / 2)
            powers.append(np.mean(spectrum_2d[mask]))

    return np.array(freqs), np.array(powers)


def compaction_curve_contiguous(dct_2d, total_energy, max_k=None):
    """Cumulative energy for contiguous [:k, :k] DCT blocks.

    This is the key signal-processing metric: for the learned ΔW from each
    method, how quickly does the energy accumulate when taking contiguous
    low-frequency DCT coefficients. Spectral's ΔW will jump to 100% at k=p
    because ALL its energy is in the [:p, :q] block by construction.
    """
    m, n = dct_2d.shape
    if max_k is None:
        max_k = min(m, n)
    ks = []
    fracs = []
    for k in range(1, max_k + 1):
        ks.append(k * k)
        frac = np.sum(dct_2d[:k, :k] ** 2) / total_energy if total_energy > 0 else 0
        fracs.append(frac)
    return np.array(ks), np.array(fracs)


# ---------------------------------------------------------------------------
# Measurement validation — numerically verify visual claims
# ---------------------------------------------------------------------------

def total_variation(arr):
    """Total variation of a 2D array (sum of absolute gradient)."""
    dy = np.abs(np.diff(arr, axis=0))
    dx = np.abs(np.diff(arr, axis=1))
    return np.sum(dy) + np.sum(dx)


def laplacian_energy(arr):
    """Mean squared discrete Laplacian (smoothness measure, lower = smoother)."""
    # Interior points only
    lap = (arr[:-2, 1:-1] + arr[2:, 1:-1] +
           arr[1:-1, :-2] + arr[1:-1, 2:] -
           4 * arr[1:-1, 1:-1])
    return np.mean(lap ** 2)


def spectral_rolloff(freq, psd, threshold=0.85):
    """Frequency below which `threshold` fraction of total PSD energy lies."""
    cumulative = np.cumsum(psd)
    total = cumulative[-1]
    if total == 0:
        return freq[-1]
    idx = np.searchsorted(cumulative, threshold * total)
    idx = min(idx, len(freq) - 1)
    return freq[idx]


def validate_results(spectral_dws, fourierft_dws, lora_dws, rep_layer):
    """Numerically validate that the figure's visual claims hold.

    Runs quantitative checks on the ΔW data and prints PASS/FAIL for each
    expected property. Does NOT assume Spectral wins — reports what the
    numbers actually say.

    Returns a dict of all measurements for programmatic inspection.
    """
    print("\n" + "=" * 70)
    print("MEASUREMENT VALIDATION")
    print("=" * 70)

    results = {}
    checks_passed = 0
    checks_failed = 0
    checks_total = 0

    def check(name, condition, detail):
        nonlocal checks_passed, checks_failed, checks_total
        checks_total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            checks_passed += 1
        else:
            checks_failed += 1
        print(f"  [{status}] {name}")
        print(f"         {detail}")

    # ===== 1. Compaction at k=p (across ALL layers) =====
    print("\n--- Compaction Analysis (all layers) ---")

    all_keys = sorted(set(spectral_dws.keys()) &
                      set(fourierft_dws.keys()) &
                      set(lora_dws.keys()))
    n_layers = len(all_keys)

    compact_at_p = {'Spectral': [], 'FourierFT': [], 'LoRA': []}
    for key in all_keys:
        for method_name, dws in [('Spectral', spectral_dws),
                                  ('FourierFT', fourierft_dws),
                                  ('LoRA', lora_dws)]:
            dct_2d = dctn(dws[key], type=2, norm='ortho')
            total_energy = np.sum(dct_2d ** 2)
            if total_energy > 0:
                block_energy = np.sum(dct_2d[:SPECTRAL_P, :SPECTRAL_Q] ** 2)
                frac = block_energy / total_energy
            else:
                frac = 0.0
            compact_at_p[method_name].append(frac)

    for method_name in ['Spectral', 'FourierFT', 'LoRA']:
        vals = compact_at_p[method_name]
        results[f'compaction_at_p_{method_name}_mean'] = np.mean(vals)
        results[f'compaction_at_p_{method_name}_min'] = np.min(vals)
        results[f'compaction_at_p_{method_name}_max'] = np.max(vals)
        print(f"  {method_name:10s} compaction at k={SPECTRAL_P}: "
              f"mean={np.mean(vals)*100:.4f}%, "
              f"min={np.min(vals)*100:.4f}%, "
              f"max={np.max(vals)*100:.4f}%")

    # Check: Spectral compaction should be ~100% (by construction)
    check("Spectral compaction at k=p >= 99.9% (all layers)",
          np.min(compact_at_p['Spectral']) >= 0.999,
          f"min={np.min(compact_at_p['Spectral'])*100:.4f}%")

    # Check: Spectral compaction should DOMINATE other methods
    spectral_mean = np.mean(compact_at_p['Spectral'])
    fourierft_mean = np.mean(compact_at_p['FourierFT'])
    lora_mean = np.mean(compact_at_p['LoRA'])

    check("Spectral compaction > FourierFT compaction (mean)",
          spectral_mean > fourierft_mean,
          f"Spectral={spectral_mean*100:.4f}% vs FourierFT={fourierft_mean*100:.4f}%")

    check("Spectral compaction > LoRA compaction (mean)",
          spectral_mean > lora_mean,
          f"Spectral={spectral_mean*100:.4f}% vs LoRA={lora_mean*100:.4f}%")

    # ===== 2. Spatial smoothness (Total Variation, Laplacian) =====
    print("\n--- Spatial Smoothness (all layers, normalized by ||ΔW||_F) ---")

    tv_vals = {'Spectral': [], 'FourierFT': [], 'LoRA': []}
    lap_vals = {'Spectral': [], 'FourierFT': [], 'LoRA': []}
    frob_vals = {'Spectral': [], 'FourierFT': [], 'LoRA': []}

    for key in all_keys:
        for method_name, dws in [('Spectral', spectral_dws),
                                  ('FourierFT', fourierft_dws),
                                  ('LoRA', lora_dws)]:
            dw = dws[key]
            frob = np.sqrt(np.sum(dw ** 2))
            frob_vals[method_name].append(frob)
            if frob > 0:
                # Normalize ΔW to unit Frobenius norm before measuring smoothness
                dw_normed = dw / frob
                tv_vals[method_name].append(total_variation(dw_normed))
                lap_vals[method_name].append(laplacian_energy(dw_normed))
            else:
                tv_vals[method_name].append(0.0)
                lap_vals[method_name].append(0.0)

    for method_name in ['Spectral', 'FourierFT', 'LoRA']:
        tv_mean = np.mean(tv_vals[method_name])
        lap_mean = np.mean(lap_vals[method_name])
        frob_mean = np.mean(frob_vals[method_name])
        results[f'tv_normalized_{method_name}'] = tv_mean
        results[f'laplacian_{method_name}'] = lap_mean
        results[f'frob_mean_{method_name}'] = frob_mean
        print(f"  {method_name:10s} TV(norm)={tv_mean:.4f}, "
              f"Lap(norm)={lap_mean:.2e}, ||ΔW||_F(mean)={frob_mean:.6f}")

    # Check: Spectral should be smoother (lower TV, lower Laplacian) than others
    check("Spectral is smoother than FourierFT (lower normalized TV)",
          np.mean(tv_vals['Spectral']) < np.mean(tv_vals['FourierFT']),
          f"Spectral TV={np.mean(tv_vals['Spectral']):.4f} vs "
          f"FourierFT TV={np.mean(tv_vals['FourierFT']):.4f}")

    check("Spectral is smoother than LoRA (lower normalized TV)",
          np.mean(tv_vals['Spectral']) < np.mean(tv_vals['LoRA']),
          f"Spectral TV={np.mean(tv_vals['Spectral']):.4f} vs "
          f"LoRA TV={np.mean(tv_vals['LoRA']):.4f}")

    check("Spectral has lower Laplacian energy than FourierFT",
          np.mean(lap_vals['Spectral']) < np.mean(lap_vals['FourierFT']),
          f"Spectral Lap={np.mean(lap_vals['Spectral']):.2e} vs "
          f"FourierFT Lap={np.mean(lap_vals['FourierFT']):.2e}")

    check("Spectral has lower Laplacian energy than LoRA",
          np.mean(lap_vals['Spectral']) < np.mean(lap_vals['LoRA']),
          f"Spectral Lap={np.mean(lap_vals['Spectral']):.2e} vs "
          f"LoRA Lap={np.mean(lap_vals['LoRA']):.2e}")

    # ===== 3. Radial PSD shape: spectral rolloff =====
    print("\n--- Radial PSD Shape (representative layer) ---")

    dw_s = spectral_dws[rep_layer]
    dw_f = fourierft_dws[rep_layer]
    dw_l = lora_dws[rep_layer]

    pow_s = dctn(dw_s, type=2, norm='ortho') ** 2
    pow_f = dctn(dw_f, type=2, norm='ortho') ** 2
    pow_l = dctn(dw_l, type=2, norm='ortho') ** 2

    freq_s, psd_s = radial_average_2d(pow_s)
    freq_f, psd_f = radial_average_2d(pow_f)
    freq_l, psd_l = radial_average_2d(pow_l)

    rolloff_s = spectral_rolloff(freq_s, psd_s, 0.85)
    rolloff_f = spectral_rolloff(freq_f, psd_f, 0.85)
    rolloff_l = spectral_rolloff(freq_l, psd_l, 0.85)

    results['rolloff_85_Spectral'] = rolloff_s
    results['rolloff_85_FourierFT'] = rolloff_f
    results['rolloff_85_LoRA'] = rolloff_l

    print(f"  85% energy rolloff frequency:")
    print(f"    Spectral:  {rolloff_s:.4f}")
    print(f"    FourierFT: {rolloff_f:.4f}")
    print(f"    LoRA:      {rolloff_l:.4f}")

    # Check: Spectral's energy should concentrate at LOWER frequencies
    check("Spectral has lower 85% rolloff than FourierFT (more low-freq concentrated)",
          rolloff_s < rolloff_f,
          f"Spectral rolloff={rolloff_s:.4f} vs FourierFT rolloff={rolloff_f:.4f}")

    check("Spectral has lower 85% rolloff than LoRA (more low-freq concentrated)",
          rolloff_s < rolloff_l,
          f"Spectral rolloff={rolloff_s:.4f} vs LoRA rolloff={rolloff_l:.4f}")

    # ===== 4. Dynamic range of normalized radial PSD =====
    print("\n--- PSD Dynamic Range (peak-normalized, representative layer) ---")

    for method_name, psd in [('Spectral', psd_s), ('FourierFT', psd_f), ('LoRA', psd_l)]:
        psd_norm = psd / (psd.max() + 1e-30)
        dr_db = 10 * np.log10(psd_norm.max() / (psd_norm.min() + 1e-30))
        results[f'psd_dynamic_range_dB_{method_name}'] = dr_db
        print(f"  {method_name:10s} dynamic range: {dr_db:.1f} dB")

    # Spectral should have larger dynamic range (sharp drop from low to high freq)
    check("Spectral has larger PSD dynamic range than FourierFT",
          results['psd_dynamic_range_dB_Spectral'] > results['psd_dynamic_range_dB_FourierFT'],
          f"Spectral={results['psd_dynamic_range_dB_Spectral']:.1f} dB vs "
          f"FourierFT={results['psd_dynamic_range_dB_FourierFT']:.1f} dB")

    # ===== 5. Low-freq vs high-freq energy ratio =====
    print("\n--- Low/High Frequency Energy Ratio (all layers) ---")

    lh_ratio = {'Spectral': [], 'FourierFT': [], 'LoRA': []}
    cutoff_k = SPECTRAL_P  # Split at the Spectral adapter's boundary

    for key in all_keys:
        for method_name, dws in [('Spectral', spectral_dws),
                                  ('FourierFT', fourierft_dws),
                                  ('LoRA', lora_dws)]:
            dct_2d = dctn(dws[key], type=2, norm='ortho')
            low_energy = np.sum(dct_2d[:cutoff_k, :cutoff_k] ** 2)
            total_energy = np.sum(dct_2d ** 2)
            high_energy = total_energy - low_energy
            if high_energy > 0:
                lh_ratio[method_name].append(low_energy / high_energy)
            else:
                lh_ratio[method_name].append(float('inf'))

    for method_name in ['Spectral', 'FourierFT', 'LoRA']:
        vals = lh_ratio[method_name]
        finite_vals = [v for v in vals if np.isfinite(v)]
        mean_val = np.mean(finite_vals) if finite_vals else float('inf')
        results[f'low_high_ratio_{method_name}'] = mean_val
        print(f"  {method_name:10s} low/high energy ratio: {mean_val:.4f} "
              f"(>{1e6:.0e} means essentially all low-freq)" if mean_val > 1e6
              else f"  {method_name:10s} low/high energy ratio: {mean_val:.6f}")

    check("Spectral low/high ratio > FourierFT",
          results['low_high_ratio_Spectral'] > results['low_high_ratio_FourierFT'],
          f"Spectral={results['low_high_ratio_Spectral']:.4f} vs "
          f"FourierFT={results['low_high_ratio_FourierFT']:.6f}")

    check("Spectral low/high ratio > LoRA",
          results['low_high_ratio_Spectral'] > results['low_high_ratio_LoRA'],
          f"Spectral={results['low_high_ratio_Spectral']:.4f} vs "
          f"LoRA={results['low_high_ratio_LoRA']:.6f}")

    # ===== 6. LoRA rank structure check =====
    print("\n--- LoRA Rank Structure (singular values of ΔW) ---")
    dw_l_rep = lora_dws[rep_layer]
    svd_vals = np.linalg.svd(dw_l_rep, compute_uv=False)
    # After merge_and_unload + float32 save/load, numerical noise fills
    # trailing singular values. Use 1% of σ_max as threshold.
    sv_threshold = 0.01 * svd_vals[0]
    significant_sv = svd_vals[svd_vals > sv_threshold]
    # Also measure energy concentration in top-r singular values
    sv_energy_top_r = np.sum(svd_vals[:LORA_R] ** 2)
    sv_energy_total = np.sum(svd_vals ** 2)
    sv_frac = sv_energy_top_r / sv_energy_total if sv_energy_total > 0 else 0
    results['lora_effective_rank'] = len(significant_sv)
    results['lora_rank_configured'] = LORA_R
    results['lora_sv_energy_top_r'] = sv_frac
    print(f"  LoRA effective rank (>1% of σ_max): {len(significant_sv)} "
          f"(configured r={LORA_R})")
    print(f"  Energy in top-{LORA_R} singular values: {sv_frac*100:.2f}%")
    print(f"  Top-5 σ: {svd_vals[:5]}")
    print(f"  σ[r]={svd_vals[LORA_R-1]:.6f}, σ[r+1]={svd_vals[LORA_R]:.6f} "
          f"(ratio={svd_vals[LORA_R]/svd_vals[LORA_R-1]:.2e})" if len(svd_vals) > LORA_R else "")
    check("LoRA ΔW energy concentrated in top-r singular values (>99%)",
          sv_frac >= 0.99,
          f"top-{LORA_R} SV energy = {sv_frac*100:.2f}% of total")

    # ===== Summary =====
    print(f"\n{'=' * 70}")
    print(f"VALIDATION SUMMARY: {checks_passed}/{checks_total} checks passed, "
          f"{checks_failed}/{checks_total} failed")
    print(f"{'=' * 70}")

    if checks_failed > 0:
        print("  WARNING: Some validation checks FAILED.")
        print("  The figure may NOT accurately support claims about Spectral's advantages.")
        print("  Review failed checks before using this figure in the paper.")
    else:
        print("  All checks passed. Quantitative measurements support the visual claims.")

    results['checks_passed'] = checks_passed
    results['checks_failed'] = checks_failed
    results['checks_total'] = checks_total

    return results


# ---------------------------------------------------------------------------
# Figure creation
# ---------------------------------------------------------------------------

def create_figure(spectral_dws, fourierft_dws, lora_dws, rep_layer):
    """Create the 2×4 signal-processing figure.

    Panels:
      (a-c) Row 1, cols 1-3: 2D DCT log-power spectrum (cropped to 128×128)
      (d)   Row 1, col 4:    Radial power spectral density (line plot)
      (e-g) Row 2, cols 1-3: Spatial domain ΔW heatmaps (cropped to 64×64)
      (h)   Row 2, col 4:    Transform compaction curves (line plot)
    """
    print("\n" + "=" * 70)
    print("CREATING FIGURE")
    print("=" * 70)
    print(f"  Representative layer: {rep_layer}")

    # --- Get representative layer ΔW ---
    dw_s = spectral_dws[rep_layer]
    dw_f = fourierft_dws[rep_layer]
    dw_l = lora_dws[rep_layer]
    m, n = dw_s.shape
    print(f"  ΔW shape: {m}×{n}")

    # --- 2D DCT of each method's ΔW ---
    dct_s = dctn(dw_s, type=2, norm='ortho')
    dct_f = dctn(dw_f, type=2, norm='ortho')
    dct_l = dctn(dw_l, type=2, norm='ortho')

    pow_s = dct_s ** 2
    pow_f = dct_f ** 2
    pow_l = dct_l ** 2

    # --- Radial PSD ---
    freq_s, psd_s = radial_average_2d(pow_s)
    freq_f, psd_f = radial_average_2d(pow_f)
    freq_l, psd_l = radial_average_2d(pow_l)

    # --- Compaction curves (averaged across all layers) ---
    print("  Computing compaction curves across all layers...")
    all_keys = sorted(set(spectral_dws.keys()) &
                      set(fourierft_dws.keys()) &
                      set(lora_dws.keys()))
    print(f"  Common layers: {len(all_keys)}")

    max_k_compact = 128  # up to 128×128 = 16384 coefficients
    all_compact_s = []
    all_compact_f = []
    all_compact_l = []

    for key in all_keys:
        d_s = dctn(spectral_dws[key], type=2, norm='ortho')
        d_f = dctn(fourierft_dws[key], type=2, norm='ortho')
        d_l = dctn(lora_dws[key], type=2, norm='ortho')

        te_s = np.sum(d_s ** 2)
        te_f = np.sum(d_f ** 2)
        te_l = np.sum(d_l ** 2)

        _, cf_s = compaction_curve_contiguous(d_s, te_s, max_k_compact)
        _, cf_f = compaction_curve_contiguous(d_f, te_f, max_k_compact)
        _, cf_l = compaction_curve_contiguous(d_l, te_l, max_k_compact)

        all_compact_s.append(cf_s)
        all_compact_f.append(cf_f)
        all_compact_l.append(cf_l)

    ks_compact = np.arange(1, max_k_compact + 1) ** 2  # number of coefficients
    compact_s = np.mean(all_compact_s, axis=0)
    compact_f = np.mean(all_compact_f, axis=0)
    compact_l = np.mean(all_compact_l, axis=0)

    # --- Print key stats ---
    print(f"\n  Compaction at k=16 ({SPECTRAL_P}×{SPECTRAL_Q} = 256 coefficients):")
    print(f"    Spectral:  {compact_s[SPECTRAL_P - 1] * 100:.2f}% energy captured")
    print(f"    FourierFT: {compact_f[SPECTRAL_P - 1] * 100:.4f}% energy captured")
    print(f"    LoRA:      {compact_l[SPECTRAL_P - 1] * 100:.4f}% energy captured")

    # --- Frobenius norms ---
    frob_s = np.sqrt(np.sum(dw_s ** 2))
    frob_f = np.sqrt(np.sum(dw_f ** 2))
    frob_l = np.sqrt(np.sum(dw_l ** 2))
    print(f"\n  ||ΔW||_F for {rep_layer}:")
    print(f"    Spectral:  {frob_s:.6f}")
    print(f"    FourierFT: {frob_f:.6f}")
    print(f"    LoRA:      {frob_l:.6f}")

    # =====================================================================
    # BUILD FIGURE
    # =====================================================================
    fig = plt.figure(figsize=(16, 6.6))
    gs = gridspec.GridSpec(
        2, 4, width_ratios=[1, 1, 1, 1.3],
        hspace=0.32, wspace=0.30,
        left=0.05, right=0.95, top=0.95, bottom=0.10,
    )

    methods = ['LYRA', 'FourierFT', 'LoRA']
    method_colors = {
        'LYRA': '#e63946',
        'FourierFT': '#457b9d',
        'LoRA': '#2a9d8f',
    }
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

    # -----------------------------------------------------------------
    # Row 1: 2D DCT Power Spectrum (log scale), cropped to 128×128
    # Per-method normalization to show each method's spectral STRUCTURE
    # -----------------------------------------------------------------
    crop = 128
    all_pow = [pow_s[:crop, :crop], pow_f[:crop, :crop], pow_l[:crop, :crop]]

    for col, (pw, method) in enumerate(zip(all_pow, methods)):
        ax = fig.add_subplot(gs[0, col])
        # Per-method color normalization: normalize by method's own energy
        pw_nonzero = pw[pw > 0]
        if len(pw_nonzero) > 0:
            vmin_m = np.percentile(pw_nonzero, 5)
            vmax_m = np.percentile(pw_nonzero, 99.5)
        else:
            vmin_m, vmax_m = 1e-20, 1.0
        pw_plot = pw.copy()
        pw_plot[pw_plot <= 0] = vmin_m * 0.01
        im = ax.imshow(
            pw_plot, norm=LogNorm(vmin=vmin_m, vmax=vmax_m),
            cmap='inferno', aspect='equal', interpolation='nearest',
            extent=[0, crop, crop, 0],
        )
        ax.set_title(f'{method} (DCT²)', fontsize=13, fontweight='bold')
        if col == 0:
            ax.set_ylabel('DCT freq (output)', fontsize=10)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel('DCT freq (input)', fontsize=10)
        ax.text(0.02, 0.98, panel_labels[col], transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='left',
                color='white', bbox=dict(boxstyle='round,pad=0.2',
                                         facecolor='black', alpha=0.5))

        # Draw the p×q contiguous block boundary for Spectral
        if method == 'LYRA':
            rect = Rectangle(
                (0, 0), SPECTRAL_Q, SPECTRAL_P,
                linewidth=2.5, edgecolor='cyan', facecolor='none',
                linestyle='--', label=f'{SPECTRAL_P}×{SPECTRAL_Q} block',
            )
            ax.add_patch(rect)
            ax.legend(loc='lower right', fontsize=8,
                      facecolor='black', edgecolor='cyan',
                      labelcolor='cyan')

    # -----------------------------------------------------------------
    # Panel (d): Radial PSD — normalized by peak to show spectral SHAPE
    # -----------------------------------------------------------------
    ax_psd = fig.add_subplot(gs[0, 3])
    for method, freq, psd in [('LYRA', freq_s, psd_s),
                                ('FourierFT', freq_f, psd_f),
                                ('LoRA', freq_l, psd_l)]:
        # Normalize by peak power to show shape (not magnitude)
        psd_norm = psd / (psd.max() + 1e-30)
        ax_psd.semilogy(freq, psd_norm, label=method,
                        color=method_colors[method], linewidth=2)

    ax_psd.set_xlabel('Radial frequency', fontsize=10)
    ax_psd.set_ylabel('Normalized power', fontsize=10)
    ax_psd.set_title('Radial PSD (normalized)', fontsize=13, fontweight='bold')
    ax_psd.legend(fontsize=9, framealpha=0.9)
    ax_psd.grid(True, alpha=0.3)
    ax_psd.set_ylim(1e-5, 2)
    ax_psd.text(0.02, 0.98, panel_labels[3], transform=ax_psd.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='left')

    # -----------------------------------------------------------------
    # Row 2: Spatial Domain ΔW (cropped to 64×64)
    # Per-method normalization to show each method's spatial PATTERN
    # -----------------------------------------------------------------
    patch = 64
    dws_plot = [dw_s[:patch, :patch], dw_f[:patch, :patch], dw_l[:patch, :patch]]

    for col, (dw, method) in enumerate(zip(dws_plot, methods)):
        ax = fig.add_subplot(gs[1, col])
        vmax_m = np.max(np.abs(dw)) if np.max(np.abs(dw)) > 0 else 1.0
        im2 = ax.imshow(
            dw, cmap='RdBu_r', vmin=-vmax_m, vmax=vmax_m,
            aspect='equal', interpolation='nearest',
            extent=[0, patch, patch, 0],
        )
        ax.set_title(f'{method} ΔW', fontsize=13, fontweight='bold')
        if col == 0:
            ax.set_ylabel('Output dim', fontsize=10)
        ax.set_xlabel('Input dim', fontsize=10)
        ax.text(0.02, 0.98, panel_labels[4 + col], transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white', alpha=0.7))

    # -----------------------------------------------------------------
    # Panel (h): Transform Compaction Curves
    # -----------------------------------------------------------------
    ax_compact = fig.add_subplot(gs[1, 3])
    for method, curve in [('LYRA', compact_s),
                          ('FourierFT', compact_f),
                          ('LoRA', compact_l)]:
        ax_compact.semilogx(ks_compact, curve * 100, label=method,
                            color=method_colors[method], linewidth=2)

    # Mark the p×q point
    idx_pq = SPECTRAL_P - 1  # index for k=16 → 16²=256 coefficients
    ax_compact.axvline(x=SPECTRAL_P ** 2, color='gray', linestyle=':',
                       alpha=0.7, linewidth=1)
    ax_compact.annotate(
        f'{SPECTRAL_P}²={SPECTRAL_P**2}',
        xy=(SPECTRAL_P ** 2, compact_s[idx_pq] * 100),
        xytext=(SPECTRAL_P ** 2 * 3, 70),
        fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'),
        color='gray',
    )

    ax_compact.set_xlabel('Number of DCT coefficients (contiguous)', fontsize=10)
    ax_compact.set_ylabel('Cumulative energy (%)', fontsize=10)
    ax_compact.set_title('Transform Compaction', fontsize=13, fontweight='bold')
    ax_compact.legend(fontsize=9, framealpha=0.9, loc='center right')
    ax_compact.grid(True, alpha=0.3)
    ax_compact.set_ylim(-5, 105)
    ax_compact.text(0.02, 0.98, panel_labels[7], transform=ax_compact.transAxes,
                    fontsize=12, fontweight='bold', va='top', ha='left')

    # Suptitle omitted: the manuscript caption already titles the figure.

    # --- Save ---
    fig_pdf = os.path.join(OUT_DIR, 'spectral_anatomy.pdf')
    fig_png = os.path.join(OUT_DIR, 'spectral_anatomy.png')
    fig.savefig(fig_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(fig_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure saved to:\n    {fig_pdf}\n    {fig_png}")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_all_delta_ws(spectral_dws, fourierft_dws, lora_dws, path):
    """Save ΔW dictionaries for all methods to a single npz file."""
    save_dict = {}
    for key, dw in spectral_dws.items():
        save_dict[f'spectral__{key}'] = dw.astype(np.float32)
    for key, dw in fourierft_dws.items():
        save_dict[f'fourierft__{key}'] = dw.astype(np.float32)
    for key, dw in lora_dws.items():
        save_dict[f'lora__{key}'] = dw.astype(np.float32)
    np.savez_compressed(path, **save_dict)
    print(f"  Saved {len(save_dict)} ΔW matrices to {path}")


def load_all_delta_ws(path):
    """Load ΔW dictionaries from npz."""
    data = np.load(path)
    spectral_dws = {}
    fourierft_dws = {}
    lora_dws = {}
    for full_key in data.files:
        method, layer_key = full_key.split('__', 1)
        arr = data[full_key].astype(np.float64)
        if method == 'spectral':
            spectral_dws[layer_key] = arr
        elif method == 'fourierft':
            fourierft_dws[layer_key] = arr
        elif method == 'lora':
            lora_dws[layer_key] = arr
    return spectral_dws, fourierft_dws, lora_dws


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip training, load saved ΔW data and replot.')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    npz_path = os.path.join(OUT_DIR, 'delta_ws.npz')

    if args.plot_only and os.path.exists(npz_path):
        print(f"Loading saved ΔW from {npz_path}")
        spectral_dws, fourierft_dws, lora_dws = load_all_delta_ws(npz_path)
        print(f"  Spectral: {len(spectral_dws)} layers")
        print(f"  FourierFT: {len(fourierft_dws)} layers")
        print(f"  LoRA: {len(lora_dws)} layers")
    else:
        # Prepare shared resources
        print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir='./data')
        tokenized = get_tokenized_dataset(tokenizer)

        # Train all three methods
        spectral_dws = train_spectral(tokenizer, tokenized)
        fourierft_dws = train_fourierft(tokenizer, tokenized)
        lora_dws = train_lora(tokenizer, tokenized)

        # Save for reuse
        save_all_delta_ws(spectral_dws, fourierft_dws, lora_dws, npz_path)

    # Verify representative layer exists
    if REP_LAYER not in spectral_dws:
        available = sorted(spectral_dws.keys())
        print(f"  WARNING: {REP_LAYER} not found. Available: {available}")
        # Fall back to middle layer query
        queries = [k for k in available if 'query' in k]
        rep = queries[len(queries) // 2] if queries else available[len(available) // 2]
        print(f"  Using: {rep}")
    else:
        rep = REP_LAYER

    # Validate measurements BEFORE creating figure
    validation = validate_results(spectral_dws, fourierft_dws, lora_dws, rep)

    if validation['checks_failed'] > 0:
        print("\n  *** WARNING: Not all validation checks passed. ***")
        print("  *** Review the failures above before trusting the figure. ***\n")

    # Create figure
    create_figure(spectral_dws, fourierft_dws, lora_dws, rep)

    total = time.time() - t0
    print(f"\nTotal time: {total:.0f}s")


if __name__ == '__main__':
    main()
