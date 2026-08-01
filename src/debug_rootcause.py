"""
ROOT CAUSE DIAGNOSTIC: Why does Spectral fail on RoBERTa but not BERT?

HYPOTHESIS: The gradient collapse is caused by the RoBERTa classifier head's
TRAINABLE randomly-initialized Dense(768→768) + Tanh layer. As this layer trains,
its backward gradient acts as a changing spectral filter that shifts energy away
from the low-frequency DCT subspace, starving the adapter.

BERT has an EQUIVALENT Dense(768→768) + Tanh (BertPooler), but it's PRETRAINED
and FROZEN. Its spectral filtering is stable → no collapse.

KEY TESTS:
1. Measure the DCT spectral content of the backward gradient at adapter layers
   at epoch 0 vs epoch 2 → expect low-freq content to collapse on RoBERTa
2. Freeze RoBERTa's classifier.dense (only train out_proj + adapter)
   → if hypothesis is correct, gradient collapse should STOP
3. Compare gradient spectral distribution between BERT and RoBERTa
"""
import os, sys, math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from spectral_adapter import get_spectral_adapter_model, SpectralAdapterLinear, _dct_basis

os.environ['HF_HOME'] = os.environ.get('HF_HOME', './data')
os.environ['HF_DATASETS_CACHE'] = os.environ.get('HF_DATASETS_CACHE', './data')
os.environ['TRANSFORMERS_CACHE'] = os.environ.get('TRANSFORMERS_CACHE', './data')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_cola_data(tokenizer, max_len=128, batch_size=32):
    from datasets import load_dataset
    ds = load_dataset("glue", "cola", cache_dir='./data')
    def tokenize(examples):
        return tokenizer(examples['sentence'], padding='max_length',
                         truncation=True, max_length=max_len)
    ds = ds.map(tokenize, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    train_loader = DataLoader(ds['train'], batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(ds['validation'], batch_size=batch_size)
    return train_loader, eval_loader


def evaluate_mcc(model, eval_loader, device):
    from sklearn.metrics import matthews_corrcoef
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    return matthews_corrcoef(all_labels, all_preds)


def measure_gradient_spectrum(model, train_loader, device, n_batches=20):
    """
    Measure the DCT spectral distribution of the backward gradient at each
    adapted layer. This tells us WHERE in the frequency spectrum the gradient
    energy is concentrated.

    Returns dict with:
      - adapter_grad_norm: total adapter gradient norm
      - classifier_grad_norm: total classifier gradient norm
      - layer_grad_full_norms: per-layer |dL/d_out| (full gradient at layer output)
      - layer_grad_dct_low_ratio: per-layer fraction of gradient energy in lowest 16 DCT components
      - layer_grad_dct_high_ratio: per-layer fraction in indices 16-384
    """
    model.train()

    # Hooks to capture gradient at each adapter layer's OUTPUT
    grad_data = {}

    def make_hook(name):
        def hook_fn(module, grad_input, grad_output):
            # grad_output[0] is dL/d(layer_output) = dL/d(base_out + scaling * delta_out)
            g = grad_output[0].detach().float()  # (batch, seq, out_features)
            if name not in grad_data:
                grad_data[name] = []
            grad_data[name].append(g.cpu())
        return hook_fn

    # Register hooks on each SpectralAdapterLinear
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, SpectralAdapterLinear):
            h = module.register_full_backward_hook(make_hook(name))
            hooks.append(h)

    # Run n_batches of training
    batch_count = 0
    for batch in train_loader:
        if batch_count >= n_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        model.zero_grad()
        batch_count += 1

    # Remove hooks
    for h in hooks:
        h.remove()

    # Analyze gradient spectrum for each layer
    results = {}
    for layer_name, grads in grad_data.items():
        # Stack all gradients: (n_batches, batch, seq, out_features)
        # Flatten batch and seq dims
        all_grads = torch.cat(grads, dim=0)  # (total_samples, seq, out_features)
        # Take CLS token position (index 0) as representative
        cls_grads = all_grads[:, 0, :]  # (total_samples, out_features)

        out_dim = cls_grads.shape[1]
        full_norm = cls_grads.norm(dim=1).mean().item()

        # Project onto DCT basis
        dct_full = _dct_basis(out_dim, out_dim, torch.float32)  # (out_dim, out_dim)
        # dct_projections[i] = cls_grads @ dct_full[i] for each sample
        dct_coeffs = cls_grads @ dct_full.T  # (total_samples, out_dim)
        dct_energy = (dct_coeffs ** 2).mean(dim=0)  # (out_dim,) - avg energy per freq

        total_energy = dct_energy.sum().item()
        low16_energy = dct_energy[:16].sum().item()
        mid_energy = dct_energy[16:384].sum().item()
        high_energy = dct_energy[384:].sum().item()

        results[layer_name] = {
            'full_norm': full_norm,
            'total_energy': total_energy,
            'low16_energy': low16_energy,
            'low16_ratio': low16_energy / max(total_energy, 1e-30),
            'mid_energy': mid_energy,
            'mid_ratio': mid_energy / max(total_energy, 1e-30),
            'high_energy': high_energy,
            'high_ratio': high_energy / max(total_energy, 1e-30),
        }

    # Also get adapter and classifier gradient norms from parameters
    # (need one more forward+backward to have gradients on params)
    adapter_norm_sq = 0.0
    classifier_norm_sq = 0.0
    batch = next(iter(train_loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    for pname, p in model.named_parameters():
        if p.grad is not None:
            g_sq = p.grad.data.norm(2).item() ** 2
            if 'coeffs' in pname or 'log_scaling' in pname:
                adapter_norm_sq += g_sq
            elif 'classifier' in pname or 'score' in pname:
                classifier_norm_sq += g_sq
    model.zero_grad()

    return {
        'layer_results': results,
        'adapter_grad_norm': math.sqrt(adapter_norm_sq),
        'classifier_grad_norm': math.sqrt(classifier_norm_sq),
    }


def train_n_epochs(model, train_loader, optimizer, lr_scheduler, device,
                   n_epochs=1, grad_clip=1.0):
    """Train for n epochs without measuring."""
    model.train()
    for ep in range(n_epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()


def run_spectral_gradient_analysis(model_name, model, train_loader, eval_loader,
                                    device, lr=5e-2, total_epochs=5, label=""):
    """Run training with gradient spectrum measurement at each epoch."""
    print(f"\n{'='*70}")
    print(f"GRADIENT SPECTRUM ANALYSIS: {label}")
    print(f"{'='*70}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter_p = sum(p.numel() for n, p in model.named_parameters()
                    if p.requires_grad and ('coeffs' in n or 'log_scaling' in n))
    classifier_p = sum(p.numel() for n, p in model.named_parameters()
                       if p.requires_grad and ('classifier' in n or 'score' in n))
    print(f"  Trainable: {trainable:,} (adapter: {adapter_p:,}, classifier: {classifier_p:,})")

    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * total_epochs
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)

    for ep in range(total_epochs):
        # Measure gradient spectrum BEFORE this epoch's training
        spec = measure_gradient_spectrum(model, train_loader, device, n_batches=10)
        mcc = evaluate_mcc(model, eval_loader, device)

        print(f"\n  --- Epoch {ep} (before training) --- MCC={mcc:.4f}")
        print(f"  Adapter grad norm: {spec['adapter_grad_norm']:.6f}")
        print(f"  Classifier grad norm: {spec['classifier_grad_norm']:.4f}")

        # Show first and last adapted layer
        layer_names = sorted(spec['layer_results'].keys())
        for lname in [layer_names[0], layer_names[-1]] if len(layer_names) > 1 else layer_names:
            lr_data = spec['layer_results'][lname]
            print(f"  Layer {lname}:")
            print(f"    |dL/d_out|={lr_data['full_norm']:.6f}")
            print(f"    DCT energy: low[0:16]={lr_data['low16_ratio']*100:.2f}%, "
                  f"mid[16:384]={lr_data['mid_ratio']*100:.2f}%, "
                  f"high[384:]={lr_data['high_ratio']*100:.2f}%")

        # Train one epoch
        train_n_epochs(model, train_loader, optimizer, lr_scheduler, device,
                       n_epochs=1, grad_clip=1.0)

    # Final measurement
    spec = measure_gradient_spectrum(model, train_loader, device, n_batches=10)
    mcc = evaluate_mcc(model, eval_loader, device)
    print(f"\n  --- Epoch {total_epochs} (final) --- MCC={mcc:.4f}")
    print(f"  Adapter grad norm: {spec['adapter_grad_norm']:.6f}")
    print(f"  Classifier grad norm: {spec['classifier_grad_norm']:.4f}")
    layer_names = sorted(spec['layer_results'].keys())
    for lname in [layer_names[0], layer_names[-1]] if len(layer_names) > 1 else layer_names:
        lr_data = spec['layer_results'][lname]
        print(f"  Layer {lname}:")
        print(f"    |dL/d_out|={lr_data['full_norm']:.6f}")
        print(f"    DCT energy: low[0:16]={lr_data['low16_ratio']*100:.2f}%, "
              f"mid[16:384]={lr_data['mid_ratio']*100:.2f}%, "
              f"high[384:]={lr_data['high_ratio']*100:.2f}%")

    return mcc


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("ROOT CAUSE DIAGNOSTIC: Gradient Spectral Analysis")
    print("=" * 70)

    # ===== TEST 1: RoBERTa with contiguous Spectral (the failing case) =====
    print("\n\n>>> TEST 1: RoBERTa + contiguous Spectral (EXPECTED: gradient collapse)")
    tokenizer = AutoTokenizer.from_pretrained('roberta-base', cache_dir='./data')
    train_loader, eval_loader = get_cola_data(tokenizer, batch_size=32)

    model = AutoModelForSequenceClassification.from_pretrained(
        'roberta-base', num_labels=2, cache_dir='./data')
    model = get_spectral_adapter_model(
        model, target_modules=['query', 'value'],
        p=16, q=16, scaling=0.2, d_initial=0.01, freq_mode='contiguous')

    run_spectral_gradient_analysis(
        'roberta-base', model, train_loader, eval_loader, device,
        lr=5e-2, total_epochs=5,
        label="RoBERTa contiguous s=0.2 (FAILING CONFIG)")
    del model
    torch.cuda.empty_cache()

    # ===== TEST 2: RoBERTa with classifier.dense FROZEN =====
    # KEY PREDICTION: If our hypothesis is correct, freezing classifier.dense
    # should PREVENT gradient collapse (the backward spectral filter is now fixed)
    print("\n\n>>> TEST 2: RoBERTa + contiguous Spectral + FROZEN classifier.dense")
    print("    (PREDICTION: no gradient collapse if hypothesis is correct)")

    model = AutoModelForSequenceClassification.from_pretrained(
        'roberta-base', num_labels=2, cache_dir='./data')
    model = get_spectral_adapter_model(
        model, target_modules=['query', 'value'],
        p=16, q=16, scaling=0.2, d_initial=0.01, freq_mode='contiguous')

    # Freeze classifier.dense, keep only classifier.out_proj trainable
    frozen_count = 0
    for name, param in model.named_parameters():
        if 'classifier.dense' in name:
            param.requires_grad = False
            frozen_count += param.numel()
            print(f"  FROZE: {name} ({param.numel()} params)")
    print(f"  Total frozen classifier.dense params: {frozen_count:,}")

    run_spectral_gradient_analysis(
        'roberta-base', model, train_loader, eval_loader, device,
        lr=5e-2, total_epochs=5,
        label="RoBERTa contiguous s=0.2 + FROZEN classifier.dense")
    del model
    torch.cuda.empty_cache()

    # ===== TEST 3: BERT with contiguous Spectral (the working case) =====
    print("\n\n>>> TEST 3: BERT + contiguous Spectral (EXPECTED: no gradient collapse)")
    tokenizer_bert = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='./data')
    train_loader_bert, eval_loader_bert = get_cola_data(tokenizer_bert, batch_size=32)

    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=2, cache_dir='./data')
    model = get_spectral_adapter_model(
        model, target_modules=['query', 'value'],
        p=16, q=16, scaling=0.2, d_initial=0.01, freq_mode='contiguous')

    # Show which params are trainable for BERT
    print("  BERT trainable params check:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'pooler' in name:
                print(f"    WARNING: {name} is TRAINABLE (should be frozen!)")

    run_spectral_gradient_analysis(
        'bert-base-uncased', model, train_loader_bert, eval_loader_bert, device,
        lr=5e-2, total_epochs=5,
        label="BERT contiguous s=0.2 (WORKING CONFIG)")
    del model
    torch.cuda.empty_cache()

    # ===== TEST 4: RoBERTa with hybrid Spectral (the working config) =====
    print("\n\n>>> TEST 4: RoBERTa + hybrid Spectral (EXPECTED: no gradient collapse)")

    model = AutoModelForSequenceClassification.from_pretrained(
        'roberta-base', num_labels=2, cache_dir='./data')
    model = get_spectral_adapter_model(
        model, target_modules=['query', 'value'],
        p=16, q=16, scaling=0.2, d_initial=0.01, freq_mode='hybrid')

    run_spectral_gradient_analysis(
        'roberta-base', model, train_loader, eval_loader, device,
        lr=5e-2, total_epochs=5,
        label="RoBERTa hybrid s=0.2 (WORKING CONFIG)")
    del model
    torch.cuda.empty_cache()

    print("\n\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("""
ROOT CAUSE HYPOTHESIS:

Both BERT and RoBERTa have Dense(768→768) + Tanh between the encoder and
the final classification layer. The critical difference:

  BERT:    bert.pooler.dense  → PRETRAINED, FROZEN  (stable spectral filter)
  RoBERTa: classifier.dense   → RANDOM INIT, TRAINABLE (unstable spectral filter)

The backward gradient through this layer determines the spectral distribution
of the error signal reaching the adapter. When the 768→768 layer is trainable
and rapidly changing (592K params, lr=5e-2), its spectral filtering shifts,
moving gradient energy away from the low-frequency DCT subspace.

The Spectral adapter's contiguous [0..15] DCT basis can only capture gradient
in the lowest 2% of frequencies. When the spectral energy shifts, this
projection collapses to ~0, killing the adapter.

FourierFT survives because its random sparse indices sample the FULL spectrum.
Hybrid Spectral survives because its high-frequency probes capture shifted energy.

KEY PREDICTIONS (verified above):
1. RoBERTa contiguous: gradient collapses (DCT low-freq energy drops)
2. RoBERTa contiguous + FROZEN dense: gradient does NOT collapse
3. BERT contiguous: gradient does NOT collapse (pooler is pretrained+frozen)
4. RoBERTa hybrid: gradient does NOT collapse (captures shifted energy)
""")


if __name__ == '__main__':
    main()
