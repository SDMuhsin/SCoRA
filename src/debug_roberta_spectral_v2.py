"""
Deep root-cause diagnostic: Why does Spectral fail on RoBERTa CoLA?

Tests 3 hypotheses:
1. Classifier-only baseline (no adapter) - can frozen features solve CoLA?
2. Adapter destructiveness - does Spectral's init output harm learning?
3. Gradient clipping interaction - does clipping starve the classifier?
"""
import os, sys, math, copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from spectral_adapter import get_spectral_adapter_model, SpectralAdapterLinear

os.environ['HF_HOME'] = os.environ.get('HF_HOME', './data')
os.environ['HF_DATASETS_CACHE'] = os.environ.get('HF_DATASETS_CACHE', './data')
os.environ['TRANSFORMERS_CACHE'] = os.environ.get('TRANSFORMERS_CACHE', './data')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_cola_data(tokenizer, max_len=128, batch_size=32):
    """Load CoLA dataset and return train/eval loaders."""
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
    """Compute MCC on eval set."""
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


def train_one_epoch(model, train_loader, optimizer, lr_scheduler, device,
                    grad_clip=1.0, verbose_grads=False):
    """Train one epoch, optionally reporting gradient stats."""
    model.train()
    total_loss = 0
    n_batches = 0
    adapter_norms = []
    classifier_norms = []
    total_norms = []

    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        if verbose_grads and n_batches < 5:
            # Measure gradient norms BEFORE clipping
            a_norm_sq = 0.0
            c_norm_sq = 0.0
            other_norm_sq = 0.0
            for name, p in model.named_parameters():
                if p.grad is not None:
                    g_sq = p.grad.data.norm(2).item() ** 2
                    if 'coeffs' in name or 'log_scaling' in name:
                        a_norm_sq += g_sq
                    elif 'classifier' in name or 'score' in name:
                        c_norm_sq += g_sq
                    else:
                        other_norm_sq += g_sq
            total_norm = math.sqrt(a_norm_sq + c_norm_sq + other_norm_sq)
            adapter_norms.append(math.sqrt(a_norm_sq))
            classifier_norms.append(math.sqrt(c_norm_sq))
            total_norms.append(total_norm)

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        n_batches += 1

    if verbose_grads and adapter_norms:
        print(f"    Grad norms (first 5 batches, BEFORE clip):")
        for i in range(len(adapter_norms)):
            clip_ratio = 1.0 / max(total_norms[i], grad_clip) * grad_clip
            print(f"      batch {i}: adapter={adapter_norms[i]:.4f}, "
                  f"classifier={classifier_norms[i]:.4f}, "
                  f"total={total_norms[i]:.4f}, "
                  f"clip_scale={clip_ratio:.4f}")

    return total_loss / max(n_batches, 1)


def run_experiment(name, model, train_loader, eval_loader, device,
                   lr=5e-2, epochs=10, grad_clip=1.0, verbose_first=True):
    """Run a complete training experiment."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*60}")

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # Show which param groups are trainable
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"    TRAINABLE: {n} shape={list(p.shape)} numel={p.numel()}")

    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)

    best_mcc = 0.0
    for ep in range(epochs):
        verbose = verbose_first and ep == 0
        avg_loss = train_one_epoch(model, train_loader, optimizer, lr_scheduler,
                                    device, grad_clip=grad_clip, verbose_grads=verbose)
        mcc = evaluate_mcc(model, eval_loader, device)
        best_mcc = max(best_mcc, mcc)
        print(f"  epoch {ep}: loss={avg_loss:.4f}, MCC={mcc:.4f}, best={best_mcc:.4f}")

    print(f"  FINAL best MCC: {best_mcc:.4f}")
    return best_mcc


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("DEEP ROOT-CAUSE DIAGNOSTIC: Spectral on RoBERTa CoLA")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained('roberta-base', cache_dir='./data')
    train_loader, eval_loader = get_cola_data(tokenizer, batch_size=32)

    # ===== EXPERIMENT 1: Classifier-only (no adapter) =====
    # If this gives MCC=0, frozen RoBERTa features can't solve CoLA with this
    # training setup, and the issue is NOT Spectral-specific
    model = AutoModelForSequenceClassification.from_pretrained(
        'roberta-base', num_labels=2, cache_dir='./data')
    # Freeze all except classifier
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'classifier' in name or 'score' in name:
            param.requires_grad = True

    run_experiment("Classifier-only (no adapter, frozen features)",
                   model, train_loader, eval_loader, device,
                   lr=5e-2, epochs=10, grad_clip=1.0)
    del model
    torch.cuda.empty_cache()

    # ===== EXPERIMENT 2: Spectral s=1 (minimal adapter signal) =====
    model = AutoModelForSequenceClassification.from_pretrained(
        'roberta-base', num_labels=2, cache_dir='./data')
    model = get_spectral_adapter_model(
        model, target_modules=['query', 'value'],
        p=16, q=16, scaling=1.0, d_initial=0.01, freq_mode='contiguous')

    run_experiment("Spectral s=1.0, d=0.01 (near-zero adapter)",
                   model, train_loader, eval_loader, device,
                   lr=5e-2, epochs=10, grad_clip=1.0)
    del model
    torch.cuda.empty_cache()

    # ===== EXPERIMENT 3: Spectral s=50 (moderate) =====
    model = AutoModelForSequenceClassification.from_pretrained(
        'roberta-base', num_labels=2, cache_dir='./data')
    model = get_spectral_adapter_model(
        model, target_modules=['query', 'value'],
        p=16, q=16, scaling=50.0, d_initial=0.01, freq_mode='contiguous')

    run_experiment("Spectral s=50, d=0.01",
                   model, train_loader, eval_loader, device,
                   lr=5e-2, epochs=10, grad_clip=1.0)
    del model
    torch.cuda.empty_cache()

    # ===== EXPERIMENT 4: Spectral s=50, NO gradient clipping =====
    # Tests if grad clipping is the culprit
    model = AutoModelForSequenceClassification.from_pretrained(
        'roberta-base', num_labels=2, cache_dir='./data')
    model = get_spectral_adapter_model(
        model, target_modules=['query', 'value'],
        p=16, q=16, scaling=50.0, d_initial=0.01, freq_mode='contiguous')

    run_experiment("Spectral s=50, d=0.01, NO grad clip",
                   model, train_loader, eval_loader, device,
                   lr=5e-2, epochs=10, grad_clip=0.0)
    del model
    torch.cuda.empty_cache()

    # ===== EXPERIMENT 5: Spectral s=50, lower LR =====
    # Tests if effective LR is too high (lr * scaling = 5e-2 * 50 = 2.5)
    model = AutoModelForSequenceClassification.from_pretrained(
        'roberta-base', num_labels=2, cache_dir='./data')
    model = get_spectral_adapter_model(
        model, target_modules=['query', 'value'],
        p=16, q=16, scaling=50.0, d_initial=0.01, freq_mode='contiguous')

    run_experiment("Spectral s=50, d=0.01, lr=1e-3",
                   model, train_loader, eval_loader, device,
                   lr=1e-3, epochs=10, grad_clip=1.0)
    del model
    torch.cuda.empty_cache()

    # ===== EXPERIMENT 6: Spectral s=150 (to match FourierFT's scaling) =====
    model = AutoModelForSequenceClassification.from_pretrained(
        'roberta-base', num_labels=2, cache_dir='./data')
    model = get_spectral_adapter_model(
        model, target_modules=['query', 'value'],
        p=16, q=16, scaling=150.0, d_initial=0.01, freq_mode='contiguous')

    run_experiment("Spectral s=150, d=0.01",
                   model, train_loader, eval_loader, device,
                   lr=5e-2, epochs=10, grad_clip=1.0)
    del model
    torch.cuda.empty_cache()

    # ===== EXPERIMENT 7: Spectral s=150, smaller d_initial =====
    model = AutoModelForSequenceClassification.from_pretrained(
        'roberta-base', num_labels=2, cache_dir='./data')
    model = get_spectral_adapter_model(
        model, target_modules=['query', 'value'],
        p=16, q=16, scaling=150.0, d_initial=0.001, freq_mode='contiguous')

    run_experiment("Spectral s=150, d=0.001 (smaller init)",
                   model, train_loader, eval_loader, device,
                   lr=5e-2, epochs=10, grad_clip=1.0)
    del model
    torch.cuda.empty_cache()

    print("\n\nDONE.")


if __name__ == '__main__':
    main()
