"""
Scaling sweep for hybrid freq_mode — the ONLY mode that works.

Previous finding: hybrid s=0.2 p=16 q=16 → MCC=0.5239 in 15 epochs!
All other freq_modes (contiguous, geometric) collapse to MCC=0.0.

Now test different scaling values with hybrid to find optimal.
Also test with 30 epochs (matching production config).
"""
import os, sys, math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from spectral_adapter import get_spectral_adapter_model

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
    train_loader = DataLoader(ds['train'], batch_size=32, shuffle=True)
    eval_loader = DataLoader(ds['validation'], batch_size=32)
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


def train_one_epoch(model, train_loader, optimizer, lr_scheduler, device,
                    grad_clip=1.0):
    model.train()
    total_loss = 0
    n_batches = 0
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
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def run_experiment(name, model, train_loader, eval_loader, device,
                   lr=5e-2, epochs=30, grad_clip=1.0):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*60}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter_p = sum(p.numel() for n, p in model.named_parameters()
                    if p.requires_grad and ('coeffs' in n or 'log_scaling' in n))
    print(f"  Trainable: {trainable:,} (adapter: {adapter_p:,}, classifier: {trainable-adapter_p:,})")

    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)

    best_mcc = 0.0
    for ep in range(epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, lr_scheduler,
                                    device, grad_clip=grad_clip)
        mcc = evaluate_mcc(model, eval_loader, device)
        best_mcc = max(best_mcc, mcc)
        print(f"  epoch {ep}: loss={avg_loss:.4f}, MCC={mcc:.4f}, best={best_mcc:.4f}")

    print(f"  FINAL best MCC: {best_mcc:.4f}")
    return best_mcc


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("HYBRID SCALING SWEEP: Finding optimal scaling for hybrid freq_mode")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained('roberta-base', cache_dir='./data')
    train_loader, eval_loader = get_cola_data(tokenizer, batch_size=32)

    configs = [
        # (name, scaling, d_initial, p, q, epochs)
        ("hybrid s=0.1 p=16 q=16", 0.1, 0.01, 16, 16, 30),
        ("hybrid s=0.2 p=16 q=16", 0.2, 0.01, 16, 16, 30),
        ("hybrid s=0.5 p=16 q=16", 0.5, 0.01, 16, 16, 30),
        ("hybrid s=1.0 p=16 q=16", 1.0, 0.01, 16, 16, 30),
        ("hybrid s=0.3 p=16 q=16", 0.3, 0.01, 16, 16, 30),
    ]

    results = {}
    for name, scaling, d_initial, p, q, epochs in configs:
        model = AutoModelForSequenceClassification.from_pretrained(
            'roberta-base', num_labels=2, cache_dir='./data')
        model = get_spectral_adapter_model(
            model, target_modules=['query', 'value'],
            p=p, q=q, scaling=scaling, d_initial=d_initial,
            freq_mode='hybrid')

        mcc = run_experiment(f"Spectral {name}",
                       model, train_loader, eval_loader, device,
                       lr=5e-2, epochs=epochs, grad_clip=1.0)
        results[name] = mcc
        del model
        torch.cuda.empty_cache()

    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  FourierFT s=150 control (15ep): MCC=0.4218")
    print(f"  FourierFT s=150 Mo5 (30ep):     MCC=0.557 (target to beat)")
    for name, mcc in results.items():
        marker = " <<<" if mcc > 0.4 else ""
        print(f"  Spectral {name}: MCC={mcc:.4f}{marker}")
    print("\nDONE.")


if __name__ == '__main__':
    main()
