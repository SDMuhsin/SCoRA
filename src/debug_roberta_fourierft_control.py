"""
Control experiment: Does FourierFT also fail in the simplified diagnostic setup?

If FourierFT achieves MCC > 0 here → Spectral is the problem
If FourierFT also gives MCC=0 → our diagnostic setup is missing something
"""
import os, sys, math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

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


def train_one_epoch(model, train_loader, optimizer, lr_scheduler, device,
                    grad_clip=1.0, verbose_grads=False):
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
            a_norm_sq = 0.0
            c_norm_sq = 0.0
            other_norm_sq = 0.0
            for name, p in model.named_parameters():
                if p.grad is not None:
                    g_sq = p.grad.data.norm(2).item() ** 2
                    if 'spectrum' in name or 'fourierft' in name.lower():
                        a_norm_sq += g_sq
                    elif 'classifier' in name or 'score' in name or 'modules_to_save' in name:
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
            clip_ratio = 1.0 / max(total_norms[i], grad_clip) * grad_clip if grad_clip > 0 else 0
            print(f"      batch {i}: adapter={adapter_norms[i]:.4f}, "
                  f"classifier={classifier_norms[i]:.4f}, "
                  f"total={total_norms[i]:.4f}, "
                  f"clip_scale={clip_ratio:.4f}")

    return total_loss / max(n_batches, 1)


def run_experiment(name, model, train_loader, eval_loader, device,
                   lr=5e-2, epochs=10, grad_clip=1.0, verbose_first=True):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*60}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    # Show trainable param groups
    adapter_params = 0
    classifier_params = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            if 'spectrum' in n or 'fourierft' in n.lower():
                adapter_params += p.numel()
            elif 'classifier' in n or 'score' in n or 'modules_to_save' in n:
                classifier_params += p.numel()
    print(f"  Adapter params: {adapter_params:,}, Classifier params: {classifier_params:,}")

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
    from peft import FourierFTConfig, get_peft_model, TaskType

    print("CONTROL EXPERIMENT: FourierFT on RoBERTa CoLA")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained('roberta-base', cache_dir='./data')
    train_loader, eval_loader = get_cola_data(tokenizer, batch_size=32)

    # ===== FourierFT with exact same config as successful run =====
    model = AutoModelForSequenceClassification.from_pretrained(
        'roberta-base', num_labels=2, cache_dir='./data')
    config = FourierFTConfig(
        n_frequency=256,
        target_modules=['query', 'value'],
        task_type=TaskType.SEQ_CLS,
        scaling=150.0,
        random_loc_seed=42,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    run_experiment("FourierFT n=256, s=150 (PEFT, matching successful config)",
                   model, train_loader, eval_loader, device,
                   lr=5e-2, epochs=10, grad_clip=1.0)
    del model
    torch.cuda.empty_cache()

    # ===== Spectral s=150 for comparison =====
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    from spectral_adapter import get_spectral_adapter_model

    model = AutoModelForSequenceClassification.from_pretrained(
        'roberta-base', num_labels=2, cache_dir='./data')
    model = get_spectral_adapter_model(
        model, target_modules=['query', 'value'],
        p=16, q=16, scaling=150.0, d_initial=0.01, freq_mode='contiguous')

    run_experiment("Spectral s=150, d=0.01 (for comparison)",
                   model, train_loader, eval_loader, device,
                   lr=5e-2, epochs=10, grad_clip=1.0)
    del model
    torch.cuda.empty_cache()

    print("\n\nDONE.")


if __name__ == '__main__':
    main()
