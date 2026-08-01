#!/usr/bin/env python3
"""
Diagnostic script for investigating why LLaMA-7B gets MCC=0.0 on CoLA
with the Spectral adapter after 10 epochs of training.

This script checks the following potential root causes:
  1. Classification token selection (pad_token = eos_token interaction)
  2. Score head dtype / initialization issues in float16
  3. Loss magnitude and gradient flow through the score head and adapter
  4. Whether logits collapse to a single prediction class
  5. Float16 gradient underflow in the backward pass

Usage:
    python debug_llama_cola.py [--device cuda]
"""
import argparse
import math
import sys
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


# ============================================================================
# Minimal re-implementation of SpectralAdapterLinear (standalone, no imports)
# ============================================================================
def _dct_basis(d: int, k: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    n = torch.arange(d, dtype=torch.float64)
    idx = torch.arange(k, dtype=torch.float64)
    basis = torch.cos(torch.pi * idx[:, None] * (2 * n[None, :] + 1) / (2 * d))
    basis[0] *= 1.0 / math.sqrt(d)
    basis[1:] *= math.sqrt(2.0 / d)
    return basis.to(dtype)


class SpectralAdapterLinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, p: int, q: int,
                 scaling: float = 1.0, d_initial: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.out_features = base_layer.out_features
        self.in_features = base_layer.in_features
        self.scaling = scaling

        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.register_buffer('dct_in', _dct_basis(self.in_features, q, torch.float32))
        self.register_buffer('dct_out', _dct_basis(self.out_features, p, torch.float32))

        self.coeffs = nn.Parameter(torch.zeros(p, q, dtype=torch.float32))
        if d_initial > 0.0:
            nn.init.normal_(self.coeffs, mean=0, std=d_initial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        base_dtype = base_out.dtype

        x_f32 = x.float()
        x_proj = F.linear(x_f32, self.dct_in)
        s_out = F.linear(x_proj, self.coeffs)
        delta_out = F.linear(s_out, self.dct_out.t())

        return base_out + self.scaling * delta_out.to(base_dtype)


def apply_spectral_adapters(model, target_modules, p=16, q=16, scaling=1.0,
                            d_initial=0.01):
    """Apply spectral adapters and freeze/unfreeze appropriately."""
    # Freeze all
    for param in model.parameters():
        param.requires_grad = False

    adapted = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in target_modules):
            continue

        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = dict(model.named_modules())[parent_name]
        else:
            attr_name = parts[0]
            parent = model

        layer_p = min(p, module.out_features)
        layer_q = min(q, module.in_features)

        adapter = SpectralAdapterLinear(
            module, p=layer_p, q=layer_q,
            scaling=scaling, d_initial=d_initial,
        )
        setattr(parent, attr_name, adapter)
        adapted.append(name)

    # Unfreeze classifier head
    for name, param in model.named_parameters():
        if 'classifier' in name or 'score' in name:
            param.requires_grad = True

    return adapted


# ============================================================================
# Diagnostic helpers
# ============================================================================
def print_section(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}\n")


def check_trainable_params(model):
    """List all trainable parameters with their dtype and shape."""
    trainable = []
    frozen = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            trainable.append((name, p.shape, p.dtype, p.numel()))
        else:
            frozen += p.numel()
    return trainable, frozen


# ============================================================================
# Main diagnostic
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    base_dtype = dtype_map[args.dtype]

    print_section("DIAGNOSTIC: LLaMA-7B + Spectral Adapter on CoLA (MCC=0.0 Bug)")
    print(f"Device: {device}")
    print(f"Base model dtype: {base_dtype}")
    print(f"Model: {args.model_name}")

    # ------------------------------------------------------------------
    # 1. Load tokenizer and check pad/eos token interaction
    # ------------------------------------------------------------------
    print_section("CHECK 1: Tokenizer pad_token / eos_token configuration")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f"Before fix:")
    print(f"  pad_token = {tokenizer.pad_token!r}  (id={tokenizer.pad_token_id})")
    print(f"  eos_token = {tokenizer.eos_token!r}  (id={tokenizer.eos_token_id})")
    print(f"  bos_token = {tokenizer.bos_token!r}  (id={tokenizer.bos_token_id})")
    print(f"  padding_side = {tokenizer.padding_side!r}")

    # Replicate training code: set pad_token = eos_token
    tokenizer.pad_token = tokenizer.eos_token
    print(f"\nAfter setting pad_token = eos_token:")
    print(f"  pad_token = {tokenizer.pad_token!r}  (id={tokenizer.pad_token_id})")
    print(f"  padding_side = {tokenizer.padding_side!r}")

    # ------------------------------------------------------------------
    # 2. Tokenize sample CoLA examples
    # ------------------------------------------------------------------
    print_section("CHECK 2: Tokenization and classification-token selection")

    cola_examples = [
        {"sentence": "The dog ran.", "label": 1},
        {"sentence": "John seems sleeping.", "label": 0},
        {"sentence": "What did you wonder who saw?", "label": 0},
        {"sentence": "The cat sat on the mat.", "label": 1},
    ]

    texts = [ex["sentence"] for ex in cola_examples]
    labels = torch.tensor([ex["label"] for ex in cola_examples])

    enc = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    print(f"Batch shape: {input_ids.shape}")
    for i in range(len(texts)):
        tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
        mask = attention_mask[i].tolist()
        print(f"\n  Example {i}: {texts[i]!r}  (label={labels[i].item()})")
        print(f"    IDs:    {input_ids[i].tolist()}")
        print(f"    Tokens: {tokens}")
        print(f"    Mask:   {mask}")

    # Replicate LlamaForSequenceClassification's token selection logic
    pad_token_id = tokenizer.pad_token_id
    sequence_lengths = torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1
    sequence_lengths = sequence_lengths % input_ids.shape[-1]
    print(f"\n  pad_token_id used for classification token selection: {pad_token_id}")
    print(f"  Computed sequence_lengths (classification token positions): {sequence_lengths.tolist()}")
    print(f"  (Should be the index of the last non-pad token in each row)")

    # Verify correctness
    for i in range(len(texts)):
        real_last = attention_mask[i].sum().item() - 1
        # For left-padded, last token is always at the rightmost position
        # unless sequence is shorter than max
        n_pad = (attention_mask[i] == 0).sum().item()
        expected_last = input_ids.shape[-1] - 1  # Last position for left-padded
        actual = sequence_lengths[i].item()
        print(f"    Example {i}: n_pad={n_pad}, last_real_token_pos={n_pad + real_last}, "
              f"selected_pos={actual}, "
              f"token_at_pos={tokenizer.convert_ids_to_tokens([input_ids[i, actual].item()])[0]!r}")

    # ------------------------------------------------------------------
    # 3. Load model, apply dtype, apply spectral adapter
    # ------------------------------------------------------------------
    print_section("CHECK 3: Model loading, dtype casting, and adapter application")

    config = AutoConfig.from_pretrained(args.model_name, num_labels=2, finetuning_task="cola")
    print(f"Config pad_token_id (before fix): {config.pad_token_id}")
    config.pad_token_id = tokenizer.pad_token_id
    print(f"Config pad_token_id (after fix):  {config.pad_token_id}")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, config=config, ignore_mismatched_sizes=True
    )

    # Check score head BEFORE dtype cast
    score_param = None
    for name, p in model.named_parameters():
        if 'score' in name:
            score_param = p
            print(f"Score head param '{name}': shape={p.shape}, dtype={p.dtype}")
            print(f"  Weight stats BEFORE dtype cast: mean={p.data.float().mean():.6f}, "
                  f"std={p.data.float().std():.6f}, "
                  f"abs_max={p.data.float().abs().max():.6f}")

    # Cast to target dtype (replicate training code)
    if base_dtype != torch.float32:
        model.to(dtype=base_dtype)
        print(f"\nCast model to {base_dtype}")

    # Check score head AFTER dtype cast
    for name, p in model.named_parameters():
        if 'score' in name:
            print(f"Score head param '{name}': dtype={p.dtype}")
            print(f"  Weight stats AFTER dtype cast: mean={p.data.float().mean():.6f}, "
                  f"std={p.data.float().std():.6f}")

    # Apply spectral adapter
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    adapted = apply_spectral_adapters(
        model, target_modules, p=16, q=16, scaling=1.0, d_initial=0.01,
    )
    print(f"\nApplied spectral adapters to {len(adapted)} modules")

    # List trainable params
    trainable_params, frozen_count = check_trainable_params(model)
    print(f"\nTrainable parameters ({len(trainable_params)} tensors):")
    total_trainable = 0
    score_trainable = 0
    adapter_trainable = 0
    for name, shape, dt, numel in trainable_params:
        kind = "SCORE" if "score" in name else "ADAPTER"
        print(f"  [{kind}] {name}: shape={list(shape)}, dtype={dt}, numel={numel}")
        total_trainable += numel
        if "score" in name:
            score_trainable += numel
        else:
            adapter_trainable += numel
    print(f"\n  Total trainable: {total_trainable:,}")
    print(f"  Score head: {score_trainable:,}")
    print(f"  Adapter coeffs: {adapter_trainable:,}")
    print(f"  Frozen: {frozen_count:,}")

    # ------------------------------------------------------------------
    # CRITICAL CHECK: score head dtype after adapter wrapping
    # ------------------------------------------------------------------
    print_section("CHECK 4: Score head dtype after adapter wrapping (CRITICAL)")
    for name, p in model.named_parameters():
        if 'score' in name:
            print(f"  {name}: dtype={p.dtype}, requires_grad={p.requires_grad}")
            if p.dtype == torch.float16:
                print(f"  *** WARNING: Score head is in float16! ***")
                print(f"      float16 has only 10 mantissa bits (vs 23 for float32).")
                print(f"      Tiny gradients from a 7B frozen backbone may underflow to 0.")
                print(f"      The score head is randomly initialized and must learn from scratch.")
                print(f"      With ~8500 CoLA training samples, gradients need to propagate")
                print(f"      through 32 frozen transformer layers in float16 -- gradients can")
                print(f"      become degenerate (all-same or all-zero) by the time they reach")
                print(f"      the adapter coefficients.")

    # ------------------------------------------------------------------
    # 4. Forward pass analysis
    # ------------------------------------------------------------------
    print_section("CHECK 5: Forward pass — logits, loss, and predictions")

    model.to(device)
    model.eval()

    batch = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }

    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    loss = outputs.loss
    probs = F.softmax(logits.float(), dim=-1)
    preds = logits.argmax(dim=-1)

    print(f"Loss: {loss.item():.6f} (dtype={loss.dtype})")
    print(f"Logits dtype: {logits.dtype}")
    print(f"Logits shape: {logits.shape}")
    print()
    for i in range(len(texts)):
        print(f"  Example {i}: label={labels[i].item()}, pred={preds[i].item()}")
        print(f"    logits = [{logits[i, 0].item():.6f}, {logits[i, 1].item():.6f}]")
        print(f"    probs  = [{probs[i, 0].item():.6f}, {probs[i, 1].item():.6f}]")
        print(f"    logit_diff = {(logits[i, 1] - logits[i, 0]).item():.8f}")

    all_same_pred = (preds == preds[0]).all().item()
    print(f"\n  All predictions identical? {all_same_pred}")
    if all_same_pred:
        print(f"  *** All examples predicted class {preds[0].item()} ***")
        print(f"      This means MCC will be exactly 0.0!")

    # Check logit variance across examples
    logit_var = logits.float().var(dim=0)
    print(f"\n  Logit variance across batch (per class): {logit_var.tolist()}")
    if logit_var.max().item() < 1e-4:
        print(f"  *** WARNING: Logits have almost no variance across examples ***")
        print(f"      The model is producing nearly identical outputs for all inputs.")
        print(f"      This suggests the score head is dominating and the backbone")
        print(f"      hidden states at the classification position are very similar.")

    # ------------------------------------------------------------------
    # 5. Gradient flow analysis
    # ------------------------------------------------------------------
    print_section("CHECK 6: Gradient flow through score head and adapter params")

    model.train()
    # Need fresh forward pass with gradients
    outputs = model(**batch)
    loss = outputs.loss

    print(f"Training loss: {loss.item():.6f} (dtype={loss.dtype})")

    # Check if loss is computed in float16
    if loss.dtype == torch.float16:
        print(f"*** WARNING: Loss is in float16! ***")
        print(f"    CrossEntropyLoss computed in float16 can have numerical issues.")
        print(f"    float16 min subnormal: {torch.finfo(torch.float16).tiny:.2e}")
        print(f"    float16 eps: {torch.finfo(torch.float16).eps:.2e}")
        print(f"    If loss gradients underflow, no learning occurs.")

    loss.backward()

    print(f"\n--- Score head gradients ---")
    for name, p in model.named_parameters():
        if 'score' in name and p.requires_grad:
            if p.grad is not None:
                g = p.grad.float()
                print(f"  {name}:")
                print(f"    grad dtype={p.grad.dtype}, shape={list(p.grad.shape)}")
                print(f"    grad mean={g.mean():.2e}, std={g.std():.2e}, "
                      f"abs_max={g.abs().max():.2e}, abs_min={g.abs().min():.2e}")
                print(f"    grad norm={g.norm():.2e}")
                n_zero = (p.grad == 0).sum().item()
                print(f"    zero elements: {n_zero}/{p.grad.numel()} "
                      f"({100*n_zero/p.grad.numel():.1f}%)")
                if p.grad.dtype == torch.float16 and g.abs().max().item() < 1e-4:
                    print(f"    *** WARNING: Gradients are tiny in float16 ***")
                    print(f"        This likely means gradient underflow is occurring.")
            else:
                print(f"  {name}: grad is None! Gradient is not flowing!")

    print(f"\n--- Adapter coefficient gradients (first 5 modules) ---")
    adapter_grad_info = []
    for name, p in model.named_parameters():
        if 'coeffs' in name and p.requires_grad:
            if p.grad is not None:
                g = p.grad.float()
                info = {
                    "name": name,
                    "grad_dtype": str(p.grad.dtype),
                    "grad_mean": g.mean().item(),
                    "grad_std": g.std().item(),
                    "grad_abs_max": g.abs().max().item(),
                    "grad_norm": g.norm().item(),
                    "n_zero": (p.grad == 0).sum().item(),
                    "numel": p.grad.numel(),
                }
                adapter_grad_info.append(info)
            else:
                adapter_grad_info.append({"name": name, "grad": "None"})

    for info in adapter_grad_info[:5]:
        name = info["name"]
        if "grad" in info and info["grad"] == "None":
            print(f"  {name}: grad is None!")
        else:
            print(f"  {name}:")
            print(f"    grad dtype={info['grad_dtype']}, "
                  f"mean={info['grad_mean']:.2e}, std={info['grad_std']:.2e}, "
                  f"abs_max={info['grad_abs_max']:.2e}, norm={info['grad_norm']:.2e}")
            pct_zero = 100 * info["n_zero"] / info["numel"]
            print(f"    zero elements: {info['n_zero']}/{info['numel']} ({pct_zero:.1f}%)")

    if adapter_grad_info:
        max_grad = max(i.get("grad_abs_max", 0) for i in adapter_grad_info if "grad_abs_max" in i)
        mean_norm = sum(i.get("grad_norm", 0) for i in adapter_grad_info if "grad_norm" in i) / len(adapter_grad_info)
        total_zero = sum(i.get("n_zero", 0) for i in adapter_grad_info if "n_zero" in i)
        total_elem = sum(i.get("numel", 0) for i in adapter_grad_info if "numel" in i)
        print(f"\n  Summary across all {len(adapter_grad_info)} adapter modules:")
        print(f"    Max gradient magnitude: {max_grad:.2e}")
        print(f"    Mean gradient norm: {mean_norm:.2e}")
        print(f"    Total zero gradients: {total_zero}/{total_elem} "
              f"({100*total_zero/total_elem:.1f}%)" if total_elem > 0 else "")

    # ------------------------------------------------------------------
    # 6. Simulate a few training steps to see if loss changes
    # ------------------------------------------------------------------
    print_section("CHECK 7: Mini training loop (5 steps) — does the loss decrease?")

    model.train()
    model.zero_grad()

    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params_list, lr=0.02, weight_decay=0.0)

    # Load a few real CoLA examples
    ds = load_dataset("glue", "cola", split="train[:32]")
    enc_train = tokenizer(
        ds["sentence"], padding=True, truncation=True, max_length=128, return_tensors="pt"
    )
    train_labels = torch.tensor(ds["label"])

    losses = []
    all_preds = []
    for step in range(5):
        batch = {
            "input_ids": enc_train["input_ids"].to(device),
            "attention_mask": enc_train["attention_mask"].to(device),
            "labels": train_labels.to(device),
        }
        outputs = model(**batch)
        loss = outputs.loss
        preds = outputs.logits.detach().argmax(dim=-1).cpu()

        losses.append(loss.item())
        pred_counts = Counter(preds.tolist())
        all_preds.append(pred_counts)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params_list, 1.0)
        optimizer.step()
        optimizer.zero_grad()

        print(f"  Step {step}: loss={loss.item():.6f}, preds={dict(pred_counts)}, "
              f"label_dist={dict(Counter(train_labels.tolist()))}")

    loss_changed = abs(losses[-1] - losses[0]) > 1e-6
    print(f"\n  Loss change over 5 steps: {losses[0]:.6f} -> {losses[-1]:.6f} "
          f"(delta={losses[-1]-losses[0]:.6f})")
    if not loss_changed:
        print(f"  *** WARNING: Loss did not change! Training is completely stuck. ***")

    # Check if predictions diversify
    final_preds = all_preds[-1]
    if len(final_preds) == 1:
        print(f"  *** WARNING: After 5 steps, still predicting only class {list(final_preds.keys())[0]} ***")
    else:
        print(f"  Good: predictions have diversified to {dict(final_preds)}")

    # ------------------------------------------------------------------
    # 7. Check the hidden states at classification position
    # ------------------------------------------------------------------
    print_section("CHECK 8: Hidden state analysis at classification position")

    model.eval()
    with torch.no_grad():
        batch = {
            "input_ids": input_ids[:4].to(device),
            "attention_mask": attention_mask[:4].to(device),
        }
        # Get internal hidden states
        out = model.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )
        last_hidden = out.last_hidden_state  # (batch, seq, hidden)

    # Extract hidden states at classification positions
    seq_lens = torch.eq(input_ids[:4], tokenizer.pad_token_id).int().argmax(-1) - 1
    seq_lens = seq_lens % input_ids[:4].shape[-1]

    print(f"Last hidden state shape: {last_hidden.shape}")
    print(f"Last hidden state dtype: {last_hidden.dtype}")

    for i in range(min(4, last_hidden.shape[0])):
        pos = seq_lens[i].item()
        h = last_hidden[i, pos].float()
        print(f"\n  Example {i} (pos={pos}):")
        print(f"    hidden mean={h.mean():.6f}, std={h.std():.6f}, "
              f"norm={h.norm():.4f}")
        print(f"    hidden abs_max={h.abs().max():.6f}, min={h.abs().min():.8f}")

    # Check similarity of hidden states across examples
    hidden_vecs = []
    for i in range(min(4, last_hidden.shape[0])):
        pos = seq_lens[i].item()
        hidden_vecs.append(last_hidden[i, pos].float())
    hidden_stack = torch.stack(hidden_vecs)
    cos_sim_matrix = F.cosine_similarity(
        hidden_stack.unsqueeze(0), hidden_stack.unsqueeze(1), dim=-1
    )
    print(f"\n  Cosine similarity between hidden states at classification positions:")
    for i in range(cos_sim_matrix.shape[0]):
        row = [f"{cos_sim_matrix[i, j].item():.4f}" for j in range(cos_sim_matrix.shape[1])]
        print(f"    [{', '.join(row)}]")

    avg_cos = (cos_sim_matrix.sum() - cos_sim_matrix.trace()) / (cos_sim_matrix.numel() - cos_sim_matrix.shape[0])
    if avg_cos > 0.99:
        print(f"\n  *** WARNING: Hidden states are nearly identical (avg cosine sim={avg_cos:.4f}) ***")
        print(f"      All inputs produce ~same hidden state at the classification position.")
        print(f"      The score head then maps these ~identical vectors to ~identical logits.")
        print(f"      => All predictions are the same class => MCC = 0.0")

    # ------------------------------------------------------------------
    # 8. Float16 specific analysis
    # ------------------------------------------------------------------
    print_section("CHECK 9: Float16-specific gradient underflow analysis")

    if base_dtype == torch.float16:
        print(f"  float16 properties:")
        print(f"    eps  = {torch.finfo(torch.float16).eps:.2e}")
        print(f"    tiny = {torch.finfo(torch.float16).tiny:.2e}")
        print(f"    max  = {torch.finfo(torch.float16).max:.2e}")
        print()
        print(f"  Problem: float16 without a GradScaler leads to gradient underflow.")
        print(f"  The training code does NOT use torch.cuda.amp.GradScaler.")
        print(f"  Instead, it casts the entire model to float16 and computes")
        print(f"  loss.backward() directly. For a 7B parameter model, gradients")
        print(f"  at the early layers (and the adapter coefficients deep in the")
        print(f"  network) can easily fall below float16's minimum representable")
        print(f"  value ({torch.finfo(torch.float16).tiny:.2e}) and become zero.")
        print()

        # Demonstrate the underflow
        model.train()
        model.zero_grad()
        outputs = model(**{
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "labels": labels.to(device),
        })
        outputs.loss.backward()

        # Count adapter params with zero gradients
        total_adapter_params = 0
        zero_grad_params = 0
        for name, p in model.named_parameters():
            if 'coeffs' in name and p.requires_grad and p.grad is not None:
                total_adapter_params += p.grad.numel()
                zero_grad_params += (p.grad == 0).sum().item()

        if total_adapter_params > 0:
            pct_zero = 100 * zero_grad_params / total_adapter_params
            print(f"  Adapter coefficient gradient analysis:")
            print(f"    Total elements:       {total_adapter_params:,}")
            print(f"    Zero-gradient elements: {zero_grad_params:,} ({pct_zero:.1f}%)")
            if pct_zero > 50:
                print(f"    *** CRITICAL: Over {pct_zero:.0f}% of adapter gradients are exactly zero ***")
                print(f"        This confirms float16 gradient underflow is the root cause.")
                print(f"        FIX: Use bfloat16 (same memory, wider dynamic range) or")
                print(f"             use float16 with torch.cuda.amp.GradScaler.")
    else:
        print(f"  Base dtype is {base_dtype}, not float16. Gradient underflow is less likely.")
        print(f"  (bfloat16 has the same exponent range as float32, so underflow is rare.)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_section("DIAGNOSIS SUMMARY")

    issues = []

    # Issue 1: float16 without GradScaler
    if base_dtype == torch.float16:
        issues.append(
            "CRITICAL — float16 without GradScaler causes gradient underflow.\n"
            "    The score head (in float16) and adapter coefficients receive\n"
            "    tiny gradients that round to zero in float16 representation.\n"
            "    FIX: Switch to --dtype bfloat16 (same 2-byte memory, wider\n"
            "    dynamic range) OR add torch.cuda.amp.GradScaler to the\n"
            "    training loop when using float16."
        )

    # Issue 2: score head dtype mismatch
    score_dtype = None
    for name, p in model.named_parameters():
        if 'score' in name:
            score_dtype = p.dtype
    if score_dtype == torch.float16:
        issues.append(
            "MODERATE — Score head weights are in float16.\n"
            "    The randomly-initialized score head maps 4096-dim hidden states\n"
            "    to 2 logits. In float16, the score head's weight updates may\n"
            "    be too small to change its behavior.\n"
            "    FIX: Cast score head to float32 after applying the adapter,\n"
            "    e.g.: model.score.to(torch.float32)"
        )

    # Issue 3: hidden state similarity
    if avg_cos > 0.95:
        issues.append(
            f"HIGH — Hidden states at classification position are very similar\n"
            f"    (avg cosine sim = {avg_cos:.4f}). For a decoder-only model,\n"
            f"    the last token's hidden state is heavily influenced by the\n"
            f"    final layers which are frozen. Different inputs produce nearly\n"
            f"    identical hidden representations, making classification almost\n"
            f"    impossible from the start."
        )

    # Issue 4: loss in float16
    if loss.dtype == torch.float16:
        issues.append(
            "HIGH — CrossEntropyLoss is computed in float16.\n"
            "    The log-softmax and NLL computation in float16 can produce\n"
            "    inaccurate gradients. The loss itself may not be NaN, but\n"
            "    the backward pass gradients suffer from reduced precision."
        )

    if not issues:
        issues.append("No critical issues detected. The problem may be in hyperparameters or training duration.")

    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}\n")

    print(f"ROOT CAUSE ANALYSIS:")
    print(f"  The most likely root cause is the combination of:")
    print(f"    (a) Using dtype=float16 for the base model without a GradScaler")
    print(f"    (b) The score head being in float16 with random initialization")
    print(f"    (c) The spectral adapter coefficients receive float32 gradients,")
    print(f"        BUT the backward pass through the frozen float16 backbone")
    print(f"        produces underflowed gradients that, once cast to float32,")
    print(f"        are zero or near-zero")
    print()
    print(f"  RECOMMENDED FIXES (in order of impact):")
    print(f"    1. Change --dtype from float16 to bfloat16")
    print(f"       bfloat16 has the same exponent range as float32 (8 bits)")
    print(f"       so gradients will not underflow, while using the same memory.")
    print(f"    2. If float16 is required, add GradScaler to the training loop:")
    print(f"       scaler = torch.cuda.amp.GradScaler()")
    print(f"       scaler.scale(loss).backward()")
    print(f"       scaler.step(optimizer)")
    print(f"       scaler.update()")
    print(f"    3. Upcast the score head to float32:")
    print(f"       model.score.to(torch.float32) after adapter wrapping")
    print(f"    4. Consider upcasting the loss computation:")
    print(f"       loss = F.cross_entropy(logits.float(), labels)")


if __name__ == "__main__":
    main()
