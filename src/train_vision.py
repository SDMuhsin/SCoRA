"""
Minimal vision fine-tuning harness for GASA conv adapters and baselines.

Loads an HF image-classification model (microsoft/resnet-18,
facebook/convnext-tiny-224, google/vit-base-patch16-224), loads CIFAR-10/100 via
HF `datasets`, injects conv adapters (GASA / FourierFT / LoRA) into the specified
conv modules, trains head + adapters with AdamW, and evaluates top-1 accuracy.

Logs (mirroring the accounting idiom of train_glue.py): trainable param count per
method, top-1 accuracy, seed, wall-time, peak GPU memory.

This harness is intentionally lean. It supports full smoke tests (--max_train_steps
with --train_subset) as well as small real runs, but the deliverable only requires
the smoke test — DO NOT use this for full-scale training.

Example (smoke):
  python src/train_vision.py --model microsoft/resnet-18 --dataset cifar10 \
      --method gasa --target_modules layer.0.convolution \
      --gasa_rank 4 --gasa_p 9 --max_train_steps 30 --train_subset 512 --eval_subset 512
"""
import argparse
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForImageClassification, AutoImageProcessor

from gasa_adapter import get_gasa_model


# ---------------------------------------------------------------------------
# memory accounting (idiom from train_glue.py)
# ---------------------------------------------------------------------------
def mib(x: int) -> float:
    return x / 1024 ** 2


@torch.no_grad()
def get_memory_breakdown(model: nn.Module, optimizer, device: torch.device) -> dict:
    stats = {}
    if device.type == "cuda":
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        stats["param_mem_mib"] = mib(param_bytes)
        opt_bytes = 0
        if optimizer and hasattr(optimizer, "state") and optimizer.state:
            for state in optimizer.state.values():
                for t in state.values():
                    if torch.is_tensor(t):
                        opt_bytes += t.numel() * t.element_size()
        stats["opt_mem_mib"] = mib(opt_bytes)
        stats["peak_memory_mib"] = mib(torch.cuda.max_memory_allocated(device))
        stats["allocated_memory_mib"] = mib(torch.cuda.memory_allocated(device))
    return stats


# ---------------------------------------------------------------------------
# args
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Minimal vision fine-tuning harness (GASA + baselines)")
    ap.add_argument("--model", type=str, default="microsoft/resnet-18")
    ap.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--method", type=str, default="gasa",
                    choices=["gasa", "fourierft", "lora", "full", "none"])
    ap.add_argument("--target_modules", type=str, default="layer.0.convolution",
                    help="comma-separated substrings matching nn.Conv2d module names")
    ap.add_argument("--gasa_rank", type=int, default=4, help="GASA channel rank R")
    ap.add_argument("--gasa_p", type=int, default=9, help="GASA spectral modes p (bottom-p of L_G)")
    ap.add_argument("--fourierft_n_frequency", type=int, default=None,
                    help="FourierFT frequencies; if None, auto-match GASA budget")
    ap.add_argument("--fourierft_scaling", type=float, default=None,
                    help="FourierFT post-real scaling; if None, uses --scaling")
    ap.add_argument("--fourierft_random_loc_seed", type=int, default=777)
    ap.add_argument("--lora_rank", type=int, default=None,
                    help="LoRA rank; if None, auto-match GASA budget")
    ap.add_argument("--lora_alpha", type=float, default=None)
    ap.add_argument("--scaling", type=float, default=1.0, help="adapter output scaling")
    ap.add_argument("--depthwise", type=str, default="auto", choices=["auto", "true", "false"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_train_steps", type=int, default=None, help="cap steps (smoke test)")
    ap.add_argument("--train_subset", type=int, default=None, help="use first N train examples")
    ap.add_argument("--eval_subset", type=int, default=1000, help="use first N eval examples")
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--cache_dir", type=str, default="data")
    ap.add_argument("--num_workers", type=int, default=0,
                    help="DataLoader worker processes (parallelizes image preprocessing)")
    ap.add_argument("--cache_pixels", action="store_true",
                    help="precompute + disk-cache processed pixel_values (identical batches, much faster)")
    ap.add_argument("--cache_pixels_dir", type=str, default=None,
                    help="directory for the pixel cache (required with --cache_pixels)")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
_IMG_KEY = {"cifar10": "img", "cifar100": "img"}
_LABEL_KEY = {"cifar10": "label", "cifar100": "fine_label"}


def build_dataloaders(args, processor):
    img_key = _IMG_KEY[args.dataset]
    label_key = _LABEL_KEY[args.dataset]
    name = "cifar10" if args.dataset == "cifar10" else "cifar100"
    ds = load_dataset(name, cache_dir=args.cache_dir)
    train_ds, eval_ds = ds["train"], ds["test"]
    if args.train_subset:
        train_ds = train_ds.select(range(min(args.train_subset, len(train_ds))))
    if args.eval_subset:
        eval_ds = eval_ds.select(range(min(args.eval_subset, len(eval_ds))))
    num_labels = ds["train"].features[label_key].num_classes

    size = processor.size
    edge = size.get("shortest_edge", size.get("height", 224))
    mean = torch.tensor(processor.image_mean).view(3, 1, 1)
    std = torch.tensor(processor.image_std).view(3, 1, 1)

    def collate(batch):
        imgs = [b[img_key].convert("RGB") for b in batch]
        # processor handles resize + rescale + normalize consistently with the model
        px = processor(images=imgs, return_tensors="pt")["pixel_values"]
        labels = torch.tensor([b[label_key] for b in batch], dtype=torch.long)
        return px, labels

    # ---- optional disk cache of processed pixel_values --------------------
    # The 32->224 PIL resize is repeated every epoch/run over identical images.
    # With --cache_pixels we precompute processor(images) once for the whole
    # (subset of the) split, store the tensor (fp16) + labels to disk, and serve
    # batches from a TensorDataset. Batch *content* (modulo fp16 storage) and,
    # under the same seed, batch *order* are identical to the on-the-fly path.
    cache_dir = getattr(args, "cache_pixels_dir", None)
    if getattr(args, "cache_pixels", False) and cache_dir:
        import os, hashlib
        os.makedirs(cache_dir, exist_ok=True)
        mslug = args.model.replace("/", "_")

        def _load_or_build(split_ds, split_name, subset):
            key = f"{mslug}_{args.dataset}_{split_name}_{subset if subset else 'all'}.pt"
            path = os.path.join(cache_dir, key)
            if os.path.exists(path):
                blob = torch.load(path)
                return blob["px"], blob["labels"]
            pxs, lbs = [], []
            bs = 256
            for i in range(0, len(split_ds), bs):
                chunk = split_ds.select(range(i, min(i + bs, len(split_ds))))
                imgs = [im.convert("RGB") for im in chunk[img_key]]
                px = processor(images=imgs, return_tensors="pt")["pixel_values"].half()
                pxs.append(px)
                lbs.append(torch.tensor(chunk[label_key], dtype=torch.long))
            px_all = torch.cat(pxs, 0)
            lb_all = torch.cat(lbs, 0)
            torch.save({"px": px_all, "labels": lb_all}, path)
            return px_all, lb_all

        from torch.utils.data import TensorDataset
        tr_px, tr_lb = _load_or_build(train_ds, "train", args.train_subset)
        ev_px, ev_lb = _load_or_build(eval_ds, "eval", args.eval_subset)

        def cache_collate(batch):
            px = torch.stack([b[0] for b in batch]).float()
            labels = torch.stack([b[1] for b in batch])
            return px, labels

        train_loader = DataLoader(TensorDataset(tr_px, tr_lb), batch_size=args.batch_size,
                                  shuffle=True, collate_fn=cache_collate)
        eval_loader = DataLoader(TensorDataset(ev_px, ev_lb), batch_size=args.batch_size,
                                 shuffle=False, collate_fn=cache_collate)
        return train_loader, eval_loader, num_labels

    nw = getattr(args, "num_workers", 0)
    dl_kwargs = dict(num_workers=nw)
    if nw > 0:
        # /dev/shm is tiny (64MB) in this container; use file_system sharing so
        # worker->main tensor transfer goes through regular tmp files, and skip
        # pin_memory to avoid extra shared-memory pressure.
        torch.multiprocessing.set_sharing_strategy("file_system")
        dl_kwargs.update(persistent_workers=True, prefetch_factor=4, pin_memory=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, **dl_kwargs)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate, **dl_kwargs)
    return train_loader, eval_loader, num_labels


# ---------------------------------------------------------------------------
# model / adapter setup
# ---------------------------------------------------------------------------
def build_model(args, num_labels):
    model = AutoModelForImageClassification.from_pretrained(
        args.model, num_labels=num_labels, ignore_mismatched_sizes=True,
        cache_dir=args.cache_dir)

    target_modules = [t.strip() for t in args.target_modules.split(",") if t.strip()]
    dw = {"auto": None, "true": True, "false": False}[args.depthwise]

    if args.method in ("gasa", "fourierft", "lora"):
        model = get_gasa_model(
            model, target_modules, method=args.method,
            rank=args.gasa_rank, p=args.gasa_p, scaling=args.scaling,
            fourierft_n_frequency=args.fourierft_n_frequency,
            fourierft_scaling=args.fourierft_scaling,
            fourierft_random_loc_seed=args.fourierft_random_loc_seed,
            lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, depthwise=dw)
        model.print_trainable_parameters()
        print("adapted conv modules:", model.adapted_modules)
        for cfg in model.module_configs:
            print(f"  {cfg['name']}: P_gasa={cfg['P_gasa']} "
                  f"ff_n_freq={cfg['fourierft_n_frequency']} lora_rank={cfg['lora_rank']} "
                  f"(lora_params={cfg['lora_params']}) depthwise={cfg['depthwise']}")
    elif args.method == "full":
        for p in model.parameters():
            p.requires_grad = True
    elif args.method == "none":
        for p in model.parameters():
            p.requires_grad = False
        for name, p in model.named_parameters():
            if "classifier" in name or "score" in name:
                p.requires_grad = True

    return model


# ---------------------------------------------------------------------------
# train / eval
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    processor = AutoImageProcessor.from_pretrained(args.model, cache_dir=args.cache_dir)
    train_loader, eval_loader, num_labels = build_dataloaders(args, processor)
    model = build_model(args, num_labels)
    model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"[setup] method={args.method} trainable={trainable_count:,} "
          f"total={total_count:,} ({100 * trainable_count / total_count:.4f}%)")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # --- training ---
    t0 = time.perf_counter()
    loss_traj: List[float] = []
    completed = 0
    mem_stats = {}
    grad_seen = {"adapter": False, "head": False}
    stop = False
    for epoch in range(args.epochs):
        model.train()
        for px, labels in train_loader:
            px, labels = px.to(device), labels.to(device)
            logits = model(pixel_values=px).logits
            loss = loss_fn(logits, labels)
            loss.backward()

            # confirm gradients reach adapters and head (once)
            if not (grad_seen["adapter"] and grad_seen["head"]):
                for name, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0:
                        if any(k in name for k in ("Phi", ".S", ".u", ".v", ".s", "spectrum", "lora_A", "lora_B")):
                            grad_seen["adapter"] = True
                        if "classifier" in name or "score" in name:
                            grad_seen["head"] = True

            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            completed += 1
            loss_traj.append(float(loss.detach()))
            if completed == 1 and device.type == "cuda":
                mem_stats = get_memory_breakdown(model, optimizer, device)
            if completed % args.log_every == 0:
                print(f"  step {completed:4d} | loss {loss.item():.4f}")
            if args.max_train_steps and completed >= args.max_train_steps:
                stop = True
                break
        if stop:
            break
    train_time = time.perf_counter() - t0

    # --- eval (top-1) ---
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for px, labels in eval_loader:
            px, labels = px.to(device), labels.to(device)
            preds = model(pixel_values=px).logits.argmax(dim=-1)
            correct += (preds == labels.to(device)).sum().item()
            total += labels.numel()
    top1 = correct / max(total, 1)

    peak_mem = mib(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0.0

    print("=" * 70)
    print(f"RESULT method={args.method} model={args.model} dataset={args.dataset}")
    print(f"  seed={args.seed} steps={completed} epochs_done<= {args.epochs}")
    print(f"  trainable_params={trainable_count:,}")
    print(f"  top1_acc={top1:.4f} ({correct}/{total})")
    print(f"  wall_time_s={train_time:.2f}")
    print(f"  peak_gpu_mem_mib={peak_mem:.1f}  param_mem_mib={mem_stats.get('param_mem_mib', 0):.1f}"
          f"  opt_mem_mib={mem_stats.get('opt_mem_mib', 0):.1f}")
    print(f"  gradients_reached: adapter={grad_seen['adapter']} head={grad_seen['head']}")
    print(f"  loss_trajectory={[round(x, 4) for x in loss_traj]}")
    print("=" * 70)
    return {
        "method": args.method, "top1": top1, "trainable_params": trainable_count,
        "seed": args.seed, "wall_time_s": train_time, "peak_gpu_mem_mib": peak_mem,
        "loss_trajectory": loss_traj,
    }


if __name__ == "__main__":
    main()
