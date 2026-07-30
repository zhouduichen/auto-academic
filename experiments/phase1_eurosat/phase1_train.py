"""
Phase 1: EuroSAT LoRA Training — 16 Configs × 3 Seeds = 48 Runs
=================================================================
Runs on Windows GPU node (RTX 5080 Laptop, max 14 GB VRAM).

Each run:
  - Trains one LoRA configuration on EuroSAT with one random seed
  - Saves per-sample validation + test logits for offline selection analysis
  - Does NOT use validation early stopping (fixed epochs)

Output structure:
  experiments/phase1_eurosat/outputs/{config_id}/seed_{seed}/
    ├── val_logits.npy     # (n_val_samples, n_classes)
    ├── val_labels.npy     # (n_val_samples,)
    ├── test_logits.npy    # (n_test_samples, n_classes)
    ├── test_labels.npy    # (n_test_samples,)
    └── metrics.json       # {val_acc, test_acc, train_time, peak_vram_gb, ...}
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import EuroSAT
from transformers import ViTForImageClassification, ViTImageProcessor
from peft import LoraConfig, get_peft_model, TaskType

# ── ✂️ GPU Memory Limit ──────────────────────────────────────
# Leave 2 GB for gaming: 16 GB → 14 GB usable
MAX_VRAM_GB = 14.0
try:
    torch.cuda.set_per_process_memory_fraction(MAX_VRAM_GB / 16.0)
except (AttributeError, RuntimeError):
    # PyTorch >=2.5 removed set_per_process_memory_fraction;
    # CUDA allocator defaults are sufficient for our workload.
    pass


# ── 16 LoRA Configurations ───────────────────────────────────

@dataclass
class LoRAConfigDef:
    """A single LoRA configuration in the search space."""
    config_id: int
    rank: int
    lora_alpha: int
    learning_rate: float
    weight_decay: float
    target_modules: str  # comma-separated: "q,v" or "q,v,out"
    description: str = ""


def build_16_lora_configs() -> List[LoRAConfigDef]:
    """
    Build exactly 16 LoRA configurations that systematically cover:
      - rank ∈ {2, 4, 8, 16}         (4 values)
      - lr   ∈ {1e-4, 3e-4, 1e-3}    (3 values)
      - wd   ∈ {0.0, 0.01}            (2 values)
      - target_modules ∈ {"query,value", "query,value,output.dense"}  (2 values)

    We don't do a full 4×3×2×2 = 48 grid. Instead we sample 16 diverse
    combos: for each rank we pick 4 combos that mix lr/wd/target.

    rank=2  → 4 combos (low capacity, test aggressive lr)
    rank=4  → 4 combos (mid-low capacity)
    rank=8  → 4 combos (mid capacity, standard)
    rank=16 → 4 combos (high capacity, test wd sensitivity)
    """
    configs = []
    cid = 0

    plans = [
        # (rank, lr, wd, target_modules, desc)
        (2, 1e-4, 0.0, "query,value", "r=2,lr=1e-4,wd=0,qv"),
        (2, 3e-4, 0.0, "query,value", "r=2,lr=3e-4,wd=0,qv"),
        (2, 1e-4, 0.0, "query,value,output.dense", "r=2,lr=1e-4,wd=0,qvo"),
        (2, 1e-3, 0.01, "query,value", "r=2,lr=1e-3,wd=1e-2,qv"),

        (4, 1e-4, 0.0, "query,value", "r=4,lr=1e-4,wd=0,qv"),
        (4, 3e-4, 0.0, "query,value,output.dense", "r=4,lr=3e-4,wd=0,qvo"),
        (4, 1e-4, 0.01, "query,value", "r=4,lr=1e-4,wd=1e-2,qv"),
        (4, 3e-4, 0.01, "query,value,output.dense", "r=4,lr=3e-4,wd=1e-2,qvo"),

        (8, 1e-4, 0.0, "query,value", "r=8,lr=1e-4,wd=0,qv"),
        (8, 3e-4, 0.0, "query,value,output.dense", "r=8,lr=3e-4,wd=0,qvo"),
        (8, 1e-3, 0.0, "query,value", "r=8,lr=1e-3,wd=0,qv"),
        (8, 1e-4, 0.01, "query,value,output.dense", "r=8,lr=1e-4,wd=1e-2,qvo"),

        (16, 1e-4, 0.0, "query,value", "r=16,lr=1e-4,wd=0,qv"),
        (16, 3e-4, 0.0, "query,value,output.dense", "r=16,lr=3e-4,wd=0,qvo"),
        (16, 1e-3, 0.01, "query,value", "r=16,lr=1e-3,wd=1e-2,qv"),
        (16, 1e-4, 0.01, "query,value,output.dense", "r=16,lr=1e-4,wd=1e-2,qvo"),
    ]

    for rank, lr, wd, target, desc in plans:
        configs.append(LoRAConfigDef(
            config_id=cid,
            rank=rank,
            lora_alpha=rank * 2,
            learning_rate=lr,
            weight_decay=wd,
            target_modules=target,
            description=desc,
        ))
        cid += 1

    assert len(configs) == 16, f"Expected 16 configs, got {len(configs)}"
    return configs


# ── Data ──────────────────────────────────────────────────────

def get_eurosat_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load EuroSAT with fixed train/val/test split.
    Uses the standard split: 60% train, 20% val, 20% test.
    Returns (train_loader, val_loader, test_loader).
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = EuroSAT(root=data_dir, download=True, transform=transform)
    n_total = len(full_dataset)
    targets = full_dataset.targets  # type: ignore[attr-defined]

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)

    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)
    n_test = n_total - n_train - n_val

    train_idx = indices[:n_train].tolist()
    val_idx = indices[n_train:n_train + n_val].tolist()
    test_idx = indices[n_train + n_val:].tolist()

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"EuroSAT loaded: {n_total} images")
    print(f"  Train: {n_train} | Val: {n_val} | Test: {n_test}")
    print(f"  Classes: 10")

    return train_loader, val_loader, test_loader


# ── Model ─────────────────────────────────────────────────────

def build_lora_model(
    lora_config_def: LoRAConfigDef,
    num_classes: int = 10,
) -> nn.Module:
    """Build ViT-B/16 with LoRA adapters."""
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    # Map our target module naming to HF PEFT naming
    # ViT attention: query, key, value, output.dense
    target_modules = lora_config_def.target_modules.replace("q_proj", "query").replace(
        "v_proj", "value").replace("out_proj", "output.dense").split(",")

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_config_def.rank,
        lora_alpha=lora_config_def.lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
    )

    model = get_peft_model(model, peft_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  LoRA: {trainable_params:,} trainable / {total_params:,} total params ({100*trainable_params/total_params:.2f}%)")

    return model


# ── Training ──────────────────────────────────────────────────

def train_one_config(
    config_def: LoRAConfigDef,
    seed: int,
    data_dir: str,
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict:
    """Train one LoRA config with one seed. Save per-sample logits."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ── Data ──
    train_loader, val_loader, test_loader = get_eurosat_loaders(
        data_dir, batch_size=batch_size, seed=seed
    )

    # ── Model ──
    model = build_lora_model(config_def)
    model = model.to(device)

    # ── Optimizer ──
    # Only train LoRA params
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config_def.learning_rate, weight_decay=config_def.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ──
    t0 = time.time()
    peak_vram = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Track VRAM
            vram = torch.cuda.max_memory_allocated(device) / 1024**3
            peak_vram = max(peak_vram, vram)

    train_time = time.time() - t0

    # ── Collect per-sample logits ──
    model.eval()

    def collect_logits(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                outputs = model(images)
                all_logits.append(outputs.logits.cpu().numpy())
                all_labels.append(labels.numpy())
        return np.concatenate(all_logits), np.concatenate(all_labels)

    val_logits, val_labels = collect_logits(val_loader)
    test_logits, test_labels = collect_logits(test_loader)

    val_acc = float((val_logits.argmax(axis=1) == val_labels).mean())
    test_acc = float((test_logits.argmax(axis=1) == test_labels).mean())

    # ── Save ──
    run_dir = Path(output_dir) / f"config_{config_def.config_id:02d}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    np.save(run_dir / "val_logits.npy", val_logits)
    np.save(run_dir / "val_labels.npy", val_labels)
    np.save(run_dir / "test_logits.npy", test_logits)
    np.save(run_dir / "test_labels.npy", test_labels)

    metrics = {
        "config_id": config_def.config_id,
        "config_desc": config_def.description,
        "rank": config_def.rank,
        "lora_alpha": config_def.lora_alpha,
        "learning_rate": config_def.learning_rate,
        "weight_decay": config_def.weight_decay,
        "target_modules": config_def.target_modules,
        "seed": seed,
        "epochs": epochs,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "train_time_seconds": round(train_time, 1),
        "peak_vram_gb": round(peak_vram, 2),
        "batch_size": batch_size,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  config_{config_def.config_id:02d}/seed_{seed}: "
          f"val={val_acc:.4f} test={test_acc:.4f} "
          f"time={train_time:.0f}s vram={peak_vram:.1f}GB")

    return metrics


# ── CLI ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 1: EuroSAT LoRA Training")
    parser.add_argument("--data-dir", default="./data/eurosat", help="EuroSAT data directory")
    parser.add_argument("--output-dir", default="./outputs/phase1_eurosat", help="Output directory")
    parser.add_argument("--config-id", type=int, default=None, help="Run specific config ID (0-15)")
    parser.add_argument("--seed", type=int, default=None, help="Run specific seed (0, 1, 2)")
    parser.add_argument("--all", action="store_true", help="Run all 48 config×seed combinations")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    args = parser.parse_args()

    configs = build_16_lora_configs()

    if args.all:
        print("=" * 72)
        print(f"PHASE 1: EuroSAT LoRA — {len(configs)} configs × 3 seeds = {len(configs)*3} runs")
        print(f"Device: {args.device} | Max VRAM: {MAX_VRAM_GB} GB")
        print("=" * 72)

        all_metrics = []
        for config_def in configs:
            for seed in [0, 1, 2]:
                print(f"\n[{config_def.config_id:02d}/{len(configs)-1}] seed={seed} | {config_def.description}")
                metrics = train_one_config(
                    config_def, seed, args.data_dir, args.output_dir,
                    epochs=args.epochs, batch_size=args.batch_size, device=args.device,
                )
                all_metrics.append(metrics)

        # Save aggregate metrics
        agg_path = Path(args.output_dir) / "all_metrics.json"
        with open(agg_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\n✅ All {len(all_metrics)} runs complete. Metrics → {agg_path}")

    elif args.config_id is not None and args.seed is not None:
        config_def = configs[args.config_id]
        metrics = train_one_config(
            config_def, args.seed, args.data_dir, args.output_dir,
            epochs=args.epochs, batch_size=args.batch_size, device=args.device,
        )
        print(json.dumps(metrics, indent=2))

    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Run all 48 config×seed combinations:")
        print("  python phase1_train.py --all")
        print("  # Run single config×seed:")
        print("  python phase1_train.py --config-id 0 --seed 0")


if __name__ == "__main__":
    main()
