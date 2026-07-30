"""
Phase 1 Offline Analysis: Selection Reliability From Saved Logits
=================================================================
Runs on Mac after GPU training completes.
Reads saved per-sample logits from Phase 1 and answers:

Q1: Does candidate pool size increase selection regret? (search-overfitting)
Q2: Can independent confirmation set reduce selection error?
Q3: Under equal GPU budget, is exploring more configs better than
    replicating fewer configs? (explore vs replicate trade-off)

Input: experiments/phase1_eurosat/outputs/{config_XX}/seed_{Y}/
           val_logits.npy, val_labels.npy, test_logits.npy, test_labels.npy, metrics.json

Run: uv run python experiments/phase1_eurosat/phase1_analyze.py
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.stats import spearmanr


# ── Load ──────────────────────────────────────────────────────

def load_run_outputs(outputs_dir: str) -> Dict[str, dict]:
    """
    Load all Phase 1 training outputs.
    Returns: {(config_id, seed): {val_logits, val_labels, test_logits, test_labels, metrics}}
    """
    outputs_dir = Path(outputs_dir)
    runs = {}

    for config_dir in sorted(outputs_dir.glob("config_*")):
        config_id = config_dir.name
        for seed_dir in sorted(config_dir.glob("seed_*")):
            seed = seed_dir.name
            key = f"{config_id}_{seed}"

            val_logits_path = seed_dir / "val_logits.npy"
            if not val_logits_path.exists():
                continue

            runs[key] = {
                "val_logits": np.load(val_logits_path),
                "val_labels": np.load(seed_dir / "val_labels.npy"),
                "test_logits": np.load(seed_dir / "test_logits.npy"),
                "test_labels": np.load(seed_dir / "test_labels.npy"),
            }
            metrics_path = seed_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    runs[key]["metrics"] = json.load(f)

    return runs


# ── Candidate Pool ────────────────────────────────────────────

def build_candidate_pools(runs: Dict[str, dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build per-config validation/test accuracy matrices.

    Returns:
      val_accs  (n_configs, n_seeds) — validation accuracy per config×seed
      test_accs (n_configs, n_seeds) — test accuracy per config×seed
      config_ids [str]               — config id strings in-order
    """
    # Group by config_id
    config_data: Dict[str, List[dict]] = {}
    for key, data in runs.items():
        config_id = key.split("_seed_")[0]
        if config_id not in config_data:
            config_data[config_id] = []
        config_data[config_id].append(data)

    config_ids = sorted(config_data.keys())
    max_seeds = max(len(d) for d in config_data.values())
    n_configs = len(config_ids)

    val_accs = np.full((n_configs, max_seeds), np.nan)
    test_accs = np.full((n_configs, max_seeds), np.nan)

    for i, cid in enumerate(config_ids):
        for j, data in enumerate(config_data[cid]):
            val_correct = (data["val_logits"].argmax(axis=1) == data["val_labels"]).sum()
            val_total = len(data["val_labels"])
            val_accs[i, j] = val_correct / val_total

            test_correct = (data["test_logits"].argmax(axis=1) == data["test_labels"]).sum()
            test_total = len(data["test_labels"])
            test_accs[i, j] = test_correct / test_total

    return val_accs, test_accs, config_ids


# ── Simulate Validation Subsampling ───────────────────────────

def simulate_val_subsample(
    runs: Dict[str, dict],
    n_val_per_class: int,
    n_classes: int = 10,
    n_trials: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """
    For each config×seed run, subsample the validation set to
    `n_val_per_class` samples per class. Returns simulated validation
    accuracies for `n_trials` different subsamples.

    Returns: (n_configs, n_seeds, n_trials) validation accuracies
    """
    rng = np.random.default_rng(seed)

    # Group by config_id
    config_data: Dict[str, List[dict]] = {}
    for key, data in runs.items():
        config_id = key.split("_seed_")[0]
        if config_id not in config_data:
            config_data[config_id] = []
        config_data[config_id].append(data)

    config_ids = sorted(config_data.keys())
    max_seeds = max(len(d) for d in config_data.values())
    n_configs = len(config_ids)

    sim_val_accs = np.zeros((n_configs, max_seeds, n_trials))

    for i, cid in enumerate(config_ids):
        for j, data in enumerate(config_data[cid]):
            val_logits = data["val_logits"]  # (n_val, n_classes)
            val_labels = data["val_labels"]  # (n_val,)
            n_total_val = len(val_labels)

            for t in range(n_trials):
                # Stratified subsample: n_val_per_class per class
                selected = []
                for cls in range(n_classes):
                    cls_idx = np.where(val_labels == cls)[0]
                    if len(cls_idx) >= n_val_per_class:
                        chosen = rng.choice(cls_idx, size=n_val_per_class, replace=False)
                    else:
                        chosen = rng.choice(cls_idx, size=n_val_per_class, replace=True)
                    selected.append(chosen)
                selected = np.concatenate(selected)

                # Compute accuracy on subsampled validation
                preds = val_logits[selected].argmax(axis=1)
                targets = val_labels[selected]
                sim_val_accs[i, j, t] = (preds == targets).mean()

    return sim_val_accs


# ── Selection Strategies ──────────────────────────────────────

@dataclass
class SelectionResult:
    selected_config: int
    selected_seed: int
    val_acc: float
    test_acc: float
    regret: float  # oracle_test_acc - selected_test_acc


def select_naive_max(
    val_accs: np.ndarray, test_accs: np.ndarray
) -> SelectionResult:
    """
    Naive Max: use all validation budget, pick config×seed with highest val acc.
    val_accs: (n_configs, n_seeds, n_trials?) or (n_configs, n_seeds)
    """
    if val_accs.ndim == 3:
        mean_val = val_accs.mean(axis=2)
    else:
        mean_val = val_accs
    flat = mean_val.flatten()
    best = np.nanargmax(flat)
    i, j = np.unravel_index(best, mean_val.shape)
    return SelectionResult(
        selected_config=int(i), selected_seed=int(j),
        val_acc=float(mean_val[i, j]), test_acc=float(test_accs[i, j]),
        regret=float(np.nanmax(test_accs) - test_accs[i, j]),
    )


def select_with_confirmation(
    val_accs: np.ndarray, test_accs: np.ndarray,
    n_trials: int = 200, select_fraction: float = 0.5, top_k: int = 4,
    seed: int = 42,
) -> SelectionResult:
    """
    Split validation budget:
      - Selection set (select_fraction of MC trials): pick top-K configs
      - Confirmation set (remaining trials): pick best among top-K

    val_accs: (n_configs, n_seeds, n_trials)
    """
    rng = np.random.default_rng(seed)
    n_configs, n_seeds, total_trials = val_accs.shape

    n_select = max(1, int(total_trials * select_fraction))
    trial_idx = rng.permutation(total_trials)
    select_idx = trial_idx[:n_select]
    confirm_idx = trial_idx[n_select:]

    # Selection phase
    select_means = val_accs[:, :, select_idx].mean(axis=2)
    flat_select = select_means.flatten()
    top_indices = np.argpartition(-flat_select, top_k - 1)[:top_k]

    # Confirmation phase
    confirm_means = val_accs[:, :, confirm_idx].mean(axis=2)
    best_confirm = top_indices[np.nanargmax(confirm_means.flatten()[top_indices])]
    i, j = np.unravel_index(best_confirm, select_means.shape)

    mean_val = val_accs.mean(axis=2)
    return SelectionResult(
        selected_config=int(i), selected_seed=int(j),
        val_acc=float(mean_val[i, j]), test_acc=float(test_accs[i, j]),
        regret=float(np.nanmax(test_accs) - test_accs[i, j]),
    )


def select_explore_vs_replicate(
    val_accs: np.ndarray, test_accs: np.ndarray,
    n_total_trials: int = 16, mode: str = "explore_all",
    n_trials: int = 200, seed: int = 42,
) -> SelectionResult:
    """
    Compare budget allocation strategies:
      - "explore_all":   16 configs × 1 seed each  (all-else-equal)
      - "half_replicate": 8 configs × 2 seeds each
      - "top2_confirm":   12 configs × 1 seed → top-2 × 2 extra seeds

    val_accs: (n_configs, n_seeds, n_trials)
    """
    rng = np.random.default_rng(seed)
    n_configs, n_seeds, _ = val_accs.shape
    mean_val = val_accs.mean(axis=2)  # (n_configs, n_seeds)

    if mode == "explore_all":
        # Use all 16 configs × 1 seed
        n_use = min(n_total_trials, n_configs)
        chosen_configs = rng.choice(n_configs, size=n_use, replace=False)
        # Pick first seed for each
        seeds_for_config = np.zeros(n_use, dtype=int)
        chosen_vals = np.array([mean_val[c, 0] for c in chosen_configs])
        best_idx = np.nanargmax(chosen_vals)
        i, j = chosen_configs[best_idx], seeds_for_config[best_idx]

    elif mode == "half_replicate":
        # 8 configs × 2 seeds each
        n_configs_use = min(n_total_trials // 2, n_configs)
        chosen_configs = rng.choice(n_configs, size=n_configs_use, replace=False)
        # Average across 2 seeds
        avg_vals = np.array([
            np.nanmean([mean_val[c, s] for s in range(min(2, n_seeds))])
            for c in chosen_configs
        ])
        best_idx = np.nanargmax(avg_vals)
        i, j = chosen_configs[best_idx], 0  # pick any seed for reporting

    elif mode == "top2_confirm":
        # 12 configs explore → top-2 confirm with extra seed
        n_explore = min(12, n_configs)
        chosen_configs = rng.choice(n_configs, size=n_explore, replace=False)
        first_pass = np.array([mean_val[c, 0] for c in chosen_configs])
        top2 = chosen_configs[np.argpartition(-first_pass, 1)[:2]]
        # "Confirm" with second seed
        confirm_vals = np.array([
            np.nanmean([first_pass[list(chosen_configs).index(t)], mean_val[t, min(1, n_seeds - 1)]])
            for t in top2
        ])
        best_idx = np.nanargmax(confirm_vals)
        i, j = top2[best_idx], 0

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return SelectionResult(
        selected_config=int(i), selected_seed=int(j),
        val_acc=float(mean_val[i, j]), test_acc=float(test_accs[i, j]),
        regret=float(np.nanmax(test_accs) - test_accs[i, j]),
    )


# ── Main Analysis ─────────────────────────────────────────────

def main():
    outputs_dir = Path(__file__).parent / "outputs"
    if not outputs_dir.exists():
        print(f"❌ Outputs directory not found: {outputs_dir}")
        print("   Run phase1_train.py --all first on the GPU node.")
        sys.exit(1)

    runs = load_run_outputs(str(outputs_dir))
    if len(runs) < 48:
        print(f"⚠️  Only {len(runs)}/48 runs found. Analysis will use available data.")
    else:
        print(f"✅ All {len(runs)} runs loaded.")

    val_accs, test_accs, config_ids = build_candidate_pools(runs)
    n_configs, n_seeds = val_accs.shape
    print(f"\nConfigs: {n_configs} | Seeds: {n_seeds} | Total runs: {len(runs)}")
    print(f"Test acc range: {np.nanmin(test_accs):.4f} - {np.nanmax(test_accs):.4f}")
    print(f"Oracle (best test): config={np.unravel_index(np.nanargmax(test_accs), test_accs.shape)}")

    # ── Simulate validation subsampling at different sizes ──
    print("\n" + "=" * 64)
    print("SIMULATING VALIDATION SUBSAMPLING")
    print("=" * 64)

    for n_val_pc in [1, 2, 4]:
        print(f"\n─── n_val/class = {n_val_pc} ───")

        sim_val = simulate_val_subsample(
            runs, n_val_per_class=n_val_pc, n_trials=200
        )
        n_trials = sim_val.shape[2]
        print(f"  Simulated {n_trials} validation subsamples")

        # Naive Max
        nm = select_naive_max(sim_val, test_accs)
        print(f"  Naive Max:          regret={nm.regret:.4f}, test={nm.test_acc:.4f} (config_{nm.selected_config:02d})")

        # Select + Confirm (try different split ratios)
        for split in [0.3, 0.5, 0.7]:
            sc = select_with_confirmation(
                sim_val, test_accs, n_trials=n_trials,
                select_fraction=split, top_k=4,
            )
            print(f"  Select+Confirm ({split:.0%}): regret={sc.regret:.4f}, test={sc.test_acc:.4f} (config_{sc.selected_config:02d})")

        # Explore vs Replicate
        for mode in ["explore_all", "half_replicate", "top2_confirm"]:
            ev = select_explore_vs_replicate(
                sim_val, test_accs, mode=mode, n_trials=n_trials,
            )
            print(f"  Explore/Repl ({mode:16s}): regret={ev.regret:.4f}, test={ev.test_acc:.4f} (config_{ev.selected_config:02d})")

    # ── Q1: Regret vs candidate pool size ──
    print("\n" + "=" * 64)
    print("Q1: REGRET vs CANDIDATE POOL SIZE")
    print("=" * 64)

    rng = np.random.default_rng(42)
    for n_candidates in [4, 8, 12, 16]:
        regrets = []
        for _ in range(100):
            chosen = rng.choice(n_configs, size=n_candidates, replace=False)
            sub_val = val_accs[chosen]
            sub_test = test_accs[chosen]
            nm = select_naive_max(sub_val, sub_test)
            regrets.append(nm.regret)
        print(f"  {n_candidates:2d} candidates: mean regret={np.mean(regrets):.4f} ± {np.std(regrets):.4f}")

    # ── Q3: Explore vs replicate with equal budget ──
    print("\n" + "=" * 64)
    print("Q3: EXPLORE vs REPLICATE TRADE-OFF")
    print("=" * 64)

    sim_val = simulate_val_subsample(runs, n_val_per_class=2, n_trials=200)

    for mode in ["explore_all", "half_replicate", "top2_confirm"]:
        regrets = []
        test_accs_selected = []
        for _ in range(100):
            ev = select_explore_vs_replicate(
                sim_val, test_accs, mode=mode, n_trials=200,
            )
            regrets.append(ev.regret)
            test_accs_selected.append(ev.test_acc)
        print(f"  {mode:16s}: regret={np.mean(regrets):.4f} ± {np.std(regrets):.4f}, "
              f"test={np.mean(test_accs_selected):.4f}")

    # ── Ranking correlation ──
    print("\n" + "=" * 64)
    print("RANKING STABILITY (validation vs test)")
    print("=" * 64)

    mean_val = val_accs.mean(axis=1)  # avg across seeds
    mean_test = test_accs.mean(axis=1)
    rho, p = spearmanr(mean_val, mean_test)
    print(f"  Val-test Spearman ρ: {rho:.3f} (p={p:.4f})")

    val_rank = np.argsort(np.argsort(-mean_val))
    test_rank = np.argsort(np.argsort(-mean_test))
    top3_val = np.argsort(-mean_val)[:3]
    top3_test = np.argsort(-mean_test)[:3]
    print(f"  Top-3 by val:  {list(top3_val)}")
    print(f"  Top-3 by test: {list(top3_test)}")
    print(f"  Overlap: {len(set(top3_val) & set(top3_test))}/3")


if __name__ == "__main__":
    main()
