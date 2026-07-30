"""
Phase 0: Zero-GPU Statistical Pre-experiment
=============================================
Synthetic per-sample binary predictions to answer:

Q1: Are Naive Max, LCB, and Bootstrap quantile equivalent
    when all configs use the same number of validation samples?
Q2: What validation set sizes cause divergence between strategies?
Q3: Does validation label noise amplify strategy differences?
Q4: Which of the 4 genuine strategies produce different decisions?

All results should inform Phase 1 experiment design without
wasting GPU time proving obvious equivalences.

Run: python experiments/phase0_synthetic/pre_experiment.py
"""

from dataclasses import dataclass, field
from typing import List, Optional
import itertools
import json
import sys
from pathlib import Path

import numpy as np

# ── Configuration ────────────────────────────────────────────

@dataclass
class PreExperimentConfig:
    """Control all synthetic experiment parameters."""
    # Number of candidate configurations to simulate
    n_configs: int = 16
    # True test accuracies (sorted, hidden from selection strategies)
    true_test_accs: Optional[List[float]] = None
    # Validation samples per class
    val_samples_per_class: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    # Number of classes
    n_classes: int = 10
    # Label noise rates on validation set
    noise_rates: List[float] = field(default_factory=lambda: [0.0, 0.10, 0.20])
    # Number of Monte Carlo resamplings of the validation set
    n_mc_samples: int = 500
    # Number of bootstrap resamples for bootstrap quantile strategy
    n_bootstrap: int = 500
    # Random seed
    seed: int = 42
    # Confidence level for LCB
    lcb_lambda: float = 1.0

    def __post_init__(self):
        if self.true_test_accs is None:
            rng = np.random.default_rng(self.seed)
            # Spread true accuracies: 55% to 75% (realistic for VTAB-1K few-shot)
            accs = 0.55 + 0.20 * rng.random(self.n_configs)
            self.true_test_accs = sorted(accs.tolist())


# ── Simulator ─────────────────────────────────────────────────

def simulate_validation_accuracies(
    true_test_accs: List[float],
    n_val_samples: int,
    n_classes: int,
    noise_rate: float,
    n_mc: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate validation set evaluation.

    For each MC trial:
    - For each config, generate `n_val_samples` predictions as Bernoulli(accuracy)
    - Inject label noise by flipping `noise_rate` fraction of predictions
    - Return observed validation accuracies

    Returns: (n_mc, n_configs) array of validation accuracies
    """
    n_configs = len(true_test_accs)
    val_accs = np.zeros((n_mc, n_configs))

    for i in range(n_configs):
        # Each validation sample: correct (1) with prob = true accuracy
        # Account for noise: observed correct = (1-noise)*true + noise*(1-true)
        # for symmetric noise (flip label → random guess among other n_classes-1)
        # Simplified: noise * (1/(n_classes-1)) chance of guessing right anyway
        effective_acc = true_test_accs[i] * (1 - noise_rate) + \
                        (1 - true_test_accs[i]) * noise_rate / (n_classes - 1)

        # Generate n_val_samples * n_classes total samples per MC trial
        total_samples = n_val_samples * n_classes
        samples = rng.binomial(total_samples, effective_acc, size=n_mc)
        val_accs[:, i] = samples / total_samples

    return val_accs


# ── Selection Strategies ──────────────────────────────────────

def strategy_naive_max(val_accs: np.ndarray) -> np.ndarray:
    """Select config with highest mean validation accuracy."""
    mean_accs = val_accs.mean(axis=0)
    return np.argmax(mean_accs)  # single selection (deterministic given MC mean)


def strategy_lcb(val_accs: np.ndarray, lmbda: float = 1.0) -> np.ndarray:
    """Select config with highest Lower Confidence Bound."""
    mean = val_accs.mean(axis=0)
    std = val_accs.std(axis=0, ddof=1)
    lcb = mean - lmbda * std
    return np.argmax(lcb)


def strategy_bootstrap_quantile(
    val_accs: np.ndarray, n_bootstrap: int = 500, quantile: float = 0.10, rng: np.random.Generator = None
) -> np.ndarray:
    """
    Select config with highest bootstrap lower quantile.
    For each config, bootstrap resample the MC trials and take the quantile.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n_mc, n_configs = val_accs.shape
    lower_bounds = np.zeros(n_configs)

    for i in range(n_configs):
        boot_means = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n_mc, size=n_mc)
            boot_means[b] = val_accs[idx, i].mean()
        lower_bounds[i] = np.percentile(boot_means, quantile * 100)

    return np.argmax(lower_bounds)


def strategy_selection_confirmation(
    val_accs: np.ndarray, split_ratio: float = 0.5, rng: np.random.Generator = None
) -> np.ndarray:
    """
    Split validation budget: selection set → top-K; confirmation set → pick best.
    K is determined by split_ratio indirectly: we use half the MC samples for
    selection, half for confirmation.
    Returns index of selected config.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n_mc, n_configs = val_accs.shape
    n_select = max(1, int(n_mc * split_ratio))

    # Shuffle MC trials
    idx = rng.permutation(n_mc)
    select_idx = idx[:n_select]
    confirm_idx = idx[n_select:]

    # Selection phase: pick top-4
    select_means = val_accs[select_idx].mean(axis=0)
    top_k = min(4, n_configs)  # fix K=4
    candidates = np.argpartition(-select_means, top_k - 1)[:top_k]

    # Confirmation phase: pick best among candidates
    confirm_means = val_accs[confirm_idx][:, candidates].mean(axis=0)
    return candidates[np.argmax(confirm_means)]


def strategy_cross_fold_ranking(
    val_accs: np.ndarray, n_folds: int = 4, criterion: str = "avg_rank", rng: np.random.Generator = None
) -> np.ndarray:
    """
    Split validation into folds. Select config with best average rank
    (or best worst rank) across folds — NOT highest total accuracy.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n_mc, n_configs = val_accs.shape
    fold_size = n_mc // n_folds

    config_ranks = np.zeros((n_folds, n_configs))
    for f in range(n_folds):
        start = f * fold_size
        end = start + fold_size if f < n_folds - 1 else n_mc
        fold_means = val_accs[start:end].mean(axis=0)
        # Rank: 1 = best, n_configs = worst
        config_ranks[f] = n_configs - np.argsort(np.argsort(fold_means))

    if criterion == "avg_rank":
        scores = -config_ranks.mean(axis=0)  # negate for argmax
    elif criterion == "worst_rank":
        scores = -config_ranks.max(axis=0)  # minimize worst rank
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    return np.argmax(scores)


def strategy_explore_replicate(
    all_config_accs: np.ndarray,
    n_total_trials: int = 16,
    n_replicates: int = 1,
    rng: np.random.Generator = None,
) -> int:
    """
    Simulate explore-vs-replicate budget allocation.

    (a) 16 configs × 1 trial each
    (b) 8 configs × 2 trials each → pick 8 configs, run each twice, average
    (c) 12 configs × 1 trial → pick top-2 → run each 2 more times

    n_total_trials: total training budget
    n_replicates: controls strategy:
        1 → (a) explore all
        2 → (b) half configs, replicate each
        'top2' → (c) explore first, then replicate top-2

    Returns index of selected config.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_configs = all_config_accs.shape[1]
    n_mc = all_config_accs.shape[0]

    if n_replicates == 1:
        # (a) Explore all: 16 configs × 1 trial
        n_chosen = min(n_total_trials, n_configs)
        chosen = rng.choice(n_configs, size=n_chosen, replace=False)
        # Each chosen config gets 1 trial
        trial_results = all_config_accs[:, chosen].mean(axis=0)
        return chosen[np.argmax(trial_results)]

    elif n_replicates == 2:
        # (b) Half configs, each run twice
        n_chosen = min(n_total_trials // 2, n_configs)
        chosen = rng.choice(n_configs, size=n_chosen, replace=False)
        # Run each config twice: average of 2 independent draws
        trial_results = np.zeros(n_chosen)
        for j, c in enumerate(chosen):
            draw1 = rng.choice(all_config_accs[:, c], size=1).item()
            draw2 = rng.choice(all_config_accs[:, c], size=1).item()
            trial_results[j] = (draw1 + draw2) / 2
        return chosen[np.argmax(trial_results)]

    elif n_replicates == 'top2':
        # (c) Explore 12, then replicate top-2
        n_explore = min(12, n_configs)
        chosen = rng.choice(n_configs, size=n_explore, replace=False)
        # First pass: 1 trial each
        first_pass = np.array([rng.choice(all_config_accs[:, c], size=1).item() for c in chosen])
        top2_idx = np.argpartition(-first_pass, 1)[:2]
        top2 = chosen[top2_idx]
        # Second pass: 2 more trials for top-2
        results = np.zeros(2)
        for j, c in enumerate(top2):
            extra1 = rng.choice(all_config_accs[:, c], size=1).item()
            extra2 = rng.choice(all_config_accs[:, c], size=1).item()
            results[j] = (first_pass[top2_idx[j]] + extra1 + extra2) / 3
        return top2[np.argmax(results)]

    else:
        raise ValueError(f"Unknown n_replicates: {n_replicates}")


# ── Oracle ────────────────────────────────────────────────────

def oracle_selection(true_test_accs: List[float]) -> int:
    """Perfect selection: pick config with highest true test accuracy."""
    return int(np.argmax(true_test_accs))


# ── Comparison Metrics ────────────────────────────────────────

def compute_regret(
    selected_idx: int,
    true_test_accs: List[float],
    oracle_idx: int,
) -> float:
    """Regret = oracle test acc - selected test acc (lower = better)."""
    return true_test_accs[oracle_idx] - true_test_accs[selected_idx]


def compute_rank_correlation(
    rankings: List[List[int]],
) -> float:
    """Kendall's W for multiple rankings. 1.0 = perfect agreement."""
    from scipy.stats import kendalltau

    if len(rankings) < 2:
        return 1.0

    # Average pairwise Kendall tau
    taus = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            tau, _ = kendalltau(rankings[i], rankings[j])
            taus.append(tau)
    return float(np.mean(taus))


# ── Main Experiment ───────────────────────────────────────────

def run_phase0(config: PreExperimentConfig = None):
    """Run all Phase 0 pre-experiments."""
    if config is None:
        config = PreExperimentConfig()

    rng = np.random.default_rng(config.seed)
    oracle_idx = oracle_selection(config.true_test_accs)
    oracle_acc = config.true_test_accs[oracle_idx]

    print("=" * 72)
    print("PHASE 0: ZERO-GPU STATISTICAL PRE-EXPERIMENT")
    print("=" * 72)
    print(f"\nConfigs: {config.n_configs}")
    print(f"True test accuracies: {[f'{a:.3f}' for a in config.true_test_accs]}")
    print(f"Oracle config: #{oracle_idx} (acc={oracle_acc:.4f})")
    print(f"Val samples/class: {config.val_samples_per_class}")
    print(f"Noise rates: {config.noise_rates}")
    print(f"MC trials: {config.n_mc_samples}")

    # ── Q1: Equivalent strategies comparison ──
    print("\n" + "─" * 72)
    print("Q1: Are Naive Max, LCB, Bootstrap Quantile equivalent?")
    print("─" * 72)

    q1_results = {}
    for n_val in config.val_samples_per_class:
        for noise in config.noise_rates:
            key = f"n_val={n_val}, noise={noise:.0%}"
            val_accs = simulate_validation_accuracies(
                config.true_test_accs, n_val, config.n_classes, noise, config.n_mc_samples, rng
            )

            naive = strategy_naive_max(val_accs)
            lcb = strategy_lcb(val_accs, config.lcb_lambda)
            boot = strategy_bootstrap_quantile(val_accs, config.n_bootstrap, 0.10, rng)

            # Rankings (not just top-1)
            naive_rank = np.argsort(np.argsort(-val_accs.mean(axis=0)))
            lcb_mean = val_accs.mean(axis=0)
            lcb_std = val_accs.std(axis=0, ddof=1)
            lcb_vals = lcb_mean - config.lcb_lambda * lcb_std
            lcb_rank = np.argsort(np.argsort(-lcb_vals))

            # Bootstrap ranking
            boot_lower = np.zeros(config.n_configs)
            for i in range(config.n_configs):
                boot_means = np.zeros(config.n_bootstrap)
                for b in range(config.n_bootstrap):
                    idx = rng.integers(0, config.n_mc_samples, size=config.n_mc_samples)
                    boot_means[b] = val_accs[idx, i].mean()
                boot_lower[i] = np.percentile(boot_means, 10)
            boot_rank = np.argsort(np.argsort(-boot_lower))

            strategies_agree = (naive == lcb == boot)
            naive_lcb_same_rank = np.array_equal(naive_rank, lcb_rank)

            # Spearman correlation between rankings
            from scipy.stats import spearmanr
            r_nb, _ = spearmanr(naive_rank, boot_rank)
            r_nl, _ = spearmanr(naive_rank, lcb_rank)
            r_lb, _ = spearmanr(lcb_rank, boot_rank)

            q1_results[key] = {
                'naive_idx': int(naive),
                'lcb_idx': int(lcb),
                'boot_idx': int(boot),
                'all_agree': bool(strategies_agree),
                'naive_lcb_same_rank': bool(naive_lcb_same_rank),
                'spearman_naive_boot': float(r_nb),
                'spearman_naive_lcb': float(r_nl),
                'spearman_lcb_boot': float(r_lb),
            }

            status = "✅ EQUIVALENT" if naive == lcb == boot else "⚠️  DIVERGE"
            print(f"  {key}: {status}")
            print(f"    Naive=#{naive}, LCB=#{lcb}, Boot=#{boot}")
            print(f"    Spearman: naive-boot={r_nb:.3f}, naive-lcb={r_nl:.3f}, lcb-boot={r_lb:.3f}")

    # ── Q2: Genuinely different strategies ──
    print("\n" + "─" * 72)
    print("Q2: Do the 4 genuine strategies differ from Naive Max?")
    print("─" * 72)

    q2_results = {}
    for n_val in config.val_samples_per_class:
        for noise in config.noise_rates:
            key = f"n_val={n_val}, noise={noise:.0%}"
            val_accs = simulate_validation_accuracies(
                config.true_test_accs, n_val, config.n_classes, noise, config.n_mc_samples, rng
            )

            naive = strategy_naive_max(val_accs)
            sc = strategy_selection_confirmation(val_accs, rng=rng)
            cross_avg = strategy_cross_fold_ranking(val_accs, criterion="avg_rank", rng=rng)
            cross_worst = strategy_cross_fold_ranking(val_accs, criterion="worst_rank", rng=rng)

            naive_regret = compute_regret(naive, config.true_test_accs, oracle_idx)
            sc_regret = compute_regret(sc, config.true_test_accs, oracle_idx)
            cross_avg_regret = compute_regret(cross_avg, config.true_test_accs, oracle_idx)
            cross_worst_regret = compute_regret(cross_worst, config.true_test_accs, oracle_idx)

            # Explore vs replicate (need multiple trials per config simulation)
            er_regrets = {}
            for n_rep in [1, 2, 'top2']:
                er_regret = compute_regret(
                    strategy_explore_replicate(val_accs, n_total_trials=16, n_replicates=n_rep, rng=rng),
                    config.true_test_accs, oracle_idx,
                )
                er_regrets[str(n_rep)] = float(er_regret)

            q2_results[key] = {
                'naive_regret': float(naive_regret),
                'sel_conf_regret': float(sc_regret),
                'cross_avg_regret': float(cross_avg_regret),
                'cross_worst_regret': float(cross_worst_regret),
                'explore_rep_regrets': er_regrets,
            }

            print(f"  {key}:")
            print(f"    Naive Max regret:          {naive_regret:.4f}")
            print(f"    Select+Confirm regret:     {sc_regret:.4f} (Δ={sc_regret - naive_regret:+.4f})")
            print(f"    Cross-fold avg rank:       {cross_avg_regret:.4f} (Δ={cross_avg_regret - naive_regret:+.4f})")
            print(f"    Cross-fold worst rank:     {cross_worst_regret:.4f} (Δ={cross_worst_regret - naive_regret:+.4f})")
            for n_rep, er_r in er_regrets.items():
                print(f"    Explore/Replicate ({n_rep}): {er_r:.4f} (Δ={er_r - naive_regret:+.4f})")

    # ── Q3: Effect of validation set size on selection quality ──
    print("\n" + "─" * 72)
    print("Q3: Selection quality vs validation set size")
    print("─" * 72)

    q3_results = {}
    for n_val in config.val_samples_per_class:
        noise = 0.0  # clean validation
        val_accs = simulate_validation_accuracies(
            config.true_test_accs, n_val, config.n_classes, noise, config.n_mc_samples, rng
        )
        naive = strategy_naive_max(val_accs)
        regret = compute_regret(naive, config.true_test_accs, oracle_idx)
        q3_results[f"n_val={n_val}"] = float(regret)
        print(f"  n_val/class={n_val}: Naive Max regret = {regret:.4f}")

    # ── Summary ──
    print("\n" + "=" * 72)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 72)

    all_agree_count = sum(1 for v in q1_results.values() if v['all_agree'])
    mean_spearman = np.mean([
        v['spearman_naive_lcb'] for v in q1_results.values()
    ])

    print(f"\n  Q1: Strategies agree on top-1 in {all_agree_count}/{len(q1_results)} conditions")
    print(f"      Mean Spearman(Naive vs LCB rank): {mean_spearman:.3f}")

    if mean_spearman > 0.98:
        print("  → RECOMMENDATION: LCB and Bootstrap are equivalent to Naive Max.")
        print("    Do NOT waste GPU proving this. Drop LCB/Bootstrap from Phase 1.")
        print("    Focus on the 4 genuinely different strategies.")
    else:
        print("  → Strategies diverge — worth testing LCB separately on real data.")

    # Identify best-performing strategy
    all_deltas = []
    for v in q2_results.values():
        all_deltas.append(v['sel_conf_regret'] - v['naive_regret'])
    mean_sel_conf_delta = np.mean(all_deltas)
    print(f"\n  Q2: Mean Select+Confirm Δ regret: {mean_sel_conf_delta:+.4f}")
    if mean_sel_conf_delta < 0:
        print("  → Select+Confirm tends to OUTPERFORM Naive Max — test in Phase 1")
    else:
        print("  → Select+Confirm tends to UNDERPERFORM Naive Max — still worth testing with real data")

    # Save results
    output_path = Path(__file__).parent / "phase0_results.json"
    results = {
        'config': {k: v for k, v in config.__dict__.items() if k != 'true_test_accs'},
        'true_test_accs': config.true_test_accs,
        'q1_equivalent_strategies': q1_results,
        'q2_genuine_strategies': q2_results,
        'q3_val_size_effect': q3_results,
    }
    # Convert numpy types for JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    print(f"\n  Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    config = PreExperimentConfig(
        n_configs=16,
        val_samples_per_class=[1, 2, 4, 8],
        n_classes=10,
        noise_rates=[0.0, 0.10, 0.20],
        n_mc_samples=500,
        lcb_lambda=1.0,
    )
    run_phase0(config)
