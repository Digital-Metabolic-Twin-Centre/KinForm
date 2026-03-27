#!/usr/bin/env python3
"""
Baseline R² using Normal Distribution Draws

This script computes a baseline R² by:
1. For each CV/SE-CV fold, fit a Normal(μ, σ²) to the training targets
2. Generate random predictions for test set by sampling from N(μ, σ²)
3. Compute R² between random predictions and true test targets

This provides a baseline of "statistical randomness" to contextualize model performance.

Datasets:
- DLKcat (kcat dataset)
- EITLEM/Shen (kcat dataset)

Splits:
- CV (KFold)
- SE-CV (GroupKFold)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import ROOT, SEQ_LOOKUP

# ═══════════════════════════ Configuration ═══════════════════════════
SEED = 42
np.random.seed(SEED)

# Paths
RAW_DLKCAT = ROOT / "data/dlkcat_raw.json"
RAW_EITLEM = ROOT / "data/EITLEM_data/KCAT/kcat_data.json"


# ═══════════════════════════ Data Loading ═══════════════════════════
def load_dlkcat() -> Tuple[List[str], np.ndarray, List[int]]:
    """
    Load DLKcat dataset.
    
    Returns:
        sequences: List of protein sequences
        y: Target values (log10 kcat)
        groups: Sequence group IDs for GroupKFold
    """
    with RAW_DLKCAT.open() as f:
        raw = json.load(f)
    
    # Filter valid entries
    valid = [
        d for d in raw
        if len(d["Sequence"]) <= 1499 
        and float(d["Value"]) > 0 
        and "." not in d["Smiles"]
    ]
    
    sequences = [d["Sequence"] for d in valid]
    y = np.array([math.log(float(d["Value"]), 10) for d in valid], dtype=np.float32)
    
    # Get sequence groups for GroupKFold
    seq_to_id = {v: k for k, v in pd.read_pickle(SEQ_LOOKUP).items()}
    groups = [seq_to_id[s] for s in sequences]
    
    return sequences, y, groups


def load_eitlem() -> Tuple[List[str], np.ndarray, List[int]]:
    """
    Load EITLEM/Shen dataset.
    
    Returns:
        sequences: List of protein sequences
        y: Target values (log10 kcat)
        groups: Sequence group IDs for GroupKFold
    """
    with RAW_EITLEM.open() as f:
        raw = json.load(f)
    
    # Filter valid entries
    valid = [
        (i, r) for i, r in enumerate(raw)
        if len(r["sequence"]) <= 1499 and float(r["value"]) > 0
    ]
    
    sequences = [r["sequence"] for _, r in valid]
    y = np.array([math.log(float(r["value"]), 10) for _, r in valid], dtype=np.float32)
    
    # Get sequence groups for GroupKFold
    seq_to_id = {v: k for k, v in pd.read_pickle(SEQ_LOOKUP).items()}
    groups = [seq_to_id[s] for s in sequences]
    
    return sequences, y, groups


# ═══════════════════════════ Baseline R² Computation ═══════════════════════════
def compute_normal_draw_baseline(
    y_full: np.ndarray,
    groups: List[int],
    split_mode: str,
    n_folds: int = 5,
    n_repeats: int = 50
) -> Tuple[List[Dict], Dict]:
    """
    Compute baseline R² using normal distribution draws with repeated sampling.
    
    For each fold:
    1. Fit Normal(μ, σ²) to training targets
    2. Repeat n_repeats times:
       - Sample n_test predictions from N(μ, σ²)
       - Compute R² against true test targets
    3. Report mean ± std R² across repeats
    
    Args:
        y_full: Full target array (log10 scale)
        groups: Sequence group IDs for GroupKFold
        split_mode: "kfold" or "groupkfold"
        n_folds: Number of CV folds
        n_repeats: Number of random draws per fold for stability
    
    Returns:
        fold_results: List of dicts with statistics for each fold
        summary_stats: Overall summary statistics
    """
    if split_mode == "groupkfold":
        cv = GroupKFold(n_splits=n_folds).split(y_full, groups=groups)
    else:  # kfold
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=SEED).split(y_full)
    
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv, 1):
        # Get training targets
        y_train = y_full[train_idx]
        y_test = y_full[test_idx]
        
        # Fit Normal distribution to training data
        mu = np.mean(y_train)
        sigma = np.std(y_train, ddof=1)  # Use sample std (N-1)
        
        # Generate random predictions n_repeats times
        n_test = len(test_idx)
        fold_r2s = []
        
        for _ in range(n_repeats):
            y_pred_random = np.random.normal(loc=mu, scale=sigma, size=n_test)
            r2 = r2_score(y_test, y_pred_random)
            fold_r2s.append(r2)
        
        # Compute statistics across repeats
        mean_r2 = np.mean(fold_r2s)
        std_r2 = np.std(fold_r2s)
        min_r2 = np.min(fold_r2s)
        max_r2 = np.max(fold_r2s)
        
        fold_results.append({
            "fold": fold_idx,
            "mu": mu,
            "sigma": sigma,
            "n_train": len(train_idx),
            "n_test": n_test,
            "mean_r2": mean_r2,
            "std_r2": std_r2,
            "min_r2": min_r2,
            "max_r2": max_r2,
            "all_r2s": fold_r2s
        })
        
        print(f"    Fold {fold_idx}: μ={mu:.3f}, σ={sigma:.3f}, "
              f"R²={mean_r2:.4f}±{std_r2:.4f} (over {n_repeats} draws)")
    
    # Aggregate across folds
    all_fold_means = [f["mean_r2"] for f in fold_results]
    
    summary_stats = {
        "median_r2": float(np.median(all_fold_means)),
        "mean_r2": float(np.mean(all_fold_means)),
        "std_r2": float(np.std(all_fold_means)),
        "min_r2": float(np.min(all_fold_means)),
        "max_r2": float(np.max(all_fold_means)),
        "grand_mean": float(np.mean([r2 for f in fold_results for r2 in f["all_r2s"]])),
        "grand_std": float(np.std([r2 for f in fold_results for r2 in f["all_r2s"]]))
    }
    
    return fold_results, summary_stats


# ═══════════════════════════ Main Execution ═══════════════════════════
def main():
    """Run normal draw baseline for both datasets and split modes."""
    
    datasets = {
        "DLKcat": load_dlkcat,
        "EITLEM/Shen": load_eitlem
    }
    
    split_modes = ["kfold", "groupkfold"]
    split_labels = {"kfold": "CV", "groupkfold": "SE-CV"}
    
    # Number of repeated draws per fold for stability
    N_REPEATS = 50
    
    print("="*70)
    print("Baseline R² using Normal Distribution Draws")
    print(f"({N_REPEATS} repeated draws per fold for robustness)")
    print("="*70)
    print()
    
    all_results = {}
    
    for dataset_name, load_func in datasets.items():
        print(f"\n{'─'*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'─'*70}")
        
        # Load data
        sequences, y, groups = load_func()
        print(f"Loaded {len(sequences)} samples")
        print(f"Target range: [{y.min():.2f}, {y.max():.2f}] (log10 scale)")
        print()
        
        dataset_results = {}
        
        for split_mode in split_modes:
            split_label = split_labels[split_mode]
            print(f"  {split_label} ({split_mode}):")
            
            fold_results, summary_stats = compute_normal_draw_baseline(
                y, groups, split_mode, n_folds=5, n_repeats=N_REPEATS
            )
            
            print(f"  → Median R² across folds: {summary_stats['median_r2']:.4f}")
            print(f"  → Mean R² across folds:   {summary_stats['mean_r2']:.4f} ± {summary_stats['std_r2']:.4f}")
            print(f"  → Grand mean (all draws): {summary_stats['grand_mean']:.4f} ± {summary_stats['grand_std']:.4f}")
            print(f"  → Range:                  [{summary_stats['min_r2']:.4f}, {summary_stats['max_r2']:.4f}]")
            print()
            
            dataset_results[split_mode] = {
                "fold_results": fold_results,
                "summary": summary_stats
            }
        
        all_results[dataset_name] = dataset_results
    
    # Print summary table
    print("\n" + "="*70)
    print("Summary Table")
    print("="*70)
    print(f"{'Dataset':<20} {'Split':<10} {'Median R²':<12} {'Mean R² ± Std':<20}")
    print("─"*70)
    
    for dataset_name, dataset_res in all_results.items():
        for split_mode in split_modes:
            summary = dataset_res[split_mode]["summary"]
            split_label = split_labels[split_mode]
            print(f"{dataset_name:<20} {split_label:<10} "
                  f"{summary['median_r2']:>8.4f}     "
                  f"{summary['mean_r2']:>6.4f} ± {summary['std_r2']:<6.4f}")
    
    print("="*70)
    print("\nInterpretation:")
    print(f"These R² values (computed with {N_REPEATS} repeats per fold) represent")
    print("the expected performance of random predictions drawn from the training")
    print("distribution. Any model should significantly exceed these baseline values")
    print("to be considered useful. The repeated sampling ensures robust estimates.")
    print("="*70)


if __name__ == "__main__":
    main()
