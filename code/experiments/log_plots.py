"""
Plot predicted log kcat vs experimental log kcat for Shen/EITLEM dataset.

Creates a two-panel figure showing:
- CV (KFold): log10(pred kcat) vs log10(exp kcat)
- SE-CV (GroupKFold): log10(pred kcat) vs log10(exp kcat)

Uses hexbin/density scatter to show point density and outliers.
Includes metrics: R², RMSE, Spearman ρ, N
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_squared_error


# ═══════════════════════════ Configuration ═══════════════════════════
ROOT = Path("/home/saleh/KinForm-1")
EITLEM_DIR = ROOT / "data/EITLEM_data"
JSON_FILE = EITLEM_DIR / "KCAT/kcat_data.json"
TRAIN_PAIRS = EITLEM_DIR / "KCAT/KCATTrainPairInfo"
TEST_PAIRS = EITLEM_DIR / "KCAT/KCATTestPairInfo"
RESULTS_PKL = ROOT / "results/unikp_comp_eitlem.pkl"
OUTPUT_PATH = ROOT / "results/plots/eitlem_cv_secv_scatter.png"

# Model to plot (change this if you want different model)
MODEL_CONFIG = "KinForm-L"


# ═══════════════════════════ Load Data ═══════════════════════════════
def load_eitlem_full_dataset() -> np.ndarray:
    """Load the full EITLEM dataset y values."""
    with JSON_FILE.open() as fp:
        raw = json.load(fp)
    valid = [(i, r) for i, r in enumerate(raw) if len(r["sequence"]) <= 1499 and float(r["value"]) > 0]
    orig_idx = [i for i, _ in valid]
    y_full = np.array([math.log(float(r["value"]), 10) for _, r in valid], np.float32)
    return y_full


def load_eitlem_results(model_config: str = "KinForm-L") -> Dict[str, Dict[str, Any]]:
    """
    Load EITLEM CV results and merge test sets across folds.
    
    Returns:
        Dictionary with 'kfold' and 'groupkfold' keys, each containing:
        - 'test_df': DataFrame with y_true, y_pred for all test samples
        - 'train_data': List of training y_true arrays from each fold
    """
    # Load full dataset
    y_full = load_eitlem_full_dataset()
    
    with open(RESULTS_PKL, "rb") as f:
        results = pickle.load(f)
    
    # Check if we need to handle KinForm-L(OS) naming
    config_data = results.get(model_config, {})
    
    output = {}
    
    for split_mode in ["kfold", "groupkfold"]:
        test_rows = []
        train_data = []
        fold_records = config_data.get(split_mode, [])
        
        for rec in fold_records:
            # Get config name (might be KinForm-L(OS) or KinForm-L)
            config = rec.get("config", model_config)
            
            # Only include the requested model config
            if config not in [model_config, f"{model_config}(OS)"]:
                continue
            
            # Test data
            test_indices = rec.get("test_indices", [])
            y_true = rec.get("y_true", [])
            y_pred = rec.get("y_pred", [])
            
            for idx, yt, yp in zip(test_indices, y_true, y_pred):
                test_rows.append({
                    "index": idx,
                    "y_true": yt,
                    "y_pred": yp,
                    "fold": rec.get("fold", -1)
                })
            
            # Training data - use train_indices to get y values from full dataset
            train_indices = rec.get("train_indices", [])
            if train_indices:
                y_train = y_full[train_indices]
                train_data.append(y_train)
        
        test_df = pd.DataFrame(test_rows)
        # Remove duplicates (if any sample appears in multiple folds, keep first)
        test_df = test_df.drop_duplicates(subset=["index"], keep="first")
        
        output[split_mode] = {
            'test_df': test_df,
            'train_data': train_data
        }
    
    return output


# ═══════════════════════════ Plotting ═══════════════════════════════
def plot_cv_comparison(data_dict: Dict[str, Dict[str, Any]], output_path: Path) -> None:
    """
    Create publication-quality four-panel figure:
    - Top row: hexbin scatter plots for CV vs SE-CV
    - Bottom row: dual-axis bar/line plots showing training counts vs test error by bin
    """
    # Enhanced styling for publication
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
    })
    
    # Create figure with 2 rows and 2 columns + colorbar
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05], 
                          wspace=0.05, hspace=0.35,
                          height_ratios=[1, 1])
    
    # Top row: scatter plots
    scatter_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])
    
    # Bottom row: bar/line plots
    bar_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    
    # Share y-axis between the scatter plots
    scatter_axes[1].sharey(scatter_axes[0])
    
    titles = {
        "kfold": "CV",
        "groupkfold": "SE-CV"
    }
    
    # Determine global plot limits for consistency
    all_data = []
    for data in data_dict.values():
        df = data['test_df']
        if not df.empty:
            all_data.extend(df["y_true"].values)
            all_data.extend(df["y_pred"].values)
    
    if all_data:
        global_min, global_max = np.min(all_data), np.max(all_data)
        margin = (global_max - global_min) * 0.05
        plot_min, plot_max = global_min - margin, global_max + margin
    else:
        plot_min, plot_max = -2, 6
    
    hexbin_objects = []
    
    # ========== Top Row: Scatter Plots ==========
    for idx, (split_mode, data) in enumerate(data_dict.items()):
        ax = scatter_axes[idx]
        df = data['test_df']
        
        if df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", 
                   transform=ax.transAxes, fontsize=14)
            continue
        
        y_true = df["y_true"].values
        y_pred = df["y_pred"].values
        
        # Calculate metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        rho, _ = spearmanr(y_true, y_pred)
        n = len(y_true)
        
        # Hexbin plot with professional colormap
        hb = ax.hexbin(
            y_true, y_pred,
            gridsize=35,
            cmap="viridis",
            mincnt=1,
            extent=[plot_min, plot_max, plot_min, plot_max],
            linewidths=0.1,
            edgecolors='face',
            alpha=0.9
        )
        hexbin_objects.append(hb)
        
        # Identity line (y = x) with better styling
        ax.plot([plot_min, plot_max], [plot_min, plot_max], 
                'k--', linewidth=2, alpha=0.6, zorder=10)
        
        # Add metrics box with professional styling
        textstr = '\n'.join([
            f'$R^2$ = {r2:.3f}',
            f'RMSE = {rmse:.3f}',
            f'$\\rho$ = {rho:.3f}',
            f'$n$ = {n:,}'
        ])
        
        ax.text(0.05, 0.95, textstr,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                         alpha=0.95, edgecolor='black', linewidth=1.2))
        
        # Set labels - only left plot gets y-label
        ax.set_xlabel('Experimental $\\mathbf{log_{10}}$($\\mathbf{k_{cat}}$)', fontsize=13, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Predicted $\\mathbf{log_{10}}$($\\mathbf{k_{cat}}$)', fontsize=13, fontweight='bold')
        else:
            ax.tick_params(labelleft=False)
        
        # Title with panel label
        panel_label = chr(97 + idx)  # a, b, c, ...
        ax.set_title(f'{panel_label}) {titles[split_mode]} - Hexbin Density    ', 
                    fontsize=14, fontweight='bold', pad=10)
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)
        
        # Subtle grid
        ax.grid(True, alpha=0.2, linewidth=0.5, linestyle='-', color='gray')
        ax.set_axisbelow(True)
    
    # Add single colorbar for scatter plots
    if hexbin_objects:
        # Use the hexbin with the maximum count for colorbar scale
        vmax = max(hb.get_array().max() for hb in hexbin_objects)
        for hb in hexbin_objects:
            hb.set_clim(vmin=1, vmax=vmax)
        
        cbar = plt.colorbar(hexbin_objects[0], cax=cax)
        cbar.set_label('Count', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
    
    # ========== Bottom Row: Bar/Line Plots ==========
    bin_edges = np.arange(-7, 8, 1)  # Bins from -6 to 6
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    for idx, (split_mode, data) in enumerate(data_dict.items()):
        ax1 = bar_axes[idx]
        df = data['test_df']
        train_data = data['train_data']
        
        if df.empty or not train_data:
            ax1.text(0.5, 0.5, "No data", ha="center", va="center", 
                    transform=ax1.transAxes, fontsize=14)
            continue
        
        # Aggregate training data across all folds
        all_train_y = np.concatenate(train_data)
        
        # Count training samples per bin
        train_counts, _ = np.histogram(all_train_y, bins=bin_edges)
        
        # Calculate test error (RMSE) per bin
        y_true = df["y_true"].values
        y_pred = df["y_pred"].values
        
        bin_errors = []
        bin_sample_counts = []
        for i in range(len(bin_edges) - 1):
            mask = (y_true >= bin_edges[i]) & (y_true < bin_edges[i + 1])
            if mask.sum() > 0:
                error = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
                bin_errors.append(error)
                bin_sample_counts.append(mask.sum())
            else:
                bin_errors.append(np.nan)
                bin_sample_counts.append(0)
        
        bin_errors = np.array(bin_errors)
        
        # Plot bars for training counts
        bars = ax1.bar(bin_centers, train_counts, width=0.8, 
                      alpha=0.6, color='steelblue', 
                      label='Training samples', edgecolor='black', linewidth=1)
        
        ax1.set_xlabel('$\\mathbf{log_{10}}$($\\mathbf{k_{cat}}$) bin', fontsize=13, fontweight='bold')
        
        # Only show y-label and ticks on the left plot (idx == 0)
        if idx == 0:
            ax1.set_ylabel('Training sample count', fontsize=12, fontweight='bold', color='steelblue')
            ax1.tick_params(axis='y', labelcolor='steelblue')
        else:
            ax1.tick_params(axis='y', labelcolor='steelblue', labelleft=False)
        
        # Create second y-axis for error
        ax2 = ax1.twinx()
        
        # Plot line for test error
        valid_mask = ~np.isnan(bin_errors)
        line = ax2.plot(bin_centers[valid_mask], bin_errors[valid_mask], 
                       color='crimson', marker='o', markersize=8, 
                       linewidth=2.5, label='Test RMSE', zorder=10)
        
        # Only show y-label and ticks on the right plot (idx == 1)
        if idx == 1:
            ax2.set_ylabel('Test RMSE', fontsize=12, fontweight='bold', color='crimson')
            ax2.tick_params(axis='y', labelcolor='crimson')
        else:
            ax2.tick_params(axis='y', labelcolor='crimson', labelright=False)
        
        # Title with panel label
        panel_label = chr(97 + idx + 2)  # c, d, e, ...
        ax1.set_title(f'{panel_label}) {titles[split_mode]} - Sample Distribution vs Error', 
                     fontsize=14, fontweight='bold', pad=10)
        
        # Set x-axis with proper bin labels (lower edge of each bin)
        ax1.set_xticks(bin_centers)
        bin_labels = [f'{int(edge)}' for edge in bin_edges[:-1]]
        ax1.set_xticklabels(bin_labels)
        ax1.set_xlim(bin_edges[0] - 0.5, bin_edges[-1] + 0.5)
        
        # Add grid
        ax1.grid(True, alpha=0.2, linewidth=0.5, linestyle='-', color='gray', axis='x')
        ax1.set_axisbelow(True)
        
        # Adjust spines
        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
    
    # Save with high quality
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white', 
                edgecolor='none', pad_inches=0.1)
    print(f"✓ Saved publication-quality plot to {output_path}")



# ═══════════════════════════ Main ═══════════════════════════════════
def main():
    """Main execution function."""
    print(f"Loading EITLEM results for {MODEL_CONFIG}...")
    data = load_eitlem_results(MODEL_CONFIG)
    
    print(f"  - CV (kfold): {len(data['kfold']['test_df'])} test samples, "
          f"{len(data['kfold']['train_data'])} folds")
    print(f"  - SE-CV (groupkfold): {len(data['groupkfold']['test_df'])} test samples, "
          f"{len(data['groupkfold']['train_data'])} folds")
    
    print("Creating plot...")
    plot_cv_comparison(data, OUTPUT_PATH)
    print("Done!")


if __name__ == "__main__":
    main()
