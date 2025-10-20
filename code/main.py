#!/usr/bin/env python3
"""
KinForm / UniKP – single-pass training & prediction script.

Usage
-----
Run from the code/ directory:

# TRAIN on all data (default)
python main.py --mode train --task kcat \
               --model_config KinForm-L

# TRAIN with cross-validation (5-fold KFold + GroupKFold)
python main.py --mode train --task kcat \
               --model_config KinForm-L \
               --train_test_split 0.8

# PREDICT
python main.py --mode predict --task kcat \
               --model_config KinForm-L \
               --save_results ./predictions/kcat_L.csv

Note: All paths are relative to the repository root and will work on any machine.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score, mean_squared_error
# ──────────────────────────── local imports ───────────────────────── #
from config import CONFIG_H, CONFIG_L, CONFIG_UniKP
from smiles_embeddings.smiles_transformer.build_vocab import WordVocab 
from utils.smiles_features import smiles_to_vec
from utils.sequence_features import sequences_to_feature_blocks
from utils.pca import make_design_matrices
from model_training import train_model

# Global paths - relative to repository root
# This script is in code/, so go up one level to get to repo root
ROOT = Path(__file__).resolve().parent.parent
DATA_KCAT = ROOT / "data/EITLEM_data/KCAT/kcat_data.json"
DATA_KM   = ROOT / "data/KM_data_raw.json"
SEQ_LOOKUP   = ROOT / "results/sequence_id_to_sequence.pkl"
BS_PRED_DIRS = [
    ROOT / "results/binding_sites/prediction.tsv"
] + [
    ROOT / f"results/binding_sites/prediction_{i}.tsv"
    for i in range(2, 8)
]

# ------------------------------------------------------------------- #
CONFIG_MAP = {
    "KinForm-H": CONFIG_H,
    "KinForm-L": CONFIG_L,
    "UniKP":     CONFIG_UniKP,
}


# ═════════════════════════ data loading ════════════════════════════ #
def load_kcat() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sequences, smiles and log10(kcat) as numpy arrays."""
    print("Loading kcat data...")
    with DATA_KCAT.open() as fp:
        raw = json.load(fp)
    valid = [(r["sequence"], r["smiles"], float(r["value"]))
            for r in raw if len(r["sequence"]) <= 1499 and float(r["value"]) > 0]
    seqs, smis, y = zip(*valid)
    y = np.array([math.log(v, 10) for v in y], dtype=np.float32)
    return np.asarray(seqs), np.asarray(smis), y


def load_km() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sequences, smiles and log10(KM) as numpy arrays."""
    with DATA_KM.open() as fp:
        raw = json.load(fp)
    good = [(r["Sequence"], r["smiles"], float(r["log10_KM"]))
            for r in raw if len(r["Sequence"]) <= 1499 and "." not in r["smiles"]]
    seqs, smis, y = zip(*good)
    return np.asarray(seqs), np.asarray(smis), np.asarray(y, dtype=np.float32)


def get_dataset(task: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if task.lower() == "kcat":
        return load_kcat()
    if task.lower() == "km":
        return load_km()
    raise ValueError(f"Unknown task: {task}")


# ═════════════ feature construction (+ optional PCA) ═══════════════ #
def build_design_matrix(
    seqs: np.ndarray,
    smis: np.ndarray,
    cfg: Dict,
    task: str = "kcat",
) -> np.ndarray:
    """
    Convert sequences & SMILES to the final feature matrix.
    Uses make_design_matrices which handles PCA internally.
    """
    # Binding-site predictions
    bs_df = pd.concat([pd.read_csv(p, sep="\t") for p in BS_PRED_DIRS], ignore_index=True)

    # map sequence → id (for GroupKFold compatibility, but we only need ids here)
    seq_lookup = pd.read_pickle(SEQ_LOOKUP)
    seq_to_id = {v: k for k, v in seq_lookup.items()}

    blocks_all, block_names = sequences_to_feature_blocks(
        sequence_list=seqs,
        binding_site_df=bs_df,
        cat_sites_df=None,
        ec_num_df=None,
        seq_to_id=seq_to_id,
        use_ec_logits=False,
        use_esmc=cfg["use_esmc"],
        use_esm2=cfg["use_esm2"],
        use_t5=cfg["use_t5"],
        t5_last_layer=cfg.get("t5_last_layer", -1),
        prot_rep_mode=cfg["prot_rep_mode"],
        task=task,
    )

    smiles_vec = smiles_to_vec(smis, method="smiles_transformer")

    # Use all indices for training
    all_idx = np.arange(len(seqs))
    X, _ = make_design_matrices(all_idx, all_idx, blocks_all, block_names, cfg, smiles_vec)
    
    return X


# ═════════════════════════ main routine ════════════════════════════ #
def train(task: str, cfg_name: str, model_dir: Path, train_test_split: float = 1.0) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG_MAP[cfg_name]

    seqs, smis, y = get_dataset(task)
    print(f"✓ Loaded {len(seqs)} {task} samples with sequences and SMILES.")
    
    # If train_test_split < 1.0, perform cross-validation
    if train_test_split < 1.0:
        print(f"\n{'='*70}")
        print(f"Running cross-validation with {int(train_test_split*100)}% train split")
        print(f"{'='*70}\n")
        
        # Get sequence groups for GroupKFold
        seq_lookup = pd.read_pickle(SEQ_LOOKUP)
        seq_to_id = {v: k for k, v in seq_lookup.items()}
        groups = [seq_to_id[s] for s in seqs]
        
        # Binding-site predictions (load once)
        bs_df = pd.concat([pd.read_csv(p, sep="\t") for p in BS_PRED_DIRS], ignore_index=True)
        
        # Build feature blocks once
        blocks_all, block_names = sequences_to_feature_blocks(
            sequence_list=seqs,
            binding_site_df=bs_df,
            cat_sites_df=None,
            ec_num_df=None,
            seq_to_id=seq_to_id,
            use_ec_logits=False,
            use_esmc=cfg["use_esmc"],
            use_esm2=cfg["use_esm2"],
            use_t5=cfg["use_t5"],
            t5_last_layer=cfg.get("t5_last_layer", -1),
            prot_rep_mode=cfg["prot_rep_mode"],
            task=task,
        )
        
        smiles_vec = smiles_to_vec(smis, method="smiles_transformer")
        
        # Run both KFold and GroupKFold
        for split_mode in ["kfold", "groupkfold"]:
            print(f"\n{'-'*70}")
            print(f"Running {split_mode.upper()}")
            print(f"{'-'*70}")
            
            if split_mode == "kfold":
                cv = KFold(n_splits=5, shuffle=True, random_state=42).split(seqs)
            else:
                cv = GroupKFold(n_splits=5).split(seqs, groups=groups)
            
            fold_results: List[Dict] = []
            
            for fold_no, (tr_idx, te_idx) in enumerate(cv, 1):
                tr_idx = np.asarray(tr_idx, int)
                te_idx = np.asarray(te_idx, int)
                
                # Build design matrices
                X_tr, X_te = make_design_matrices(
                    tr_idx, te_idx, blocks_all, block_names, cfg, smiles_vec
                )
                y_tr, y_te = y[tr_idx], y[te_idx]
                
                # Train model
                et_params = cfg.get("et_params", None)
                model, y_pred, metrics = train_model(
                    X_tr, y_tr, X_te, y_te, fold=fold_no, et_params=et_params
                )
                
                # Save model
                fold_model_dir = model_dir / split_mode / f"fold{fold_no}"
                fold_model_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, fold_model_dir / "model.joblib")
                
                fold_results.append({
                    "fold": fold_no,
                    "r2": metrics["r2"],
                    "mse": metrics["mse"],
                    "rmse": metrics["rmse"],
                })
                
                print(f"  Fold {fold_no}: R²={metrics['r2']:.4f}, "
                      f"MSE={metrics['mse']:.4f}, RMSE={metrics['rmse']:.4f}")
            
            # Print summary statistics
            r2_scores = [r["r2"] for r in fold_results]
            mse_scores = [r["mse"] for r in fold_results]
            rmse_scores = [r["rmse"] for r in fold_results]
            
            print(f"\n{split_mode.upper()} Summary:")
            print(f"  R²   : {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
            print(f"  MSE  : {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
            print(f"  RMSE : {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
        
        print(f"\n{'='*70}")
        print(f"✓ Cross-validation complete. Models saved to {model_dir}")
        print(f"{'='*70}\n")
        
    else:
        # Original behavior: train on all data
        X = build_design_matrix(seqs, smis, cfg, task=task)
        print(f"✓ Built design matrix with shape {X.shape}.")
        
        if cfg_name == "KinForm-L" and task == "kcat":
            from utils.oversampling import (
                oversample_similarity_balanced_indices,
                oversample_kcat_balanced_indices,
            )
            print("↪ Performing similarity-based oversampling...")
            indices = np.arange(len(seqs))
            indices = oversample_similarity_balanced_indices(indices, seqs)
            print(f"  ↪ After similarity oversampling: {len(indices)} samples")
            indices = oversample_kcat_balanced_indices(indices, y)
            print(f"  ↪ After kcat oversampling: {len(indices)} samples")
            X = X[indices]
            y = y[indices]
        
        model, _, metrics = train_model(X, y, X, y, fold=0)
        print(f"✓ Training finished – R² on full data: {metrics['r2']:.3f}")

        joblib.dump(model, model_dir / "model.joblib")
        print(f"✓ Model saved to {model_dir}")


def predict(task: str, cfg_name: str, model_dir: Path, csv_out: Path) -> None:
    model = joblib.load(model_dir / "model.joblib")

    seqs, smis, y_true = get_dataset(task)
    cfg = CONFIG_MAP[cfg_name]

    X = build_design_matrix(seqs, smis, cfg, task=task)
    y_pred = model.predict(X)

    out = pd.DataFrame({
        "sequence": seqs,
        "smiles":   smis,
        "y_true":   y_true,
        "y_pred":   y_pred,
    })
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_out, index=False)
    print(f"✓ Predictions saved to {csv_out}")


# ══════════════════════════ CLI parser ═════════════════════════════ #
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Single-pass training / inference for KinForm & UniKP.")
    p.add_argument("--mode", required=True, choices=["train", "predict"],
                help="'train' – fit model on all data; 'predict' – run inference with a saved model")
    p.add_argument("--task", required=True, choices=["kcat", "KM"],
                help="What to train/predict on (kcat or KM)")
    p.add_argument("--model_config", required=True, choices=["KinForm-H", "KinForm-L", "UniKP"],
                help="Which model configuration to use")
    p.add_argument("--save_results", type=Path,
                help="CSV path for predictions (required in predict mode)")
    p.add_argument("--train_test_split", type=float, default=1.0,
                help="Proportion of data to use for training (default: 1.0 = all data). "
                     "If < 1.0, performs 5-fold KFold and GroupKFold cross-validation.")

    args = p.parse_args()
    model_dir = Path(f"./models/{args.task}_{args.model_config}")

    if args.mode == "train":
        if args.train_test_split <= 0.0 or args.train_test_split > 1.0:
            p.error("--train_test_split must be in range (0.0, 1.0]")
        train(args.task, args.model_config, model_dir, args.train_test_split)
    else:  # predict
        if args.save_results is None:
            p.error("--save_results is required in predict mode")
        predict(args.task, args.model_config, model_dir, args.save_results)
