#!/usr/bin/env python3
"""
KinForm / UniKP – single-pass training & prediction script.

Usage
-----
# TRAIN
python main.py --mode train --task kcat \
               --model_dir ./models/kcat_KinForm-L \
               --model_config KinForm-L

# PREDICT
python main.py --mode predict --task kcat \
               --model_dir ./models/kcat_KinForm-L \
               --save_results ./predictions/kcat_L.csv
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple
import joblib
import numpy as np
import pandas as pd
# ──────────────────────────── local imports ───────────────────────── #
from config import CONFIG_H, CONFIG_L, CONFIG_UniKP
from smiles_embeddings.smiles_transformer.build_vocab import WordVocab 
from utils.smiles_features import smiles_to_vec
from utils.sequence_features import sequences_to_feature_blocks
from utils.pca import split_blocks
from model_training import train_model

# Global paths (reuse what is already in your repo)
ROOT = Path("/home/msp/saleh/KinForm")
DATA_KCAT = ROOT / "data/EITLEM_data/KCAT/kcat_data.json"
DATA_KM   = ROOT / "data/KM_data_raw.json"
SEQ_LOOKUP = ROOT / "data/SEQ_LOOKUP.pkl"         
BS_PRED_DIRS = ROOT.glob("binding_site_preds/*.tsv")

# ------------------------------------------------------------------- #
CONFIG_MAP = {
    "KinForm-H": CONFIG_H,
    "KinForm-L": CONFIG_L,
    "UniKP":     CONFIG_UniKP,
}


# ═════════════════════════ data loading ════════════════════════════ #
def load_kcat() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sequences, smiles and log10(kcat) as numpy arrays."""
    with DATA_KCAT.open() as fp:
        raw = json.load(fp)
    good = [(r["sequence"], r["smiles"], float(r["value"]))
            for r in raw if len(r["sequence"]) <= 1499 and float(r["value"]) > 0]
    seqs, smis, y = zip(*good)
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
    fit_pipeline: bool = True,
    pipeline: Dict | None = None,
    task: str = "kcat",
) -> Tuple[np.ndarray, Dict | None]:
    """
    Convert sequences & SMILES to the final feature matrix.
    If `fit_pipeline` is True and cfg uses PCA => fit scalers + PCA and return them.
    If False, the provided `pipeline` (dict) is applied instead.
    """
    # Binding-site predictions
    bs_df = pd.concat([pd.read_csv(p, sep="\t") for p in BS_PRED_DIRS], ignore_index=True)

    # map sequence → id (for GroupKFold compatibility, but we only need ids here)
    seq_lookup = pd.read_pickle(SEQ_LOOKUP)
    seq_to_id = {v: k for k, v in seq_lookup.items()}

    blocks_all, block_names = sequences_to_feature_blocks(
        seqs,
        bs_df,
        seq_to_id,
        use_esmc=cfg["use_esmc"],
        use_esm2=cfg["use_esm2"],
        use_t5=cfg["use_t5"],
        t5_last_layer=cfg.get("t5_last_layer", -1),
        prot_rep_mode=cfg["prot_rep_mode"],
        task=task,
    )

    smiles_vec = smiles_to_vec(smis, method="smiles_transformer")

    # ---- PCA branch ------------------------------------------------ #
    if cfg["use_pca"]:
        if fit_pipeline:
            # Fit scalers & PCA exactly as in pca.scale_and_reduce_blocks,
            # but keep the fitted objects so we can persist them.
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import RobustScaler, StandardScaler

            bind_blocks, glob_blocks = split_blocks(block_names, blocks_all)

            def scale_group(blocks):
                scaled, scalers = [], []
                for b in blocks:
                    sc = RobustScaler().fit(b)
                    scaled.append(sc.transform(b))
                    scalers.append(sc)
                return np.concatenate(scaled, axis=1), scalers

            Xg, scalers_g = scale_group(glob_blocks)
            Xb, scalers_b = scale_group(bind_blocks)

            sc_g = StandardScaler().fit(Xg)
            sc_b = StandardScaler().fit(Xb)
            Xg = sc_g.transform(Xg)
            Xb = sc_b.transform(Xb)

            pca_g = PCA(cfg["n_comps"], random_state=42).fit(Xg)
            pca_b = PCA(cfg["n_comps"], random_state=42).fit(Xb)
            X_seq = np.concatenate([pca_b.transform(Xb), pca_g.transform(Xg)], axis=1)

            pipeline_out = dict(
                scalers_g=scalers_g,
                scalers_b=scalers_b,
                sc_g=sc_g,
                sc_b=sc_b,
                pca_g=pca_g,
                pca_b=pca_b,
                block_names=block_names,
            )
        else:
            # ---- apply existing pipeline ----------------------------- #
            scalers_g = pipeline["scalers_g"]
            scalers_b = pipeline["scalers_b"]
            sc_g = pipeline["sc_g"]
            sc_b = pipeline["sc_b"]
            pca_g = pipeline["pca_g"]
            pca_b = pipeline["pca_b"]

            bind_blocks, glob_blocks = split_blocks(block_names, blocks_all)

            def apply_scale(blocks, rscalers):
                scaled = []
                for b, sc in zip(blocks, rscalers):
                    scaled.append(sc.transform(b))
                return np.concatenate(scaled, axis=1)

            Xg = sc_g.transform(apply_scale(glob_blocks, scalers_g))
            Xb = sc_b.transform(apply_scale(bind_blocks, scalers_b))
            X_seq = np.concatenate([pca_b.transform(Xb), pca_g.transform(Xg)], axis=1)
            pipeline_out = None  # nothing new to return
    else:
        # ---- simple concatenation ---------------------------------- #
        bind_blocks, glob_blocks = split_blocks(block_names, blocks_all)
        X_seq = np.concatenate(bind_blocks + glob_blocks, axis=1)
        pipeline_out = None

    # Final feature matrix
    X = np.concatenate([smiles_vec, X_seq], axis=1).astype(np.float32)
    return X, pipeline_out


# ═════════════════════════ main routine ════════════════════════════ #
def train(task: str, cfg_name: str, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG_MAP[cfg_name]

    seqs, smis, y = get_dataset(task)
    X, pipe = build_design_matrix(seqs, smis, cfg, fit_pipeline=True, pipeline=None, task=task)

    model, _, metrics = train_model(X, y, X, y, fold=0)
    print(f"✓ Training finished – R² on full data: {metrics['r2']:.3f}")

    joblib.dump(model, model_dir / "model.joblib")
    if pipe is not None:
        joblib.dump(pipe, model_dir / "pipeline.joblib")
    print(f"✓ Model saved to {model_dir}")


def predict(task: str, cfg_name: str, model_dir: Path, csv_out: Path) -> None:
    model = joblib.load(model_dir / "model.joblib")
    pipe = None
    pipe_file = model_dir / "pipeline.joblib"
    if pipe_file.exists():
        pipe = joblib.load(pipe_file)

    seqs, smis, y_true = get_dataset(task)
    cfg = CONFIG_MAP[cfg_name]

    X, _ = build_design_matrix(seqs, smis, cfg, fit_pipeline=False, pipeline=pipe, task=task)
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

    args = p.parse_args()
    model_dir = Path(f"./models/{args.task}_{args.model_config}")

    if args.mode == "train":
        train(args.task, args.model_config, model_dir)
    else:  # predict
        if args.save_results is None:
            p.error("--save_results is required in predict mode")
        predict(args.task, args.model_config, model_dir, args.save_results)
