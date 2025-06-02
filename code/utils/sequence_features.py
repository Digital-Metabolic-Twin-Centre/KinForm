
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple


def _weighted_mean(arr: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Length‑L weights → weighted mean over axis‑0."""
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    return (arr * w[:, None]).sum(axis=0)

def _load_residue_embeddings(seq_id: str,
                             *,
                             use_t5: bool,
                             use_esmc: bool,
                             use_esm2: bool,
                             t5_last_layer: bool = False,
                             task: str,
                             ) -> np.ndarray:
    base_dir = Path("/home/msp/saleh/KinForm/results/embeddings")

    if task == "kcat":
        t5_dir = base_dir / "prot_t5_layer_17" if not t5_last_layer else base_dir / "prot_t5_res"
        esmc_layer_dir = base_dir / "esmc_layer_24"
        esm2_dir = base_dir / "esm2_layer_26"
    elif task == "KM":
        t5_dir = base_dir / "prot_t5_layer_19" if not t5_last_layer else base_dir / "prot_t5_res"
        esmc_layer_dir = base_dir / "esmc_layer_32"
        esm2_dir = base_dir / "esm2_layer_29"
    else:
        raise ValueError(f"Unsupported task: {task}")

    emb_list: List[np.ndarray] = []

    if use_t5:
        emb_list.append(np.load(t5_dir / f"{seq_id}.npy"))
    if use_esmc:
        emb = np.load(esmc_layer_dir / f"{seq_id}.npy", allow_pickle=True)
        emb_list.append(emb)
    if use_esm2:
        emb_list.append(np.load(esm2_dir / f"{seq_id}.npy"))

    if not emb_list:
        raise ValueError("No embedding selected for loading.")
    return np.concatenate(emb_list, axis=1) if len(emb_list) > 1 else emb_list[0]



# ---------- main extraction routine -----------------------------------------

def sequences_to_features(sequence_list: List[str],
                          binding_site_df: pd.DataFrame,
                          seq_to_id: Dict[str, str],
                          *,
                          use_t5: bool,
                          use_esmc: bool,
                          use_esm2: bool,
                          t5_last_layer: bool = False,
                          prot_rep_mode: str = "global", 
                          task: str = "kcat",  # or "KM"
                          ) -> np.ndarray:
    features: List[np.ndarray] = []

    for seq in sequence_list:
        seq_id = seq_to_id.get(seq)
        if seq_id is None:
            raise KeyError(f"Sequence not found in lookup: {seq[:20]}...")
        resid_emb = _load_residue_embeddings(seq_id,
                                             use_t5=use_t5,
                                             use_esmc=use_esmc,
                                             use_esm2=use_esm2,
                                            task=task,
                                            t5_last_layer=t5_last_layer)
        # ---------- pooling --------------------------------------------------
        if prot_rep_mode in {"binding", "both"}:
            bs_row = binding_site_df.loc[binding_site_df['PDB'] == seq_id, 'Pred_BS_Scores']
            if bs_row.empty:
                raise ValueError(f"No binding-site scores for {seq_id}")
            weights_full = np.fromiter(map(float, bs_row.iloc[0].split(',')), dtype=float)
            weights = weights_full
            assert len(weights) == len(resid_emb), f"Length mismatch: {len(weights)} != {len(resid_emb)}"

        if prot_rep_mode == "global":
            pooled = resid_emb.mean(axis=0)
        elif prot_rep_mode == "binding":
            pooled = _weighted_mean(resid_emb, weights)
        elif prot_rep_mode == "both":
            binding = _weighted_mean(resid_emb, weights)
            global_mean = resid_emb.mean(axis=0)
            pooled = np.concatenate([binding, global_mean])
        else:
            raise ValueError(f"Unknown prot_rep_mode: {prot_rep_mode}")


        features.append(pooled)

    return np.vstack(features)



def _load_single_embedding(seq_id: str, model: str, task: str, t5_last_layer: bool = False) -> np.ndarray:
    base_dir = Path("/home/msp/saleh/KinForm/results/embeddings")
    if task == "kcat":
        layer_map = {
            "t5": base_dir / "prot_t5_layer_17" if not t5_last_layer else base_dir / "prot_t5_res",
            "esm2": base_dir / "esm2_layer_26",
            "esmc": base_dir / "esmc_layer_24"
        }
    elif task == "KM":
        layer_map = {
            "t5": base_dir / "prot_t5_layer_19" if not t5_last_layer else base_dir / "prot_t5_res",
            "esm2": base_dir / "esm2_layer_29",
            "esmc": base_dir / "esmc_layer_32"
        }
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    emb_path = layer_map[model] / f"{seq_id}.npy"
    if model == "esmc":
        return np.load(emb_path, allow_pickle=True)
    return np.load(emb_path)


def sequences_to_feature_blocks(
    sequence_list: List[str],
    binding_site_df: pd.DataFrame,
    seq_to_id: Dict[str, str],
    *,
    use_t5: bool,
    use_esmc: bool,
    use_esm2: bool,
    t5_last_layer: bool = False,
    prot_rep_mode: str = "both",
    task: str = "kcat",  # or "KM"
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Extracts structured blocks of protein sequence features.
    Returns list of [N, D_block] arrays and corresponding block names.
    """
    block_dict = {}  # block name -> list of pooled vectors

    for seq in sequence_list:
        seq_id = seq_to_id[seq]

        if prot_rep_mode in {"binding", "both"}:
            bs_row = binding_site_df.loc[binding_site_df['PDB'] == seq_id, 'Pred_BS_Scores']
            if bs_row.empty:
                raise ValueError(f"No binding-site scores for {seq_id}")
            weights_full = np.fromiter(map(float, bs_row.iloc[0].split(',')), dtype=float)
            weights = weights_full

        def pool(rep: np.ndarray, kind: str) -> np.ndarray:
            if kind == "global":
                return rep.mean(axis=0)
            elif kind == "binding":
                return (rep * (weights / (weights.sum() + 1e-6))[:, None]).sum(axis=0)
            else:
                raise ValueError(f"Invalid pooling kind: {kind}")

        for rep_type in ("binding", "global") if prot_rep_mode == "both" else (prot_rep_mode,):
            if use_t5:
                rep = _load_single_embedding(seq_id, "t5",task=task, t5_last_layer=t5_last_layer)
                block_dict.setdefault(f"T5_{rep_type}", []).append(pool(rep, rep_type))
            if use_esmc:
                rep = _load_single_embedding(seq_id, "esmc", task=task)
                block_dict.setdefault(f"ESMC_{rep_type}", []).append(pool(rep, rep_type))
            if use_esm2:
                rep = _load_single_embedding(seq_id, "esm2", task=task)
                block_dict.setdefault(f"ESM2_{rep_type}", []).append(pool(rep, rep_type))

    ordered_keys = []
    for rep_type in ("binding", "global"):
        for model in ["T5", "ESMC", "ESM2"]:
            key = f"{model}_{rep_type}"
            if key in block_dict:
                ordered_keys.append(key)

    blocks = [np.vstack(block_dict[k]) for k in ordered_keys]
    names = ordered_keys
    return blocks, names
