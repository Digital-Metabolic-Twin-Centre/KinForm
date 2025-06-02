#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract ProtT5‐XL UniRef50 embeddings.

Three operating modes (mutually exclusive):
    1. --setting mean
       → compute & save the per-sequence **mean vector of the last layer**.
    2. --setting residue [--layer N]
       → compute & save the **per-residue matrix** from either the last
         layer (default) or the user–specified layer *N* (0-based, 0-23).
    3. --all_layers
       → compute & save the **mean vector of every encoder layer**;
         result shape: [24, hidden_size].

Embeddings are written under:
    KinForm/results/embeddings/
        prot_t5/                     (mean vectors, last layer)
        prot_t5_res/                 (per-residue, layer n)
        prot_t5_layer_{n}/           (alias for ^)
        prot_t5_all_layers/          (24×mean vectors)

If all required files already exist the model is *not* loaded.
"""

import gc
import json
import os
import pickle
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer


# --------------------------------------------------------------------------- #
#                              EMBEDDING BACK-END                             #
# --------------------------------------------------------------------------- #
def get_prot_t5_embeddings(
    seq_dict: Dict[str, str],
    *,
    batch_size = 2,
    setting  = "mean",              # 'mean' | 'residue'
    all_layers = False,
    layer = None,           # 0-23 for residue setting
    only_save = False,
    id_to_seq = None):

    # ----------------------- sanity checks -------------------------------- #
    assert setting in {"mean", "residue"}, f"Invalid setting: {setting}"
    if all_layers:
        assert layer is None, "--layer is invalid when --all_layers is set"

    # ------------------------- path handling ------------------------------ #
    base_path = os.getcwd().split("KinForm")[0]
    idseq_path = os.path.join(base_path,
                              "KinForm/results/sequence_id_to_sequence.pkl")
    id_to_seq = pickle.load(open(idseq_path, "rb")) if id_to_seq is None else id_to_seq
    assert all(k in id_to_seq and id_to_seq[k] == v for k, v in seq_dict.items()), (
        "Sequence mismatch between provided seq_dict and id_to_seq"
    )

    paths = {
        "mean": os.path.join(base_path, "KinForm/results/embeddings/prot_t5"),
        "residue": os.path.join(base_path, "KinForm/results/embeddings/prot_t5_res"),
        "all_layers": os.path.join(base_path, "KinForm/results/embeddings/prot_t5_all_layers"),
    }
    # layer-specific directory
    if layer is not None:
        paths[f"layer_{layer}"] = os.path.join(
            base_path, f"KinForm/results/embeddings/prot_t5_layer_{layer}"
        )
    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    # ------------------------ skip existing files ------------------------- #
    if all_layers:
        key_to_exist = {
            k: os.path.exists(os.path.join(paths["all_layers"], f"{k}.npy"))
            for k in seq_dict
        }
    elif setting == "residue" and layer is not None:
        key_to_exist = {
            k: os.path.exists(os.path.join(paths[f"layer_{layer}"], f"{k}.npy"))
            for k in seq_dict
        }
    else:
        key_to_exist = {
            k: os.path.exists(os.path.join(paths[setting], f"{k}.npy"))
            for k in seq_dict
        }

    if all(key_to_exist.values()):
        print("All required ProtT5 embeddings already on disk — skipping model load.")
        if only_save:
            return None
        return _load_existing_embeddings(seq_dict, paths, setting, all_layers, layer)

    # --------------------------- model load ------------------------------- #
    print("Loading ProtT5-XL UniRef50 ...")
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained(
        "Rostlab/prot_t5_xl_uniref50", output_hidden_states=True
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) // 1e6:,} M")

    # ----------------------- batching & encoding -------------------------- #
    missing_keys = [k for k, ok in key_to_exist.items() if not ok]
    print(f"Generating embeddings for {len(missing_keys)} new sequences")
    batches = [missing_keys[i:i + batch_size] for i in range(0, len(missing_keys), batch_size)]

    for batch_keys in tqdm(batches, desc="ProtT5 batches"):
        batch_seqs = [seq_dict[k] for k in batch_keys]
        # ProtT5 expects amino acids separated by space & ambiguous tokens as 'X'
        batch_strs = [
            " ".join(list(re.sub(r"[UZOB]", "X", s))) for s in batch_seqs
        ]
        token_data = tokenizer(
            batch_strs,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**token_data)
        hidden_states: Tuple[torch.Tensor, ...] = outputs.hidden_states  # len=25, 0=embeddings

        # lengths incl. <eos>; exclude from downstream
        seq_lens = (token_data["attention_mask"] == 1).sum(dim=1) - 1  # [batch]

        for idx, key in enumerate(batch_keys):
            L = seq_lens[idx].item()
            if all_layers:
                layer_means: List[np.ndarray] = []
                for hs in hidden_states[1:]:  # skip embedding layer
                    vec = hs[idx, :L].mean(dim=0).cpu().numpy()
                    layer_means.append(vec)
                stack = np.stack(layer_means)             # [24, hidden]
                np.save(os.path.join(paths["all_layers"], f"{key}.npy"), stack)

            elif setting == "mean":
                last_hidden = hidden_states[-1][idx, :L]
                mean_vec = last_hidden.mean(dim=0).cpu().numpy()
                np.save(os.path.join(paths["mean"], f"{key}.npy"), mean_vec)

            elif setting == "residue":
                if layer is None:
                    layer_tensor = hidden_states[-1]           # last layer
                else:
                    layer_tensor = hidden_states[layer + 1]    # skip embed layer
                
                residue_emb = layer_tensor[idx, :L].cpu().numpy()  # [L, H]
                target_dir = paths["residue"] if layer is None else paths[f"layer_{layer}"]
                np.save(os.path.join(target_dir, f"{key}.npy"), residue_emb)

        # ------------------- memory hygiene per batch -------------------- #
        del token_data, outputs, hidden_states
        torch.cuda.empty_cache()
        gc.collect()

    # ---------------------- return (optional) ----------------------------- #
    if only_save:
        return None
    return _load_existing_embeddings(seq_dict, paths, setting, all_layers, layer)


# --------------------------------------------------------------------------- #
#                               HELPER ROUTINE                                #
# --------------------------------------------------------------------------- #
def _load_existing_embeddings(
    seq_dict: Dict[str, str],
    paths: Dict[str, str],
    setting: str,
    all_layers: bool,
    layer):
    """
    Load embeddings that are now guaranteed to exist.
    """
    if all_layers:
        return {
            k: np.load(os.path.join(paths["all_layers"], f"{k}.npy"))
            for k in seq_dict
        }
    elif setting == "mean":
        return {
            k: np.load(os.path.join(paths["mean"], f"{k}.npy"))
            for k in seq_dict
        }
    elif setting == "residue":
        dir_key = "residue" if layer is None else f"layer_{layer}"
        return {
            k: np.load(os.path.join(paths[dir_key], f"{k}.npy"))
            for k in seq_dict
        }
    else:  # should not occur
        raise RuntimeError("Unsupported combination for loading embeddings.")


# --------------------------------------------------------------------------- #
#                             SCRIPT ENTRY POINT                              #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    """
    This main block mirrors the structure of the example provided.
    Modify the variables `SETTING`, `ALL_LAYERS`, and `LAYER`
    to switch between the three operating modes described in the
    module docstring.
    """
    from pathlib import Path
    # ------------------------ user-editable options ---------------------- #
    BATCH_SIZE: int = 1

    # # Choose exactly ONE of the following:
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("layer", type=int, help="Which ProtT5 encoder layer to extract (0–23)")
    # args = parser.parse_args()

    # Embedding config
    SETTING = "residue"
    ALL_LAYERS = False
    LAYER = None

    # ------------------------------------------------------------------- #
    dlkcat_df_path = "/home/msp/saleh/KinForm/results/dlkcat_df.pkl"
    seq_id_to_seq_path = "/home/msp/saleh/KinForm/results/sequence_id_to_sequence.pkl"
    eitlem_csv_path = "/home/msp/saleh/KinForm/results/eitlemall_subset.csv"
    km_raw_json = "/home/msp/saleh/KinForm/data/KM_data_raw.json"

    # -------------------- build the {id: sequence} dict ----------------- #
    RAW_DLKCAT = Path("/home/msp/saleh/KinForm/data/dlkcat_raw.json")
    raw = [d for d in json.loads(RAW_DLKCAT.read_text())
           if len(d["Sequence"]) <= 1499 and float(d["Value"]) > 0 and "." not in d["Smiles"]]

    sequences = [d["Sequence"] for d in raw]

    id_to_seq: Dict[str, str] = pickle.load(open(seq_id_to_seq_path, "rb"))
    seq_to_id = {seq: sid for sid, seq in id_to_seq.items()}
    dlkcat_seq_ids = list(set([seq_to_id[seq] for seq in sequences]))

    eitlem_df = pd.read_csv(eitlem_csv_path)
    eitlem_seqs = set(eitlem_df["sequence"].unique())
    eitlem_ids = {seq_to_id[s] for s in eitlem_seqs if s in seq_to_id}

    with open(km_raw_json, "r") as fp:
        km_raw = json.load(fp)
    km_seqs = {d["Sequence"] for d in km_raw if len(d["Sequence"]) <= 1499 and "." not in d["smiles"]}
    km_ids = {seq_to_id[s] for s in km_seqs if s in seq_to_id}


    with open('/home/msp/saleh/KinForm/data/EITLEM_data/KM/km_data.json', 'r') as fp:
        raw = json.load(fp)
    sequences = [d["sequence"] for d in raw if len(d["sequence"]) <= 1499]
    sequences = list(set(sequences))
    eit_km_seq_ids = [seq_to_id[seq] for seq in sequences if seq in seq_to_id]

    sequences_df = pd.read_csv("/home/msp/saleh/KinForm/results/synthetic_data/filtered_dlkcat_sequences.csv")
    sequences_list = sequences_df["sequence"].tolist()
    syn_seq_ids = [seq_to_id[seq] for seq in sequences_list if seq in seq_to_id]

    seq_ids = dlkcat_seq_ids + list(eitlem_ids) + list(km_ids) + list(eit_km_seq_ids) + list(syn_seq_ids)
    seq_ids = list(set(seq_ids))
    seq_dict = {sid: id_to_seq[sid] for sid in seq_ids}

    # --------------------------- run embedding -------------------------- #
    embeddings = get_prot_t5_embeddings(
        seq_dict,
        batch_size=BATCH_SIZE,
        setting=SETTING,
        all_layers=ALL_LAYERS,
        layer=LAYER,
        only_save=True,          # change to True if you don't need the return value
        id_to_seq=id_to_seq,
    )

    print(f"Completed ProtT5 embedding extraction for {len(seq_dict)} sequences.")
