
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import List, Tuple

def split_blocks(names: List[str], blocks: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Splits feature blocks into (binding_blocks, global_blocks) using block names.
    """
    binding_blocks, global_blocks = [], []
    for name, b in zip(names, blocks):
        if "binding" in name:
            binding_blocks.append(b)
        elif "global" in name:
            global_blocks.append(b)
    return binding_blocks, global_blocks

def scale_and_reduce_blocks(
    blocks_train: List[np.ndarray],
    blocks_test: List[np.ndarray],
    block_names: List[str],
    n_comps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply RobustScaler + PCA per group (global/binding) if PCA is enabled.
    Returns transformed features: concat(PCA(global), PCA(binding))
    """
    def scale_blocks(train_blocks, test_blocks):
        train_scaled, test_scaled = [], []
        for b_tr, b_te in zip(train_blocks, test_blocks):
            scaler = RobustScaler().fit(b_tr)
            train_scaled.append(scaler.transform(b_tr))
            test_scaled.append(scaler.transform(b_te))
        return np.concatenate(train_scaled, axis=1), np.concatenate(test_scaled, axis=1)

    binding_train, global_train = split_blocks(block_names, blocks_train)
    binding_test,  global_test  = split_blocks(block_names, blocks_test)

    Xg_train, Xg_test = scale_blocks(global_train, global_test)
    Xb_train, Xb_test = scale_blocks(binding_train, binding_test)
    
    Xg_scaler = StandardScaler().fit(Xg_train)
    Xg_train = Xg_scaler.transform(Xg_train)
    Xg_test  = Xg_scaler.transform(Xg_test)

    Xb_scaler = StandardScaler().fit(Xb_train)
    Xb_train = Xb_scaler.transform(Xb_train)
    Xb_test  = Xb_scaler.transform(Xb_test)

    pca_g = PCA(n_components=n_comps, random_state=42).fit(Xg_train)
    pca_b = PCA(n_components=n_comps, random_state=42).fit(Xb_train)

    Xg_train = pca_g.transform(Xg_train)
    Xg_test  = pca_g.transform(Xg_test)
    Xb_train = pca_b.transform(Xb_train)
    Xb_test  = pca_b.transform(Xb_test)

    return np.concatenate([Xb_train, Xg_train], axis=1), np.concatenate([Xb_test, Xg_test], axis=1)


def make_design_matrices(tr, te, blocks_all, names, cfg, smiles_vec):
    b_tr = [b[tr] for b in blocks_all]
    b_te = [b[te] for b in blocks_all]
    if cfg["use_pca"]:
        seq_tr, seq_te = scale_and_reduce_blocks(b_tr, b_te, names, cfg["n_comps"])
    else:
        bl_tr, gl_tr = split_blocks(names, b_tr)
        bl_te, gl_te = split_blocks(names, b_te)
        seq_tr = np.concatenate(bl_tr + gl_tr, 1)
        seq_te = np.concatenate(bl_te + gl_te, 1)
    X_tr = np.concatenate([smiles_vec[tr], seq_tr], 1)
    X_te = np.concatenate([smiles_vec[te], seq_te], 1)
    return X_tr, X_te
