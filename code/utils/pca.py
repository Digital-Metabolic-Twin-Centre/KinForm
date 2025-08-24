
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import List, Tuple

CANONICAL_ORDER = ("binding", "cat", "ec", "global")

def reorder_blocks(block_names: List[str],
                   blocks: List[np.ndarray]) -> List[np.ndarray]:
    """
    Return `blocks` reordered so that all blocks whose *name* contains
    'binding' come first (original internal order preserved),
    followed by 'cat', then 'ec', and finally 'global'.

    Parameters
    ----------
    block_names : List[str]
        Names returned by `sequences_to_feature_blocks`
        (e.g. ["T5_binding", "ESMC_cat", ...]).
    blocks : List[np.ndarray]
        Feature-block arrays in the same order as `block_names`.

    Returns
    -------
    List[np.ndarray]
        The reordered list of blocks.
    """
    ordered = []
    for tag in CANONICAL_ORDER:
        for nm, blk in zip(block_names, blocks):
            if tag in nm.lower():
                ordered.append(blk)
    return ordered

def group_blocks(names: List[str],
                 blocks: List[np.ndarray]):
    """
    Groups feature blocks by their semantic tag.

    Returns a dict whose keys are one (or more) of
        'binding', 'cat', 'ec', 'global'
    and whose values are lists of np.ndarrays in their original order.
    """
    groups = {}
    tag_map = {
        "binding": "binding",
        "cat":     "cat",
        "ec":      "ec",
        "global":  "global",
    }
    for name, block in zip(names, blocks):
        lname = name.lower()
        for tag, key in tag_map.items():
            if tag in lname:
                groups.setdefault(key, []).append(block)
                break
    return groups

def _robust_scale_per_block(train_blocks, test_blocks):
    tr_scaled, te_scaled = [], []
    for b_tr, b_te in zip(train_blocks, test_blocks):
        scaler = RobustScaler().fit(b_tr)
        tr_scaled.append(scaler.transform(b_tr))
        te_scaled.append(scaler.transform(b_te))
    return np.concatenate(tr_scaled, axis=1), np.concatenate(te_scaled, axis=1)

def scale_and_reduce_blocks(
    blocks_train: List[np.ndarray],
    blocks_test:  List[np.ndarray],
    block_names:  List[str],
    n_comps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For every representation family (`binding`, `cat`, `ec`, `global`)
    run:

        RobustScaler (per block) → concat →
        StandardScaler          → PCA(n_comps)

    The final design matrices are the left-to-right concatenation
    of the *transformed* groups in the fixed order
        binding → cat → ec → global
    (groups that are not present are silently skipped).
    """

    train_groups = group_blocks(block_names, blocks_train)
    test_groups  = group_blocks(block_names, blocks_test)

    X_tr_parts, X_te_parts = [], []

    for grp in CANONICAL_ORDER:
        if grp not in train_groups:
            continue

        # 1️⃣ Robust-scale each individual block
        X_tr_grp, X_te_grp = _robust_scale_per_block(
            train_groups[grp], test_groups[grp])

        # 2️⃣ Standard-scale the concatenated group
        std_scaler = StandardScaler().fit(X_tr_grp)
        X_tr_grp = std_scaler.transform(X_tr_grp)
        X_te_grp = std_scaler.transform(X_te_grp)

        # 3️⃣ PCA – keep at most the available column count
        n_keep = n_comps
        pca = PCA(n_components=n_keep, random_state=42).fit(X_tr_grp)
        X_tr_parts.append(pca.transform(X_tr_grp))
        X_te_parts.append(pca.transform(X_te_grp))

    return np.concatenate(X_tr_parts, axis=1), np.concatenate(X_te_parts, axis=1)


def make_design_matrices(tr, te, blocks_all, names, cfg, smiles_vec):
    b_tr = [b[tr] for b in blocks_all]
    b_te = [b[te] for b in blocks_all]
    if cfg["use_pca"]:
        seq_tr, seq_te = scale_and_reduce_blocks(b_tr, b_te, names, cfg["n_comps"])
    else:
        seq_tr = np.concatenate(reorder_blocks(names, b_tr), axis=1)
        seq_te = np.concatenate(reorder_blocks(names, b_te), axis=1)
    X_tr = np.concatenate([smiles_vec[tr], seq_tr], 1)
    X_te = np.concatenate([smiles_vec[te], seq_te], 1)
    return X_tr, X_te
