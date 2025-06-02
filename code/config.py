from pathlib import Path
CONDA_EXE   = Path("/home/msp/miniconda3/bin/conda")   # path to conda binary
MMSEQS_ENV  = "mmseqs"                                 # name of env containing mmseqs

DATA_DIR     = Path("/home/msp/saleh/KinForm")
RAW_DLKCAT     = DATA_DIR / "data/dlkcat_raw.json"
BS_PRED_DIRS = [
    DATA_DIR / "results/binding_sites/prediction.tsv"
] + [
    DATA_DIR / f"results/binding_sites/prediction_{i}.tsv"
    for i in range(2, 7)
]
SEQ_LOOKUP   = DATA_DIR / "results/sequence_id_to_sequence.pkl"

EMB_COMBOS = {
    "ESMC":               dict(use_t5=False, use_esmc=True,  use_esm2=False),
    "T5":                 dict(use_t5=True,  use_esmc=False, use_esm2=False),
    "ESM2":              dict(use_t5=False, use_esmc=False, use_esm2=True),
    "ESMC+T5":            dict(use_t5=True,  use_esmc=True,  use_esm2=False),
    "T5+ESM2":            dict(use_t5=True,  use_esmc=False, use_esm2=True),
    "ESMC+ESM2":         dict(use_t5=False, use_esmc=True,  use_esm2=True),
    "ESMC+T5+ESM2":      dict(use_t5=True,  use_esmc=True,  use_esm2=True),
}

REP_MODES = {
    "global":        dict(prot_rep_mode="global"),
    "binding":       dict(prot_rep_mode="binding"),
    "both":          dict(prot_rep_mode="both"),
}

CONFIGS = []
for emb_name, emb_flags in EMB_COMBOS.items():
    for rep_name, rep_flags in REP_MODES.items():
        name = f"{emb_name}|{rep_name}"
        CONFIGS.append(dict(name=name, **emb_flags, **rep_flags))

# config.py or main_pca.py (before main)

PCA_VALUES = [100, 200, 300, 400, 500, 750, 1000, 1750, None]  # None = no PCA

CONFIGS_PCA = []

for emb_name, emb_flags in {
    "ESMC+ESM2+T5": dict(use_esmc=True, use_esm2=True, use_t5=True, t5_last_layer=True),
    "ESMC+T5":    dict(use_esmc=True, use_esm2=False, use_t5=True,t5_last_layer=True),
    
}.items():
    for n_comps in PCA_VALUES:
        use_pca = n_comps is not None
        label = (
            f"{emb_name}|+PCA|k={n_comps}"
            if use_pca else
            f"{emb_name}|-PCA"
        )
        CONFIGS_PCA.append(dict(
            name=label,
            use_pca=use_pca,
            n_comps=n_comps,
            prot_rep_mode="both",
            **emb_flags
        ))

SMILES_REPS = ['smiles_transformer', 'MFP', 'UniMol', 'FARM', 'molformer',
        'TopologicalTorsion', 'MinHash', 'MACCS', 'AtomPair', 'Avalon']

CONFIGS_SMILES_KCAT = [
    dict(
        name=f"{method}|PCA (ESMC+ESM2+T5, k=300)",
        smiles_method=method,
        use_pca=True,
        n_comps=300,
        use_esmc=True, use_esm2=True, use_t5=True,
        prot_rep_mode="both",
        t5_last_layer=True,
    )
    for method in SMILES_REPS
] 
CONFIGS_SMILES_KM = [
    dict(
        name=f"{method}|ESMC(both)",
        smiles_method=method,
        use_pca=False,
        n_comps=None,
        use_esmc=True, use_esm2=False, use_t5=False,
        prot_rep_mode="both",
    )
    for method in SMILES_REPS
]

CONFIG_L = dict(
    name="KinForm-L",
    use_pca=True,
    n_comps=300,
    prot_rep_mode="both",
    use_esmc=True,
    use_esm2=True,
    use_t5=True,
    t5_last_layer=True,
)
CONFIG_H = dict(
    name="KinForm-H",
    use_pca=False,
    n_comps=None,
    prot_rep_mode="both",
    use_esmc=True,
    use_esm2=True,
    use_t5=True,
    t5_last_layer=True,
)
CONFIG_UniKP = dict(
    name="UniKP",
    use_pca=False,
    n_comps=None,
    prot_rep_mode="global",
    use_esmc=False,
    use_esm2=False,
    use_t5=True,
    t5_last_layer=True,
)