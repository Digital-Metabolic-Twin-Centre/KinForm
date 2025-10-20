
KinForm – Enzyme Kinetics Prediction
====================================
Code for the implementation and experiments of KinForm models for predicting enzyme–substrate kinetic parameters (kcat and KM) from protein sequence and SMILES.

If you just want to get predictions for your own data using our trained models, you can use the hosted web app (no setup required):
https://kineticxpredictor.humanmetabolism.org/

For details on the models and experiments, see the preprint:
https://arxiv.org/abs/2507.14639


Setup
-----
```bash
git clone https://github.com/Digital-Metabolic-Twin-Centre/KinForm.git
cd KinForm
pip install -r requirements.txt
```


Path A — Train and run with the Zenodo bundle
----------------------------------------------------------
Use this if you want to train/evaluate/predict with the same assets used in the paper.

1) Download the Zenodo bundle(s) and extract under `results/` (repo root)

- Trained models, binding-site predictions, mapping files, etc.: <placeholder-zenodo-link> (includes binding-site TSVs for transparency/inspection; not required to run)
- Precomputed protein embeddings: <placeholder-zenodo-link>

After extraction, your tree should include (examples):

- `results/sequence_id_to_sequence.pkl`
- `results/protein_embeddings/...` (e.g., `esm2_layer_26/`, `esmc_layer_32/`, `prot_t5_res/`)
- Optional (for inspection only): `results/binding_sites/prediction.tsv`, `prediction_2.tsv` … `prediction_7.tsv` (Pseq2Sites outputs for all proteins)

2) Train models (run commands from `code/`)

```bash
cd code

# kcat
python main.py --mode train --task kcat --model_config UniKP
python main.py --mode train --task kcat --model_config KinForm-L
python main.py --mode train --task kcat --model_config KinForm-H

# KM
python main.py --mode train --task KM --model_config UniKP
python main.py --mode train --task KM --model_config KinForm-L
python main.py --mode train --task KM --model_config KinForm-H

# Optional: 5-fold KFold + GroupKFold evaluation (--train_test_split < 1.0 triggers cross-validation; 1.0 trains on all data)
python main.py --mode train --task kcat --model_config KinForm-L --train_test_split 0.8
```

3) Predict (predictions are saved in original/non-log units)

```bash
# Default dataset
python main.py --mode predict --task kcat --model_config KinForm-L --save_results ../predictions/kcat_L.csv

# Custom JSON
python main.py --mode predict --task KM --model_config UniKP --save_results ../predictions/km_unikp.csv --data_path ../my_km.json
```

Custom JSON format

- For kcat: array of {"sequence": str, "smiles": str, "value": float} (value is raw kcat, NOT log)
- For KM:   array of {"Sequence": str, "smiles": str, "log10_KM": float}


Path B — New proteins or full regeneration
-----------------------------------------
Use this if you want to run on sequences not in the bundle, or rebuild all features locally.

About `results/sequence_id_to_sequence.pkl`

- This file maps a stable sequence ID (e.g., "Sequence 11894") to its amino-acid string.
- It is included in the Zenodo bundle.
- If you have new sequences, append entries to this mapping so that downstream scripts can locate embeddings by ID.

1) Add or verify your sequence IDs

- Place IDs (one per line) in `data/unique_seq_ids.txt`.
- Ensure each ID exists in `results/sequence_id_to_sequence.pkl`; if not, add it.

2) Generate protein embeddings (if not using the precomputed set)

```bash
cd code/protein_embeddings

# ESM2 layers 26/29 and ESMC layers 24/32
python prot_embeddings.py

# ProtT5 residue embeddings (layer 19 and last layer)
python t5_embeddings.py
```

Embeddings are saved under `results/embeddings/` (these are full per-residue embeddings for each protein).

3) Generate binding-site predictions (if not using the precomputed set)

Use Pseq2Sites (https://github.com/Blue1993/Pseq2Sites) and save TSV outputs under `results/binding_sites/`. You can use one TSV file or multiple (e.g., if running Pseq2Sites in batches).

TSV format:
- Column 1: `PDB` (sequence ID matching your `results/sequence_id_to_sequence.pkl` keys)
- Column 2: `Pred_BS_Scores` (string representation of a list with L values, where L = sequence length; the i-th value is the probability that residue i is in the binding site)

Example files: `prediction.tsv`, `prediction_2.tsv`, … `prediction_7.tsv`

4) Train and predict

- Use the same commands as in Path A.


