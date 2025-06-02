import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch
import pickle
import gc

def get_embeddings(seq_dict, batch_size=2, model=None, id_to_seq=None, setting='mean',all_layers=False, only_save=False, layer=None):
    assert setting in ['mean', 'residue', 'both'], f"Invalid setting: {setting}"
    """
    Get ESM-2 embeddings for sequences in seq_dict.
    input: seq_dict: dictionary with sequence IDs as keys and sequences as values
            batch_size: batch size for processing sequences
    output: dictionary with sequence IDs as keys and embeddings as values
    """
    accepted_models = ['esm2', 'esm1v', 'esmc']
    assert model in accepted_models, f"Invalid model: {model}. Accepted models: {accepted_models}"
    
    base_path = os.getcwd().split('KinForm')[0]
    idseq_path = base_path + 'KinForm/results/sequence_id_to_sequence.pkl'
    id_to_seq = pickle.load(open(idseq_path, 'rb')) if id_to_seq is None else id_to_seq
    assert all([key in id_to_seq.keys() and id_to_seq[key] == value for key, value in seq_dict.items()]), "Sequences must be in id_to_seq dictionary"
    print(f"Loaded {len(seq_dict)} sequences")

    paths = {
        "mean": os.path.join(base_path, f'KinForm/results/embeddings/{model}'),
        "residue": os.path.join(base_path, f'KinForm/results/embeddings/{model}_res'),
        "all_layers": os.path.join(base_path, f'KinForm/results/embeddings/{model}_all_layers'),
        f"layer_{layer}": os.path.join(base_path, f'KinForm/results/embeddings/{model}_layer_{layer}')
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    if all_layers:
        key_to_exist = {
            key: os.path.exists(os.path.join(paths["all_layers"], f"{key}.npy"))
            for key in seq_dict
        }
    elif layer is not None:
        key_to_exist = {
            key: os.path.exists(os.path.join(paths[f"layer_{layer}"], f"{key}.npy"))
            for key in seq_dict
        }
    else:
        key_to_exist = {
            key: all(os.path.exists(os.path.join(paths[t], f"{key}.npy")) for t in (["mean", "residue"] if setting == "both" else [setting]))
            for key in seq_dict
        }

    if all(key_to_exist.values()):
        print(f"Skipping {model} model loading, all embeddings already exist")
        if not only_save:
            if setting == "both":
                embeddings = {
                    key: {
                        "mean": np.load(os.path.join(paths["mean"], f"{key}.npy")),
                        "residue": np.load(os.path.join(paths["residue"], f"{key}.npy"))
                    }
                    for key in seq_dict
                }
            else:
                embeddings = {
                    key: np.load(os.path.join(paths[setting], f"{key}.npy"))
                    for key in seq_dict
                }
            return embeddings
    else:
        import esm
        torch.cuda.empty_cache()
        not_exist = [key for key, value in key_to_exist.items() if not value]
        print(f"Generating {model} embeddings for {len(not_exist)} sequences")
        print(F"Loading {model} model...")#
        if model == 'esm1v':
            assert not all_layers, "esm1v model does not support all_layers=True"
            model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
            seq_dict = {
                k: (v[:500] + v[-500:] if len(v) > 1022 else v)
                for k, v in seq_dict.items()
            }
        elif model == 'esm2':
            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            batch_converter = alphabet.get_batch_converter()
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            print(f"Using device: {device}")
            print(f"Loaded model with {sum(p.numel() for p in model.parameters()) // 1e6} million parameters")

            keys = list(seq_dict.keys())
            keys = [key for key in keys if not key_to_exist[key]]
            print(f"Skipping {len(seq_dict) - len(keys)} sequences with existing embeddings")

            batches = [keys[i:i + batch_size] for i in range(0, len(keys), batch_size)]

            for batch_keys in tqdm(batches):
                batch = {key: seq_dict[key] for key in batch_keys}
                data = [(label, seq) for label, seq in batch.items()]
                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.to(device)

                with torch.no_grad():
                    if all_layers:
                        n_layers = 34
                        results = model(batch_tokens, repr_layers=list(range(n_layers)), return_contacts=False)
                        token_reps = results["representations"]
                        for i, (label, seq) in enumerate(data):
                            all_layer_means = [
                                token_reps[layer][i, 1:len(seq) + 1].mean(dim=0).cpu().numpy()
                                for layer in range(n_layers)
                            ]
                            all_layer_means = np.stack(all_layer_means)  # [n_layers, H]
                            np.save(os.path.join(paths["all_layers"], f"{label}.npy"), all_layer_means)
                    else:
                        assert layer is not None, "Layer must be specified when all_layers is False"
                        results = model(batch_tokens, repr_layers=[layer], return_contacts=False)
                        token_rep = results["representations"][layer]
                        for i, (label, seq) in enumerate(data):
                            res_emb = token_rep[i, 1:len(seq) + 1].cpu().numpy()
                            if setting in ['residue', 'both']:
                                np.save(os.path.join(paths[f'layer_{layer}'], f'{label}.npy'), res_emb)
                            if setting in ['mean', 'both']:
                                mean_emb = res_emb.mean(0)
                                np.save(os.path.join(paths['mean'], f'{label}.npy'), mean_emb)

                del data, batch_tokens, results
                gc.collect()
                torch.cuda.empty_cache()

            if not only_save:
                if all_layers:
                    return {
                        key: np.load(os.path.join(paths["all_layers"], f"{key}.npy"))
                        for key in seq_dict
                    }
                if setting == "both":
                    return {
                        key: {
                            "mean": np.load(os.path.join(paths["mean"], f"{key}.npy")),
                            "residue": np.load(os.path.join(paths[f"layer_{layer}"], f"{key}.npy"))
                        }
                        for key in seq_dict
                    }
                elif setting == "residue":
                    return {
                        key: np.load(os.path.join(paths[f"layer_{layer}"], f"{key}.npy"))
                        for key in seq_dict
                    }
                else:  # setting == "mean"
                    return {
                        key: np.load(os.path.join(paths["mean"], f"{key}.npy"))
                        for key in seq_dict
                    }
            else:
                return None


        elif model == 'esmc':
            from esm_sdk.models.esmc import ESMC
            from esm_sdk.sdk.api import ESMProtein, LogitsConfig

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = ESMC.from_pretrained("esmc_600m").to(device)
            model.eval()
            config = LogitsConfig(sequence=True, return_hidden_states=True, return_embeddings=True)

            if all_layers:
                assert layer is None, "Layer argument is not used when all_layers=True"
            else:
                assert layer is not None, "Layer argument is required when all_layers=False"
            keys = list(seq_dict.keys())
            keys = [key for key in keys if not key_to_exist[key]]
            for key in tqdm(keys):  # keys already defined as sequences needing embedding
                sequence = seq_dict[key]
                protein = ESMProtein(sequence=sequence)
                # try:
                tensor = model.encode(protein)
                logits_out = model.logits(tensor, config)
                if all_layers:
                    hidden_states = logits_out.hidden_states  # [36, 1, L+2, 1152]
                    all_layer_embs = []
                    for layer_emb in hidden_states:
                        layer_emb = layer_emb.squeeze(0).to(torch.float32).cpu().numpy()  # [L+2, H]
                        cls_emb = layer_emb[0]
                        mean_emb = layer_emb[1:-1].mean(0)
                        all_layer_embs.append((cls_emb, mean_emb))
                    np.save(os.path.join(paths['all_layers'], f'{key}.npy'), all_layer_embs)
                else:
                    layer_emb = logits_out.hidden_states[layer]  # tensor of shape [1, L+2, H]
                    layer_emb = layer_emb.squeeze(0).to(torch.float32).cpu().numpy()  # → [L+2, H]
                    # Remove CLS and END
                    residue_emb = layer_emb[1:-1]  # → [L, H]
                    if setting in ['residue', 'both']:
                        np.save(os.path.join(paths[f'layer_{layer}'], f'{key}.npy'), residue_emb)
                    if setting in ['mean', 'both']:
                        mean_emb = residue_emb.mean(0)
                        np.save(os.path.join(paths['mean'], f'{key}.npy'), mean_emb)

                #     print(f"Error processing {key}: {e}")
            if not only_save:
                if setting == "both":
                    return {
                        key: {
                            "mean": np.load(os.path.join(paths["mean"], f"{key}.npy")),
                            "residue": np.load(os.path.join(paths["residue"], f"{key}.npy"))
                        }
                        for key in seq_dict
                    }
                else:
                    return {
                        key: np.load(os.path.join(paths[setting], f"{key}.npy"))
                        for key in seq_dict
                    }
            else:
                return None
        else:
            raise ValueError(f"Invalid model: {model}. Accepted models: {accepted_models}")
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # Disables dropout for deterministic results
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = model.to(device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Loaded model with {num_params // 1e6} million parameters")

        keys = list(seq_dict.keys())
        keys = [key for key in keys if not key_to_exist[key]]
        print(f"Skipping {len(seq_dict) - len(keys)} sequences with existing embeddings")

        batches = [keys[i:i + batch_size] for i in range(0, len(keys), batch_size)]
        for batch_keys in tqdm(batches):
            batch = {key: seq_dict[key] for key in batch_keys}
            data = [(label, seq) for label, seq in batch.items()]

            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)

                token_representations = results["representations"][33]

                for i, (label, seq) in enumerate(data):
                    res_emb = token_representations[i, 1:len(seq) + 1].cpu().numpy()
                    if setting in ['residue', 'both']:
                        np.save(os.path.join(paths['residue'], f'{label}.npy'), res_emb)
                    if setting in ['mean', 'both']:
                        mean_emb = res_emb.mean(0)
                        np.save(os.path.join(paths['mean'], f'{label}.npy'), mean_emb)


                del data
                del batch_labels
                del batch_strs
                del results
                token_representations = token_representations.cpu()
                del token_representations
                del batch_tokens

            gc.collect()
            torch.cuda.empty_cache()  # Free GPU memory
        if not only_save:
            if setting == "both":
                return {
                    key: {
                        "mean": np.load(os.path.join(paths["mean"], f"{key}.npy")),
                        "residue": np.load(os.path.join(paths["residue"], f"{key}.npy"))
                    }
                    for key in seq_dict
                }
            else:
                return {
                    key: np.load(os.path.join(paths[setting], f"{key}.npy"))
                    for key in seq_dict
                }
if __name__ == '__main__':
#     test_seq_dict = {'Sequence 0': 'AAAATQAVPAPNQQPEVFYNQIFINNEWHDAVSKKTFPTVNPSTGEVICQVAAGDKEDVDRAVKAARAAFQLGSPWRRMDASDRGRLLNRLADLIERDRTYLAALETLDNGKPYVISYLVDLDMVLKCLRYYAGWADKYHGKTIPIDGDFFSYTRHEPVGVCGQIIPWNFPLLMQAAKLGPALATGNVVVMKVAEQTPLTALYVANLTKEAGFPPGVVNVVPGFGPTAGAAIASHEDVDKVAFTGSTEVGHLIQVAAGRSNLKKVTLELGGKSPNIIVSDADMDWAVEQAHFALFFNQGQCCGAGSRTFVQEDVYAEFVERSVARAKSRVVGNPFDSQTEQGPQVDETQFNKVLGYIKSGKEEGAKLLCGGGAAADRGYFIQPTVFGDVQDGMTIAKEEIFGPVMQILKFKTIEEVVGRANNSKYGLAAAVFTKDLDKANYLSQALQAGTVWINCYDVFGAQSPFGGYKMSGNGRELGEYGLQAYTEVKTVTIKVPQKNS',
#  'Sequence 1': 'AADIFAKFKTSMEVK',
#  'Sequence 2': 'ADDINPKEECFFEDDYYEFE',
#  'Sequence 3': 'ADDKNPLEECFCEDDDYCEG',
#  'Sequence 4': 'ADDKNPLEECFREADYEEFLEIAKVTVLEASERNDKEDWYANLGPMRLPEKLNEFVQETENGWYFIKYPVKPSEEGKSAGQLYEESLRKSAGQLYQESLGKAHDDIFAYEKRFDEIVDGMDKLPTSMYQAIQERINFKPPLPPKKYAMGAITTFTPYQFQHFSEALTAPVGR',
#  'Sequence 5': 'ADDKNPLEECFREDDYEEFLEIAKNGLEGWYANLGPMRYPVKPSEEGKHDDIFAYEKFDEIVGGMDKKFWEDDGIHGGKETFCYSPMIQKPYQFQHFSEALTAPVGR',
#  'Sequence 6': 'ADDKNPLEECFREDDYEEFLEIAKNGLKKTSNPKHIVYPVKPSEQLYEESLRDQLPTSMHRYPSMIQKIFFAGEYTANAHGWIDSTIK'}

    import json
    import pandas as pd


    # dlkcat_df_path = '/home/msp/saleh/KinForm/results/dlkcat_df.pkl'
    # import pandas as pd
    # df = pd.read_pickle(dlkcat_df_path)

    # seq_ids = df['Sequence ID'].unique().tolist()
    # seq_ids = list(set(seq_ids))
    seq_id_to_seq_path = '/home/msp/saleh/KinForm/results/sequence_id_to_sequence.pkl'
    seq_id_to_seq = pickle.load(open(seq_id_to_seq_path, 'rb'))
    seq_to_seq_id = {v: k for k, v in seq_id_to_seq.items()}

    # with open('/home/msp/saleh/KinForm/data/EITLEM_data/KM/km_data.json', 'r') as fp:
    #     raw = json.load(fp)
    # sequences = [d["sequence"] for d in raw if len(d["sequence"]) <= 1499]
    # sequences = list(set(sequences))
    # seq_ids = [seq_to_seq_id[seq] for seq in sequences if seq in seq_to_seq_id]
    # for s_id in seq_ids:
    #     assert seq_id_to_seq[s_id] == df[df['Sequence ID'] == s_id]['sequence'].values[0], f"Sequence ID {s_id} not found in sequence_id_to_sequence.pkl"
    # eitlem_csv_path = '/home/msp/saleh/KinForm/results/eitlemall_subset.csv'
    # eitlem_df = pd.read_csv(eitlem_csv_path)
    # eitlem_seqs = eitlem_df['sequence'].unique().tolist()
    # eitlem_seqs = list(set(eitlem_seqs))
    # eitlem_seq_ids = [seq_to_seq_id[seq] for seq in eitlem_seqs if seq in seq_to_seq_id]
    # eitlem_seq_ids = list(set(eitlem_seq_ids))


    # with open('/home/msp/saleh/KinForm/data/KM_data_raw.json', 'r') as fp:
    #     raw = json.load(fp)
    # raw = [d for d in raw if len(d["Sequence"]) <= 1499 and "." not in d["smiles"]]
    # sequences = [d["Sequence"] for d in raw]
    # km_sequences = list(set(sequences))
    # km_seq_ids = [seq_to_seq_id[seq] for seq in km_sequences]

    # seq_ids = seq_ids + eitlem_seq_ids + km_seq_ids
    sequences_df = pd.read_csv("/home/msp/saleh/KinForm/results/synthetic_data/filtered_dlkcat_sequences.csv")
    sequences_list = sequences_df["sequence"].tolist()
    seq_ids = [seq_to_seq_id[seq] for seq in sequences_list if seq in seq_to_seq_id]
    seq_ids = list(set(seq_ids))

    seq_dict = {s_id:seq_id_to_seq[s_id] for s_id in seq_ids}
    
    embd = get_embeddings(seq_dict, model='esm2', batch_size=1, setting='residue', all_layers=False, only_save=True,layer=26)
    # embd = get_embeddings(seq_dict, model='esm2', batch_size=1, setting='residue', all_layers=False, only_save=True,layer=29)

    # embd = get_embeddings(seq_dict, model='esmc', batch_size=1, setting='residue', all_layers=False, only_save=True,layer=32)
    embd = get_embeddings(seq_dict, model='esmc', batch_size=1, setting='residue', all_layers=False, only_save=True,layer=24)
    # embd = get_embeddings(seq_dict, model='esm1v', batch_size=1, setting='both', only_save=True)
    # embd = get_embeddings(seq_dict, model='esm2', batch_size=1, setting='both', only_save=True)
