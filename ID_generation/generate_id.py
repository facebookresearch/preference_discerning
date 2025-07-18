"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import json
import os
import pickle
from collections import defaultdict
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from utils import set_weight_decay, unwrap_dict, WandbManager


def train_rqvae(model, x, device, config, seed):
    model.to(device)
    print(model)
    batch_size = config["batch_size"]
    num_epochs = config["epochs"]
    beta = config["beta"]
    lr = config["lr"]
    if not config["original_impl"]:
        model.generate_codebook(torch.Tensor(x).to(device), device, seed)
    if hasattr(torch.optim, config["optimizer"]):
        optimizer = getattr(torch.optim, config["optimizer"])(model.parameters(), lr=lr)
        if "weight_decay" in optimizer.param_groups[0]:
            set_weight_decay(optimizer, config["weight_decay"])
    else:
        raise NotImplementedError(
            f"Specified Optimizer {config['optimizer']} not implemented!!"
        )
    trainset, validationset = train_test_split(x, test_size=0.05, random_state=42)
    train_dataset = TensorDataset(torch.Tensor(trainset).to(device))
    val_dataset = TensorDataset(torch.Tensor(validationset).to(device))
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0.0
        total_rec_loss = 0.0
        total_quant_loss = 0.0
        total_emb_loss = 0.0
        total_commit_loss = 0.0
        total_count = 0
        for batch in dataloader:
            x_batch = batch[0]
            optimizer.zero_grad()
            if config["original_impl"]:
                count = 0
                recon_x, commitment_loss, indices = model(x_batch)
                reconstruction_mse_loss = F.mse_loss(recon_x, x_batch, reduction="mean")
                loss = reconstruction_mse_loss + commitment_loss
                quantization_loss = torch.Tensor([0])
                embedding_loss = torch.Tensor([0])
            else:
                recon_x, r, e, count, indices = model(x_batch)
                reconstruction_mse_loss = F.mse_loss(recon_x, x_batch, reduction="mean")
                embedding_loss = F.mse_loss(r.detach(), e, reduction="mean")
                commitment_loss = beta * F.mse_loss(r, e.detach(), reduction="mean")
                quantization_loss = embedding_loss + commitment_loss
                loss = reconstruction_mse_loss + quantization_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_rec_loss += reconstruction_mse_loss.item()
            total_quant_loss += quantization_loss.item()
            total_emb_loss += embedding_loss.item()
            total_commit_loss += commitment_loss.item()
            total_count += count

        if (epoch + 1) % 100 == 0:
            total_val_loss = 0.0
            total_val_rec_loss = 0.0
            total_val_quant_loss = 0.0
            total_val_count = 0
            total_val_emb_loss = 0.0
            total_val_comm_loss = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    x_batch = batch[0]
                    if config["original_impl"]:
                        count = 0
                        recon_x, commitment_loss, indices = model(x_batch)
                        reconstruction_mse_loss = F.mse_loss(
                            recon_x, x_batch, reduction="mean"
                        )
                        loss = reconstruction_mse_loss + commitment_loss
                        quantization_loss = torch.Tensor([0])
                        embedding_loss = torch.Tensor([0])
                    else:
                        recon_x, r, e, count, indices = model(x_batch)
                        reconstruction_mse_loss = F.mse_loss(
                            recon_x, x_batch, reduction="mean"
                        )
                        embedding_loss = F.mse_loss(r, e, reduction="mean")
                        commitment_loss = beta * F.mse_loss(r, e, reduction="mean")
                        quantization_loss = embedding_loss + commitment_loss
                        loss = reconstruction_mse_loss + quantization_loss
                    total_val_loss += loss.item()
                    total_val_rec_loss += reconstruction_mse_loss.item()
                    total_val_quant_loss += quantization_loss.item()
                    total_val_count += count

            print(
                f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {total_loss/ len(dataloader)}, unused_codebook:{total_count/ len(dataloader)}"
            )
            print(
                f"Epoch [{epoch+1}/{num_epochs}], recosntruction_loss: {total_rec_loss/ len(dataloader)}, quantization_loss: {total_quant_loss/ len(dataloader)}"
            )
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {total_val_loss/ len(val_dataloader)}, unused_codebook:{total_val_count/ len(val_dataloader)}"
            )
            print(
                f"Epoch [{epoch+1}/{num_epochs}], recosntruction_loss: {total_val_rec_loss/ len(val_dataloader)}, quantization_loss: {total_val_quant_loss/ len(val_dataloader)}"
            )
    print("Training complete.")


def train(config, device, output_path):
    dataset_name, save_location = config["name"], config["saved_id_path"]
    content_model, use_prompt = config["content_model"], config["use_prompt"]
    model_config = config["RQ-VAE"]
    seed = config["seed"]
    features_used = "_".join(config["features_needed"])
    if not use_prompt:
        emb_save_dir = f"./ID_generation/preprocessing/processed/{dataset_name}_{content_model}_{features_used}_embeddings.pkl"
    else:
        emb_save_dir = f"./ID_generation/preprocessing/processed/{dataset_name}_{content_model}_{features_used}_embeddings.pkl"
        emb_save_dir = emb_save_dir.replace(",pkl", "_prompted.pkl")
    if not (os.path.exists(emb_save_dir)):
        print("Embeddings not found, generating embeddings...")
        if not use_prompt:
            item_2_text = json.loads(
                open(
                    f"./ID_generation/preprocessing/processed/{dataset_name}_{features_used}_id2meta.json"
                ).read()
            )
        else:
            item_2_text = json.loads(
                open(
                    f"./ID_generation/preprocessing/processed/{dataset_name}_{features_used}_id2meta_prompted.json"
                ).read()
            )
        text_embedding_model = SentenceTransformer(
            f"sentence-transformers/{content_model}"
        ).to(device)

        if "review" in config["features_needed"]:
            # filter all reviews for each item and concatenate them
            item2review = json.loads(
                open(f"dataset/item2review_{dataset_name}.json").read()
            )
            item2review = unwrap_dict(item2review)
            for itemid in item_2_text.keys():
                item_2_text[itemid] += f" Reviews: {', '.join(item2review[itemid])}"

        if "instructions" in config["features_needed"]:
            # filter all reviews for each item and concatenate them
            item2instruct = json.loads(
                open(f"dataset/instruction_dict_{dataset_name}.json").read()
            )
            item2instruct = unwrap_dict(item2instruct)
            for itemid in item_2_text.keys():
                if len(config["features_needed"]) == 1:
                    # only instructions
                    item_2_text[itemid] = ". ".join(item2instruct[itemid])
                else:
                    item_2_text[
                        itemid
                    ] += f" Instructions: {'. '.join(item2instruct[itemid])}"

        if "properties" in config["features_needed"]:
            with open(f"dataset/reviews_to_properties_{dataset_name}.json", "r") as f:
                item2prop = json.load(f)
            for itemid in item_2_text.keys():
                item_2_text[itemid] += f" Properties: {', '.join(item2prop[itemid])}"

        item_id_2_text = {}
        for k, v in item_2_text.items():
            item_id_2_text[int(k)] = v

        if dataset_name == "lastfm":
            sorted_text = [
                ", ".join(value) for key, value in sorted(item_id_2_text.items())
            ]
        else:
            sorted_text = [value for key, value in sorted(item_id_2_text.items())]
        bs = 512 if content_model == "sentence-t5-base" else 32
        with torch.no_grad():
            embeddings = text_embedding_model.encode(
                sorted_text,
                convert_to_numpy=True,
                batch_size=bs,
                show_progress_bar=True,
            )
        # embeddings_map = {i:embeddings[i] for i in range(len(embeddings))}
        with open(emb_save_dir, "wb") as f:
            pickle.dump(embeddings, f)

    with open(emb_save_dir, "rb") as f:
        embeddings = pickle.load(f)
    input_size = model_config["input_dim"]
    hidden_sizes = model_config["hidden_dim"]
    latent_size = model_config["latent_dim"]
    num_levels = model_config["num_layers"]
    codebook_size = model_config["code_book_size"]
    dropout = model_config["dropout"]
    if model_config["standardize"]:
        embeddings = StandardScaler().fit_transform(embeddings)
    if model_config["pca"]:
        pca = PCA(n_components=input_size, whiten=True)
        embeddings = pca.fit_transform(embeddings)
    if model_config["original_impl"]:
        from .models.rqvae import RQVAE
    else:
        from .models.RQ_VAE import RQVAE
    rqvae = RQVAE(
        input_size, hidden_sizes, latent_size, num_levels, codebook_size, dropout
    )

    if dataset_name == "steam":
        user_sequence = []
        with open(
            f'{os.path.abspath(".")}/ID_generation/preprocessing/processed/{dataset_name}.txt',
            "r",
        ) as f:
            for line in f.readlines():
                full_seq = line.split(" ")
                user_sequence.append([int(x) for x in full_seq[1:]])
        user_sequence = user_sequence[::7]
        item_2_text = json.loads(
            open(
                f"./ID_generation/preprocessing/processed/{dataset_name}_{features_used}_id2meta.json"
            ).read()
        )
        item_id_2_text = {int(k): v for k, v in item_2_text.items()}
        item_to_ind = {
            int(key): i for i, (key, _) in enumerate(sorted(item_id_2_text.items()))
        }
        all_items = list(chain(*user_sequence))
        train_inds = [item_to_ind[item] for item in np.unique(all_items)]
    else:
        train_inds = np.arange(len(embeddings))

    train_rqvae(rqvae, embeddings[train_inds], device, model_config, seed)
    rqvae.to(device)
    embeddings_tensor = torch.Tensor(embeddings).to(device)
    rqvae.eval()
    if model_config["original_impl"]:
        ids = rqvae.get_codes(embeddings_tensor).cpu().numpy()
        codebook_embs = torch.cat(
            [
                rqvae.quantizer.codebooks[i].weight.data
                for i in range(len(rqvae.quantizer.codebooks))
            ]
        )
        if config["encode_instructs"]:
            raise NotImplementedError(
                "Encoding instructions with alternative RQ-VAE implementation not supported!"
            )
    else:
        ids = rqvae.encode(embeddings_tensor)
        codebook_embs = rqvae.quantization_layer.codebooks
        if config["encode_instructs"]:
            if config["cluster_users"]:
                file = f"dataset/item2review_{dataset_name}_cluster.pkl"
            else:
                file = (
                    f"dataset/embedded_instruction_dict_{dataset_name}.json"
                    if not config["GenRet"]["accumulate_inst"]
                    else f"dataset/accumulated_embedded_instruction_dict_{dataset_name}.json"
                )
            with open(file, "rb") as f:
                encoded_instruct_embs = pickle.load(f)

            if not config["cluster_users"]:
                encoded_instruct_list = [
                    (encoded_instruct_embs[uid][iid], uid, iid)
                    for uid in encoded_instruct_embs.keys()
                    for iid in encoded_instruct_embs[uid].keys()
                ]
                inst_tensor = torch.Tensor(
                    np.array([x[0] for x in encoded_instruct_list])
                )
            else:
                user_ids = encoded_instruct_embs["user_id"]
                inst_tensor = torch.Tensor(
                    [emb for emb in encoded_instruct_embs["centroids"]]
                )

            bs = 32
            instruct_ids = []
            for i in tqdm(
                range(0, len(inst_tensor), bs),
                desc="Encoding instructions in semid space...",
            ):
                batch = inst_tensor[i : i + bs].to(device)
                instruct_ids.append(rqvae.encode(batch))
            instruct_ids = np.concatenate(instruct_ids)

            if not config["cluster_users"]:
                instruct_semid_map = defaultdict(dict)
                for (_, uid, iid), semid in zip(encoded_instruct_list, instruct_ids):
                    instruct_semid_map[uid][iid] = semid
            else:
                instruct_semid_map = {
                    uid: semid for uid, semid in zip(user_ids, instruct_ids)
                }

            with open(
                f"{output_path}/results/embedded_instruction_dict.pkl", "wb"
            ) as f:
                pickle.dump(instruct_semid_map, f)

    # If the ID directory does not exist, create it
    os.makedirs("./ID_generation/ID", exist_ok=True)
    save_location = f'{output_path}/results/{save_location.replace(".pkl", f"_{features_used}_{content_model}_{seed}")}'
    if model_config["original_impl"]:
        save_location += "_original"
    if model_config["pca"]:
        save_location += "_pca"
    save_location += f"_{model_config['optimizer']}"
    torch.save(codebook_embs, f"{save_location}.pth")
    with open(f"{save_location}.pkl", "wb") as f:
        pickle.dump(ids, f)
