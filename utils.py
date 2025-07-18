"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import glob
import importlib
import os
import socket
import zipfile
from collections import defaultdict

import faiss
import numpy as np
import requests
import tqdm
from torch.utils.data import Dataset


def wandb_available():
    # any value of WANDB_DISABLED disables wandb
    if os.getenv("WANDB_DISABLED", "").upper():
        print(
            "Not using wandb for logging, if this is not intended, unset WANDB_DISABLED env var"
        )
        return False
    return importlib.util.find_spec("wandb") is not None


assert (
    wandb_available
), "wandb is not installed but is selected as default for logging, please install via pip install wandb"
import wandb


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.data.items()}
        return item


class WandbManager:
    def __init__(self) -> None:
        self._initialized = False

    def setup(self, args, **kwargs):
        if not isinstance(args, dict):
            args = args.__dict__
        project_name = args.get("project", "debug")

        combined_dict = {**args, **kwargs}
        wandb.init(
            # set the wandb project where this run will be logged
            project=project_name,
            entity=args.get("entity", None),
            # track hyperparameters and run metadata
            config=combined_dict,
            id=args.get("run_id", None),
            resume="allow",
            reinit=False,
        )
        self._initialized = True

    def log(self, logs):
        wandb.log(logs)

    def close(self):
        pass

    def summarize(self, outputs):
        # add values to the wandb summary => only works for scalars
        for k, v in outputs.items():
            self._wandb.run.summary[k] = v.item()


def download_file(url, path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {os.path.basename(path)}")
    else:
        print(f"Failed to download {os.path.basename(path)}")


def setup_logging(config):
    logger_conf = config["logging"]
    model_config = config["setting"]
    if logger_conf["writer"] == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=f"./logs/tiger_exp{config['exp_id']}")
    elif logger_conf["writer"] == "wandb":
        if logger_conf["mode"] == "offline":
            os.environ["WANDB_MODE"] = "offline"
        from utils import WandbManager

        writer = WandbManager()
        writer.setup(
            {
                **logger_conf,
                **model_config,
                "experiment_id": config["experiment_id"],
                "seed": config["seed"],
                "output_path": config["output_path"],
            }
        )
    else:
        raise NotImplementedError("Specified writer not recognized!")
    return writer


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def set_weight_decay(optimizer, weight_decay):
    for param_group in optimizer.param_groups:
        param_group["weight_decay"] = weight_decay


def yield_trainable_params(model):
    for p in model.parameters():
        if p.requires_grad:
            yield p


def set_module_params_trainable(module):
    for p in module.parameters():
        p.requires_grad = True


def set_module_params_not_trainable(module):
    for p in module.parameters():
        p.requires_grad = False


def prepend_tag(dict, tag):
    return {f"{tag}/{k}": v for k, v in dict.items()}


def unwrap_dict(dict):
    new_dict = defaultdict(list)
    for user_id in dict.keys():
        if isinstance(dict[user_id], list):
            for item in dict[user_id]:
                new_dict[item["itemid"]].append(item["review"])
        else:
            for item in dict[user_id].keys():
                new_dict[item].extend(dict[user_id][item])
    return new_dict


def defaultdict_list():
    return defaultdict(list)


def defaultdict_defaultdict_defaultdict_int():
    def defaultdict_defaultdict_int():
        def defaultdict_int():
            return defaultdict(int)

        return defaultdict(defaultdict_int)

    return defaultdict(defaultdict_defaultdict_int)


def find_free_port():
    with socket.socket() as s:
        s.bind(("", 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.


def check_overlap(training_data, test_data):
    inst_label_pairs = [
        (training_data["input_ids"][i], training_data["labels"][i])
        for i in range(len(training_data["input_ids"]))
    ]
    test_inst_label_pairs = [
        (test_data["input_ids"][i], test_data["labels"][i])
        for i in range(len(test_data["input_ids"]))
    ]

    test_matches = []
    test_ct = 0
    for i, (train_inst, train_label) in tqdm(
        enumerate(inst_label_pairs), desc=f"Test Matches: {test_ct}"
    ):
        for j, (val_inst, val_label) in enumerate(test_inst_label_pairs):
            if (train_inst == val_inst).all() and (val_label == train_label).all():
                test_matches.append((i, j))
                test_ct += 1


def compress_src(path):
    files = glob.glob("**", recursive=True)
    # Read all directory, subdirectories and list files
    zf = zipfile.ZipFile(
        os.path.join(path, "src.zip"),
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    )
    for name in files:
        if name.endswith(".py") or name.endswith(".yaml"):
            zf.write(name, arcname=name)
    zf.close()


def get_item_vector_index(embedded_items_dict):
    tmp_key = list(embedded_items_dict.keys())[0]
    dim = embedded_items_dict[tmp_key].shape[0]
    index = faiss.IndexFlatIP(dim)
    values = np.array(list(embedded_items_dict.values()))
    index.add(values)
    return index


def get_preference_vector_index(pref_emb_map):
    tmp_key = list(pref_emb_map.keys())[0]
    dim = pref_emb_map[tmp_key].shape[0]
    index = faiss.IndexFlatIP(dim)
    keys = list(pref_emb_map.keys())
    _, idx = np.unique(keys, return_index=True)
    keys = np.array(keys)[idx]
    values = np.array(list(pref_emb_map.values()))[idx]
    index.add(values)
    return index, keys


# padding:0 first id:1-256 second id:257-512 third id:513-768 fourth id:769-1024 user id:1025-3024 <EOS>:3025
def expand_id(id, offset=0, add_special_tokens=False):
    new_id = (
        id[0] + offset + 1,
        id[1] + offset + 257,
        id[2] + offset + 513,
        id[3] + offset + 769,
    )
    if add_special_tokens:
        new_id = (offset + 1025,) + new_id + (offset + 1026,)
    return list(new_id)


def expand_id_arr(id_arr):
    id_arr[:, 0] += 1
    id_arr[:, 1] += 257
    id_arr[:, 2] += 513
    id_arr[:, 3] += 769
