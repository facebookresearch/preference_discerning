"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import gc
import json
import os
import random
import sys
import traceback
import uuid

import hydra
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from GenRet.training import train_genret
from hydra.core.hydra_config import HydraConfig
from ID_generation.generate_id import train
from ID_generation.preprocessing.data_process import preprocessing
from omegaconf import DictConfig, OmegaConf
from utils import compress_src, download_file, find_free_port


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(config: DictConfig) -> None:
    urls = [
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz",
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz",
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz",
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Sports_and_Outdoors.json.gz",
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Toys_and_Games.json.gz",
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz",
    ]
    print(config)
    config = OmegaConf.to_container(config)
    hydra_cfg = HydraConfig.get()
    launcher = hydra_cfg["runtime"]["choices"]["hydra/launcher"]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    directory = "./ID_generation/preprocessing/raw_data/"
    directory_processed = "./ID_generation/preprocessing/processed/"
    os.makedirs(directory, exist_ok=True)
    os.makedirs(directory_processed, exist_ok=True)
    os.makedirs("./ID_generation/ID/", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./results", exist_ok=True)

    if launcher == "rfai":
        cache_dir = "/data/shared/fpaischer/cache"
        output_path = "/data/shared/fpaischer/checkpoints"
    elif launcher == "devfair":
        cache_dir = "/checkpoint/fpaischer/cache"
        output_path = "/checkpoint/fpaischer"
    else:
        cache_dir = "./cache"
        output_path = "./outputs"

    if config["output_path"]:
        # output path is pre-set -> continue run from that directory
        output_path = config["output_path"]
        use_ddp = config["use_ddp"]
        try:
            output_path = os.path.join(output_path, hydra_cfg["sweep"]["subdir"])
        except:
            # we do not run a sweep and assume that run directory is explicitly given in the output path
            cache_dir = "./cache"
        # load config from output path
        print(f"Loading config from {output_path}")
        with open(os.path.join(output_path, "config.json"), "r") as f:
            config = json.load(f)
        assert os.path.exists(
            os.path.join(output_path, "results", "best_ckpt.pt")
        ), "No checkpoint dumped to this directory"
        config["logging"]["run_id"] = "_".join(output_path.split("/")[-3:])
        config["use_ddp"] = use_ddp
        # if one of the runs in sweep has already finished -> exit
        if os.path.exists(os.path.join(output_path, "results", "result_dict.pkl")):
            print("Run has already finished... exiting")
            exit(0)
    else:
        try:
            output_path = os.path.join(
                output_path, hydra_cfg["sweep"]["dir"], hydra_cfg["sweep"]["subdir"]
            )
        except:
            output_path = os.path.join(output_path, str(uuid.uuid4()).split("-")[0])

        try:
            config["logging"]["run_id"] = "_".join(output_path.split("/")[-3:])
            os.makedirs(f"{output_path}/logs", exist_ok=True)
            os.makedirs(f"{output_path}/results", exist_ok=True)
        except OSError:
            cache_dir = "./cache"
            output_path = f"./{output_path}"
            os.makedirs(f"{output_path}/logs", exist_ok=True)
            os.makedirs(f"{output_path}/results", exist_ok=True)
        os.makedirs(os.path.join(output_path, "plots"), exist_ok=True)
        config["output_path"] = output_path

        with open(os.path.join(config["output_path"], "config.json"), "w") as f:
            json.dump(config, f)

        if "use_ddp" not in config:
            config["use_ddp"] = False

    compress_src(output_path)
    for url in urls:
        # Extract the filename from the URL
        filename = url.split("/")[-1]
        filepath = os.path.join(directory, filename)
        if not os.path.exists(filepath):
            print(f"{filename} not found, downloading...")
            download_file(url, filepath)
    try:
        resume_from_ckpt = False
        if os.path.exists(f"{output_path}/results/best_ckpt.pt"):
            resume_from_ckpt = True
        train_config = {
            **config["setting"],
            **{k: v for k, v in config.items() if k not in ["logging", "setting"]},
        }
        if not resume_from_ckpt:
            preprocessing(config["setting"], require_attributes=True)
            if config["setting"]["use_prompt"]:
                config["setting"]["features_needed"] = [
                    "title",
                    "brand",
                    "price",
                    "categories",
                ]

            if config["setting"]["train_rqvae"]:
                train(train_config, device, output_path)

        if torch.cuda.is_available():
            world_size = torch.cuda.device_count()
            torch.cuda.empty_cache()
        else:
            world_size = 1

        if config["use_ddp"] and world_size > 1:
            # launch DDP processes
            print(f"----------Using {world_size} data-parallel GPUs----------")
            # assume we only use one node with multiple GPUs
            if not "SLURM_NODELIST" in os.environ:
                os.environ["MASTER_ADDR"] = "localhost"
            else:
                # os.system(f'export MASTER_ADDR=$(scontrol show hostname {os.environ["SLURM_NODELIST"]})')
                # only works for single node so far, adapt above for multinode
                os.environ["MASTER_ADDR"] = os.environ["SLURM_NODELIST"]
            os.environ["MASTER_PORT"] = str(find_free_port())
            if "NCCL_SOCKET_IFNAME" in os.environ:
                # FAIR cluster sets NCCL_SOCKET_IFNAME on login node
                del os.environ["NCCL_SOCKET_IFNAME"]
            mp.spawn(
                train_genret,
                args=(config, output_path, cache_dir, resume_from_ckpt),
                nprocs=world_size,
            )
        else:
            rank = 0
            train_genret(
                rank,
                config,
                output_path,
                cache_dir,
                resume_from_checkpoint=resume_from_ckpt,
            )
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        # fflush everything
        sys.stdout.flush()
        sys.stderr.flush()
        gc.collect()


if __name__ == "__main__":
    main()
