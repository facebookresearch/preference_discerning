"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5Config,
    T5ForConditionalGeneration,
)
from transformers.optimization import get_scheduler

from .evaluation import evaluate
from .load_data import *
from functools import partial

from huggingface_hub import login
from peft import get_peft_model, LoraConfig, TaskType
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils import (
    CustomDataset,
    get_lr,
    prepend_tag,
    set_module_params_trainable,
    setup_logging,
    WandbManager,
    yield_trainable_params,
)

from .models import (
    expand_emb_layer,
    expand_lm_head,
    initialize_embeddings,
    LangEncSemIDDecModel,
    LinearEncDecModel,
)


def ddp_setup(rank, world_size):
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def train_genret(rank, config, output_path, cache_dir, resume_from_checkpoint):
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else "cpu"
    if not rank:
        writer = setup_logging(config)
    config = {**config["setting"], **{k: v for k, v in config.items()}}
    dataset, save_location, seed = (
        config["name"],
        config["saved_id_path"],
        config["seed"],
    )
    use_first, most_recent, use_amp = (
        config["use_first"],
        config["most_recent"],
        config["use_amp"],
    )
    max_items_per_seq, content_model, features_used = (
        config["max_items_per_seq"],
        config["content_model"],
        config["features_needed"],
    )
    encode_instructions, cluster_users = (
        config["encode_instructs"],
        config["cluster_users"],
    )
    use_ddp = config["use_ddp"]
    features_used = "_".join(features_used)
    used_rqvae = config["train_rqvae"]
    codebook_size = config["RQ-VAE"]["code_book_size"]
    n_codebooks = config["RQ-VAE"]["num_layers"] + 1

    # by default use all available gpus for dataparallel
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if use_ddp and world_size > 1:
        ddp_setup(rank, world_size)

    if used_rqvae:
        save_location = save_location.replace(
            ".pkl", f"_{features_used}_{content_model}_{seed}"
        )
        if config["RQ-VAE"]["original_impl"]:
            save_location += "_original"
        if config["RQ-VAE"]["pca"]:
            save_location += "_pca"
        save_location += f"_{config['RQ-VAE']['optimizer']}"
        save_location = f"{output_path}/results/{save_location}.pkl"
    else:
        training_data, val_data, test_data, n_items = load_data_itemids(
            dataset,
            max_items_per_seq=max_items_per_seq,
            use_first=use_first,
            most_recent=most_recent,
        )
        n_users = 2000

    config = config["GenRet"]
    t5_config = config["T5"]
    trainer_config = config["trainer"]
    if config["hf_name"] is None:
        if config["instruct_tune"] and config["hf_encoder"] is None:
            tokenizer = AutoTokenizer.from_pretrained(
                "google/flan-t5-small", cache_dir=cache_dir
            )
            vocab_size = tokenizer.vocab_size + 1026
            if config["add_user_emb"]:
                vocab_size += 2000
            pad_token = tokenizer.pad_token_id
            eos_token = tokenizer.eos_token_id
        elif used_rqvae:
            vocab_size = 1026
            if config["add_user_emb"]:
                vocab_size += 2000
            pad_token = 0
            eos_token = 1025 if not config["add_user_emb"] else 3025
            tokenizer = None
        else:
            vocab_size = n_users + n_items + 2
            pad_token = 0
            eos_token = n_users + n_items + 1
            tokenizer = None

        model_config = T5Config(
            num_layers=t5_config["encoder_layers"],
            num_decoder_layers=t5_config["decoder_layers"],
            d_model=t5_config["d_model"],
            d_ff=t5_config["d_ff"],
            num_heads=t5_config["num_heads"],
            d_kv=t5_config["d_kv"],
            dropout_rate=t5_config["dropout_rate"],
            activation_function=t5_config["activation_function"],
            vocab_size=vocab_size,
            pad_token_id=pad_token,
            eos_token_id=eos_token,
            decoder_start_token_id=pad_token,
            feed_forward_proj=t5_config["feed_forward_proj"],
            n_positions=config["n_positions"],
        )

        if config["hf_encoder"] is not None and config["instruct_tune"]:
            model = LangEncSemIDDecModel(config=model_config)
            model.register_lang_encoder(
                config["hf_encoder"],
                cache_dir,
                set_trainable=config["train_encoder"],
                lora_config=config["lora"],
                semantic_ids_in_encoder=config["semantic_ids_in_encoder"],
                path_to_embs=save_location.replace(".pkl", ".pth"),
                center_and_scale=config["center_and_scale"],
                embedded_history=config["embedded_history"],
                embedding_model=config["embedding_model"],
                cache_hidden_states=config.get("cache_hidden_states", False),
                neg_weight=config.get("neg_weight", None),
            )
            if config["embedded_history"]:
                config["embedding_model"] = config["embedding_model"].split("/")[-1]
            tokenizer = (
                AutoTokenizer.from_pretrained(config["hf_encoder"], cache_dir=cache_dir)
                if not config["embedded_history"]
                else None
            )
        else:
            # Initialize the model with the custom configuration
            print("Initializing standard TIGER model and train from scratch")
            model = T5ForConditionalGeneration(config=model_config).to(device)
            if config["instruct_tune"]:
                # TIGER with vocabulary extension
                model.decoder.embed_tokens = torch.nn.Embedding(
                    num_embeddings=1026, embedding_dim=t5_config["d_model"]
                )
                model.lm_head = torch.nn.Linear(t5_config["d_model"], 1026)
    else:
        # fine-tune a pretrained language model
        if "Llama-3" in config["hf_name"]:
            if not os.path.exists(os.path.expanduser("~/.cache/huggingface/token")):
                print("llama3 models require authorized access... exiting")
                exit(1)
            # login with hf authentication token
            auth_token = open(
                os.path.expanduser("~/.cache/huggingface/token"), "r"
            ).read()
            login(auth_token)
        tokenizer = AutoTokenizer.from_pretrained(
            config["hf_name"], cache_dir=cache_dir
        )
        if config["from_pretrained"]:
            if config["vocab_extension"]:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    config["hf_name"], cache_dir=cache_dir
                )
            elif config["mlp_encoder"]:
                model_config = AutoConfig.from_pretrained(config["hf_name"])
                assert (
                    used_rqvae
                ), "Cannot use linear encoder for semantic id embeddings if RQ-VAE wasnt trained"
                model = LinearEncDecModel.from_pretrained(config["hf_name"])
                model.register_new_params(
                    tokenizer,
                    save_location.replace(".pkl", ".pth"),
                    enc_hidden_sizes=[model_config.d_model],
                    tie_weights=config["tie_weights"],
                )
            else:
                raise NotImplementedError(
                    "Conditioning stratey must be one of vocab_extension of mlp_encoder"
                )
        else:
            # load config of pretrained model, but do not initialize it from pretrained
            model_config = AutoConfig.from_pretrained(config["hf_name"])
            if config["vocab_extension"]:
                model = AutoModelForSeq2SeqLM.from_config(
                    model_config, torch_dtype=torch.float16
                )
            elif config["mlp_encoder"]:
                assert (
                    used_rqvae
                ), "Cannot use linear encoder for semantic id embeddings if RQ-VAE wasnt trained"
                model = LinearEncDecModel(model_config)
                model.register_new_params(
                    tokenizer,
                    save_location.replace(".pkl", ".pth"),
                    enc_hidden_sizes=[model_config.d_model],
                    tie_weights=config["tie_weights"],
                )
            else:
                raise NotImplementedError(
                    "Conditioning stratey must be one of vocab_extension of mlp_encoder"
                )

        # + 2 because of special tokens for bos and eos of semantic ids
        n_new_embeddings = codebook_size * n_codebooks + 2
        if config["add_user_emb"]:
            n_new_embeddings += 2000
        if config["vocab_extension"]:
            if config["semantic_ids_in_encoder"]:
                expand_emb_layer(model, n_new_embeddings, config["hf_name"])
            expand_lm_head(
                model,
                n_new_embeddings,
                config["hf_name"],
                rand_lm_head=config["rand_lm_head"],
            )

        if not hasattr(model, "semid_decoder") and config["mlp_encoder"]:
            # we only use input projection, but re-purpose the pre-trained lm head
            expand_lm_head(
                model,
                n_new_embeddings,
                config["hf_name"],
                rand_lm_head=config["rand_lm_head"],
            )

    if t5_config["initialize_pretrained"]:
        assert not config[
            "vocab_extension"
        ], "Vocabulary extension not supported with initialize pretrained"
        initialize_embeddings(
            model, save_location, instruct_tune=config["instruct_tune"], device=device
        )
    model = model.to(device)

    if config["use_lora"]:
        lora_conf = config["lora"]
        if config["rand_lm_head"] or hasattr(model, "semid_decoder"):
            # do not wrap lm_head if we train it from scratch
            lora_conf["target"] = [w for w in lora_conf["target"] if w != "lm_head"]
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_conf["lora_r"],
            lora_alpha=lora_conf["lora_alpha"],
            target_modules=lora_conf["target"],
            lora_dropout=lora_conf["lora_dropout"],
        )
        model = get_peft_model(model, peft_config)

    if (config["vocab_extension"] or config["mlp_encoder"]) and config[
        "hf_name"
    ] is not None:
        if config["vocab_extension"] and "t5" in config["hf_name"]:
            # if we do vocabulary extension, set new embeddings and lm head as trainable
            set_module_params_trainable(model.base_model.encoder.embed_tokens)
            set_module_params_trainable(model.base_model.decoder.embed_tokens)
            set_module_params_trainable(model.base_model.lm_head)
        elif config["mlp_encoder"]:
            if hasattr(model.base_model, "semid_encoder"):
                set_module_params_trainable(model.base_model.semid_encoder)
            if config["tie_weights"]:
                set_module_params_trainable(model.base_model.codebook_embs)
            if hasattr(model.base_model, "semid_decoder"):
                set_module_params_trainable(model.base_model.semid_decoder)

    if config["compile"]:
        # compile model
        model = torch.compile(model)

    total_steps = trainer_config["steps"]
    batch_size = trainer_config["batch_size"] // world_size
    learning_rate = trainer_config["lr"]
    eval_batch_size = trainer_config["eval_batch_size"]

    if config["instruct_tune"]:
        assert not (
            config["only_attend_inst"] and config["semantic_ids_in_encoder"]
        ), "Cannot put semantic ids in encoder when only conditioning on instructions!"
        max_length = tokenizer.model_max_length if tokenizer else config["n_positions"]

        if (
            not config["train_encoder"]
            and config["hf_encoder"] is not None
            and not config["embedded_history"]
            and config.get("cache_hidden_states", False)
        ):
            encoder = T5EncoderModel.from_pretrained(
                config["hf_encoder"], cache_dir=cache_dir
            )
            encoder.eval()
            encoder = encoder.to(device)
        else:
            encoder = None

        (
            training_data,
            val_data,
            test_data,
            retrieval_semantic_ids,
            item_2_semantic_id,
            semantic_id_2_item,
            user_sequence,
            users,
        ) = load_data_instruct_tune(
            dataset,
            save_location,
            codebook_size,
            tokenizer,
            max_items_per_seq=max_items_per_seq,
            only_attend_inst=config["only_attend_inst"],
            most_recent=most_recent,
            semantic_ids_in_encoder=config["semantic_ids_in_encoder"]
            or config["vocab_extension"],
            accumulate_inst=config["accumulate_inst"],
            item_as_text=config["item_as_text"],
            max_length=max_length,
            item_title_plus_inst=config["item_title_plus_inst"],
            item_repr=config["item_repr"],
            preference_type=config["preference_type"],
            embedded_history=config["embedded_history"],
            embedding_model=config["embedding_model"],
            encoder=encoder,
        )

        # needle in haystack evaluation
        haystack_val_data, haystack_test_data = load_haystack_eval_data(
            dataset,
            user_sequence,
            users,
            item_2_semantic_id,
            tokenizer,
            semantic_ids_in_encoder=config["semantic_ids_in_encoder"]
            or config["vocab_extension"],
            embedded_history=config["embedded_history"],
            embedding_model=config["embedding_model"],
            encoder=encoder,
        )
        haystack_val_dataset = CustomDataset(haystack_val_data)
        haystak_test_dataset = CustomDataset(haystack_test_data)
        if not use_ddp:
            haystack_val_dataloader = DataLoader(
                haystack_val_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=default_collate,
            )
            haystack_test_dataloader = DataLoader(
                haystak_test_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=default_collate,
            )
        else:
            haystack_val_dataloader = DataLoader(
                haystack_val_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=default_collate,
                sampler=DistributedSampler(haystack_val_dataset),
            )
            haystack_test_dataloader = DataLoader(
                haystak_test_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=default_collate,
                sampler=DistributedSampler(haystak_test_dataset),
            )

        if (
            config["item_as_text"]
            or config["item_title_plus_inst"]
            or config["semantic_ids_in_encoder"]
            or config["embedded_history"]
        ):

            fine_datapoints, coarse_datapoints = load_fine_coarse_eval_sets(
                dataset,
                tokenizer,
                item_2_semantic_id,
                user_sequence,
                item_as_text=config["item_as_text"] or config["item_title_plus_inst"],
                item_repr=config["item_repr"],
                semantic_ids_in_encoder=config["semantic_ids_in_encoder"],
                embedded_history=config["embedded_history"],
                embedding_model=config["embedding_model"],
                encoder=encoder,
                add_user_emb=config["add_user_emb"],
            )
            fine_coarse_train_data = {
                "fine": fine_datapoints["train"],
                "coarse": coarse_datapoints["train"],
            }
            fine_val_dataset = CustomDataset(fine_datapoints["val"])
            fine_test_dataset = CustomDataset(fine_datapoints["test"])
            coarse_val_dataset = CustomDataset(coarse_datapoints["val"])
            coarse_test_dataset = CustomDataset(coarse_datapoints["test"])
            if not use_ddp or not config["ddp_during_inference"]:
                fine_val_dataloader = DataLoader(
                    fine_val_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    collate_fn=default_collate,
                )
                fine_test_dataloader = DataLoader(
                    fine_test_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    collate_fn=default_collate,
                )
                coarse_val_dataloader = DataLoader(
                    coarse_val_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    collate_fn=default_collate,
                )
                coarse_test_dataloader = DataLoader(
                    coarse_test_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    collate_fn=default_collate,
                )
            else:
                fine_val_dataloader = DataLoader(
                    fine_val_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    collate_fn=default_collate,
                    sampler=DistributedSampler(fine_val_dataset),
                )
                fine_test_dataloader = DataLoader(
                    fine_test_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    collate_fn=default_collate,
                    sampler=DistributedSampler(fine_test_dataset),
                )
                coarse_val_dataloader = DataLoader(
                    coarse_val_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    collate_fn=default_collate,
                    sampler=DistributedSampler(coarse_val_dataset),
                )
                coarse_test_dataloader = DataLoader(
                    coarse_test_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    collate_fn=default_collate,
                    sampler=DistributedSampler(coarse_test_dataset),
                )

            train_pos_neg, val_pos_neg, test_pos_neg = load_pos_neg_sets(
                dataset,
                tokenizer,
                item_2_semantic_id,
                only_attend_inst=True,
                semantic_ids_in_encoder=config["semantic_ids_in_encoder"],
                embedded_history=config["embedded_history"],
                embedding_model=config["embedding_model"],
                load_train=config["add_pos_data"] or config["add_neg_data"],
                encoder=encoder,
                add_user_emb=config["add_user_emb"],
            )

            pos_val_dataset = CustomDataset(val_pos_neg["positive"])
            pos_test_dataset = CustomDataset(test_pos_neg["positive"])
            neg_val_dataset = CustomDataset(val_pos_neg["negative"])
            neg_test_dataset = CustomDataset(test_pos_neg["negative"])
            if not use_ddp or not config["ddp_during_inference"]:
                neg_val_dataloader = DataLoader(
                    neg_val_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    collate_fn=default_collate,
                )
                neg_test_dataloader = DataLoader(
                    neg_test_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    collate_fn=default_collate,
                )
                pos_val_dataloader = DataLoader(
                    pos_val_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    collate_fn=default_collate,
                )
                pos_test_dataloader = DataLoader(
                    pos_test_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    collate_fn=default_collate,
                )
            else:
                neg_val_dataloader = DataLoader(
                    neg_val_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    collate_fn=default_collate,
                    sampler=DistributedSampler(neg_val_dataset),
                )
                neg_test_dataloader = DataLoader(
                    neg_test_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    collate_fn=default_collate,
                    sampler=DistributedSampler(neg_test_dataset),
                )
                pos_val_dataloader = DataLoader(
                    pos_val_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    collate_fn=default_collate,
                    sampler=DistributedSampler(pos_val_dataset),
                )
                pos_test_dataloader = DataLoader(
                    pos_test_dataset,
                    batch_size=eval_batch_size,
                    shuffle=False,
                    collate_fn=default_collate,
                    sampler=DistributedSampler(pos_test_dataset),
                )

            to_add = []
            if config["add_neg_data"]:
                to_add.append("negative")
                print(f"Adding negatives to training data")
            if config["add_pos_data"]:
                print(f"Adding positives to training data")
                to_add.append("positive")
            if config["add_fine_data"]:
                print("Adding fine data to training data")
                to_add.append("fine")
            if config["add_coarse_data"]:
                print("Adding coarse data to training data")
                to_add.append("coarse")

            for key in training_data.keys():
                for data in to_add:
                    if isinstance(training_data[key], torch.Tensor):
                        if data in ["positive", "negative"]:
                            training_data[key] = torch.cat(
                                [training_data[key], train_pos_neg[data][key]], dim=0
                            )
                        else:
                            training_data[key] = torch.cat(
                                [training_data[key], fine_coarse_train_data[data][key]],
                                dim=0,
                            )
                    else:
                        if data in ["positive", "negative"]:
                            training_data[key].extend(train_pos_neg[data][key])
                        else:
                            training_data[key].extend(fine_coarse_train_data[data][key])

        if config["add_instruction_item_pairs"]:
            (
                additional_data,
                *_,
            ) = load_data_instruct_tune(
                dataset,
                save_location,
                codebook_size,
                tokenizer,
                max_items_per_seq=max_items_per_seq,
                only_attend_inst=True,
                most_recent=most_recent,
                semantic_ids_in_encoder=False,
                accumulate_inst=False,
                item_as_text=False,
                max_length=max_length,
                item_title_plus_inst=False,
                item_repr=config["item_repr"],
                preference_type=config["preference_type"],
                embedded_history=config["embedded_history"],
                embedding_model=config["embedding_model"],
                encoder=encoder,
            )

            for key in additional_data.keys():
                if isinstance(training_data[key], torch.Tensor):
                    training_data[key] = torch.cat(
                        [training_data[key], additional_data[key]], dim=0
                    )
                else:
                    training_data[key].extend(additional_data[key])

    elif used_rqvae:
        encoder = None
        (
            training_data,
            val_data,
            test_data,
            retrieval_semantic_ids,
            semantic_id_2_item,
            item_2_semantic_id,
            user_sequence,
        ) = load_data(
            dataset,
            save_location,
            codebook_size,
            max_items_per_seq=max_items_per_seq,
            use_first=use_first,
            add_user_emb=config["add_user_emb"],
            most_recent=most_recent,
            encode_instructs=encode_instructions,
            cluster_users=cluster_users,
            predict_all_from_bucket=config["predict_all_from_bucket"],
        )
        fine_datapoints, coarse_datapoints = load_fine_coarse_eval_sets(
            dataset,
            tokenizer,
            item_2_semantic_id,
            user_sequence,
            item_as_text=False,
            item_repr=config["item_repr"],
            no_instruct=True,
            add_user_emb=config["add_user_emb"],
        )
        fine_val_dataset = CustomDataset(fine_datapoints["val"])
        fine_test_dataset = CustomDataset(fine_datapoints["test"])
        coarse_val_dataset = CustomDataset(coarse_datapoints["val"])
        coarse_test_dataset = CustomDataset(coarse_datapoints["test"])
        if not use_ddp or not config["ddp_during_inference"]:
            fine_val_dataloader = DataLoader(
                fine_val_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=default_collate,
            )
            fine_test_dataloader = DataLoader(
                fine_test_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=default_collate,
            )
            coarse_val_dataloader = DataLoader(
                coarse_val_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=default_collate,
            )
            coarse_test_dataloader = DataLoader(
                coarse_test_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=default_collate,
            )
        else:
            fine_val_dataloader = DataLoader(
                fine_val_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=default_collate,
                sampler=DistributedSampler(fine_val_dataset),
            )
            fine_test_dataloader = DataLoader(
                fine_test_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=default_collate,
                sampler=DistributedSampler(fine_test_dataset),
            )
            coarse_val_dataloader = DataLoader(
                coarse_val_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=default_collate,
                sampler=DistributedSampler(coarse_val_dataset),
            )
            coarse_test_dataloader = DataLoader(
                coarse_test_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=default_collate,
                sampler=DistributedSampler(coarse_test_dataset),
            )

    train_dataset = CustomDataset(training_data)
    val_dataset = CustomDataset(val_data)
    test_dataset = CustomDataset(test_data)

    if encoder is not None:
        del encoder
        torch.cuda.empty_cache()

    if config["fine_coarse_data_augmentation"]:
        if not use_ddp:
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=partial(
                    augmented_collate, tokenizer=tokenizer, p=config["bernoulli_p"]
                ),
            )
        else:
            train_sampler = DistributedSampler(train_dataset)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=partial(
                    augmented_collate, tokenizer=tokenizer, p=config["bernoulli_p"]
                ),
                sampler=train_sampler,
            )
    else:
        if not use_ddp:
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=default_collate,
            )
        else:
            train_sampler = DistributedSampler(train_dataset)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=default_collate,
                sampler=train_sampler,
            )

    if not use_ddp or not config["ddp_during_inference"]:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=default_collate,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=default_collate,
        )
    else:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=default_collate,
            sampler=DistributedSampler(val_dataset),
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=default_collate,
            sampler=DistributedSampler(test_dataset),
        )

    optimizer = AdamW(
        yield_trainable_params(model),
        lr=learning_rate,
        weight_decay=trainer_config["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    # train in bfloat16 if we finetune a t5 model (see https://github.com/huggingface/transformers/pull/10956) and it is supported
    use_bf16 = (
        use_amp
        and (config["hf_name"] is not None or config["hf_encoder"] is not None)
        and torch.cuda.is_bf16_supported()
    )

    scheduler = get_scheduler(
        name=trainer_config["scheduler"],
        optimizer=optimizer,
        num_warmup_steps=trainer_config["warmup_steps"],
        num_training_steps=total_steps,
    )

    model.train()
    if config["hf_encoder"] and not config["train_encoder"]:
        model.encoder.eval()
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")
    n_epochs = int(np.ceil(total_steps / len(train_dataloader)))
    print(f"Total number of epochs: {n_epochs}")
    gen_kwargs = {
        "num_beams": config["num_beams"],
        "max_new_tokens": 4,
        "num_return_sequences": config["num_beams"],
    }
    best_metric = -np.inf
    global_step = 0
    best_epoch = 0

    if resume_from_checkpoint:
        print(f"Resuming run from {output_path}/results/best_ckpt.pt")
        checkpoint = torch.load(
            f"{output_path}/results/best_ckpt.pt", map_location=device
        )
        optimizer.load_state_dict(checkpoint["optim"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        state_dict = {k: v for k, v in checkpoint["network"].items()}
        best_metric = checkpoint["best_metric"] if "best_metric" in checkpoint else 0.0
        if isinstance(model, LangEncSemIDDecModel):
            # delete encoder parameters from state dict
            enc_keys = [k for k, _ in model.encoder.state_dict().items()]
            state_dict = {
                k: v
                for k, v in checkpoint["network"].items()
                if ".".join(k.split(".")[1:]) not in enc_keys and k not in enc_keys
            }
        model.load_state_dict(state_dict, strict=False)
        epoch = checkpoint["epoch"]
        train_range = range(epoch, n_epochs)
        np.random.set_state(checkpoint["np_rng"])
        torch.random.set_rng_state(checkpoint["torch_rng"].cpu())
    else:
        train_range = range(n_epochs)

    # by default use all available gpus for dataparallel
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if world_size > 1 and use_ddp:
        model = DDP(model, device_ids=[rank])

    for epoch in train_range:
        progress_bar = tqdm(range(len(train_dataloader)))
        total_loss = pos_loss = neg_loss = 0.0
        total_fine_cts = total_coarse_cts = 0.0
        batch_num = 0
        if use_ddp:
            train_sampler.set_epoch(epoch)
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.cuda.amp.autocast(
                dtype=torch.float16 if not use_bf16 else torch.bfloat16, enabled=use_amp
            ):
                if config["instruct_tune"]:
                    decoder_input_ids = (
                        batch["decoder_input_ids"].to(device)
                        if len(batch["decoder_input_ids"])
                        else None
                    )
                    if batch["input_ids"].dtype == torch.float32:
                        encoder_outputs = input_ids
                        input_ids = None
                        attention_mask = None
                    else:
                        encoder_outputs = (
                            batch["encoder_outputs"].to(device)
                            if "encoder_outputs" in batch
                            else None
                        )
                    # we only need sentiment if we added negative samples
                    sentiment = (
                        batch["sentiment"]
                        if "sentiment" in batch and config["add_neg_data"]
                        else None
                    )
                    if config["vocab_extension"] or config["hf_encoder"] is None:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids,
                            labels=labels,
                        )
                    elif encoder_outputs is not None:
                        outputs = model(
                            encoder_outputs=encoder_outputs,
                            decoder_input_ids=decoder_input_ids,
                            labels=labels,
                            sentiment=sentiment,
                        )
                    else:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids,
                            labels=labels,
                            sentiment=sentiment,
                        )
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

            loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if hasattr(outputs, "neg_loss") and outputs.neg_loss != 0.0:
                neg_loss += outputs.neg_loss
                pos_loss += outputs.pos_loss

            if config["fine_coarse_data_augmentation"]:
                fine_cts, coarse_cts = batch["counts"]
                total_fine_cts += fine_cts
                total_coarse_cts += coarse_cts

            total_loss += loss.item()
            batch_num += 1
            global_step += 1
            progress_bar.set_description(
                f"Epoch {epoch+1}, Loss: {(total_loss/batch_num):.4f}"
            )
            progress_bar.update(1)

        if neg_loss != 0.0:
            add_metrics = {
                "train/neg_loss": outputs.neg_loss,
                "train/pos_loss": outputs.pos_loss,
            }
        else:
            add_metrics = {}

        if config["fine_coarse_data_augmentation"]:
            add_metrics = {
                **add_metrics,
                "train/avg_num_fine_cts": fine_cts / len(train_dataloader),
                "train/avg_num_coarse_cts": coarse_cts / len(train_dataloader),
            }

        if not rank:
            if isinstance(writer, WandbManager):
                logs = {
                    "train/loss": total_loss / len(train_dataloader),
                    "train/epoch": epoch + 1,
                    "train/lr": get_lr(optimizer),
                    **add_metrics,
                }
                writer.log(logs)
            else:
                raise NotImplementedError("Writer not implemented!!")

        progress_bar.close()
        eval_condition = (
            (not rank and not config["ddp_during_inference"])
            if not config["ddp_during_inference"]
            else True
        )
        if (epoch + 1) % config["eval_every"] == 0 and eval_condition:
            if config["instruct_tune"]:
                if config["constrained_beam"]:
                    gen_kwargs["force_words_ids"] = [
                        np.unique(retrieval_semantic_ids.flatten()).tolist()
                    ]
                metrics, _ = evaluate(
                    world_size,
                    model,
                    val_dataloader,
                    device,
                    retrieval_semantic_ids,
                    gen_kwargs,
                    rerank=config["rerank"],
                    filter_ids=config["filter_ids"],
                    instruct_tune=config["instruct_tune"],
                    use_amp=use_amp,
                    use_bf16=use_bf16,
                    use_ddp=config["ddp_during_inference"],
                )

                if config["evaluate_all"]:
                    haystack_metrics, _ = evaluate(
                        world_size,
                        model,
                        haystack_val_dataloader,
                        device,
                        retrieval_semantic_ids,
                        gen_kwargs,
                        rerank=config["rerank"],
                        filter_ids=config["filter_ids"],
                        instruct_tune=config["instruct_tune"],
                        use_amp=use_amp,
                        use_bf16=use_bf16,
                        use_ddp=config["ddp_during_inference"],
                    )

                    neg_metrics, (neg_dict_at5, neg_dict_at10) = evaluate(
                        world_size,
                        model,
                        neg_val_dataloader,
                        device,
                        retrieval_semantic_ids,
                        gen_kwargs,
                        rerank=config["rerank"],
                        filter_ids=config["filter_ids"],
                        instruct_tune=config["instruct_tune"],
                        negatives=True,
                        use_amp=use_amp,
                        use_bf16=use_bf16,
                        use_ddp=config["ddp_during_inference"],
                    )
                    pos_metrics, (pos_dict_at5, pos_dict_at10) = evaluate(
                        world_size,
                        model,
                        pos_val_dataloader,
                        device,
                        retrieval_semantic_ids,
                        gen_kwargs,
                        rerank=config["rerank"],
                        filter_ids=config["filter_ids"],
                        instruct_tune=config["instruct_tune"],
                        use_amp=use_amp,
                        use_bf16=use_bf16,
                        use_ddp=config["ddp_during_inference"],
                    )

                    pos_neg_combined = {
                        "combined_Recall@10": np.mean(
                            [
                                pos_dict_at10[ind] and neg_dict_at10[ind]
                                for ind in pos_dict_at10.keys()
                            ]
                        ),
                        "combined_Recall@5": np.mean(
                            [
                                pos_dict_at5[ind] and neg_dict_at5[ind]
                                for ind in pos_dict_at5.keys()
                            ]
                        ),
                    }

                    fine_metrics, _ = evaluate(
                        world_size,
                        model,
                        fine_val_dataloader,
                        device,
                        retrieval_semantic_ids,
                        gen_kwargs,
                        rerank=config["rerank"],
                        filter_ids=config["filter_ids"],
                        instruct_tune=config["instruct_tune"],
                        use_amp=use_amp,
                        use_bf16=use_bf16,
                        use_ddp=config["ddp_during_inference"],
                    )
                    coarse_metrics, _ = evaluate(
                        world_size,
                        model,
                        coarse_val_dataloader,
                        device,
                        retrieval_semantic_ids,
                        gen_kwargs,
                        rerank=config["rerank"],
                        filter_ids=config["filter_ids"],
                        instruct_tune=config["instruct_tune"],
                        use_amp=use_amp,
                        use_bf16=use_bf16,
                        use_ddp=config["ddp_during_inference"],
                    )
                    # for negative eval noramlize match ranks by number of items
                    for key in ["avg_rank_at_5", "avg_rank_at_10"]:
                        neg_metrics[key] = np.mean(neg_metrics[key])
                        pos_metrics[key] = np.mean(pos_metrics[key])
                        fine_metrics[key] = np.mean(fine_metrics[key])
                        coarse_metrics[key] = np.mean(coarse_metrics[key])
                        haystack_metrics[key] = np.mean(haystack_metrics[key])
                    neg_metrics = prepend_tag(neg_metrics, "neg_eval")
                    pos_metrics = prepend_tag(pos_metrics, "pos_eval")
                    fine_metrics = prepend_tag(fine_metrics, "fine_eval")
                    coarse_metrics = prepend_tag(coarse_metrics, "coarse_eval")
                    combined_metrics = prepend_tag(pos_neg_combined, "pos_neg_eval")
                    haystack_metrics = prepend_tag(haystack_metrics, "haystack_eval")
                else:
                    neg_metrics = {}
                    pos_metrics = {}
                    fine_metrics = {}
                    coarse_metrics = {}
                    combined_metrics = {}
                    haystack_metrics = {}

                for key in ["avg_rank_at_5", "avg_rank_at_10"]:
                    metrics[key] = np.mean(metrics[key])

                if (
                    hasattr(model, "module")
                    and isinstance(model.module, LangEncSemIDDecModel)
                    and not config["train_encoder"]
                ):
                    model.module.encoder.eval()
                elif (
                    isinstance(model, LangEncSemIDDecModel)
                    and not config["train_encoder"]
                ):
                    model.encoder.eval()
                cs_metrics = {}
                overall_metrics = {}
            else:
                if config["constrained_beam"]:
                    gen_kwargs["force_words_ids"] = [
                        np.unique(retrieval_semantic_ids.flatten()).tolist()
                    ]
                metrics, _ = evaluate(
                    world_size,
                    model,
                    val_dataloader,
                    device,
                    retrieval_semantic_ids,
                    gen_kwargs,
                    rerank=config["rerank"],
                    filter_ids=config["filter_ids"],
                    use_ddp=config["ddp_during_inference"],
                    use_amp=use_amp,
                    use_bf16=use_bf16,
                )
                if config["evaluate_all"]:
                    fine_metrics, _ = evaluate(
                        world_size,
                        model,
                        fine_val_dataloader,
                        device,
                        retrieval_semantic_ids,
                        gen_kwargs,
                        rerank=config["rerank"],
                        filter_ids=config["filter_ids"],
                        instruct_tune=config["instruct_tune"],
                        use_amp=use_amp,
                        use_bf16=use_bf16,
                        use_ddp=config["ddp_during_inference"],
                    )
                    coarse_metrics, _ = evaluate(
                        world_size,
                        model,
                        coarse_val_dataloader,
                        device,
                        retrieval_semantic_ids,
                        gen_kwargs,
                        rerank=config["rerank"],
                        filter_ids=config["filter_ids"],
                        instruct_tune=config["instruct_tune"],
                        use_ddp=config["ddp_during_inference"],
                        use_amp=use_amp,
                        use_bf16=use_bf16,
                    )
                    for key in ["avg_rank_at_5", "avg_rank_at_10"]:
                        fine_metrics[key] = np.mean(fine_metrics[key])
                        coarse_metrics[key] = np.mean(coarse_metrics[key])
                    fine_metrics = prepend_tag(fine_metrics, "fine_eval")
                    coarse_metrics = prepend_tag(coarse_metrics, "coarse_eval")
                else:
                    fine_metrics = {}
                    coarse_metrics = {}

                for key in ["avg_rank_at_5", "avg_rank_at_10"]:
                    metrics[key] = np.mean(metrics[key])
                cs_metrics = {}
                overall_metrics = {}
                neg_metrics = {}
                pos_metrics = {}
                haystack_metrics = {}
                combined_metrics = {}

            if not rank:
                metrics = prepend_tag(metrics, "eval")

                if isinstance(writer, WandbManager) and not rank:
                    logs = {
                        **metrics,
                        **cs_metrics,
                        **overall_metrics,
                        "train/step": global_step,
                        **neg_metrics,
                        **pos_metrics,
                        **fine_metrics,
                        **coarse_metrics,
                        **haystack_metrics,
                        **combined_metrics,
                    }
                    writer.log(logs)
                else:
                    raise NotImplementedError("Writer not implemented!!")

                if metrics[f"eval/{config['save_best']}"] > best_metric:
                    best_metric = metrics[f"eval/{config['save_best']}"]
                    best_epoch = epoch
                    model.to("cpu")
                    state_dict = (
                        model.state_dict()
                        if not hasattr(model, "module")
                        else model.module.state_dict()
                    )
                    checkpoint_dict = {
                        "optim": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "network": state_dict,
                        "torch_rng": torch.random.get_rng_state(),
                        "np_rng": np.random.get_state(),
                        "best_metric": best_metric,
                    }
                    torch.save(checkpoint_dict, f"{output_path}/results/best_ckpt.pt")
                    model.to(device)

        if (
            best_epoch + trainer_config["patience"] < epoch
        ) and global_step > trainer_config["warmup_steps"]:
            break

    eval_condition = (
        (not rank and not config["ddp_during_inference"])
        if not config["ddp_during_inference"]
        else True
    )
    if eval_condition:
        print("Testing...")
        print(f"Loading model from {output_path}")
        best_ckpt = torch.load(f"{output_path}/results/best_ckpt.pt")
        if hasattr(model, "module"):
            model.module.load_state_dict(best_ckpt["network"])
        else:
            model.load_state_dict(best_ckpt["network"])

        if config["instruct_tune"]:
            if config["constrained_beam"]:
                gen_kwargs["force_words_ids"] = [
                    np.unique(retrieval_semantic_ids.flatten()).tolist()
                ]
            metrics, _ = evaluate(
                world_size,
                model,
                test_dataloader,
                device,
                retrieval_semantic_ids,
                gen_kwargs,
                rerank=config["rerank"],
                filter_ids=config["filter_ids"],
                instruct_tune=config["instruct_tune"],
                use_amp=use_amp,
                use_bf16=use_bf16,
                use_ddp=config["ddp_during_inference"],
            )

            haystack_metrics, _ = evaluate(
                world_size,
                model,
                haystack_test_dataloader,
                device,
                retrieval_semantic_ids,
                gen_kwargs,
                rerank=config["rerank"],
                filter_ids=config["filter_ids"],
                instruct_tune=config["instruct_tune"],
                use_amp=use_amp,
                use_bf16=use_bf16,
                use_ddp=config["ddp_during_inference"],
            )

            neg_metrics, (neg_dict_at5, neg_dict_at10) = evaluate(
                world_size,
                model,
                neg_test_dataloader,
                device,
                retrieval_semantic_ids,
                gen_kwargs,
                rerank=config["rerank"],
                filter_ids=config["filter_ids"],
                instruct_tune=config["instruct_tune"],
                negatives=True,
                use_amp=use_amp,
                use_bf16=use_bf16,
                use_ddp=config["ddp_during_inference"],
            )
            pos_metrics, (pos_dict_at5, pos_dict_at10) = evaluate(
                world_size,
                model,
                pos_test_dataloader,
                device,
                retrieval_semantic_ids,
                gen_kwargs,
                rerank=config["rerank"],
                filter_ids=config["filter_ids"],
                instruct_tune=config["instruct_tune"],
                use_amp=use_amp,
                use_bf16=use_bf16,
                use_ddp=config["ddp_during_inference"],
            )

            pos_neg_combined = {
                "combined_Recall@10": np.mean(
                    [
                        pos_dict_at10[ind] and neg_dict_at10[ind]
                        for ind in pos_dict_at10.keys()
                    ]
                ),
                "combined_Recall@5": np.mean(
                    [
                        pos_dict_at5[ind] and neg_dict_at5[ind]
                        for ind in pos_dict_at5.keys()
                    ]
                ),
            }

            fine_metrics, _ = evaluate(
                world_size,
                model,
                fine_test_dataloader,
                device,
                retrieval_semantic_ids,
                gen_kwargs,
                rerank=config["rerank"],
                filter_ids=config["filter_ids"],
                instruct_tune=config["instruct_tune"],
                use_amp=use_amp,
                use_bf16=use_bf16,
                use_ddp=config["ddp_during_inference"],
            )
            coarse_metrics, _ = evaluate(
                world_size,
                model,
                coarse_test_dataloader,
                device,
                retrieval_semantic_ids,
                gen_kwargs,
                rerank=config["rerank"],
                filter_ids=config["filter_ids"],
                instruct_tune=config["instruct_tune"],
                use_amp=use_amp,
                use_bf16=use_bf16,
                use_ddp=config["ddp_during_inference"],
            )

            for key in ["avg_rank_at_5", "avg_rank_at_10"]:
                neg_metrics[key] = np.mean(neg_metrics[key])
                pos_metrics[key] = np.mean(pos_metrics[key])
                fine_metrics[key] = np.mean(fine_metrics[key])
                coarse_metrics[key] = np.mean(coarse_metrics[key])
                haystack_metrics[key] = np.mean(haystack_metrics[key])
            neg_metrics = prepend_tag(neg_metrics, "neg_test")
            pos_metrics = prepend_tag(pos_metrics, "pos_test")
            fine_metrics = prepend_tag(fine_metrics, "fine_test")
            coarse_metrics = prepend_tag(coarse_metrics, "coarse_test")
            combined_metrics = prepend_tag(pos_neg_combined, "pos_neg_test")
            haystack_metrics = prepend_tag(haystack_metrics, "haystack_test")
            overall_metrics = {}
        else:
            if config["constrained_beam"]:
                gen_kwargs["force_words_ids"] = [
                    np.unique(retrieval_semantic_ids.flatten()).tolist()
                ]
            metrics, _ = evaluate(
                world_size,
                model,
                test_dataloader,
                device,
                retrieval_semantic_ids,
                gen_kwargs,
                rerank=config["rerank"],
                filter_ids=config["filter_ids"],
                use_amp=use_amp,
                use_bf16=use_bf16,
                use_ddp=config["ddp_during_inference"],
            )
            fine_metrics, _ = evaluate(
                world_size,
                model,
                fine_test_dataloader,
                device,
                retrieval_semantic_ids,
                gen_kwargs,
                rerank=config["rerank"],
                filter_ids=config["filter_ids"],
                instruct_tune=config["instruct_tune"],
                use_amp=use_amp,
                use_bf16=use_bf16,
                use_ddp=config["ddp_during_inference"],
            )
            coarse_metrics, _ = evaluate(
                world_size,
                model,
                coarse_test_dataloader,
                device,
                retrieval_semantic_ids,
                gen_kwargs,
                rerank=config["rerank"],
                filter_ids=config["filter_ids"],
                instruct_tune=config["instruct_tune"],
                use_amp=use_amp,
                use_bf16=use_bf16,
                use_ddp=config["ddp_during_inference"],
            )

            for key in ["avg_rank_at_5", "avg_rank_at_10"]:
                fine_metrics[key] = np.mean(fine_metrics[key])
                coarse_metrics[key] = np.mean(coarse_metrics[key])
            fine_metrics = prepend_tag(fine_metrics, "fine_test")
            coarse_metrics = prepend_tag(coarse_metrics, "coarse_test")
            cs_metrics = {}
            overall_metrics = {}
            neg_metrics = {}
            pos_metrics = {}
            haystack_metrics = {}
            combined_metrics = {}

    if not rank:
        for key in ["avg_rank_at_5", "avg_rank_at_10"]:
            metrics[key] = np.mean(metrics[key])
        metrics = prepend_tag(metrics, "test")
        result_dict = {
            **metrics,
            **cs_metrics,
            **overall_metrics,
            "train/step": global_step,
            **neg_metrics,
            **pos_metrics,
            **fine_metrics,
            **coarse_metrics,
            **haystack_metrics,
            **combined_metrics,
        }
        if isinstance(writer, WandbManager):
            writer.log(result_dict)
        else:
            raise NotImplementedError("Writer not implemented!!")

        with open(os.path.join(output_path, "results", "result_dict.pkl"), "wb") as f:
            pickle.dump(result_dict, f)

        writer.close()

    if use_ddp:
        destroy_process_group()
