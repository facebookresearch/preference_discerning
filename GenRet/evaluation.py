"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
from tqdm import tqdm
from argparse import ArgumentParser
import sys
import os
from torch.nn.functional import softmax
parent_dir = os.path.dirname('/'.join(os.path.realpath(__file__).split("/")[:-1]))
sys.path.append(parent_dir)
from utils import CustomDataset, prepend_tag
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from transformers import T5Config, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from collections import defaultdict
from transformers.modeling_outputs import BaseModelOutput
import torch.distributed as dist
import glob
import pickle
import torch.distributed as dist
from peft import get_peft_model, TaskType, LoraConfig
import time

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--eval_sentiment', action='store_true')
    parser.add_argument('--eval_steering', action='store_true')
    parser.add_argument('--eval_hist_cons', action='store_true')
    parser.add_argument('--eval_pref_cons', action='store_true')
    parser.add_argument('--eval_rec', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()

def apk(predicted, k=10):
    """ Computes the average precision at k. """

    predicted = np.asfarray(predicted)
    if len(predicted)>k:
        predicted = predicted[:k]
    actual = [1]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.)

    return score / min(len(actual), k)

def dcg(scores):
    """Compute the Discounted Cumulative Gain."""
    scores = np.asfarray(scores)  # Ensure scores is an array of floats
    return np.sum((2**scores - 1) / np.log2(np.arange(2, scores.size + 2)))

def ndcg_at_k(r, k):
    """Compute NDCG at rank k."""
    r = np.asfarray(r)[:k]  # Ensure r is an array of floats and take top k scores
    dcg_max = dcg(sorted(r, reverse=True))
    if not dcg_max:
        return 0.
    return dcg(r) / dcg_max

def mrr(matches, k):
    """ Compute Mean reciprocal rank at k"""
    rs = np.asfarray(matches[:k]).nonzero()[0]
    return 1. / (rs[0] + 1) if rs.size else 0.

def calculate_metrics(outputs, labels, negatives=False):
    batch_size = len(outputs)  # Assuming outputs is [batch_size, num_beams, seq_len] => seqlen = 
    recall_at_5, recall_at_10 = [], []
    ndcg_at_5, ndcg_at_10 = [], []
    mrr_at_5, mrr_at_10 = [], []
    map_at_5, map_at_10 = [], []
    match_ranks_at_5 = []
    all_matches_at_5 = []
    match_ranks_at_10 = []
    all_matches_at_10 = []

    for i in range(batch_size):
        if not isinstance(outputs[i], torch.Tensor):
            out = torch.Tensor(outputs[i]) # [num_beams, seq_len]
            label = labels[i].cpu().unsqueeze(0)  # [1, seq_len]
        else:
            out = outputs[i]  # [num_beams, seq_len]
            label = labels[i].unsqueeze(0)  # [1, seq_len]

        matches = torch.all(torch.eq(out.unsqueeze(1), label.unsqueeze(0)), dim=2)  # [num_beams, 1, seq_len] -> [num_beams, 1]
        matches = matches.any(dim=1).cpu().numpy()  # [num_beams]

        rank_at_5 = np.nonzero(matches[:5])[0]
        if len(rank_at_5):
            match_ranks_at_5.append(rank_at_5[0])
            all_matches_at_5.append(True)
        else:
            all_matches_at_5.append(False)

        rank_at_10 = np.nonzero(matches[:10])[0]
        if len(rank_at_10):
            match_ranks_at_10.append(rank_at_10[0])
            all_matches_at_10.append(True)
        else:
            all_matches_at_10.append(False)

        # Recall
        if not negatives:
            recall_at_5.append(matches[:5].sum() / 1.0)  # Assuming each label has only 1 correct match.
            recall_at_10.append(matches[:10].sum() / 1.0)
        else:
            # we check if the item is not in the retrieval set
            recall_at_5.append(1 - (matches[:5].sum() / 1.0))  # Assuming each label has only 1 correct match.
            recall_at_10.append(1 - (matches[:10].sum() / 1.0))

        # NDCG (binary relevance)
        ndcg_at_5.append(ndcg_at_k(matches, 5))
        ndcg_at_10.append(ndcg_at_k(matches, 10))

        # MRR
        mrr_at_5.append(mrr(matches, 5))
        mrr_at_10.append(mrr(matches, 10))

        # MAP
        map_at_5.append(apk(matches, 5))
        map_at_10.append(apk(matches, 10))

    # Calculate mean metrics
    metrics = (
        np.mean(recall_at_5),
        np.mean(recall_at_10),
        np.mean(ndcg_at_5),
        np.mean(ndcg_at_10),
        match_ranks_at_5,
        all_matches_at_5,
        match_ranks_at_10,
        all_matches_at_10,
        np.mean(mrr_at_5),
        np.mean(mrr_at_10),
        np.mean(map_at_5),
        np.mean(map_at_10)
    )

    return metrics

def rerank_ids(model, input_id, attn_mask, semantic_ids, batch_size=32):
    outputs = []
    _, seqlen = input_id.shape
    device = input_id.device
    batch_scores = []
    for i in range(0, len(semantic_ids), batch_size):
        label_batch = torch.LongTensor(semantic_ids[i:i+batch_size]).to(device)
        input = input_id.expand(len(label_batch), seqlen)
        att_mask = attn_mask.expand(len(label_batch), seqlen)
        out = model(input_ids=input, attention_mask=att_mask, labels=label_batch)
        logits = torch.log(softmax(out.logits[:, :-1], dim=-1))
        label_batch = label_batch[:, :-1]
        logits = torch.sum(torch.gather(logits, dim=-1, index=label_batch.unsqueeze(-1)), dim=1)
        batch_scores.append(logits)
    inds = torch.argsort(torch.cat(batch_scores).squeeze(), descending=True)[:10]
    selected = semantic_ids[inds.cpu().numpy()][:, :-1]
    outputs.append(selected)
    return outputs

def prep_encoder_outputs(encoder_outputs):
    encoder_outputs = BaseModelOutput(
        last_hidden_state=encoder_outputs,
        hidden_states=None,
        attentions=None,
    )
    return encoder_outputs

@torch.no_grad()
def evaluate(world_size, model, dataloader, device, retrieval_semantic_ids, gen_kwargs, rerank=False, filter_ids=False, instruct_tune=False,
             negatives=False, use_bf16=False, use_amp=True, use_ddp=False):
    model.eval()
    recall_at_5s = []
    recall_at_10s = []
    ndcg_at_5s = []
    ndcg_at_10s = []
    mrr_at_5s, mrr_at_10s = [], []
    maps_at_5, maps_at_10 = [], []
    losses = []
    avg_ranks_at_5, avg_ranks_at_10 = [], []
    all_matches_at_5, all_matches_at_10 = [], []
    match_dict_at_5, match_dict_at_10 = {}, {}
    times = []
    
    progress_bar = tqdm(range(len(dataloader)))
    if rerank:
        retrieval_semantic_ids = np.hstack((retrieval_semantic_ids, np.full((len(retrieval_semantic_ids), 1), fill_value=3025)))

    for batch in dataloader:
        batch_size, *_ = batch['input_ids'].shape
        input_ids = batch['input_ids'].to(device) if batch['input_ids'].dtype == torch.long else None
        attention_mask = batch['attention_mask'].to(device) if batch['input_ids'].dtype == torch.long else None
        labels = batch['labels'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device) if 'decoder_input_ids' in batch and len(batch['decoder_input_ids']) else None
        encoder_outputs = prep_encoder_outputs(batch['input_ids'].to(device)) if batch['input_ids'].dtype == torch.float32 else None

        with torch.cuda.amp.autocast(dtype=torch.float16 if not use_bf16 else torch.bfloat16, enabled=use_amp):
            if not instruct_tune or decoder_input_ids is None:
                if encoder_outputs is not None:
                    outputs = model(encoder_outputs=encoder_outputs, labels=labels)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                losses.append(outputs.loss.item())
            else:
                losses.append(0.)

            if rerank:
                outputs = rerank_ids(model, input_ids, attention_mask, retrieval_semantic_ids)
            else:
                start = time.time()
                if not instruct_tune or decoder_input_ids is None:
                    if encoder_outputs is None:
                        if hasattr(model, 'module'):
                            outputs = model.module.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
                        else:
                            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
                    else:
                        if hasattr(model, 'module'):
                            outputs = model.module.generate(encoder_outputs=encoder_outputs, **gen_kwargs)
                        else:
                            outputs = model.generate(encoder_outputs=encoder_outputs, **gen_kwargs)
                        
                    outputs = outputs[:, 1:5]
                else:
                    if encoder_outputs is not None:
                        if hasattr(model, 'module'):
                            outputs = model.module.generate(encoder_outputs=encoder_outputs, **gen_kwargs)
                        else:
                            outputs = model.generate(encoder_outputs=encoder_outputs, **gen_kwargs)
                        outputs = outputs[:, 1:5]
                    else:
                        _, init_seq_len = decoder_input_ids.shape
                        if hasattr(model, 'module'):
                            outputs = model.module.generate(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, **gen_kwargs)
                        else:
                            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, **gen_kwargs)
                        outputs = outputs[:, init_seq_len:init_seq_len+4]
                end = time.time()
                times.append(end - start)
                if not use_ddp:
                    outputs = outputs.reshape(batch_size, gen_kwargs['num_return_sequences'], -1).cpu().numpy()
                else:
                    outputs = outputs.reshape(batch_size, gen_kwargs['num_return_sequences'], -1)
                    all_outputs = [torch.zeros_like(outputs, dtype=outputs.dtype, device=device) for _ in range(world_size)]
                    all_labels = [torch.zeros_like(labels, dtype=labels.dtype, device=device) for _ in range(world_size)]
                    dist.all_gather(all_labels, labels)
                    dist.all_gather(all_outputs, outputs.contiguous())
                    outputs = torch.cat(all_outputs) # 2xbs, n_seqs, 4
                    outputs = outputs.cpu().numpy()
                    labels = torch.cat(all_labels)
                
                labels = labels[:, :-1]
                if outputs.shape[-1] < 4:
                    # if output shape is smaller than label shape, pad with zeros to label shape
                    # can happen when the LM predicts bos for semids too late or 
                    to_pad = 4 - outputs.shape[-1]
                    pad_tensor = np.zeros((batch_size, gen_kwargs['num_return_sequences'], to_pad))
                    outputs = np.concatenate([outputs, pad_tensor], axis=-1)

                if filter_ids:
                    new_outputs = outputs.copy().tolist()
                    for sample in range(batch_size):
                        all_matches = []
                        for id in outputs[sample]:
                            matches = (id == retrieval_semantic_ids).all(axis=-1)
                            all_matches.append(matches.any())
                        if any(all_matches):
                            # only select items that match an item in the item set
                            # if no items match, retrieval set remains untouched
                            # results in zero recall/ndcg
                            new_outputs[sample] = outputs[sample][all_matches]
                    outputs = new_outputs
        metrics = calculate_metrics(outputs, labels, negatives=negatives)
        recall_at_5, recall_at_10, ndcg_at_5, ndcg_at_10, avg_rank_at_5, has_match_at_5, avg_rank_at_10, has_match_at_10, mrr_at_5, mrr_at_10, map_at_5, map_at_10 = metrics
        if 'sample_index' in batch:
            for index, is_match_at_10, is_match_at_5 in zip(batch['sample_index'], has_match_at_10, has_match_at_5):
                match_dict_at_5[index.item()] = not is_match_at_5 if negatives else is_match_at_5
                match_dict_at_10[index.item()] = not is_match_at_10 if negatives else is_match_at_10
        avg_ranks_at_5.extend(avg_rank_at_5)
        avg_ranks_at_10.extend(avg_rank_at_10)
        all_matches_at_5.extend(has_match_at_5)
        all_matches_at_10.extend(has_match_at_10)
        recall_at_5s.append(recall_at_5)
        recall_at_10s.append(recall_at_10)
        ndcg_at_5s.append(ndcg_at_5)
        ndcg_at_10s.append(ndcg_at_10)
        mrr_at_5s.append(mrr_at_5)
        mrr_at_10s.append(mrr_at_10)
        maps_at_5.append(map_at_5)
        maps_at_10.append(map_at_10)
        progress_bar.set_description(f"recall@10: {(sum(recall_at_10s) / len(recall_at_10s)):.4f}, NDCG@10: {(sum(ndcg_at_10s) / len(ndcg_at_10s)):.4f}")
        progress_bar.update(1)
    progress_bar.close()
    print(f"Validation Loss: {sum(losses) / len(losses)}")
    print(f"recall@5: {sum(recall_at_5s) / len(recall_at_5s)}")
    print(f"recall@10: {sum(recall_at_10s) / len(recall_at_10s)}")
    print(f"NDCG@5: {sum(ndcg_at_5s) / len(ndcg_at_5s)}")
    print(f"NDCG@10: {sum(ndcg_at_10s) / len(ndcg_at_10s)}")
    print("Avg Time for forward pass: ", np.mean(times))
    metrics = {
        'Recall@5': sum(recall_at_5s) / len(recall_at_5s),
        'Recall@10': sum(recall_at_10s) / len(recall_at_10s),
        'NDCG@5': sum(ndcg_at_5s) / len(ndcg_at_5s),
        'NDCG@10': sum(ndcg_at_10s) / len(ndcg_at_10s),
        'loss': sum(losses) / len(losses),
        'avg_rank_at_5': avg_ranks_at_5,
        'avg_rank_at_10': avg_ranks_at_10,
        'MRR@5': np.mean(mrr_at_5s),
        'MRR@10': np.mean(mrr_at_10s),
        'MAP@5': np.mean(maps_at_5),
        'MAP@10': np.mean(maps_at_10)
    }
    if hasattr(model, 'module'):
        model.module.train()
    else:
        model.train()
    return metrics, (match_dict_at_5, match_dict_at_10)

def main(args):
    eval_dirs = ['/'.join(f.split('/')[:-2]) for f in glob.glob(os.path.join(args.ckpt_path, '**'), recursive=True) if f.endswith('best_ckpt.pt')]
    for eval_dir in eval_dirs:
        print(f'Evaluating: {eval_dir}')
        config = OmegaConf.to_container(OmegaConf.load(os.path.join(eval_dir, 'config.json')))
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        from load_data import load_data, load_data_instruct_tune, load_fine_coarse_eval_sets, load_pos_neg_sets, default_collate, load_haystack_eval_data
        from models import LangEncSemIDDecModel, PretrainedCodebookEmbedding, expand_emb_layer, expand_lm_head
        from huggingface_hub import login
        world_size = 1
        
        dataset_config = config['dataset'] if 'dataset' in config else config['setting']
        genret_conf = dataset_config['GenRet']
        if args.batch_size:
            genret_conf['trainer']['eval_batch_size'] = args.batch_size
        t5_config = genret_conf['T5']
        features_used = "_".join(dataset_config['features_needed'])
        save_location = dataset_config['saved_id_path'].replace(".pkl", f"_{features_used}_{dataset_config['content_model']}_{config['seed']}")
        rqvae_conf = dataset_config['RQ-VAE']
        if rqvae_conf['original_impl']:
            save_location += '_original'
        if rqvae_conf['pca']:
            save_location += '_pca'
        save_location += f"_{rqvae_conf['optimizer']}"
        save_location = os.path.join(eval_dir, 'results', f'{save_location}.pkl')

        if genret_conf['instruct_tune']:
            if os.path.exists('/data/shared/fpaischer'):
                cache_dir = '/data/shared/fpaischer/cache'
            elif os.path.exists('/checkpoint/fpaischer'):
                cache_dir = '/checkpoint/fpaischer/cache'
            else:
                cache_dir = os.path.expanduser('~/.cache/huggingface/hub')

        if genret_conf['hf_name'] is None:
            # standard TIGER model, but can use pretrained encoder, from-pretrained indicates we fine-tune entire model
            if genret_conf['instruct_tune'] and genret_conf['hf_encoder'] is None:
                tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small', cache_dir=cache_dir)
                vocab_size = tokenizer.vocab_size + 1026
                if genret_conf['add_user_emb']:
                    vocab_size += 2000
                pad_token = tokenizer.pad_token_id
                eos_token = tokenizer.eos_token_id
            else:
                vocab_size = 1026
                if config['add_user_emb']:
                    vocab_size += 2000
                pad_token = 0
                eos_token = 1025 if not config['add_user_emb'] else 3025
                tokenizer = None

            model_config = T5Config(
                num_layers=t5_config['encoder_layers'], 
                num_decoder_layers=t5_config['decoder_layers'],
                d_model=t5_config['d_model'],
                d_ff=t5_config['d_ff'],
                num_heads=t5_config['num_heads'],
                d_kv=t5_config['d_kv'],
                dropout_rate=t5_config['dropout_rate'],
                activation_function=t5_config['activation_function'],
                vocab_size=vocab_size,
                pad_token_id=pad_token,
                eos_token_id=eos_token,
                decoder_start_token_id=pad_token,
                feed_forward_proj=t5_config['feed_forward_proj'],
                n_positions=genret_conf['n_positions'],
            )
            if genret_conf['hf_encoder'] is not None and genret_conf['instruct_tune']:
                model = LangEncSemIDDecModel(config=model_config)
                embedding_model = 'hkunlp/instructor-base' if genret_conf['embedding_model'] == 'instructor-base' else genret_conf['embedding_model']
                model.register_lang_encoder(genret_conf['hf_encoder'], cache_dir, set_trainable=genret_conf['train_encoder'], lora_config=genret_conf['lora'],
                                            semantic_ids_in_encoder=genret_conf['semantic_ids_in_encoder'], path_to_embs=save_location.replace('.pkl', '.pth'),
                                            center_and_scale=genret_conf['center_and_scale'], embedded_history=genret_conf['embedded_history'], 
                                            embedding_model=embedding_model)
                if genret_conf['embedded_history']:
                    genret_conf['embedding_model'] = genret_conf['embedding_model'].split('/')[-1]
                tokenizer = AutoTokenizer.from_pretrained(genret_conf['hf_encoder'], cache_dir=cache_dir) if not genret_conf['embedded_history'] else None
            else:
                # Initialize the model with the custom configuration
                print("Initializing standard TIGER model")
                model = T5ForConditionalGeneration(config=model_config).to(device)
                if genret_conf['instruct_tune']:
                    # TIGER with vocabulary extension
                    model.decoder.embed_tokens = torch.nn.Embedding(num_embeddings=1026, embedding_dim=t5_config['d_model'])
                    model.lm_head = torch.nn.Linear(t5_config['d_model'], 1026)
        else:
            # fine-tune a pretrained language model
            if 'Llama-3' in genret_conf['hf_name']:
                if not os.path.exists(os.path.expanduser('~/.cache/huggingface/token')):
                    print("llama3 models require authorized access... exiting")
                    exit(1)
                # login with hf authentication token
                auth_token = open(os.path.expanduser('~/.cache/huggingface/token'), 'r').read()
                login(auth_token)
            tokenizer = AutoTokenizer.from_pretrained(genret_conf['hf_name'], cache_dir=cache_dir)
            if genret_conf['from_pretrained']:
                if genret_conf['vocab_extension']:
                    model = AutoModelForSeq2SeqLM.from_pretrained(genret_conf['hf_name'], cache_dir=cache_dir)
                else:
                    raise NotImplementedError('Conditioning stratey must be one of vocab_extension of mlp_encoder')
            else:
                # load config of pretrained model, but do not initialize it from pretrained
                model_config = AutoConfig.from_pretrained(genret_conf['hf_name'])
                if genret_conf['vocab_extension']:
                    model = AutoModelForSeq2SeqLM.from_config(model_config, torch_dtype=torch.float16)
                else:
                    raise NotImplementedError('Conditioning stratey must be one of vocab_extension of mlp_encoder')
                
            # + 2 because of special tokens for bos and eos of semantic ids
            n_new_embeddings = 1026
            if genret_conf['add_user_emb']:
                n_new_embeddings += 2000
            if genret_conf['vocab_extension']:
                if genret_conf['semantic_ids_in_encoder']:
                    expand_emb_layer(model, n_new_embeddings, config['hf_name'])
                expand_lm_head(model, n_new_embeddings, genret_conf['hf_name'], rand_lm_head=genret_conf['rand_lm_head'])

        if t5_config["initialize_pretrained"]:
            assert not genret_conf['vocab_extension'], "Vocabulary extension not supported with initialize pretrained"
            if t5_config['d_model'] != dataset_config['RQ-VAE']['latent_dim']:
                new_embs = PretrainedCodebookEmbedding(model.decoder.embed_tokens.num_embeddings, 128, t5_config['d_model'])
                model.decoder.embed_tokens = new_embs
                if not genret_conf['instruct_tune']:
                    model.shared = model.decoder.embed_tokens
                    model.encoder.embed_tokens = model.decoder.embed_tokens

        if genret_conf['use_lora']:
            lora_conf = genret_conf['lora']
            if genret_conf['rand_lm_head'] or hasattr(genret_conf, "semid_decoder"):
                # do not wrap lm_head if we train it from scratch
                lora_conf['target'] = [w for w in lora_conf['target'] if w != 'lm_head']
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=lora_conf['lora_r'],
                lora_alpha=lora_conf['lora_alpha'],
                target_modules=lora_conf['target'],
                lora_dropout=lora_conf['lora_dropout'],
            )
            model = get_peft_model(model, peft_config)

        if genret_conf['compile']:
            # compile model
            model = torch.compile(model)

        model = model.to(device)
        ckpt_path = [f for f in glob.glob(os.path.join(eval_dir, 'results', '**')) if f.endswith('pt')][0]
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(model, LangEncSemIDDecModel):
            enc_keys = [k for k, _ in model.encoder.state_dict().items() if not 'lora' in k]
            state_dict = {k: v for k, v in ckpt['network'].items() if '.'.join(k.split('.')[1:]) not in enc_keys and k not in enc_keys}
        elif hasattr(model, 'base_model') and genret_conf['vocab_extension']:
            state_dict = {k: v for k, v in ckpt['network'].items() if 'lora' in k or 'encoder.embed_tokens' in k or 'lm_head' in k or 'decoder.embed_tokens' in k}
        else:
            state_dict = ckpt['network']
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        print('Preparing evaluation sequences...')
        if genret_conf['instruct_tune']:
            _, _, test_data, all_sem_ids, item_2_semantic_id, semantic_id_2_item, user_sequence, users = load_data_instruct_tune(dataset_config['name'], save_location, 256, tokenizer=tokenizer,
                                                                                    max_items_per_seq=dataset_config['max_items_per_seq'], 
                                                                                    only_attend_inst=genret_conf.get('only_attend_inst', False), 
                                                                                    most_recent=dataset_config['most_recent'], semantic_ids_in_encoder=genret_conf.get('semantic_ids_in_encoder', False),
                                                                                    accumulate_inst=genret_conf.get('accumulate_inst', False), 
                                                                                    item_as_text=genret_conf.get('item_as_text', True), item_title_plus_inst=genret_conf.get('item_title_plus_inst', False),
                                                                                    item_repr=genret_conf.get('item_repr', ['title']), 
                                                                                    preference_type=genret_conf.get('preference_type', 'default'),
                                                                                    embedded_history=genret_conf['embedded_history'],
                                                                                    embedding_model=genret_conf['embedding_model'])
            test_dataset = CustomDataset(test_data)
            test_dataloader = DataLoader(test_dataset, batch_size=genret_conf['trainer']['eval_batch_size'], shuffle=False, collate_fn=default_collate)
            
            if args.eval_hist_cons:
                # needle in haystack evaluation
                _, haystack_test_data = load_haystack_eval_data(dataset_config['name'], user_sequence, users, item_2_semantic_id, tokenizer,
                                                                                semantic_ids_in_encoder=genret_conf['semantic_ids_in_encoder'],
                                                                                embedded_history=genret_conf['embedded_history'],
                                                                                embedding_model=genret_conf['embedding_model'])
                haystak_test_dataset = CustomDataset(haystack_test_data)
                haystack_test_dataloader = DataLoader(haystak_test_dataset, batch_size=genret_conf['trainer']['eval_batch_size'], shuffle=False, collate_fn=default_collate)

            if args.eval_steering or args.eval_pref_cons:
                fine_datapoints, coarse_datapoints = load_fine_coarse_eval_sets(dataset_config['name'], tokenizer, item_2_semantic_id,
                                                                                user_sequence,
                                                                                item_as_text=True,
                                                                                item_repr=genret_conf['item_repr'],
                                                                                semantic_ids_in_encoder=genret_conf['semantic_ids_in_encoder'],
                                                                                embedded_history=genret_conf['embedded_history'],
                                                                                embedding_model=genret_conf['embedding_model'],
                                                                                add_user_emb=genret_conf['add_user_emb'])

                fine_test_dataset = CustomDataset(fine_datapoints['test'])
                coarse_test_dataset = CustomDataset(coarse_datapoints['test'])
                fine_test_dataloader = DataLoader(fine_test_dataset, batch_size=genret_conf['trainer']['eval_batch_size'], shuffle=False, collate_fn=default_collate)
                coarse_test_dataloader = DataLoader(coarse_test_dataset, batch_size=genret_conf['trainer']['eval_batch_size'], shuffle=False, collate_fn=default_collate)
            
            if args.eval_sentiment:
                _, _, test_pos_neg = load_pos_neg_sets(dataset_config['name'], tokenizer, item_2_semantic_id, only_attend_inst=True, 
                                                                                semantic_ids_in_encoder=genret_conf['semantic_ids_in_encoder'],
                                                                                embedded_history=genret_conf['embedded_history'],
                                                                                embedding_model=genret_conf['embedding_model'],
                                                                                load_train=False, add_user_emb=genret_conf['add_user_emb'])
                pos_test_dataset = CustomDataset(test_pos_neg['positive'])
                neg_test_dataset = CustomDataset(test_pos_neg['negative'])
                neg_test_dataloader = DataLoader(neg_test_dataset, batch_size=genret_conf['trainer']['eval_batch_size'], shuffle=False, collate_fn=default_collate)
                pos_test_dataloader = DataLoader(pos_test_dataset, batch_size=genret_conf['trainer']['eval_batch_size'], shuffle=False, collate_fn=default_collate)
        else:
            tokenizer = None
            _, _, test_data, all_sem_ids, semantic_id_2_item, item_2_semantic_id = load_data(dataset_config['name'], save_location, 256, add_user_emb=genret_conf['add_user_emb'],
                                                    max_items_per_seq=dataset_config['max_items_per_seq'])
            test_dataset = CustomDataset(test_data)
            test_dataloader = DataLoader(test_dataset, batch_size=genret_conf['trainer']['eval_batch_size'], shuffle=False, collate_fn=default_collate)

            if args.eval_steering or args.eval_pref_cons:
                fine_datapoints, coarse_datapoints = load_fine_coarse_eval_sets(dataset_config['name'], tokenizer, item_2_semantic_id, user_sequence,
                                                                                item_as_text=False, item_repr=genret_conf['item_repr'], no_instruct=True,
                                                                                add_user_emb=genret_conf['add_user_emb'])
                fine_test_dataset = CustomDataset(fine_datapoints['test'])
                coarse_test_dataset = CustomDataset(coarse_datapoints['test'])
                fine_test_dataloader = DataLoader(fine_test_dataset, batch_size=genret_conf['trainer']['eval_batch_size'], shuffle=False, collate_fn=default_collate)
                coarse_test_dataloader = DataLoader(coarse_test_dataset, batch_size=genret_conf['trainer']['eval_batch_size'], shuffle=False, collate_fn=default_collate)

        
        use_amp = dataset_config['use_amp']
        use_bf16 = use_amp and (genret_conf['hf_name'] is not None or genret_conf['hf_encoder'] is not None) and torch.cuda.is_bf16_supported()
        gen_kwargs = {'num_beams': genret_conf['num_beams'], "max_new_tokens": 4, "num_return_sequences": genret_conf['num_beams']}
        if genret_conf['instruct_tune']:
            if args.eval_sentiment:
                neg_metrics, (neg_dict_at5, neg_dict_at10) = evaluate(world_size, model, neg_test_dataloader, device, all_sem_ids, gen_kwargs,
                                    rerank=genret_conf['rerank'], filter_ids=genret_conf['filter_ids'], instruct_tune=genret_conf['instruct_tune'],
                                    negatives=True, use_amp=use_amp, use_bf16=use_bf16)
                
                pos_metrics, (pos_dict_at5, pos_dict_at10) = evaluate(world_size, model, pos_test_dataloader, device, all_sem_ids, gen_kwargs, use_amp=use_amp, use_bf16=use_bf16,
                                    rerank=genret_conf['rerank'], filter_ids=genret_conf['filter_ids'], instruct_tune=genret_conf['instruct_tune'])
                
                pos_neg_combined = {
                    'combined_Recall@10': np.mean([pos_dict_at10[ind] and neg_dict_at10[ind] for ind in pos_dict_at10.keys()]),
                    'combined_Recall@5': np.mean([pos_dict_at5[ind] and neg_dict_at5[ind] for ind in pos_dict_at5.keys()])
                }
                neg_metrics = prepend_tag(neg_metrics, 'neg_test')
                pos_metrics = prepend_tag(pos_metrics, 'pos_test')
                combined_metrics = prepend_tag(pos_neg_combined, 'pos_neg_test')
            else:
                pos_metrics = neg_metrics = combined_metrics = {}

            if args.eval_rec:
                metrics, _ = evaluate(world_size, model, test_dataloader, device, all_sem_ids, gen_kwargs, rerank=genret_conf['rerank'], filter_ids=genret_conf['filter_ids'],
                                instruct_tune=genret_conf['instruct_tune'], use_amp=use_amp, use_bf16=use_bf16)
                metrics = prepend_tag(metrics, 'test')
            else:
                metrics = {}
            
            if args.eval_hist_cons:
                haystack_metrics, _ = evaluate(world_size, model, haystack_test_dataloader, device, all_sem_ids, gen_kwargs,
                                        rerank=genret_conf['rerank'], filter_ids=genret_conf['filter_ids'], instruct_tune=genret_conf['instruct_tune'],
                                        use_amp=use_amp, use_bf16=use_bf16)
                haystack_metrics = prepend_tag(haystack_metrics, 'haystack_test')
            else:
                haystack_metrics = {}       

            if args.eval_steering:
                fine_metrics, _ = evaluate(world_size, model, fine_test_dataloader, device, all_sem_ids, gen_kwargs, use_amp=use_amp, use_bf16=use_bf16,
                                        rerank=genret_conf['rerank'], filter_ids=genret_conf['filter_ids'], instruct_tune=genret_conf['instruct_tune'])
                fine_metrics = prepend_tag(fine_metrics, 'fine_test')
            else:
                fine_metrics = {}

            if args.eval_pref_cons:
                coarse_metrics, _ = evaluate(world_size, model, coarse_test_dataloader, device, all_sem_ids, gen_kwargs, use_amp=use_amp, use_bf16=use_bf16,
                                        rerank=genret_conf['rerank'], filter_ids=genret_conf['filter_ids'], instruct_tune=genret_conf['instruct_tune'])
                coarse_metrics = prepend_tag(coarse_metrics, 'coarse_test')
            else:
                coarse_metrics = {}
            
            metrics = {
                **metrics,
                **neg_metrics,
                **pos_metrics,
                **fine_metrics,
                **coarse_metrics,
                **haystack_metrics,
                **combined_metrics
            }
        else:
            if args.eval_rec:
                metrics, _ = evaluate(world_size, model, test_dataloader, device, all_sem_ids, gen_kwargs, use_amp=use_amp, use_bf16=use_bf16,
                                rerank=genret_conf['rerank'], filter_ids=genret_conf['filter_ids'], instruct_tune=genret_conf['instruct_tune'])
                metrics = prepend_tag(metrics, 'test')
            else:
                metrics = {}
            
            if args.eval_steering:
                fine_metrics, _ = evaluate(world_size, model, fine_test_dataloader, device, all_sem_ids, gen_kwargs,
                                                rerank=genret_conf['rerank'], filter_ids=genret_conf['filter_ids'], instruct_tune=genret_conf['instruct_tune'],
                                                use_amp=use_amp, use_bf16=use_bf16, use_ddp=genret_conf['ddp_during_inference'])
                fine_metrics = prepend_tag(fine_metrics, 'fine_test')
            else:
                fine_metrics = {}

            if args.eval_pref_cons:
                coarse_metrics, _ = evaluate(world_size, model, coarse_test_dataloader, device, all_sem_ids, gen_kwargs,
                                            rerank=genret_conf['rerank'], filter_ids=genret_conf['filter_ids'], instruct_tune=genret_conf['instruct_tune'], use_ddp=genret_conf['ddp_during_inference'],
                                            use_amp=use_amp, use_bf16=use_bf16)
                coarse_metrics = prepend_tag(coarse_metrics, 'coarse_test')
            else:
                coarse_metrics = {}

            metrics = {
                **metrics,
                **fine_metrics,
                **coarse_metrics
            }
        
        #to_exclude = ['avg_rank_at_5', 'avg_rank_at_10']
        #for k, v in metrics.items():
        #    if k not in to_exclude:
        #        print(f'{k}: {v}\n')

        if args.overwrite:
            res_dict_path = [f for f in glob.glob(os.path.join(eval_dir, 'results', '**')) if f.endswith('result_dict.pkl')][0]
            with open(res_dict_path, 'rb') as f:
                res_dict = pickle.load(f)

            for key in metrics.keys():
                if key in res_dict:
                    res_dict[key] = metrics[key]
            
            with open(res_dict_path, 'wb') as f:
                pickle.dump(res_dict, f)
        else:
            with open(os.path.join(eval_dir, 'results', 'new_result_dict.pkl'), 'wb') as f:
                pickle.dump(metrics, f)

if __name__ == "__main__":
    main(parse_args())
