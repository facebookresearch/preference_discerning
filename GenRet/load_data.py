"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from collections import defaultdict
import pickle
import numpy as np
import torch
import os
from itertools import chain, product
import json
from utils import defaultdict_list, defaultdict_defaultdict_defaultdict_int, expand_id_arr, expand_id
from tqdm import tqdm, trange
import random

def add_special_tokens(semantic_ids):
    semantic_ids = np.hstack((np.zeros((len(semantic_ids), 1)), semantic_ids))
    semantic_ids = np.hstack((semantic_ids, np.full(shape=(len(semantic_ids), 1), fill_value=3025)))
    return semantic_ids

def unexpand_ids(ids):
    if isinstance(ids, torch.Tensor) or isinstance(ids, np.ndarray):
        ids[..., 0] -= 1
        ids[..., 1] -= 257
        ids[..., 2] -= 513
        ids[..., 3] -= 769
    elif isinstance(ids, list):
        bs = len(ids)
        for sample in range(bs):
            for id in ids[sample]:
                id[0] -= 1
                id[1] -= 257
                id[2] -= 513
                id[3] -= 769
    else:
        raise NotImplementedError(f"Cannot unexpand ids of datatype {type(ids)}")
    
def isvalid(seq):
    # Skip this sequence if its length is smaller than the minimum number of items we use
    return len(seq) > 2

def map_semid2item(ids, semantic_id2item):
    if isinstance(ids, list):
        new_id_list = []
        bs = len(ids)
        for sample in range(bs):
            new_ids = []
            for id in ids[sample]:
                new_ids.append(semantic_id2item[id[0]][id[1]][id[2]][id[3]])
            new_id_list.append(new_ids)
        return new_id_list
    else:
        assert len(ids.shape) == 2, f"ID tensor must be 2-dimensional, but is {len(ids.shape)}-dimensional"
        item_ids = np.zeros((ids.shape[0], 1))
        for i in range(len(ids)):
            item_ids[i] = semantic_id2item[ids[i, 0].item()][ids[i, 1].item()][ids[i, 2].item()][ids[i, 3].item()]
        return torch.LongTensor(item_ids)

def pad_sequence(sequence, context_length, eos_id, used_user_emb=False):
    # Input sequence format: [<user_id>, <item_1_code_1>, <item_1_code_2>, <item_1_code_3>, <item_1_code_4>, <item_2_code_1>, ....]
    # Here, we have to reserve a position for <EOS>
    if len(sequence) + 1 >= context_length:
        # No need to pad
        if used_user_emb:
            sequence = [sequence[0]] + sequence[len(sequence) - (context_length-2):len(sequence)] + [eos_id]
        else:
            sequence = sequence[len(sequence) - (context_length-1):len(sequence)] + [eos_id]
        return sequence
    return sequence + [eos_id] + [0]*(context_length-len(sequence) - 1)

def pad_labels(labels, left_pad=False, mask=True, shift_right=False):
    # Get max sequence lenght and pad with zeros
    lens = [len(l) for l in labels]
    max_label_len = np.max(lens)

    if shift_right:
        labels = [[0] + lab for lab in labels]

    if left_pad:
        labels = np.array([[0] * (max_label_len - len) + lab  for lab, len in zip(labels, lens)])
    else:
        labels = np.array([lab + [0]*(max_label_len - len) for lab, len in zip(labels, lens)])

    if mask:
        labels[labels == 0] = -100
    return labels

def pad_sequence_preferences(sequence, context_length, pad_id, eos_id, end_lengths, left_pad=False):
    # Input sequence format: [<user_id>, <item_1_code_1>, <item_1_code_2>, <item_1_code_3>, <item_1_code_4>, <item_2_code_1>, ....]
    # Here, we have to reserve a position for <EOS>
    end_lengths = np.array(end_lengths)
    if len(sequence) + 1 >= context_length:
        if not end_lengths[0]:
            cur_context = len(sequence) - (context_length-2)
            start_ind = end_lengths[np.nonzero(end_lengths > cur_context)[0][0]]
            sequence = sequence[start_ind:len(sequence)]
        else:
            cur_context = len(sequence) - (context_length-2-end_lengths[0])
            start_ind = end_lengths[np.nonzero(end_lengths > cur_context)[0][0]]
            sequence = sequence[:end_lengths[0]] + sequence[start_ind:len(sequence)]
    if left_pad:
        sequence = [pad_id] * (context_length - len(sequence) - 1) + sequence
    else:
        sequence = sequence + [eos_id] + [pad_id] * (context_length - len(sequence) - 1)
    return sequence

def pad_sequence_attention(sequence, length, used_user_emb=False):
    if len(sequence) + 1 >= length:
        # No need to pad 
        if used_user_emb:
            sequence = [sequence[0]] + sequence[len(sequence) - (length-2):len(sequence)] + [1]
        else:
            sequence = sequence[len(sequence) - (length-1):len(sequence)] + [1]
        return sequence
    return sequence +  [1] + [0]*(length-len(sequence) - 1)

def pad_sequence_attention_preferences(sequence, context_length, end_lengths, left_pad=False):
    end_lengths = np.array(end_lengths)
    if len(sequence) + 1 >= context_length:
        if not end_lengths[0]:
            cur_context = len(sequence) - (context_length-2)
            start_ind = end_lengths[np.nonzero(end_lengths > cur_context)[0][0]]
            sequence = sequence[start_ind:len(sequence)]
        else:
            cur_context = len(sequence) - (context_length-2-end_lengths[0])
            start_ind = end_lengths[np.nonzero(end_lengths > cur_context)[0][0]]
            sequence = sequence[:end_lengths[0]] + sequence[start_ind:len(sequence)]
    if left_pad:
        sequence = [0] * (context_length - len(sequence) - 1) + sequence
    else:
        sequence = sequence + [1] + [0]*(context_length - len(sequence) - 1)
    return sequence

def generate_input_sequence(user_id, user_sequence, item_2_semantic_id, max_sequence_length, add_user_emb=True, offset=0,
                            instruct_semid_map=None, cluster_users=False, predict_all_from_bucket=False, semantic_id_2_item=None):
    eos_id = 1025 if not add_user_emb else 3025
    if instruct_semid_map is not None:
        if cluster_users:
            input_ids = instruct_semid_map[user_id].tolist()
        else:
            input_ids = instruct_semid_map[user_id][str(user_sequence[len(user_sequence)-2])].tolist()
    elif add_user_emb:
        input_ids = [user_id]
    else:
        input_ids = []
    attention_mask = [1] if add_user_emb else []
    labels = []
    for i in range(len(user_sequence)):
        if i == len(user_sequence) - 1:
            label = item_2_semantic_id[user_sequence[i]]
            if predict_all_from_bucket:
                assert semantic_id_2_item is not None, "need to provide all semantic ids for predicting all items from same bucket"
                bucket_itemids = list(semantic_id_2_item[label[0]][label[1]][label[2]].values())
                alternative_labels = [expand_id(item_2_semantic_id[item], offset=offset) + [eos_id] for item in bucket_itemids]
            else:
                alternative_labels = []
            labels.extend(expand_id(label, offset=offset))
        else:
            input_ids.extend(expand_id(item_2_semantic_id[user_sequence[i]], offset=offset))
            attention_mask.extend([1]*4)
    labels = labels + [eos_id]
    input_ids = np.array(pad_sequence(input_ids, max_sequence_length, eos_id=eos_id, used_user_emb=add_user_emb))
    attention_mask = np.array(pad_sequence_attention(attention_mask, max_sequence_length, used_user_emb=add_user_emb))
    return input_ids, attention_mask, labels, alternative_labels

def generate_embedded_input_sequence(user_sequence, instruct_emb, item_2_emb, item_2_semantic_id, max_sequence_length):
    if isinstance(instruct_emb, list):
        emb_dim = instruct_emb[0].shape[0]
    else:    
        emb_dim = instruct_emb.shape[0]
    embed_seq = np.zeros((max_sequence_length, emb_dim))
    if isinstance(instruct_emb, list):
        embed_seq[:len(instruct_emb)] = instruct_emb
        start = len(instruct_emb)
    else:
        embed_seq[0] = instruct_emb
        start = 1
    labels = []

    for i in range(len(user_sequence)):
        if i == len(user_sequence)-1:
            id_tokens = item_2_semantic_id[user_sequence[-1]]
            labels.extend(expand_id(id_tokens) + [1025])
        else:
            cur_item = user_sequence[i]
            item_emb = item_2_emb[str(cur_item)]
            embed_seq[i+start] = item_emb
    return embed_seq, labels

def generate_input_sequence_preferences(user_sequence, item_2_semantic_id, instruct, tokenizer, max_sequence_length, test_seq=False,
                                         only_attend_inst=False, semantic_ids_in_encoder=False, titles=None):
    input_ids = []
    attention_mask = []
    decoder_input_ids = []
    labels = []
    end_lengths = [0]
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id
    eos_decoder = 1025
    
    if instruct is not None:
        # construct sequence like [PREFERENCE]<ITEM1><ITEM2>....<ITEMn>
        encoded_inst = tokenizer.encode(instruct, add_special_tokens=False)
        if bos_id is not None:
            encoded_inst = [bos_id] + encoded_inst
        input_ids.extend(encoded_inst)
        end_lengths = [len(encoded_inst)]
        attention_mask.extend([1] * len(encoded_inst))

    if titles is not None:
        assert only_attend_inst, "Item titles as input only supported with only_attend_inst flag"
        encoded_titles = [tokenizer.encode(f"{title} ", add_special_tokens=False) for title in titles]
        for i, title in enumerate(encoded_titles):
            input_ids.extend(title)
            attention_mask.extend([1] * len(title))
            end_lengths.append(end_lengths[i] + len(title))
    
    if not only_attend_inst:
        for i in range(len(user_sequence)):
            if (i == len(user_sequence)-1 and semantic_ids_in_encoder) or (i == len(user_sequence)-1 and test_seq):
                # decoder has smaller vocabulary
                id_tokens = expand_id(item_2_semantic_id[user_sequence[i]])
                labels.extend(id_tokens)
            elif not semantic_ids_in_encoder:
                id_tokens = expand_id(item_2_semantic_id[user_sequence[i]])
                if test_seq:
                    decoder_input_ids.extend(id_tokens)
                else:
                    # train sequence we need both labels and decoder_input_ids
                    decoder_input_ids.extend(id_tokens)
                    labels.extend(id_tokens)
            else:
                id_tokens = expand_id(item_2_semantic_id[user_sequence[i]], offset=tokenizer.vocab_size-1)
                input_ids.extend(id_tokens)
                attention_mask.extend([1] * len(id_tokens))
                end_lengths.append(end_lengths[i] + len(id_tokens))
    else:
        id_tokens = item_2_semantic_id[user_sequence[-1]]
        labels.extend(expand_id(id_tokens))
    
    input_ids = np.array(pad_sequence_preferences(input_ids, max_sequence_length, pad_id, eos_id, end_lengths))
    attention_mask = np.array(pad_sequence_attention_preferences(attention_mask, max_sequence_length, end_lengths))
    # decoder is always only in semantic id space!
    labels = labels + [eos_decoder]
    return input_ids, attention_mask, decoder_input_ids, labels, end_lengths

def generate_full_seq_with_instruct(user_sequence, item_2_semantic_id, instruct, tokenizer, max_sequence_length, test_seq=False, user_id=None,
                                    only_attend_inst=False):
    encoder_input_ids = []
    decoder_input_ids = [user_id] if user_id else []
    attention_mask = []
    labels = []
    end_lengths = [0]
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    # language encoder part
    encoded_inst = tokenizer.encode(instruct, add_special_tokens=False)
    encoder_input_ids.extend(encoded_inst)
    end_lengths = [end_lengths[0] + len(encoded_inst)]
    attention_mask.extend([1] * len(encoded_inst))
    encoder_input_ids = np.array(pad_sequence_preferences(encoder_input_ids, max_sequence_length, pad_id, eos_id, end_lengths))
    attention_mask = np.array(pad_sequence_attention_preferences(attention_mask, max_sequence_length, end_lengths))

    if not only_attend_inst:
        for i in range(len(user_sequence)):
            id_tokens = expand_id(item_2_semantic_id[user_sequence[i]])
            if i == len(user_sequence)-1 and test_seq:
                labels.extend(id_tokens)
            elif test_seq:
                decoder_input_ids.extend(id_tokens)
            else:
                # train sequence we need both labels and decoder_input_ids
                decoder_input_ids.extend(id_tokens)
                labels.extend(id_tokens)
    else:
        id_tokens = expand_id(item_2_semantic_id[user_sequence[-1]])
        labels.extend(id_tokens)
    
    labels = labels + [3025]
    return encoder_input_ids, attention_mask, decoder_input_ids, labels

def generate_input_sequence_itemids(user_id, user_sequence, max_sequence_length):
    input_ids = [user_id]
    attention_mask = [1]
    labels = []
    for i in range(len(user_sequence)):
        if i == len(user_sequence) - 1:
            labels.extend([user_sequence[i]])
        else:
            input_ids.extend([user_sequence[i]])
            attention_mask.extend([1])
    labels = np.array(labels + [3025])
    input_ids = np.array(pad_sequence(input_ids, max_sequence_length, eos_id=3025, used_user_emb=user_id is not None))
    attention_mask = np.array(pad_sequence_attention(attention_mask, max_sequence_length))
    assert not np.any(labels == 0)
    labels[labels == 0] = -100
    return input_ids, attention_mask, labels

def load_data(dataset, path, codebook_size, max_length=258, max_items_per_seq=np.inf, use_first=False, add_user_emb=True, offset=0, most_recent=False,
              encode_instructs=False, cluster_users=False, predict_all_from_bucket=False):
    semantic_id_2_item = defaultdict(defaultdict_defaultdict_defaultdict_int)
    item_2_semantic_id = {}
    semantic_ids = pickle.load(open(path, 'rb'))

    for i in range(len(semantic_ids)):
        id = semantic_ids[i]
        id_dict = semantic_id_2_item[id[0]][id[1]][id[2]]
        id_dict[len(id_dict)] = i+1
        item_2_semantic_id[i+1] = (*id, len(id_dict))

    assert len(item_2_semantic_id) == semantic_ids.shape[0], "Not all semanticid -> item collisions have been avoided!"

    # create list of all semantic ids => needed for retrieval to avoid generation of invalid ids
    all_semantic_ids = np.array([item_2_semantic_id[idx] for idx in np.arange(1, len(semantic_ids)+1)])
    assert not (all_semantic_ids[..., :] > codebook_size).any(), "Fourth dimension of semantic ids exceeds vocabulary size, check proper RQ-VAE training!"
    expand_id_arr(all_semantic_ids)

    user_sequence = []
    users = []
    with open(f'{os.path.abspath(".")}/ID_generation/preprocessing/processed/{dataset}.txt', 'r') as f:
        for line in f.readlines():
            user_sequence.append([int(x) for x in line.split(' ')[1:]])
            users.append(line.split(' ')[0])

    if most_recent:
        assert max_items_per_seq < np.inf, "You must set max_items_per_seq to use most_recent items only"
        user_sequence = [seq if len(seq) <= max_items_per_seq else seq[-max_items_per_seq:] for seq in user_sequence]
    else:
        user_sequence = [seq if len(seq) <= max_items_per_seq else seq[:max_items_per_seq] for seq in user_sequence]
    max_sequence_length = max_length

    if dataset == 'steam':
        user_sequence = user_sequence[::7]
        users = users[::7]

    if encode_instructs:
        output_path = '/'.join(path.split('/')[:-1])
        with open(f'{output_path}/embedded_preference_dict.pkl', 'rb') as f:
            instruct_semid_map = pickle.load(f)
    else:
        instruct_semid_map = None

    training_data = {'input_ids': [], 'attention_mask': [], 'labels': []}
    val_data = {'input_ids': [], 'attention_mask': [], 'labels': []}
    test_data = {'input_ids': [], 'attention_mask': [], 'labels': []}
    for i in trange(len(user_sequence), desc="Preparing sequences..."):
        user_id = 1025 + i % 2000 if not encode_instructs else str(i+1)
        train_sequence = []
        train_attention_mask = []
        train_label = []
        val_sequence = []
        val_attention_mask = []
        val_label = []
        test_sequence = []
        test_attention_mask = []
        test_label = []
        
        # user sequence = [1,2,3,4,5]
        # train: j = 2,3 => [1,2], [1,2,3]
        # val: j = 4 =>[1,2,3,4]
        # test: j = 5 => [1,2,3,4,5]
        start = 1 if use_first else 2
        for j in range(start, len(user_sequence[i])+1):
            input_ids, attention_mask, labels, alt_labels = generate_input_sequence(user_id, user_sequence[i][:j], item_2_semantic_id, max_sequence_length,
                                                                        add_user_emb=add_user_emb, offset=offset, instruct_semid_map=instruct_semid_map,
                                                                        cluster_users=cluster_users, predict_all_from_bucket=predict_all_from_bucket,
                                                                        semantic_id_2_item=semantic_id_2_item)
            
            if predict_all_from_bucket:
                input_ids = np.repeat(input_ids.reshape(1, -1), repeats=len(alt_labels)+1, axis=0)
                attention_mask = np.repeat(attention_mask.reshape(1, -1), repeats=len(alt_labels)+1, axis=0)
                input_ids = [in_ids for in_ids in input_ids]
                attention_mask = [attn_mask for attn_mask in attention_mask]
                labels = [labels] + alt_labels
            
            if j == len(user_sequence[i]) - 1:
                if predict_all_from_bucket:
                    val_sequence.extend(input_ids)
                    val_attention_mask.extend(attention_mask)
                    val_label.extend(labels)
                else:
                    val_sequence.append(input_ids)
                    val_attention_mask.append(attention_mask)
                    val_label.append(labels)
            elif j == len(user_sequence[i]):
                if predict_all_from_bucket:
                    test_sequence.extend(input_ids)
                    test_attention_mask.extend(attention_mask)
                    test_label.extend(labels)
                else:
                    test_sequence.append(input_ids)
                    test_attention_mask.append(attention_mask)
                    test_label.append(labels)
            else:
                if predict_all_from_bucket:
                    train_sequence.extend(input_ids)
                    train_attention_mask.extend(attention_mask)
                    train_label.extend(labels)
                else:
                    train_sequence.append(input_ids)
                    train_attention_mask.append(attention_mask)
                    train_label.append(labels)
        
        training_data['input_ids'].extend(train_sequence)
        training_data['attention_mask'].extend(train_attention_mask)
        training_data['labels'].extend(train_label)
        val_data['input_ids'].extend(val_sequence)
        val_data['attention_mask'].extend(val_attention_mask)
        val_data['labels'].extend(val_label)
        test_data['input_ids'].extend(test_sequence)
        test_data['attention_mask'].extend(test_attention_mask)
        test_data['labels'].extend(test_label)

    training_data['input_ids'] = torch.tensor(np.array(training_data['input_ids']), dtype=torch.long)
    training_data['attention_mask'] = torch.tensor(np.array(training_data['attention_mask']), dtype=torch.long)
    training_data['labels'] = torch.tensor(np.array(training_data['labels']), dtype=torch.long)
    val_data['input_ids'] = torch.tensor(np.array(val_data['input_ids']), dtype=torch.long)
    val_data['attention_mask'] = torch.tensor(np.array(val_data['attention_mask']), dtype=torch.long)
    val_data['labels'] = torch.tensor(np.array(val_data['labels']), dtype=torch.long)
    test_data['input_ids'] = torch.tensor(np.array(test_data['input_ids']), dtype=torch.long)
    test_data['attention_mask'] = torch.tensor(np.array(test_data['attention_mask']), dtype=torch.long)
    test_data['labels'] = torch.tensor(np.array(test_data['labels']), dtype=torch.long)
    return training_data, val_data, test_data, all_semantic_ids, semantic_id_2_item, item_2_semantic_id, user_sequence

def prepare_instruction(instruct, titles, model_name):
    if 't5' in model_name and instruct is not None:
        instruct = f"Instructions: {instruct}."

        if titles is not None:
            instruct = f"{instruct} Items: "
    elif 'instructor' in model_name and instruct is not None:
        instruct = f"Represent the Amazon query for retrieving items: {instruct}."
        if titles is not None:
            instruct += f" Represent the Amazon items for retrieval: "
    elif 'GritLM' in model_name and instruct is not None:
        raise NotImplementedError(f'Instruction preparation for GritLM not implemented!')
    
    return instruct

def encode_sequences(data, encoder, batch_size=256):
    # encode all sequences and return hidden states
    device = encoder.device
    with torch.no_grad():
        input_ids = data['input_ids']
        attn_mask = data['attention_mask']
        hidden_states = []
        for i in trange(0, len(input_ids), batch_size, desc=f"Encoding sequences..."):
            batch_input_ids = torch.LongTensor(np.array(input_ids[i:i+batch_size])).to(device)
            batch_attn_mask = torch.LongTensor(np.array(attn_mask[i:i+batch_size])).to(device)
            encoder_outputs = encoder(input_ids=batch_input_ids, attention_mask=batch_attn_mask)[0].cpu().numpy().astype(np.float32).tolist()
            hidden_states.extend(encoder_outputs)
    return hidden_states

def load_data_instruct_tune(dataset, path, codebook_size, tokenizer, max_length=258, max_items_per_seq=np.inf,
                            only_attend_inst=False, most_recent=False, semantic_ids_in_encoder=False,
                            accumulate_inst=False, item_as_text=False, item_title_plus_inst=False,
                            item_repr=['title'], preference_type='default', embedded_history=False,
                            embedding_model='instructor-base', encoder=None):
    semantic_id_2_item = defaultdict(defaultdict_defaultdict_defaultdict_int)
    item_2_semantic_id = {}
    semantic_ids = pickle.load(open(path, 'rb'))

    for i in range(len(semantic_ids)):
        id = semantic_ids[i]
        id_dict = semantic_id_2_item[id[0]][id[1]][id[2]]
        id_dict[len(id_dict)] = i+1
        item_2_semantic_id[i+1] = (*id, len(id_dict))

    assert len(item_2_semantic_id) == semantic_ids.shape[0], "Not all semanticid -> item collisions have been avoided!"

    # create list of all semantic ids => needed for retrieval to avoid generation of invalid ids
    all_semantic_ids = np.array([item_2_semantic_id[idx] for idx in np.arange(1, len(semantic_ids)+1)])
    assert not (all_semantic_ids[..., :] > codebook_size).any(), "Fourth dimension of semantic ids exceeds vocabulary size, check proper RQ-VAE training!"
    expand_id_arr(all_semantic_ids)

    user_sequence = []
    users = []
    with open(f'{os.path.abspath(".")}/ID_generation/preprocessing/processed/{dataset}.txt', 'r') as f:
        for line in f.readlines():
            user_sequence.append([int(x) for x in line.split(' ')[1:]])
            users.append(line.split(' ')[0])
    if most_recent:
        assert max_items_per_seq < np.inf, "You must set max_items_per_seq to use most_recent items only"
        user_sequence = [seq if len(seq) <= max_items_per_seq else seq[-max_items_per_seq:] for seq in user_sequence]
    else:
        user_sequence = [seq if len(seq) <= max_items_per_seq else seq[:max_items_per_seq] for seq in user_sequence]
    max_sequence_length = max_length

    if dataset == 'steam':
        user_sequence = user_sequence[::7]
        users = users[::7]

    # load preference tuning data
    if preference_type == 'default':
        assert os.path.exists(f'dataset/preference_dict_{dataset}.json'), f"preference tuning file for {dataset} not found"
        with open(f'dataset/preference_dict_{dataset}.json', 'r') as f:
            instruct_dict = json.load(f)
    elif preference_type == 'coarse':
        assert os.path.exists(f'dataset/matched_preference_dict_{dataset}_coarse.json'), f"preference tuning file for {dataset} not found"
        with open(f'dataset/matched_preference_dict_{dataset}_coarse.json', 'r') as f:
            instruct_dict = json.load(f)
    elif preference_type == 'granular':
        assert os.path.exists(f'dataset/matched_preference_dict_{dataset}_granular.json'), f"preference tuning file for {dataset} not found"
        with open(f'dataset/matched_preference_dict_{dataset}_granular.json', 'r') as f:
            instruct_dict = json.load(f)
    elif preference_type == 'matched':
        if 'properties' in item_repr:
            assert os.path.exists(f'dataset/matched_preference_dict_properties_{dataset}.json'), f"preference tuning file for {dataset} not found"
            with open(f'dataset/matched_preference_dict_properties_{dataset}.json', 'r') as f:
                instruct_dict = json.load(f)
        elif embedded_history:
            with open(f'dataset/embedded_matched_preference_dict_{dataset}_{embedding_model}.json', 'rb') as f:
                instruct_dict = pickle.load(f)
        else:
            assert os.path.exists(f'dataset/matched_preference_dict_{dataset}.json'), f"preference tuning file for {dataset} not found"
            with open(f'dataset/matched_preference_dict_{dataset}.json', 'r') as f:
                instruct_dict = json.load(f)
    else:
        raise NotImplementedError(f"preference type {preference_type} not supported!!")

    if item_as_text or item_title_plus_inst:
        # load metadata
        with open(f'dataset/item2review_{dataset}.json', 'r') as f:
            item2meta = json.load(f)

        if 'properties' in item_repr:
            with open(f'dataset/reviews_to_properties_{dataset}.json', 'r') as f:
                item2prop = json.load(f)
                for itemid in item2prop.keys():
                    item2meta[itemid]["properties"] = item2prop[itemid]
        
        item2title = {}
        for uid in item2meta.keys():
            for item in item2meta[uid]:
                if item['itemid'] not in item2title:
                    item2title[item['itemid']] = ''
                    for key in item_repr:
                        if key in item:
                            item2title[item['itemid']] += f"{key.capitalize():}: {item[key]}"

    if embedded_history:
        assert not item_as_text and not item_title_plus_inst and not accumulate_inst and preference_type == 'matched', "invalid combination of dataloader args"
        with open(f'dataset/embedded_items_dict_{dataset}_{embedding_model}.json', 'rb') as f:
            embedded_items_dict = pickle.load(f) 
        max_sequence_length = 20

    training_data = {'input_ids': [], 'attention_mask': [], 'labels': [], 'decoder_input_ids': [], 'end_lengths': [], 'sentiment': []}
    val_data = {'input_ids': [], 'attention_mask': [], 'labels': [], 'decoder_input_ids': [], 'end_lengths': []}
    test_data = {'input_ids': [], 'attention_mask': [], 'labels': [], 'decoder_input_ids': [], 'end_lengths': []}
    for i in trange(len(user_sequence), desc='Preparing train sequences...'):
        train_sequence = []
        train_attention_mask = []
        train_label = []
        train_dec_sequence = []
        train_end_lengths = []
        train_sentiment = []
        val_sequence = []
        val_attention_mask = []
        val_label = []
        val_dec_sequence = []
        val_end_lengths = []
        test_sequence = []
        test_attention_mask = []
        test_label = []
        test_dec_sequence = []
        test_end_lengths = []
        
        # user sequence = [1,2,3,4,5]
        # train: j = 2,3 => [1,2], [1,2,3]
        # val: j = 4 =>[1,2,3,4]
        # test: j = 5 => [1,2,3,4,5]
        # cannot use first item for instruct tuning
        start = 2
        for j in range(start, len(user_sequence[i])+1):
            if j == len(user_sequence[i]) - 1:
                if item_as_text:
                    instruct = None
                    titles = [item2title[str(k)] for k in user_sequence[i][:j-1]]
                elif accumulate_inst:
                    instruct = '. '.join([instruct_dict[str(users[i])][str(user_sequence[i][j-2])][k] for k in range(5)])
                    titles = None
                elif item_title_plus_inst:
                    if isinstance(instruct_dict[str(users[i])][str(user_sequence[i][j-2])], list):
                        instruct = '. '.join([instruct_dict[str(users[i])][str(user_sequence[i][j-2])][k] for k in range(5)])
                    else:
                        # matched preferences are strings
                        instruct = instruct_dict[str(users[i])][str(user_sequence[i][j-2])]
                    titles = [item2title[str(k)] for k in user_sequence[i][:j-1]]
                elif embedded_history:
                    instruct_emb = instruct_dict[str(users[i])][str(user_sequence[i][j-2])]
                else:
                    if isinstance(instruct_dict[str(users[i])][str(user_sequence[i][j-2])], list):
                        instruct = instruct_dict[str(users[i])][str(user_sequence[i][j-2])][-2]
                    else:
                        # matched instructs are strings
                        instruct = instruct_dict[str(users[i])][str(user_sequence[i][j-2])]
                    titles = None

                if not embedded_history:
                    instruct = prepare_instruction(instruct, titles, tokenizer.name_or_path)
                    input_ids, attention_mask, decoder_input_ids, labels, end_lengths = generate_input_sequence_preferences(user_sequence[i][:j], item_2_semantic_id, instruct,
                                                                                            tokenizer, max_sequence_length, test_seq=True,
                                                                                            only_attend_inst=only_attend_inst,
                                                                                            semantic_ids_in_encoder=semantic_ids_in_encoder,
                                                                                            titles=titles)

                    val_sequence.append(input_ids)
                    val_attention_mask.append(attention_mask)
                    val_label.append(labels)
                    val_dec_sequence.append(decoder_input_ids)
                    val_end_lengths.append(end_lengths)
                else:
                    encoder_outputs, labels = generate_embedded_input_sequence(user_sequence[i][:j], instruct_emb, embedded_items_dict,
                                                                               item_2_semantic_id, max_sequence_length)
                    val_sequence.append(encoder_outputs)
                    val_attention_mask.append([])
                    val_label.append(labels)
                    val_dec_sequence.append([])
                    val_end_lengths.append([1])
            elif j == len(user_sequence[i]):
                if item_as_text:
                    instruct = None
                    titles = [item2title[str(k)] for k in user_sequence[i][:j-1]]
                elif accumulate_inst:
                    instruct = '. '.join([instruct_dict[str(users[i])][str(user_sequence[i][j-2])][k] for k in range(5)])
                    titles = None
                elif item_title_plus_inst:
                    titles = [item2title[str(k)] for k in user_sequence[i][:j-1]]
                    if isinstance(instruct_dict[str(users[i])][str(user_sequence[i][j-2])], list):
                        instruct = '. '.join([instruct_dict[str(users[i])][str(user_sequence[i][j-2])][k] for k in range(5)])
                    else:
                        # matched preferences are strings
                        instruct = instruct_dict[str(users[i])][str(user_sequence[i][j-2])]
                elif embedded_history:
                    instruct_emb = instruct_dict[str(users[i])][str(user_sequence[i][j-2])]
                else:
                    if isinstance(instruct_dict[str(users[i])][str(user_sequence[i][j-2])], list):
                        instruct = instruct_dict[str(users[i])][str(user_sequence[i][j-2])][-1]
                    else:
                        instruct = instruct_dict[str(users[i])][str(user_sequence[i][j-2])]
                    titles = None

                if not embedded_history:
                    instruct = prepare_instruction(instruct, titles, tokenizer.name_or_path)
                    input_ids, attention_mask, decoder_input_ids, labels, end_lengths = generate_input_sequence_preferences(user_sequence[i][:j], item_2_semantic_id, instruct,
                                                                                            tokenizer, max_sequence_length, test_seq=True,
                                                                                            only_attend_inst=only_attend_inst,
                                                                                            semantic_ids_in_encoder=semantic_ids_in_encoder,
                                                                                            titles=titles)
                    
                    test_sequence.append(input_ids)
                    test_attention_mask.append(attention_mask)
                    test_label.append(labels)
                    test_dec_sequence.append(decoder_input_ids)
                    test_end_lengths.append(end_lengths)
                else:
                    encoder_outputs, labels = generate_embedded_input_sequence(user_sequence[i][:j], instruct_emb, embedded_items_dict,
                                                                               item_2_semantic_id, max_sequence_length)
                    test_sequence.append(encoder_outputs)
                    test_attention_mask.append([])
                    test_label.append(labels)
                    test_dec_sequence.append([])
                    test_end_lengths.append([1])
            else:
                if item_as_text:
                    instructs = [None]
                    titles = [item2title[str(k)] for k in user_sequence[i][:j-1]]
                elif accumulate_inst:
                    instructs = ['. '.join([instruct_dict[str(users[i])][str(user_sequence[i][j-2])][k] for k in range(5)])]
                    titles = None
                elif item_title_plus_inst:
                    titles = [item2title[str(k)] for k in user_sequence[i][:j-1]]
                    if isinstance(instruct_dict[str(users[i])][str(user_sequence[i][j-2])], list):
                        instructs = ['. '.join([instruct_dict[str(users[i])][str(user_sequence[i][j-2])][k] for k in range(5)])]
                    else:
                        # matched preferences are strings
                        instructs = [instruct_dict[str(users[i])][str(user_sequence[i][j-2])]]
                elif embedded_history:
                    instruct_emb = instruct_dict[str(users[i])][str(user_sequence[i][j-2])]
                else:
                    if isinstance(instruct_dict[str(users[i])][str(user_sequence[i][j-2])], list):
                        instructs = instruct_dict[str(users[i])][str(user_sequence[i][j-2])][:3]
                    else:
                        instructs = [instruct_dict[str(users[i])][str(user_sequence[i][j-2])]]
                    titles = None
                
                if not embedded_history:
                    for instruct in instructs:
                        instruct = prepare_instruction(instruct, titles, tokenizer.name_or_path)
                        input_ids, attention_mask, decoder_input_ids, labels, end_lengths = generate_input_sequence_preferences(user_sequence[i][:j], item_2_semantic_id, instruct,
                                                                                                tokenizer, max_sequence_length, test_seq=False,
                                                                                                only_attend_inst=only_attend_inst,
                                                                                                semantic_ids_in_encoder=semantic_ids_in_encoder,
                                                                                                titles=titles)

                        train_sequence.append(input_ids)
                        train_attention_mask.append(attention_mask)
                        train_label.append(labels)
                        train_dec_sequence.append(decoder_input_ids)
                        train_end_lengths.append(end_lengths)
                        train_sentiment.append('positive')
                else:
                    encoder_outputs, labels = generate_embedded_input_sequence(user_sequence[i][:j], instruct_emb, embedded_items_dict,
                                                                               item_2_semantic_id, max_sequence_length)
                    train_sequence.append(encoder_outputs)
                    train_attention_mask.append([])
                    train_label.append(labels)
                    train_dec_sequence.append([])
                    train_end_lengths.append([1])
                    train_sentiment.append('positive')
        
        training_data['input_ids'].extend(train_sequence)
        training_data['attention_mask'].extend(train_attention_mask)
        training_data['labels'].extend(train_label)
        training_data['decoder_input_ids'].extend(train_dec_sequence)
        training_data['end_lengths'].extend(train_end_lengths)
        training_data['sentiment'].extend(train_sentiment)
        val_data['input_ids'].extend(val_sequence)
        val_data['attention_mask'].extend(val_attention_mask)
        val_data['labels'].extend(val_label)
        val_data['decoder_input_ids'].extend(val_dec_sequence)
        val_data['end_lengths'].extend(val_end_lengths)
        test_data['input_ids'].extend(test_sequence)
        test_data['attention_mask'].extend(test_attention_mask)
        test_data['labels'].extend(test_label)
        test_data['decoder_input_ids'].extend(test_dec_sequence)
        test_data['end_lengths'].extend(test_end_lengths)

    if encoder is not None:
        training_data['input_ids'] = encode_sequences(training_data, encoder)
        val_data['input_ids'] = encode_sequences(val_data, encoder)
        test_data['input_ids'] = encode_sequences(test_data, encoder)

    if not only_attend_inst and not semantic_ids_in_encoder:
        # we additionally condition on items
        training_data['decoder_input_ids'] = torch.tensor(pad_labels(training_data['decoder_input_ids'], left_pad=True, mask=False, shift_right=True), dtype=torch.long)
        val_data['decoder_input_ids'] = torch.tensor(pad_labels(val_data['decoder_input_ids'], left_pad=True, mask=False, shift_right=True), dtype=torch.long)
        test_data['decoder_input_ids'] = torch.tensor(pad_labels(test_data['decoder_input_ids'], left_pad=True, mask=False, shift_right=True), dtype=torch.long)
    input_ids_dtype = torch.float32 if embedded_history or encoder is not None else torch.long
    training_data['input_ids'] = torch.tensor(np.array(training_data['input_ids']), dtype=input_ids_dtype)
    training_data['attention_mask'] = torch.tensor(np.array(training_data['attention_mask']), dtype=torch.long)
    training_data['labels'] = torch.tensor(pad_labels(training_data['labels'], left_pad=True), dtype=torch.long)
    val_data['input_ids'] = torch.tensor(np.array(val_data['input_ids']), dtype=input_ids_dtype)
    val_data['attention_mask'] = torch.tensor(np.array(val_data['attention_mask']), dtype=torch.long)
    val_data['labels'] = torch.tensor(np.array(val_data['labels']), dtype=torch.long)
    test_data['input_ids'] = torch.tensor(np.array(test_data['input_ids']), dtype=input_ids_dtype)
    test_data['attention_mask'] = torch.tensor(np.array(test_data['attention_mask']), dtype=torch.long)
    test_data['labels'] = torch.tensor(np.array(test_data['labels']), dtype=torch.long)
    return training_data, val_data, test_data, all_semantic_ids, item_2_semantic_id, semantic_id_2_item, user_sequence, users

def load_haystack_eval_data(dataset, user_sequence, users, item_2_semantic_id, tokenizer,
                            semantic_ids_in_encoder=False, only_inst=False, item_repr=['title'], 
                            embedded_history=False, embedding_model='instructor-base',
                            encoder=None):

    # load preference tuning data
    if not embedded_history:
        assert os.path.exists(f'dataset/preference_dict_{dataset}.json'), f"preference tuning file for {dataset} not found"
        with open(f'dataset/preference_dict_{dataset}.json', 'r') as f:
            instruct_dict = json.load(f)
    else:
        with open(f'dataset/embedded_all_single_preference_dict_{dataset}_{embedding_model}.json', 'rb') as f:
            instruct_dict = pickle.load(f)

    if not semantic_ids_in_encoder:
        # load metadata
        with open(f'dataset/item2review_{dataset}.json', 'r') as f:
            item2meta = json.load(f)
        if 'properties' in item_repr:
            with open(f'dataset/reviews_to_properties_{dataset}.json', 'r') as f:
                item2prop = json.load(f)
                for itemid in item2prop.keys():
                    item2meta[itemid]["properties"] = item2prop[itemid]
        item2title = {}
        for uid in item2meta.keys():
            for item in item2meta[uid]:
                if item['itemid'] not in item2title:
                    item2title[item['itemid']] = ''
                    for key in item_repr:
                        if key in item:
                            item2title[item['itemid']] += f"{key.capitalize():}: {item[key]}"

    if embedded_history:
        with open(f'dataset/embedded_items_dict_{dataset}_{embedding_model}.json', 'rb') as f:
            embedded_items_dict = pickle.load(f) 
        max_sequence_length = 25
    else:
        max_sequence_length = tokenizer.model_max_length

    val_data = {'input_ids': [], 'attention_mask': [], 'labels': [], 'decoder_input_ids': [], 'end_lengths': []}
    test_data = {'input_ids': [], 'attention_mask': [], 'labels': [], 'decoder_input_ids': [], 'end_lengths': []}
    for i in trange(len(user_sequence), desc='Preparing train sequences...'):
        val_sequence = []
        val_attention_mask = []
        val_label = []
        val_dec_sequence = []
        val_end_lengths = []
        test_sequence = []
        test_attention_mask = []
        test_label = []
        test_dec_sequence = []
        test_end_lengths = []
        
        # user sequence = [1,2,3,4,5]
        # train: j = 2,3 => [1,2], [1,2,3]
        # val: j = 4 =>[1,2,3,4]
        # test: j = 5 => [1,2,3,4,5]
        # cannot use first item for instruct tuning
        start = 2
        for j in range(start, len(user_sequence[i])+1):
            if j == len(user_sequence[i]) - 1:
                
                titles = [item2title[str(k)] for k in user_sequence[i][:j-1]] if not semantic_ids_in_encoder and not only_inst else None
                if not embedded_history:
                    instruct = '. '.join([instruct_dict[str(users[i])][str(user_sequence[i][j-2])][k] for k in range(5)])
                else:
                    # preferences are a list of embeddings
                    instruct = instruct_dict[str(users[i])][str(user_sequence[i][j-2])]
                
                if not embedded_history:
                    instruct = prepare_instruction(instruct, titles, tokenizer.name_or_path)
                    input_ids, attention_mask, decoder_input_ids, labels, end_lengths = generate_input_sequence_preferences(user_sequence[i][:j], item_2_semantic_id, instruct,
                                                                                            tokenizer, max_sequence_length, test_seq=True,
                                                                                            only_attend_inst=not semantic_ids_in_encoder,
                                                                                            semantic_ids_in_encoder=semantic_ids_in_encoder,
                                                                                            titles=titles)

                    val_sequence.append(input_ids)
                    val_attention_mask.append(attention_mask)
                    val_label.append(labels)
                    val_dec_sequence.append(decoder_input_ids)
                    val_end_lengths.append(end_lengths)
                else:
                    instruct_emb = instruct_dict[str(users[i])][str(user_sequence[i][j-2])]
                    encoder_outputs, labels = generate_embedded_input_sequence(user_sequence[i][:j], instruct_emb, embedded_items_dict,
                                                                               item_2_semantic_id, max_sequence_length)
                    val_sequence.append(encoder_outputs)
                    val_attention_mask.append([])
                    val_label.append(labels)
                    val_dec_sequence.append([])
                    val_end_lengths.append([1])
            elif j == len(user_sequence[i]):

                titles = [item2title[str(k)] for k in user_sequence[i][:j-1]] if not semantic_ids_in_encoder else None                    
                if not embedded_history:
                    instruct = '. '.join([instruct_dict[str(users[i])][str(user_sequence[i][j-2])][k] for k in range(5)])
                    instruct = prepare_instruction(instruct, titles, tokenizer.name_or_path)
                    input_ids, attention_mask, decoder_input_ids, labels, end_lengths = generate_input_sequence_preferences(user_sequence[i][:j], item_2_semantic_id, instruct,
                                                                                            tokenizer, max_sequence_length, test_seq=True,
                                                                                            only_attend_inst=not semantic_ids_in_encoder,
                                                                                            semantic_ids_in_encoder=semantic_ids_in_encoder,
                                                                                            titles=titles)
                    
                    test_sequence.append(input_ids)
                    test_attention_mask.append(attention_mask)
                    test_label.append(labels)
                    test_dec_sequence.append(decoder_input_ids)
                    test_end_lengths.append(end_lengths)
                else:
                    instruct_emb = instruct_dict[str(users[i])][str(user_sequence[i][j-2])]
                    encoder_outputs, labels = generate_embedded_input_sequence(user_sequence[i][:j], instruct_emb, embedded_items_dict,
                                                                               item_2_semantic_id, max_sequence_length)
                    test_sequence.append(encoder_outputs)
                    test_attention_mask.append([])
                    test_label.append(labels)
                    test_dec_sequence.append([])
                    test_end_lengths.append([1])
        
        val_data['input_ids'].extend(val_sequence)
        val_data['attention_mask'].extend(val_attention_mask)
        val_data['labels'].extend(val_label)
        val_data['decoder_input_ids'].extend(val_dec_sequence)
        val_data['end_lengths'].extend(val_end_lengths)
        test_data['input_ids'].extend(test_sequence)
        test_data['attention_mask'].extend(test_attention_mask)
        test_data['labels'].extend(test_label)
        test_data['decoder_input_ids'].extend(test_dec_sequence)
        test_data['end_lengths'].extend(test_end_lengths)

    if encoder is not None:
        val_data['input_ids'] = encode_sequences(val_data, encoder)
        test_data['input_ids'] = encode_sequences(test_data, encoder)

    input_ids_dtype = torch.float32 if embedded_history or encoder is not None else torch.long
    val_data['input_ids'] = torch.tensor(np.array(val_data['input_ids']), dtype=input_ids_dtype)
    test_data['input_ids'] = torch.tensor(np.array(test_data['input_ids']), dtype=input_ids_dtype)
    val_data['attention_mask'] = torch.tensor(np.array(val_data['attention_mask']), dtype=torch.long)
    val_data['labels'] = torch.tensor(np.array(val_data['labels']), dtype=torch.long)
    test_data['attention_mask'] = torch.tensor(np.array(test_data['attention_mask']), dtype=torch.long)
    test_data['labels'] = torch.tensor(np.array(test_data['labels']), dtype=torch.long)
    return val_data, test_data

def load_fine_coarse_eval_sets(dataset, tokenizer, item_2_semantic_id, user_sequence,
                               item_as_text=True, item_repr=['title'], no_instruct=False, max_length=258, 
                               semantic_ids_in_encoder=False, embedded_history=False, embedding_model='instructor-base',
                               encoder=None, add_user_emb=False):

    all_items = list(chain(*[user_sequence[i] for i in range(len(user_sequence))]))
    semantic_id_map = { item: expand_id(item_2_semantic_id[item]) for item in np.unique(all_items) }
    
    if item_as_text:
        # load metadata
        with open(f'dataset/item2review_{dataset}.json', 'r') as f:
            item2meta = json.load(f)
        if 'properties' in item_repr:
            with open(f'dataset/reviews_to_properties_{dataset}.json', 'r') as f:
                item2prop = json.load(f)
                for itemid in item2prop.keys():
                    item2meta[itemid]["properties"] = item2prop[itemid]
        item2title = {}
        for uid in item2meta.keys():
            for item in item2meta[uid]:
                if item['itemid'] not in item2title:
                    item2title[item['itemid']] = ''
                    for key in item_repr:
                        if key in item:
                            item2title[item['itemid']] += f"{key.capitalize():}: {item[key]}"

    if not embedded_history:
        with open(f'dataset/fine_coarse_preference_splits_{dataset}.pkl', 'rb') as f:
            fine_coarse_data = pickle.load(f)
    else:
        with open(f'dataset/embedded_fine_coarse_split_{dataset}_{embedding_model}.json', 'rb') as f:
            fine_coarse_data = pickle.load(f)
        
        with open(f'dataset/embedded_items_dict_{dataset}_{embedding_model}.json', 'rb') as f:
            embedded_items_dict = pickle.load(f)

    coarse_datapoints = defaultdict(defaultdict_list)    
    fine_datapoints = defaultdict(defaultdict_list)
    # prepare validation/test splits
    offset = tokenizer.vocab_size - 1 if tokenizer else 0
    label_offset = 0 if semantic_ids_in_encoder or tokenizer is None else tokenizer.vocab_size - 1
    eos_id = 1025 if not add_user_emb else 3025
    max_length = tokenizer.model_max_length if tokenizer else max_length
    pad_token_id = tokenizer.pad_token_id if tokenizer else 0
    eos_token_id = tokenizer.eos_token_id if tokenizer else 1

    for split in fine_coarse_data.keys():

        for type in fine_coarse_data[split]:
            for info in tqdm(fine_coarse_data[split][type], desc=f"Generating {type} for {split} set..."):

                if split == 'train':
                    instruct, user_ind, item_id, target_index = info
                else:
                    instruct, user_ind, item_id = info
                    target_index = -2 if split == 'val' else -1

                cur_seq = user_sequence[user_ind][:target_index]
                if not no_instruct and not embedded_history:
                    instruct = prepare_instruction(instruct, None if not item_as_text else "", tokenizer.name_or_path)
                    encoded_inst = tokenizer.encode(instruct, add_special_tokens=False)
                    input_ids = encoded_inst
                    end_length = [len(encoded_inst)]
                else:
                    input_ids = []
                    end_length = [0]

                if item_as_text:
                    # represent items in text form
                    items = [item2title[str(k)] for k in cur_seq]
                    encoded_titles = [tokenizer.encode(f"{title} ", add_special_tokens=False) for title in items]
                    for i, title in enumerate(encoded_titles):
                        input_ids.extend(title)
                        end_length.append(end_length[i] + len(title))
                    labels = semantic_id_map[int(item_id)] + [eos_id]
                elif embedded_history:
                    # instruct is already an embedding
                    encoder_outputs, labels = generate_embedded_input_sequence(cur_seq + [int(item_id)], instruct, embedded_items_dict,
                                                                               item_2_semantic_id, 20)
                    input_ids = encoder_outputs
                    attn_mask = []
                else:
                    # represent items as semantic ids
                    encoded_items = [expand_id(item_2_semantic_id[k], offset=offset) for k in cur_seq]
                    encoded_items = list(chain(*encoded_items))
                    end_length = np.arange(end_length[0], end_length[0]+len(encoded_items)+4, 4).tolist()
                    input_ids += encoded_items
                    labels = expand_id(item_2_semantic_id[int(item_id)], offset=label_offset) + [eos_id]
                
                if not embedded_history:
                    # padding only required for non-embedded history
                    attn_mask = [1] * len(input_ids)
                    input_ids = pad_sequence_preferences(input_ids, max_length, pad_token_id,
                                                        eos_token_id, end_lengths=end_length)
                    attn_mask = pad_sequence_attention_preferences(attn_mask, max_length, end_lengths=end_length)
                        
                if type == 'fine':
                    fine_datapoints[split]['input_ids'].append(input_ids)
                    fine_datapoints[split]['attention_mask'].append(attn_mask)
                    fine_datapoints[split]['labels'].append(labels)
                    if split == 'train':
                        fine_datapoints[split]['decoder_input_ids'].append([])
                        fine_datapoints[split]['sentiment'].append('positive')
                        fine_datapoints[split]['end_lengths'].append(end_length)
                else:
                    coarse_datapoints[split]['input_ids'].append(input_ids)
                    coarse_datapoints[split]['attention_mask'].append(attn_mask)
                    coarse_datapoints[split]['labels'].append(labels)
                    if split == 'train':
                        coarse_datapoints[split]['decoder_input_ids'].append([])
                        coarse_datapoints[split]['sentiment'].append('positive')
                        coarse_datapoints[split]['end_lengths'].append(end_length)

    if encoder is not None:
        for split in fine_coarse_data.keys():
            fine_datapoints[split]['input_ids'] = encode_sequences(fine_datapoints[split], encoder)
            coarse_datapoints[split]['input_ids'] = encode_sequences(coarse_datapoints[split], encoder)

    input_ids_dtype = torch.float32 if embedded_history or encoder is not None else torch.long
    for split in fine_coarse_data.keys():
        fine_datapoints[split]['input_ids'] = torch.tensor(np.array(fine_datapoints[split]['input_ids']), dtype=input_ids_dtype)
        fine_datapoints[split]['attention_mask'] = torch.tensor(fine_datapoints[split]['attention_mask'], dtype=torch.long)
        fine_datapoints[split]['labels'] = torch.tensor(fine_datapoints[split]['labels'], dtype=torch.long)
        coarse_datapoints[split]['input_ids'] = torch.tensor(np.array(coarse_datapoints[split]['input_ids']), dtype=input_ids_dtype)
        coarse_datapoints[split]['attention_mask'] = torch.tensor(coarse_datapoints[split]['attention_mask'], dtype=torch.long)
        coarse_datapoints[split]['labels'] = torch.tensor(coarse_datapoints[split]['labels'], dtype=torch.long)
    return fine_datapoints, coarse_datapoints

def load_pos_neg_sets(dataset, tokenizer, item_2_semantic_id, only_attend_inst=True, semantic_ids_in_encoder=False,
                      embedded_history=False, embedding_model='instructor-base', load_train=True, encoder=None,
                      add_user_emb=False):
    assert os.path.exists(f'dataset/pos_neg_splits_{dataset}.pkl'), 'data for positive/negative item splits not found'
    if not embedded_history:
        with open(f'dataset/pos_neg_splits_{dataset}.pkl', 'rb') as f:
            data_file = pickle.load(f)
    else:
        with open(f'dataset/embedded_pos_neg_split_{dataset}_{embedding_model}.json', 'rb') as f:
            data_file = pickle.load(f)

    pos_neg_train_datapoints = defaultdict(defaultdict_list)
    val_datapoints = defaultdict(defaultdict_list)
    test_datapoints = defaultdict(defaultdict_list)
    dec_eos_id = 1025 if not add_user_emb else 3025
    for sentiment in data_file.keys():
        for split in data_file[sentiment].keys():
            if split == 'train' and not load_train:
                continue
            for index, (inst, item_id) in tqdm(enumerate(data_file[sentiment][split]), desc=f"Generating {sentiment} for {split} set..."):
                if isinstance(item_id, list):
                    item_id = item_id[0]

                if not embedded_history:
                    inst = prepare_instruction(inst, "", tokenizer.name_or_path)
                    input_ids = tokenizer.encode(inst, add_special_tokens=False)
                    end_length = [len(input_ids)]
                    attn_mask = [1] * len(input_ids)
                    
                    labels = expand_id(item_2_semantic_id[int(item_id)]) + [dec_eos_id]
                    if not only_attend_inst and not semantic_ids_in_encoder:
                        decoder_input_ids = labels
                    else:
                        decoder_input_ids = []
                    input_ids = pad_sequence_preferences(input_ids, tokenizer.model_max_length, tokenizer.pad_token_id,
                                                        tokenizer.eos_token_id, end_length)
                    attn_mask = pad_sequence_attention_preferences(attn_mask, tokenizer.model_max_length, end_length)
                else:
                    # instruct is already an embedding
                    input_ids, _ = generate_embedded_input_sequence([], inst, None, item_2_semantic_id, 20)
                    attn_mask = []
                    decoder_input_ids = []
                    labels = expand_id(item_2_semantic_id[int(item_id)]) + [dec_eos_id]
                    end_length = [1]

                if split == 'train':
                    pos_neg_train_datapoints[sentiment]['input_ids'].append(input_ids)
                    pos_neg_train_datapoints[sentiment]['sentiment'].append(sentiment)
                    pos_neg_train_datapoints[sentiment]['attention_mask'].append(attn_mask)
                    pos_neg_train_datapoints[sentiment]['decoder_input_ids'].append(decoder_input_ids)
                    pos_neg_train_datapoints[sentiment]['labels'].append(labels)
                    pos_neg_train_datapoints[sentiment]['end_lengths'].append(end_length)
                elif split == 'val':
                    val_datapoints[sentiment]['input_ids'].append(input_ids)
                    val_datapoints[sentiment]['attention_mask'].append(attn_mask)
                    val_datapoints[sentiment]['decoder_input_ids'].append(decoder_input_ids)
                    val_datapoints[sentiment]['labels'].append(labels)
                    val_datapoints[sentiment]['sentiment'].append(sentiment)
                    val_datapoints[sentiment]['end_lengths'].append(end_length)
                    val_datapoints[sentiment]['sample_index'].append(index)
                else:
                    test_datapoints[sentiment]['input_ids'].append(input_ids)
                    test_datapoints[sentiment]['attention_mask'].append(attn_mask)
                    test_datapoints[sentiment]['decoder_input_ids'].append(decoder_input_ids)
                    test_datapoints[sentiment]['labels'].append(labels)
                    test_datapoints[sentiment]['sentiment'].append(sentiment)
                    test_datapoints[sentiment]['end_lengths'].append(end_length)
                    test_datapoints[sentiment]['sample_index'].append(index)
    
    for sentiment in ['positive', 'negative']:
        if not only_attend_inst and not semantic_ids_in_encoder:
            pos_neg_train_datapoints[sentiment]['decoder_input_ids'] = torch.tensor(pos_neg_train_datapoints[sentiment]['decoder_input_ids'], dtype=torch.long)
            val_datapoints[sentiment]['decoder_input_ids'] = torch.tensor(val_datapoints[sentiment]['decoder_input_ids'], dtype=torch.long)
            test_datapoints[sentiment]['decoder_input_ids'] = torch.tensor(test_datapoints[sentiment]['decoder_input_ids'], dtype=torch.long)
       
        if encoder is not None:
            if load_train:
                pos_neg_train_datapoints[sentiment]['input_ids'] = encode_sequences(pos_neg_train_datapoints[sentiment], encoder)
            val_datapoints[sentiment]['input_ids'] = encode_sequences(val_datapoints[sentiment], encoder)
            test_datapoints[sentiment]['input_ids'] = encode_sequences(test_datapoints[sentiment], encoder)

        input_ids_dtype = torch.float32 if embedded_history or encoder is not None else torch.long
        pos_neg_train_datapoints[sentiment]['input_ids'] = torch.tensor(pos_neg_train_datapoints[sentiment]['input_ids'], dtype=input_ids_dtype)
        val_datapoints[sentiment]['input_ids'] = torch.tensor(np.array(val_datapoints[sentiment]['input_ids']), dtype=input_ids_dtype)
        test_datapoints[sentiment]['input_ids'] = torch.tensor(np.array(test_datapoints[sentiment]['input_ids']), dtype=input_ids_dtype)
        pos_neg_train_datapoints[sentiment]['attention_mask'] = torch.tensor(pos_neg_train_datapoints[sentiment]['attention_mask'], dtype=torch.long)
        pos_neg_train_datapoints[sentiment]['labels'] = torch.tensor(pos_neg_train_datapoints[sentiment]['labels'], dtype=torch.long)
        val_datapoints[sentiment]['attention_mask'] = torch.tensor(val_datapoints[sentiment]['attention_mask'], dtype=torch.long)
        val_datapoints[sentiment]['labels'] = torch.tensor(val_datapoints[sentiment]['labels'], dtype=torch.long)
        val_datapoints[sentiment]['sample_index'] = torch.tensor(val_datapoints[sentiment]['sample_index'], dtype=torch.long)
        test_datapoints[sentiment]['attention_mask'] = torch.tensor(test_datapoints[sentiment]['attention_mask'], dtype=torch.long)
        test_datapoints[sentiment]['labels'] = torch.tensor(test_datapoints[sentiment]['labels'], dtype=torch.long)
        test_datapoints[sentiment]['sample_index'] = torch.tensor(test_datapoints[sentiment]['sample_index'], dtype=torch.long)
    return pos_neg_train_datapoints, val_datapoints, test_datapoints

def augmented_collate(batch, tokenizer, p=0.1):
    batch_size = len(batch)
    to_augment = torch.bernoulli(torch.full((batch_size,), fill_value=p))
    to_augment_inds = torch.nonzero(to_augment)
    fine_cts = coarse_cts = 0.

    for src_ind in to_augment_inds:
        src_label = batch[src_ind]['labels'][:-1]
        # by default do fine-grained preference following
        tar_inds = [index for index in range(len(batch)) if (batch[index]['labels'][:-2] == src_label[:-1]).all() and index != src_ind]
        if len(tar_inds):
            fine_cts += 1
        else:
            # no very similar labels found, do coarse-grained instead
            tar_inds = [index for index in range(len(batch)) if not (batch[index]['labels'][:-1] == src_label).any() and index != src_ind]
    
        if not len(tar_inds):
            # if there is still no matching sample, skip this sample
            continue
        else:
            coarse_cts += 1

        # randomly choose a target index
        tar_ind = random.choice(tar_inds)
        input_ids, attn_mask, labels, _, sentiment = fine_coarse_instruct_augmentation(src_ind, tar_ind, batch, tokenizer)
        batch[src_ind]['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
        batch[src_ind]['attention_mask'] = torch.tensor(attn_mask, dtype=torch.long)
        batch[src_ind]['labels'] = torch.tensor(labels, dtype=torch.long)
        batch[src_ind]['sentiment'] = sentiment
    
    prepped_batch = { key: [] for key in batch[0].keys() }
    for ind in range(batch_size):
        for key in batch[ind].keys():
            if key != 'decoder_input_ids':
                prepped_batch[key].append(batch[ind][key])
    prepped_batch['input_ids'] = torch.stack(prepped_batch['input_ids'])
    prepped_batch['labels'] = torch.stack(prepped_batch['labels'])
    prepped_batch['attention_mask'] = torch.stack(prepped_batch['attention_mask'])
    prepped_batch['decoder_input_ids'] = []
    prepped_batch['counts'] = (fine_cts, coarse_cts)
    return prepped_batch

def default_collate(batch):
    batch_size = len(batch)
    prepped_batch = { key: [] for key in batch[0].keys() }
    for ind in range(batch_size):
        for key in batch[ind].keys():
            if key != 'decoder_input_ids':
                prepped_batch[key].append(batch[ind][key])
    prepped_batch['input_ids'] = torch.stack(prepped_batch['input_ids'])
    prepped_batch['labels'] = torch.stack(prepped_batch['labels'])
    prepped_batch['attention_mask'] = torch.stack(prepped_batch['attention_mask'])
    prepped_batch['decoder_input_ids'] = []
    return prepped_batch

def fine_coarse_instruct_augmentation(src_ind, tar_ind, data, tokenizer):

    src_end_lengths = data[src_ind]['end_lengths']
    tar_end_lengths = data[tar_ind]['end_lengths']
    tar_instruct = data[tar_ind]['input_ids'][:tar_end_lengths[0]].tolist()
    nz = torch.nonzero(data[src_ind]['input_ids'] == tokenizer.eos_token_id)[0].item()
    src_input_ids = data[src_ind]['input_ids'][src_end_lengths[0]:nz].tolist()
    new_end_lengths = [len(tar_instruct)] + (np.array(src_end_lengths[1:]) - src_end_lengths[0] + tar_end_lengths[0]).tolist()
    input_ids = pad_sequence_preferences(tar_instruct + src_input_ids, tokenizer.model_max_length, tokenizer.pad_token_id,
                                          tokenizer.eos_token_id, new_end_lengths)
    attn_mask = pad_sequence_attention_preferences([1] * len(tar_instruct + src_input_ids), tokenizer.model_max_length, new_end_lengths)
    tar_label = data[tar_ind]['labels']
    sentiment = data[tar_ind]['sentiment']
    return input_ids, attn_mask, tar_label, new_end_lengths, sentiment

def load_data_full_seqs(dataset, path, codebook_size, tokenizer, max_length=258, max_items_per_seq=np.inf, add_user_emb=False, 
                        most_recent=False, only_attend_inst=False):
    semantic_id_2_item = defaultdict(defaultdict_defaultdict_defaultdict_int)
    item_2_semantic_id = {}
    semantic_ids = pickle.load(open(path, 'rb'))

    for i in range(len(semantic_ids)):
        id = semantic_ids[i]
        id_dict = semantic_id_2_item[id[0]][id[1]][id[2]]
        id_dict[len(id_dict)] = i+1
        item_2_semantic_id[i+1] = (*id, len(id_dict))

    assert len(item_2_semantic_id) == semantic_ids.shape[0], "Not all semanticid -> item collisions have been avoided!"

    # create list of all semantic ids => needed for retrieval to avoid generation of invalid ids
    all_semantic_ids = np.array([item_2_semantic_id[idx] for idx in np.arange(1, len(semantic_ids)+1)])
    assert not (all_semantic_ids[..., :] > codebook_size).any(), "Fourth dimension of semantic ids exceeds vocabulary size, check proper RQ-VAE training!"
    expand_id_arr(all_semantic_ids)

    user_sequence = []
    users = []
    with open(f'{os.path.abspath(".")}/ID_generation/preprocessing/processed/{dataset}.txt', 'r') as f:
        for line in f.readlines():
            user_sequence.append([int(x) for x in line.split(' ')[1:]])
            users.append(line.split(' ')[0])
    if most_recent:
        assert max_items_per_seq < np.inf, "You must set max_items_per_seq to use most_recent items only"
        user_sequence = [seq if len(seq) <= max_items_per_seq else seq[-max_items_per_seq:] for seq in user_sequence]
    else:
        user_sequence = [seq if len(seq) <= max_items_per_seq else seq[:max_items_per_seq] for seq in user_sequence]
    max_sequence_length = max_length

    # load preference tuning data
    dataset = path.split("_")[0]
    assert os.path.exists(f'dataset/preference_dict_{dataset}.json'), f"preference tuning file for {dataset} not found"
    with open(f'dataset/preference_dict_{dataset}.json', 'r') as f:
        instruct_dict = json.load(f)

    training_data = {'input_ids': [], 'attention_mask': [], 'labels': [], 'decoder_input_ids': []}
    val_data = {'input_ids': [], 'attention_mask': [], 'labels': [], 'decoder_input_ids': []}
    test_data = {'input_ids': [], 'attention_mask': [], 'labels': [], 'decoder_input_ids': []}

    for i in range(len(user_sequence)):
        user_id = tokenizer.vocab_size + (1026 + i % 2000) if add_user_emb else None
        # cur_user = ind_to_user[i]
        train_enc_sequence = []
        train_dec_sequence = []
        train_attention_mask = []
        train_label = []
        val_enc_sequence = []
        val_dec_sequence = []
        val_attention_mask = []
        val_label = []
        test_enc_sequence = []
        test_dec_sequence = []
        test_attention_mask = []
        test_label = []
        
        # user sequence = [1,2,3,4,5]
        # train: j = 2,3 => [1,2], [1,2,3]
        # val: j = 4 =>[1,2,3,4]
        # test: j = 5 => [1,2,3,4,5]

        # cannot use first item for instruct tuning
        # if user_cold_start:
        #     # always take the whole sequence if we do cold-start on users
        #     instructs = [instruct_dict[str(users[i])][str(item)] for item in user_sequence[i][:-1]]
        #     train_instructs = instructs[-1][:3]
        #     train_seq = user_sequence[i][1:]
        instructs = [instruct_dict[str(users[i])][str(item)] for item in user_sequence[i][:-1]]
        train_instructs = instructs[-3][:3]
        val_instruct = instructs[-2][-2]
        test_instruct = instructs[-1][-1]
        test_seq = user_sequence[i][1:]
        train_seq = test_seq[:-2]
        val_seq = test_seq[:-1]

        for instruct in train_instructs:
            encoder_input_ids, attention_mask, decoder_input_ids, labels = generate_full_seq_with_instruct(train_seq, item_2_semantic_id, instruct, 
                                                                                    tokenizer, max_sequence_length, test_seq=False, user_id=user_id,
                                                                                    only_attend_inst=only_attend_inst)
            train_enc_sequence.append(encoder_input_ids)
            train_dec_sequence.append(decoder_input_ids)
            train_attention_mask.append(attention_mask)
            train_label.append(labels)    
        
        encoder_input_ids, attention_mask, decoder_input_ids, labels = generate_full_seq_with_instruct(val_seq, item_2_semantic_id, val_instruct, 
                                                                                    tokenizer, max_sequence_length, test_seq=True, user_id=user_id,
                                                                                    only_attend_inst=only_attend_inst)
        val_enc_sequence.append(encoder_input_ids)
        val_dec_sequence.append(decoder_input_ids)
        val_attention_mask.append(attention_mask)
        val_label.append(labels)
        
        encoder_input_ids, attention_mask, decoder_input_ids, labels = generate_full_seq_with_instruct(test_seq, item_2_semantic_id, test_instruct, 
                                                                            tokenizer, max_sequence_length, test_seq=True, user_id=user_id,
                                                                            only_attend_inst=only_attend_inst)
        test_enc_sequence.append(encoder_input_ids)
        test_dec_sequence.append(decoder_input_ids)
        test_attention_mask.append(attention_mask)
        test_label.append(labels)

        training_data['input_ids'].extend(train_enc_sequence)
        training_data['attention_mask'].extend(train_attention_mask)
        training_data['labels'].extend(train_label)
        training_data['decoder_input_ids'].extend(train_dec_sequence)
        val_data['input_ids'].extend(val_enc_sequence)
        val_data['decoder_input_ids'].extend(val_dec_sequence)
        val_data['attention_mask'].extend(val_attention_mask)
        val_data['labels'].extend(val_label)
        test_data['input_ids'].extend(test_enc_sequence)
        test_data['decoder_input_ids'].extend(test_dec_sequence)
        test_data['attention_mask'].extend(test_attention_mask)
        test_data['labels'].extend(test_label)

    if not only_attend_inst:
        # we additionally condition on items
        training_data['decoder_input_ids'] = torch.tensor(pad_labels(training_data['decoder_input_ids'], left_pad=True, mask=False, shift_right=True), dtype=torch.long)
        val_data['decoder_input_ids'] = torch.tensor(pad_labels(val_data['decoder_input_ids'], left_pad=True, mask=False, shift_right=True), dtype=torch.long)
        test_data['decoder_input_ids'] = torch.tensor(pad_labels(test_data['decoder_input_ids'], left_pad=True, mask=False, shift_right=True), dtype=torch.long)

    training_data['input_ids'] = torch.tensor(training_data['input_ids'], dtype=torch.long)
    training_data['attention_mask'] = torch.tensor(training_data['attention_mask'], dtype=torch.long)
    training_data['labels'] = torch.tensor(pad_labels(training_data['labels'], left_pad=True), dtype=torch.long)
    val_data['input_ids'] = torch.tensor(val_data['input_ids'], dtype=torch.long)
    val_data['attention_mask'] = torch.tensor(val_data['attention_mask'], dtype=torch.long)
    val_data['labels'] = torch.tensor(val_data['labels'], dtype=torch.long)
    test_data['input_ids'] = torch.tensor(test_data['input_ids'], dtype=torch.long)
    test_data['attention_mask'] = torch.tensor(test_data['attention_mask'], dtype=torch.long)
    test_data['labels'] = torch.tensor(test_data['labels'], dtype=torch.long)
    return training_data, val_data, test_data, all_semantic_ids

def load_data_itemids(dataset, max_length=258, max_items_per_seq=np.inf, use_first=False, most_recent=False):

    user_sequence = []
    with open(f'{os.path.abspath(".")}/ID_generation/preprocessing/processed/{dataset}.txt', 'r') as f:
        for line in f.readlines():
            user_sequence.append([int(x) for x in line.split(' ')[1:]])
    if most_recent:
        assert max_items_per_seq < np.inf, "You must set max_items_per_seq to use most_recent items only"
        user_sequence = [seq if len(seq) <= max_items_per_seq else seq[-max_items_per_seq:] for seq in user_sequence]
    else:
        user_sequence = [seq if len(seq) <= max_items_per_seq else seq[:max_items_per_seq] for seq in user_sequence]
    max_sequence_length = max_length
    n_items = np.unique(list(chain(*user_sequence))).shape[0]

    training_data = {'input_ids': [], 'attention_mask': [], 'labels': []}
    val_data = {'input_ids': [], 'attention_mask': [], 'labels': []}
    test_data = {'input_ids': [], 'attention_mask': [], 'labels': []}
    for i in range(len(user_sequence)):
        user_id = n_items + i % 2000
        train_sequence = []
        train_attention_mask = []
        train_label = []
        val_sequence = []
        val_attention_mask = []
        val_label = []
        test_sequence = []
        test_attention_mask = []
        test_label = []
        
        # user sequence = [1,2,3,4,5]
        # train: j = 2,3 => [1,2], [1,2,3]
        # val: j = 4 =>[1,2,3,4]
        # test: j = 5 => [1,2,3,4,5]
        start = 1 if use_first else 2
        for j in range(start, len(user_sequence[i])+1):
            input_ids, attention_mask, labels = generate_input_sequence_itemids(user_id, user_sequence[i][:j], max_sequence_length)
            if j == len(user_sequence[i]) - 1:
                val_sequence.append(input_ids)
                val_attention_mask.append(attention_mask)
                val_label.append(labels)
            elif j == len(user_sequence[i]):
                test_sequence.append(input_ids)
                test_attention_mask.append(attention_mask)
                test_label.append(labels)
            else:
                train_sequence.append(input_ids)
                train_attention_mask.append(attention_mask)
                train_label.append(labels)
        
        training_data['input_ids'].extend(train_sequence)
        training_data['attention_mask'].extend(train_attention_mask)
        training_data['labels'].extend(train_label)
        val_data['input_ids'].extend(val_sequence)
        val_data['attention_mask'].extend(val_attention_mask)
        val_data['labels'].extend(val_label)
        test_data['input_ids'].extend(test_sequence)
        test_data['attention_mask'].extend(test_attention_mask)
        test_data['labels'].extend(test_label)

    training_data['input_ids'] = torch.tensor(training_data['input_ids'], dtype=torch.long)
    training_data['attention_mask'] = torch.tensor(training_data['attention_mask'], dtype=torch.long)
    training_data['labels'] = torch.tensor(training_data['labels'], dtype=torch.long)
    val_data['input_ids'] = torch.tensor(val_data['input_ids'], dtype=torch.long)
    val_data['attention_mask'] = torch.tensor(val_data['attention_mask'], dtype=torch.long)
    val_data['labels'] = torch.tensor(val_data['labels'], dtype=torch.long)
    test_data['input_ids'] = torch.tensor(test_data['input_ids'], dtype=torch.long)
    test_data['attention_mask'] = torch.tensor(test_data['attention_mask'], dtype=torch.long)
    test_data['labels'] = torch.tensor(test_data['labels'], dtype=torch.long)
    return training_data, val_data, test_data, n_items

def load_data_item_cold_start(dataset, path, codebook_size, unseen_val, unseen_test, max_length=258, max_items_per_seq=np.inf, use_first=False, most_recent=False):
    semantic_id_2_item = defaultdict(defaultdict_defaultdict_defaultdict_int)
    item_2_semantic_id = {}
    semantic_ids = pickle.load(open(path, 'rb'))

    for i in range(len(semantic_ids)):
        id = semantic_ids[i]
        id_dict = semantic_id_2_item[id[0]][id[1]][id[2]]
        id_dict[len(id_dict)] = i+1
        item_2_semantic_id[i+1] = (*id, len(id_dict))
    # item_2_semantic_id: {item_id: [semantic_ids]}, item_id start from 1

    assert len(item_2_semantic_id) == semantic_ids.shape[0], "Not all semanticid -> item collisions have been avoided!"
    
    # check if last dimension of semantic ids exceeds the number of available codebook entries
    all_semantic_ids = np.array([item_2_semantic_id[idx] for idx in np.arange(1, len(semantic_ids)+1)])
    assert not (all_semantic_ids[..., :] > codebook_size).any(), "Fourth dimension of semantic ids exceeds vocabulary size, check proper RQ-VAE training!"
    
    # # split into train/val/test items
    # unseen_val_idxs = unseen_val
    # Sanity check is made after the dataloader, to see if the `unseen_val_data`'s label is in `val_unseen_semantic_ids`.
    # unseen_test_idxs = unseen_test
    unseens = np.concatenate((unseen_test, unseen_val))
    val_unseen_semantic_ids = np.array([item_2_semantic_id[idx] for idx in unseen_val])
    expand_id_arr(val_unseen_semantic_ids)
    test_unseen_semantic_ids = np.array([item_2_semantic_id[idx] for idx in unseen_test])
    expand_id_arr(test_unseen_semantic_ids)
    seen_semantic_ids = np.array([value for key, value in item_2_semantic_id.items() if key not in unseens])
    expand_id_arr(seen_semantic_ids)

    assert not any([(val_id == seen_semantic_ids).all() for val_id in val_unseen_semantic_ids]), "Overlap between seen and unseen val items"
    assert not any([(test_id == seen_semantic_ids).all() for test_id in val_unseen_semantic_ids]), "Overlap between seen and unseen val items"

    user_sequence = []
    with open(f'{os.path.abspath(".")}/ID_generation/preprocessing/processed/{dataset}.txt', 'r') as f:
        for line in f.readlines():
            user_sequence.append([int(x) for x in line.split(' ')[1:]])
    if most_recent:
        assert max_items_per_seq < np.inf, "You must set max_items_per_seq to use most_recent items only"
        user_sequence = [seq if len(seq) <= max_items_per_seq else seq[-max_items_per_seq:] for seq in user_sequence]
    else:
        user_sequence = [seq if len(seq) <= max_items_per_seq else seq[:max_items_per_seq] for seq in user_sequence]
    max_sequence_length = max_length

    training_data = {'input_ids': [], 'attention_mask': [], 'labels': []}
    val_data = {'input_ids': [], 'attention_mask': [], 'labels': []}
    unseen_val_data = {'input_ids': [], 'attention_mask': [], 'labels': []}
    test_data = {'input_ids': [], 'attention_mask': [], 'labels': []}
    unseen_test_data = {'input_ids': [], 'attention_mask': [], 'labels': []}
    for i in range(len(user_sequence)):
        user_id = 1025 + i % 2000  # use Hashing Trick to map the user to 2000 user tokens
        train_sequence = []
        train_attention_mask = []
        train_label = []
        val_sequence = []
        val_attention_mask = []
        val_label = []
        unseen_val_sequence = []
        unseen_val_attention_mask = []
        unseen_val_label = []
        test_sequence = []
        test_attention_mask = []
        test_label = []
        unseen_test_sequence = []
        unseen_test_attention_mask = []
        unseen_test_label = []
        
        # user sequence = [1,2,3,4,5]
        # train: j = 2,3 => [1,2], [1,2,3]
        # val: j = 4 =>[1,2,3,4]
        # test: j = 5 => [1,2,3,4,5]
        start = 1 if use_first else 2
        for j in range(2, len(user_sequence[i])+1):
            this_sequence = user_sequence[i][:j]
            if j == len(user_sequence[i]) - 1:
                # remove unseen test of the current trajectory
                mask_test = np.isin(this_sequence, unseen_test)
                this_seq_wo_test = np.array(this_sequence)[~mask_test]
                # remove every unseen val, except it is in the last position
                mask_val = np.isin(this_seq_wo_test, unseen_val)
                mask_val[-1] = False
                this_seq_wo_test_val = np.array(this_seq_wo_test)[~mask_val]
                
                if isvalid(this_seq_wo_test_val):
                    input_ids, attention_mask, labels = generate_input_sequence(user_id, this_seq_wo_test_val, item_2_semantic_id, max_sequence_length)
                    if this_seq_wo_test_val[-1] in unseen_val:
                    
                        unseen_val_sequence.append(input_ids)
                        unseen_val_attention_mask.append(attention_mask)
                        unseen_val_label.append(labels)
                    else:
                        val_sequence.append(input_ids)
                        val_attention_mask.append(attention_mask)
                        val_label.append(labels)

            elif j == len(user_sequence[i]):
                # test split: remove items that appear in unseen eval set first, and then remove the items in unseen test set (except the last one)
                # remove items in unseen eval because it is not trained
                mask_val = np.isin(this_sequence, unseen_val)
                this_seq_wo_val = np.array(this_sequence)[~mask_val]
                mask_test = np.isin(this_seq_wo_val, unseen_test)
                mask_test[-1] = False
                this_seq_wo_test_val = np.array(this_seq_wo_val)[~mask_test]
                if isvalid(this_seq_wo_test_val):    
                    input_ids, attention_mask, labels = generate_input_sequence(user_id, this_seq_wo_test_val, item_2_semantic_id, max_sequence_length)
                    if this_seq_wo_test_val[-1] in unseen_test:
                        unseen_test_sequence.append(input_ids)
                        unseen_test_attention_mask.append(attention_mask)
                        unseen_test_label.append(labels)
                    else:
                        test_sequence.append(input_ids)
                        test_attention_mask.append(attention_mask)
                        test_label.append(labels)

            else:
                # for training split: remove items that appear in unseen val/test set
                mask = np.isin(this_sequence, unseens)
                cur_seq = np.array(this_sequence)[~mask] if mask.any() else this_sequence
                if isvalid(cur_seq):
                    # cur_seq is [--context--, label] such that context is free of unseens, and label is either seen or unseen.
                    input_ids, attention_mask, labels = generate_input_sequence(user_id, cur_seq, item_2_semantic_id, max_sequence_length)
                    train_sequence.append(input_ids)
                    train_attention_mask.append(attention_mask)
                    train_label.append(labels)
        
        training_data['input_ids'].extend(train_sequence)
        training_data['attention_mask'].extend(train_attention_mask)
        training_data['labels'].extend(train_label)
        val_data['input_ids'].extend(val_sequence)
        val_data['attention_mask'].extend(val_attention_mask)
        val_data['labels'].extend(val_label)
        unseen_val_data['input_ids'].extend(unseen_val_sequence)
        unseen_val_data['attention_mask'].extend(unseen_val_attention_mask)
        unseen_val_data['labels'].extend(unseen_val_label)
        test_data['input_ids'].extend(test_sequence)
        test_data['attention_mask'].extend(test_attention_mask)
        test_data['labels'].extend(test_label)
        unseen_test_data['input_ids'].extend(unseen_test_sequence)
        unseen_test_data['attention_mask'].extend(unseen_test_attention_mask)
        unseen_test_data['labels'].extend(unseen_test_label)

    training_data['input_ids'] = torch.tensor(training_data['input_ids'], dtype=torch.long)
    training_data['attention_mask'] = torch.tensor(training_data['attention_mask'], dtype=torch.long)
    training_data['labels'] = torch.tensor(training_data['labels'], dtype=torch.long)
    val_data['input_ids'] = torch.tensor(val_data['input_ids'], dtype=torch.long)
    val_data['attention_mask'] = torch.tensor(val_data['attention_mask'], dtype=torch.long)
    val_data['labels'] = torch.tensor(val_data['labels'], dtype=torch.long)
    unseen_val_data['input_ids'] = torch.tensor(unseen_val_data['input_ids'], dtype=torch.long)
    unseen_val_data['attention_mask'] = torch.tensor(unseen_val_data['attention_mask'], dtype=torch.long)
    unseen_val_data['labels'] = torch.tensor(unseen_val_data['labels'], dtype=torch.long)
    test_data['input_ids'] = torch.tensor(test_data['input_ids'], dtype=torch.long)
    test_data['attention_mask'] = torch.tensor(test_data['attention_mask'], dtype=torch.long)
    test_data['labels'] = torch.tensor(test_data['labels'], dtype=torch.long)
    unseen_test_data['input_ids'] = torch.tensor(unseen_test_data['input_ids'], dtype=torch.long)
    unseen_test_data['attention_mask'] = torch.tensor(unseen_test_data['attention_mask'], dtype=torch.long)
    unseen_test_data['labels'] = torch.tensor(unseen_test_data['labels'], dtype=torch.long)
    return training_data, val_data, test_data, val_unseen_semantic_ids, test_unseen_semantic_ids, unseen_val_data, unseen_test_data, all_semantic_ids
    