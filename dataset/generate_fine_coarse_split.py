"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import json
import pickle
from argparse import ArgumentParser
import numpy as np
np.random.seed(101)
from collections import defaultdict
import os
from tqdm import tqdm
import torch
from itertools import chain

def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--dataset', choices=['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games', 'yelp', 'steam'])
    parser.add_argument('--embedding_model', required=False, type=str, default='sentence-t5-xxl')
    return parser.parse_args()

def main(args):

    user_sequence = []
    users = []
    with open(f'../ID_generation/preprocessing/processed/{args.dataset}.txt', 'r') as f:
        for line in f.readlines():
            user_sequence.append([int(x) for x in line.split(' ')[1:]])
            users.append(line.split(' ')[0])
    user_sequence = [seq if len(seq) <= 20 else seq[-20:] for seq in user_sequence]
    if args.dataset == "steam":
        user_sequence = user_sequence[::7]
        users = users[::7]
    train_items = [user_sequence[i][:-2] for i in range(len(user_sequence))]
    val_items = np.array([user_sequence[i][-2] for i in range(len(user_sequence))])
    test_items = np.array([user_sequence[i][-1] for i in range(len(user_sequence))])
    
    # load preference data
    assert os.path.exists(f'./preference_dict_{args.dataset}.json'), f"preference file for {args.dataset} not found"
    with open(f'./preference_dict_{args.dataset}.json', 'r') as f:
        pref_dict = json.load(f)

    with open(f'./matched_preference_dict_{args.dataset}.json', 'r') as f:
        matched_pref_dict = json.load(f)

    with open(f'embedded_all_single_preference_dict_{args.dataset}_{args.embedding_model.split("/")[-1]}.json', 'rb') as f:
        embedded_pref_dict = pickle.load(f)

    with open(f'embedded_matched_preference_dict_{args.dataset}_{args.embedding_model.split("/")[-1]}.json', 'rb') as f:
        embedded_matched_pref_dict = pickle.load(f)

    pref_emb_map = {}
    seen_prefs = set()
    unseen_prefs = set()
    for user in pref_dict.keys():
        for item in pref_dict[user].keys():
            if item in matched_pref_dict[user]:
                seen_prefs.add(matched_pref_dict[user][item])
            for i, p in enumerate(pref_dict[user][item]):
                if item in matched_pref_dict[user] and p != matched_pref_dict[user][item]:
                    unseen_prefs.add(p)
                if not p in pref_emb_map:
                    pref_emb_map[p] = embedded_pref_dict[user][item][i]
                if item in matched_pref_dict[user] and (not matched_pref_dict[user][item] in pref_emb_map):
                    pref_emb_map[matched_pref_dict[user][item]] = embedded_matched_pref_dict[user][item]

    if not os.path.exists('./seen_preferences.pkl'):
        with open('./seen_preferences.pkl', 'wb') as f:
            pickle.dump(seen_prefs, f)
        with open('./unseen_preferences.pkl', 'wb') as f:
            pickle.dump(unseen_prefs, f)

    all_train_items = list(chain(*train_items))
    with open(f'embedded_items_dict_{args.dataset}_{args.embedding_model.split("/")[-1]}.json', 'rb') as f:
        embedded_items_dict = pickle.load(f)
        embedded_train_items_dict = { k: v for k, v in embedded_items_dict.items() if int(k) in all_train_items}
        embedded_val_items_dict = { k: v for k, v in embedded_items_dict.items() if int(k) in val_items}
        embedded_test_items_dict = { k: v for k, v in embedded_items_dict.items() if int(k) in test_items}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_item_vector_idx = torch.tensor(np.array(list(embedded_train_items_dict.values()))).to(device)
    val_item_vector_idx = torch.tensor(np.array(list(embedded_val_items_dict.values()))).to(device)
    test_item_vector_idx = torch.tensor(np.array(list(embedded_test_items_dict.values()))).to(device)
    train_item_keys = list(embedded_train_items_dict.keys())
    val_item_keys = list(embedded_val_items_dict.keys())
    test_item_keys = list(embedded_test_items_dict.keys())
    preference_vector_idx = torch.tensor(np.array(list(pref_emb_map.values()))).to(device)
    preference_keys = list(pref_emb_map.keys())
    splits = {'train': train_items, 'val': val_items, 'test': test_items}
    split_vectors = {'train': train_item_vector_idx, 'val': val_item_vector_idx, 'test': test_item_vector_idx}
    split_item_keys = {'train': train_item_keys, 'val': val_item_keys, 'test': test_item_keys}
    new_data_splits = defaultdict(dict)
    for split in splits.keys():
        prev_src_items = []
        if 'fine' not in new_data_splits[split]:
            new_data_splits[split]['fine'] = []
        if 'coarse' not in new_data_splits[split]:
            new_data_splits[split]['coarse'] = []
        fine_seen_cts, fine_unseen_cts = 0., 0.
        coarse_seen_cts, coarse_unseen_cts = 0., 0.

        for src_user_ind, src_item in tqdm(enumerate(splits[split]), desc=f'Assemble fine/coarse data for {split} split'):

            src_items = [src_item] if not isinstance(src_item, list) else src_item
            for src_item in src_items:

                if src_item in prev_src_items:
                    # ensure balanced dataset by avoiding duplicate ground truth items
                    # also skip if a val/test item appears in training set to avoid contamination
                    continue
                
                prev_src_items.append(src_item)
                # find a new user preference that fits the item
                src_item_emb = torch.tensor(embedded_items_dict[str(src_item)]).to(device)
                src_item_pos = user_sequence[src_user_ind].index(src_item)
                if split == 'train':
                    closest_pref_idx = (src_item_emb @ preference_vector_idx.T).argmax()
                    closest_pref_idx = closest_pref_idx.squeeze().item()
                else:
                    # take the second-closest preference to the item for val/test split to avoid contamination
                    closest_pref_idx = torch.argsort(src_item_emb @ preference_vector_idx.T, descending=True)[1]

                src_pref = preference_keys[closest_pref_idx]
                # find a very similar ground truth item
                closest_item_idx = torch.argsort(src_item_emb @ split_vectors[split].T, descending=True)[1]
                tar_item = split_item_keys[split][closest_item_idx]
                # get target preference
                tar_item_emb = torch.tensor(embedded_items_dict[str(tar_item)]).to(device)

                closest_pref_idx = torch.argsort(tar_item_emb @ preference_vector_idx.T, descending=True)
                if split == "train":
                    tar_pref = preference_keys[closest_pref_idx[0]] if preference_keys[closest_pref_idx[0]] != src_pref else preference_keys[closest_pref_idx[1]]
                else:
                    tar_pref = preference_keys[closest_pref_idx[2]]

                if split != 'train':
                    tar_user_ind = np.nonzero(splits[split] == int(tar_item))[0]
                else:
                    tar_user_ind = [i for i, tmp in enumerate(splits[split]) if int(tar_item) in tmp]
                if len(tar_user_ind) > 1:
                    tar_user_ind = np.random.choice(tar_user_ind, size=1).item()
                else:
                    tar_user_ind = tar_user_ind[0]

                # preference + item to predict stays the same, user is swapped
                if src_pref in seen_prefs:
                    fine_seen_cts += 1
                else:
                    fine_unseen_cts += 1
                if tar_pref in seen_prefs:
                    fine_seen_cts +=1
                else:
                    fine_unseen_cts +=1

                if split != 'train':
                    new_src_sample = (src_pref, tar_user_ind, src_item)
                    new_tar_sample = (tar_pref, src_user_ind, tar_item)
                else:
                    new_src_sample = (src_pref, tar_user_ind, src_item, src_item_pos)
                    new_tar_sample = (tar_pref, src_user_ind, tar_item, src_item_pos)
                new_data_splits[split]['fine'].extend([new_src_sample, new_tar_sample])
                
                # find a very distinct ground truth item
                distinct_item_idx = (src_item_emb @ split_vectors[split].T).argmin().item()
                tar_item = split_item_keys[split][distinct_item_idx]
                # get target preference
                tar_item_emb = torch.tensor(embedded_items_dict[str(tar_item)]).to(device)
                closest_pref_idx = torch.argsort(tar_item_emb @ preference_vector_idx.T, descending=True)
                if split == "train":
                    tar_pref = preference_keys[closest_pref_idx[0]] if preference_keys[closest_pref_idx[0]] != src_pref else preference_keys[closest_pref_idx[1]]
                else:
                    tar_pref = preference_keys[closest_pref_idx[2]]

                if split != 'train':
                    tar_user_ind = np.nonzero(splits[split] == int(tar_item))[0]
                else:
                    tar_user_ind = [i for i, tmp in enumerate(splits[split]) if int(tar_item) in tmp]
                if len(tar_user_ind) > 1:
                    tar_user_ind = np.random.choice(tar_user_ind, size=1).item()
                else:
                    tar_user_ind = tar_user_ind[0]
                
                # preference + item to predict stays the same, user is swapped
                if src_pref in seen_prefs:
                    coarse_seen_cts += 1
                else:
                    coarse_unseen_cts += 1
                if tar_pref in seen_prefs:
                    coarse_seen_cts += 1
                else:
                    coarse_unseen_cts += 1

                if split != 'train':
                    new_src_sample = (src_pref, tar_user_ind, src_item)
                    new_tar_sample = (tar_pref, src_user_ind, tar_item)
                else:
                    new_src_sample = (src_pref, tar_user_ind, src_item, src_item_pos)
                    new_tar_sample = (tar_pref, src_user_ind, tar_item, src_item_pos)
                new_data_splits[split]['coarse'].extend([new_src_sample, new_tar_sample])

        print(f"Number of seen samples for fine-grained evaluation:", fine_seen_cts)
        print(f"Number of unseen samples for fine-grained evaluation:", fine_unseen_cts)

        print(f"Number of seen samples for coarse-grained evaluation:", coarse_seen_cts)
        print(f"Number of unseen samples for coarse-grained evaluation:", coarse_unseen_cts)
    
    print(f"Number of validation samples for fine-grained evaluation:", len(new_data_splits['val']['fine']))
    print(f"Number of test samples for fine-grained evaluation:", len(new_data_splits['test']['fine']))

    print(f"Number of validation samples for coarse-grained evaluation:", len(new_data_splits['val']['coarse']))
    print(f"Number of test samples for coarse-grained evaluation:", len(new_data_splits['test']['coarse']))

    with open(f'fine_coarse_preference_splits_{args.dataset}.pkl', 'wb') as f:
        pickle.dump(new_data_splits, f)

if __name__ == '__main__':
    main(create_parser())
