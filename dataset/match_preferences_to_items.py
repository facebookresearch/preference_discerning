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
from collections import defaultdict
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
parent_dir = os.path.dirname('/'.join(os.path.realpath(__file__).split("/")[:-1]))
import sys
sys.path.append(parent_dir)
from utils import defaultdict_list

def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--dataset', choices=['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games', 'yelp', 'steam'])
    parser.add_argument('--type', default='default', type=str, choices=['default', 'granular', 'coarse'])
    parser.add_argument('--cache_dir', type=str, required=True)
    return parser.parse_args()

def main(args):

    with open(f'./pos_neg_dict_{args.dataset}.pkl', 'rb') as f:
        pos_neg_dict = pickle.load(f)

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
    
    # load preference data
    if args.type == 'coarse':
        assert os.path.exists(f'./preference_dict_{args.dataset}_prompt_v1_coarser.json'), f"preference tuning file for {args.dataset} not found"
        with open(f'./preference_dict_{args.dataset}_prompt_v1_coarser.json', 'r') as f:
            pref_dict = json.load(f)
    elif args.type == 'granular':
        assert os.path.exists(f'./preference_dict_{args.dataset}_prompt_v1_granular.json'), f"preference tuning file for {args.dataset} not found"
        with open(f'./preference_dict_{args.dataset}_prompt_v1_granular.json', 'r') as f:
            pref_dict = json.load(f)
    else:
        with open(f'./preference_dict_{args.dataset}.json', 'r') as f:
            pref_dict = json.load(f)

    # load metadata
    with open(f'item2review_{args.dataset}.json', 'r') as f:
        item2meta = json.load(f)

    item2title = {}
    for uid in item2meta.keys():
        for item in item2meta[uid]:
            if item['itemid'] not in item2title:
                if args.dataset == 'yelp':
                    item2title[item['itemid']] = f"Name: {item['name']} Categories: {item['categories']} "
                    if 'attributes' in item:
                        item2title[item['itemid']] += f"Attributes: {item['attributes']}"
                elif args.dataset == "steam":
                    item2title[item['itemid']] = f"Title: {item['title']} Tags: {item['tags']} Genre: {item['genre']}"
                else:    
                    item2title[item['itemid']] = f"Title: {item['title']}."

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
    text_embedding_model = SentenceTransformer(f'sentence-transformers/sentence-t5-large').to(device)
    bs = 32
    matched_pref_dict = defaultdict(dict)
    for user_ind in tqdm(range(len(user_sequence)), desc='Matching preferences to items...'):

        for j in range(2, len(user_sequence[user_ind])+1):

            if args.dataset == 'steam':
                prefs = [pref_dict[str(users[user_ind])][str(item)][k] for item in user_sequence[user_ind][:j-1] for k in range(5)
                                  if pos_neg_dict[str(users[user_ind])][str(item)][k] == 'positive']
            else:
                prefs = [pref_dict[str(users[user_ind])][str(user_sequence[user_ind][j-2])][k] for k in range(5)
                         if pos_neg_dict[str(users[user_ind])][str(user_sequence[user_ind][j-2])][k] == 'positive']

            if not len(prefs):
                # all preferences are negative, do rule-based inversion
                for p in pref_dict[str(users[user_ind])][str(user_sequence[user_ind][j-2])]:
                    if p.lower().startswith('avoid'):
                        prefs.append(p.lower().replace('avoid', 'Find'))
                    elif p.lower().startswith('exclude'):
                        prefs.append(p.lower().replace('exclude', 'Search for'))
                    elif p.lower().startswith('no'):
                        prefs.append(p.lower().replace('no ', 'Search for'))
                if not len(prefs):
                    # no rule-based inversion possible, just take all preferences
                    prefs = pref_dict[str(users[user_ind])][str(user_sequence[user_ind][j-2])]

            title = [item2title[str(user_sequence[user_ind][j-1])]]
            to_embed = prefs + title
            embeddings = text_embedding_model.encode(to_embed, convert_to_numpy=True, batch_size=bs, show_progress_bar=False)
            pref_embs = embeddings[:-1]
            title_emb = embeddings[-1]
            pref_ind = (pref_embs @ title_emb).argmax()
            matched_pref_dict[str(users[user_ind])][str(user_sequence[user_ind][j-2])] = prefs[pref_ind]

    if args.type == 'default':
        with open(f'matched_preference_dict_{args.dataset}_all_prev_prefs.json', 'w') as f:
            json.dump(matched_pref_dict, f)
    elif args.type == 'coarse':
        with open(f'matched_preference_dict_{args.dataset}_coarse.json', 'w') as f:
            json.dump(matched_pref_dict, f)
    else:
        with open(f'matched_preference_dict_{args.dataset}_granular.json', 'w') as f:
            json.dump(matched_pref_dict, f)


if __name__ == '__main__':
    main(create_parser())
