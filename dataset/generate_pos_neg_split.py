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
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import os

def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--dataset', choices=['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games', 'yelp', 'steam'])
    return parser.parse_args()

def is_negative(instruct):
    return instruct.lower().startswith('avoid') or instruct.lower().startswith('no') or instruct.lower().startswith('exclude')

def main(args):
    assert os.path.exists(f'preference_dict_{args.dataset}.json'), "Matched preference dict not found"
    with open(f'preference_dict_{args.dataset}.json') as f:
        instruct_dict = json.load(f)

    with open(f'pos_neg_review_dict_{args.dataset}.pkl', 'rb') as f:
        pos_neg_review_dict = pickle.load(f)

    with open(f'item2review_{args.dataset}.json', 'r') as f:
        item2review = json.load(f)

    users = []
    with open(f'../ID_generation/preprocessing/processed/{args.dataset}.txt', 'r') as f:
        for line in f.readlines():
            users.append(line.split(' ')[0])
    if args.dataset == "steam":
        users = users[::7]
    item2review = {u: item2review[u][-20:] for u in users}

    train_instructs = []
    train_neg = 0
    for user_id in instruct_dict.keys():
        for item_id in instruct_dict[user_id]:
            train_instructs.extend(instruct_dict[user_id][item_id][:3])
            train_neg += np.sum([is_negative(inst) for inst in instruct_dict[user_id][item_id]])

    print(f"Total number of preferences: ", len(train_instructs))
    print(f"Fraction of negatives: ", train_neg / len(train_instructs))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    text_embedding_model = SentenceTransformer(f'sentence-transformers/sentence-t5-large').to(device)
    pos_neg_preference_data = defaultdict(dict)
    for user_id in tqdm(item2review.keys()):
        item_seq = item2review[user_id]
        test_instruct_item = item_seq[-1]['itemid']
        val_instruct_item = item_seq[-2]['itemid']
        items = [(ind, item) for ind, item in enumerate(item_seq) if pos_neg_review_dict[user_id][item['itemid']][0] == 'negative']
        for seq_ind, item in items:
            cur_item_id = item['itemid']
            if cur_item_id == test_instruct_item:
                set = 'test'
            elif cur_item_id == val_instruct_item:
                set = 'val'
            else:
                set = 'train'

            if set not in pos_neg_preference_data['positive']:
                pos_neg_preference_data['positive'][set] = []
            if set not in pos_neg_preference_data['negative']:
                pos_neg_preference_data['negative'][set] = []

            # collect all preferences from current point onward and check what negative preference was caused by the item
            all_insts = [inst for iid in item_seq[seq_ind:] for inst in instruct_dict[user_id][iid['itemid']]]
            neg_insts = np.nonzero([is_negative(inst) for inst in all_insts])[0]
            all_neg_insts = np.array(all_insts)[neg_insts]
            if not len(neg_insts):
                # if there is no negative review, leave this item out
                continue

            if len(neg_insts) == 1:
                # We have a single negative preference and a single negative review => add (preference, item) pair

                # negative preferences only start with either of Avoid, Exclude, or No
                if all_insts[neg_insts.item()].lower().startswith('avoid'):
                    inverted_instruct = all_insts[neg_insts.item()].lower().replace('avoid', 'Find')
                elif all_insts[neg_insts.item()].lower().startswith('exclude'):
                    inverted_instruct = all_insts[neg_insts.item()].lower().replace('exclude', 'Search for')
                else:
                    inverted_instruct = all_insts[neg_insts.item()].lower().replace('no ', 'Search for')
                
                pos_neg_preference_data['negative'][set].append((all_insts[neg_insts.item()], cur_item_id))
                pos_neg_preference_data['positive'][set].append((inverted_instruct, cur_item_id))
            else:
                # we have multiple negative preferences and multiple items in this sequence => perform matching
                to_embed = all_neg_insts.tolist()
                emb_dim = text_embedding_model[1].word_embedding_dimension
                instruct_inds = len(to_embed)
                item_repr = ''
                keys = ['review', 'title'] if args.dataset in ['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games'] else ['review']
                for key in keys:
                    item_repr += f"{key.capitalize()}: {item[key]}, "
                to_embed.extend([item_repr])

                with torch.no_grad():
                    embeddings = text_embedding_model.encode(to_embed, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
                    inst_embs = embeddings[:instruct_inds].reshape(-1, emb_dim)
                    review_embs = embeddings[instruct_inds:].reshape(-1, emb_dim)
                    out = (inst_embs @ review_embs.T).argmax()
                    
                # negative preferences only start with either of Avoid, Exclude, or No
                if to_embed[out].lower().startswith('avoid'):
                    inverted_instruct = to_embed[out].lower().replace('avoid', 'Find')
                elif to_embed[out].lower().startswith('exclude'):
                    inverted_instruct = to_embed[out].lower().replace('exclude', 'Search for')
                else:
                    inverted_instruct = to_embed[out].lower().replace('no ', 'Search for')

                pos_neg_preference_data['positive'][set].append((inverted_instruct, cur_item_id))
                pos_neg_preference_data['negative'][set].append((to_embed[out], cur_item_id))

    unique_dict = defaultdict(dict)
    for split in pos_neg_preference_data['positive'].keys():
        n_original = len(pos_neg_preference_data['positive'][split])
        _, idx = np.unique([sample[0] for sample in pos_neg_preference_data['positive'][split]], return_index=True)
        print(f"Split: {split}, Number of samples: {n_original}, Number of unique samples: {len(idx)}")
        for sentiment in pos_neg_preference_data.keys():
            unique_dict[sentiment][split] = [pos_neg_preference_data[sentiment][split][ind] for ind in np.sort(idx)]
    
    print("Number of positive train datapoints: ", len(unique_dict['positive']['train']))
    print("Number of positive val datapoints: ", len(unique_dict['positive']['val']))
    print("Number of positive test datapoints: ", len(unique_dict['positive']['test']))

    print("Number of negative train datapoints: ", len(unique_dict['negative']['train']))
    print("Number of negative val datapoints: ", len(unique_dict['negative']['val']))
    print("Number of negative test datapoints: ", len(unique_dict['negative']['test']))

    with open(f'pos_neg_splits_{args.dataset}.pkl', 'wb') as f:
        pickle.dump(unique_dict, f)

if __name__ == '__main__':
    main(create_parser())
