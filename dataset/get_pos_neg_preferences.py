"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import json
import tqdm
import pickle
from collections import defaultdict
from argparse import ArgumentParser

import torch
torch.set_float32_matmul_precision('high')
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SENTIMENT_MAP = {1: "positive", 0: 'negative'}

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', choices=['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games', 'yelp', 'steam'], required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cache_dir', type=str, required=True)
    return parser.parse_args()

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(f"preference_dict_{args.dataset}.json", "r") as f:
        preference_dict = json.load(f)

    # load pre-trained sentiment classification model
    tokenizer = AutoTokenizer.from_pretrained('siebert/sentiment-roberta-large-english')
    sentiment_model = AutoModelForSequenceClassification.from_pretrained('siebert/sentiment-roberta-large-english',
                                                                         cache_dir=args.cache_dir)
    sentiment_model = torch.compile(sentiment_model)
    sentiment_model = sentiment_model.to(device)
    sentiment_model.eval()

    preference_list = []
    for user in preference_dict.keys():
        for item in preference_dict[user].keys():
            for inst in preference_dict[user][item]:
                preference_list.append((inst, user, item))

    pos_neg_dict = defaultdict(dict)
    pos_ct = 0
    neg_ct = 0
    for i in tqdm.trange(0, len(preference_list), args.batch_size):
        batch = [inst[0] for inst in preference_list[i:i+args.batch_size]]
        user_ids = [inst[1] for inst in preference_list[i:i+args.batch_size]]
        item_ids = [inst[-1] for inst in preference_list[i:i+args.batch_size]]
        tokenized = tokenizer(batch, padding='longest', truncation=True, return_tensors='pt')
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        with torch.no_grad():
            output = softmax(sentiment_model(**tokenized).logits, dim=-1).argmax(dim=-1)
        pos_ct += output.sum()
        neg_ct += len(output) - output.sum()
        for user, item, sentiment in zip(user_ids, item_ids, output):
            if item not in pos_neg_dict[user]:
                pos_neg_dict[user][item] = []
            pos_neg_dict[user][item].append('positive' if sentiment else 'negative')

    print("Total number of preferences: ", pos_ct + neg_ct)
    print(f"Fraction of positive preferences: {pos_ct / (pos_ct + neg_ct)}")
    print(f"Fraction of negative preferences: {neg_ct / (pos_ct + neg_ct)}")

    with open(f'./pos_neg_dict_{args.dataset}.pkl', 'wb') as f:
        pickle.dump(pos_neg_dict, f)

if __name__ == '__main__':
    main(parse_args())
