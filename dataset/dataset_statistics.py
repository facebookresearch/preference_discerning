"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import json
import pickle
from argparse import ArgumentParser
from collections import defaultdict

def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--dataset', choices=['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games', 'yelp', 'steam'])
    return parser.parse_args()

def main(args):
    with open(f'pos_neg_dict_{args.dataset}.pkl', 'rb') as f:
        pos_neg_dict = pickle.load(f)

    pos_ct = 0
    neg_ct = 0
    for user in pos_neg_dict.keys():
        for item in pos_neg_dict[user].keys():
            for sentiment in pos_neg_dict[user][item]:
                if sentiment == 'positive':
                    pos_ct += 1
                else:
                    neg_ct += 1
    
    print(f"Number of generated preferences: {pos_ct + neg_ct}")
    print(f"Number of generated positive preferences: {pos_ct}")
    print(f"Number of generated negative preferences: {neg_ct}")

    with open(f'fine_coarse_preference_splits_{args.dataset}.pkl', 'rb') as f:
        fine_coarse = pickle.load(f)

    ct_dict = defaultdict(dict)
    for split in fine_coarse.keys():
        for kind in fine_coarse[split].keys():
            ct_dict[split][kind] = len(fine_coarse[split][kind])

    print("Fine/coarse counts: ")
    print(ct_dict)

    with open(f'pos_neg_splits_{args.dataset}.pkl', 'rb') as f:
        pos_neg = pickle.load(f)

    ct_dict = defaultdict(dict)
    for sentiment in pos_neg.keys():
        for split in pos_neg[sentiment].keys():
            ct_dict[split][sentiment] = len(pos_neg[sentiment][split])

    print("Pos/neg counts: ")
    print(ct_dict)

if __name__ == '__main__':
    main(create_parser())
