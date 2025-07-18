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
from instruct_encoder import InstructEncoderModel, get_instruction_preference

def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--dataset', choices=['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games', 'yelp', 'steam'])
    parser.add_argument('--embedding_model', type=str, required=True, choices=['sentence-transformers/sentence-t5-xxl', 'sentence-transformers/sentence-t5-base',
                                                                               'hkunlp/instructor-base', 'hkunlp/instructor-xl', 'hkunlp/instructor-large',
                                                                               'GritLM/GritLM-7B', 'google/flan-t5-small', 'google/flan-t5-base', 'google/flan-t5-large',
                                                                            'google/flan-t5-xl', 'google/flan-t5-xxl', 'hyp1231/blair-roberta-base', 'hyp1231/blair-roberta-large'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--cache_dir', type=str, required=True)
    return parser.parse_args()


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if 'sentence-t5' in args.embedding_model:
        os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
        text_embedding_model = SentenceTransformer(args.embedding_model).to(device)
    else:
        pooling_method = "cls" if 'roberta' in args.embedding_model else "mean"
        text_embedding_model = InstructEncoderModel(args.embedding_model, mode='embedding', pooling_method=pooling_method,
                                                    cache_dir=args.cache_dir)

    for dataset in ['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games', 'yelp', 'steam']:

        with open(f'pos_neg_splits_{dataset}.pkl', 'rb') as f:
            instruct_dict = pickle.load(f)

        to_embed_list = []
        user_and_item_ids = []
        for sentiment in instruct_dict.keys():
            for split in instruct_dict[sentiment].keys():
                for inst, item_id in instruct_dict[sentiment][split]:
                    to_embed_list.extend([inst])
                    user_and_item_ids.extend([(sentiment, split, item_id)])
                
        if 't5' in args.embedding_model:
            embeddings = text_embedding_model.encode(to_embed_list, convert_to_numpy=True, batch_size=args.batch_size, show_progress_bar=True)
        else:
            embeddings = text_embedding_model.encode(to_embed_list, instruction=get_instruction_preference(args.embedding_model), batch_size=args.batch_size)

        emb_dict = defaultdict(dict)
        for emb, (sentiment, split, iid) in zip(embeddings, user_and_item_ids):
            # only have a single embedding
            if split not in emb_dict[sentiment]:
                emb_dict[sentiment][split] = []
            emb_dict[sentiment][split].append((emb, iid))
            
        with open(f'embedded_pos_neg_split_{dataset}_{args.embedding_model.split("/")[-1]}.json', 'wb') as f:
            pickle.dump(emb_dict, f)

        torch.cuda.empty_cache()
    

if __name__ == '__main__':
    main(create_parser())
