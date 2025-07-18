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
from typing import List, cast, Union
from tqdm import tqdm
import os

import torch
from transformers import AutoModel, T5EncoderModel, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import models, SentenceTransformer
from instruct_encoder import InstructEncoderModel, get_instruction_item

def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--embedding_model', type=str, required=True, choices=['sentence-transformers/sentence-t5-xxl', 'sentence-transformers/sentence-t5-base',
                                                                            'hkunlp/instructor-base', 'hkunlp/instructor-large', 'hkunlp/instructor-xl',
                                                                            'GritLM/GritLM-7B', 'google/flan-t5-small', 'google/flan-t5-base', 'google/flan-t5-large',
                                                                            'google/flan-t5-xl', 'google/flan-t5-xxl', 'hyp1231/blair-roberta-base', 'hyp1231/blair-roberta-large'])
    parser.add_argument('--add_properties', action="store_true")
    parser.add_argument('--cache_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    return parser.parse_args()
    
def main(args):
    pooling_method = "cls" if 'roberta' in args.embedding_model else "mean"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if 'sentence-t5' in args.embedding_model:
        os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
        text_embedding_model = SentenceTransformer(args.embedding_model).to(device)
    else:
        text_embedding_model = InstructEncoderModel(args.embedding_model, cache_dir=args.cache_dir, mode='embedding', pooling_method=pooling_method)
    for dataset in ['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games', 'yelp', 'steam']:
        # load metadata
        with open(f'item2review_{dataset}.json', 'r') as f:
            item2meta = json.load(f)
        
        if args.add_properties:
            with open(f'reviews_to_properties_{dataset}.json', 'r') as f:
                item2prop = json.load(f)

        item2title = {}
        for uid in item2meta.keys():
            for item in item2meta[uid]:
                if item['itemid'] not in item2title:
                    if dataset in ['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games']:
                        if not 'roberta' in args.embedding_model:
                            if args.add_properties:
                                item2title[item['itemid']] = f"Title: {item['title']}. Properties: {', '.join(item2prop[item['itemid']])}"
                            else:
                                item2title[item['itemid']] = f"Title: {item['title']}."
                        else:
                            item2title[item['itemid']] = item['title']
                    elif dataset == 'yelp':
                        item2title[item['itemid']] = ''
                        for key in ['name', 'city', 'state']:
                            if key in item:
                                if 'roberta' in args.embedding_model:
                                    item2title[item['itemid']] += f" | {item[key]}"
                                else:
                                    item2title[item['itemid']] += f"{key.capitalize():}: {item[key]}"
                    else:
                        if 'roberta' in args.embedding_model:
                            item2title[item['itemid']] = f"{item['title']} | {item['tags']} | {item['genre']}"
                        else:
                            item2title[item['itemid']] = f"Title: {item['title']}, Tags: {item['tags']}, Genre: {item['genre']}"

        to_embed_list = list(item2title.values())
        item_ids = list(item2title.keys())

        if 'sentence-t5' in args.embedding_model:
            d_rep = text_embedding_model.encode(to_embed_list, convert_to_numpy=True, batch_size=args.batch_size, show_progress_bar=True)
        else:
            d_rep = text_embedding_model.encode(to_embed_list, instruction=get_instruction_item(args.embedding_model), batch_size=args.batch_size)

        emb_dict = defaultdict(list)
        for emb, iid in zip(d_rep, item_ids):
            emb_dict[iid] = emb

        with open(f'embedded_items_dict_{dataset}_{args.embedding_model.split("/")[-1]}.json', 'wb') as f:
            pickle.dump(emb_dict, f) 

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main(create_parser())
