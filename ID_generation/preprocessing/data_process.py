"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import ast
import gzip
import html
import json
import os
import random
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
import tqdm
import wget

PROMPT = 'The item with the title "{}" is of brand "{}" and  costs {} dollars. It  belongs to categories "{}".'


def parse(path):  # for Amazon
    g = gzip.open(path, "r")
    for l in g:
        l = l.replace(b"true", b"True").replace(b"false", b"False")
        yield eval(l)


def Amazon(dataset_name, rating_score):
    """
    reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
    asin - ID of the product, e.g. 0000013714
    reviewerName - name of the reviewer
    helpful - helpfulness rating of the review, e.g. 2/3
    --"helpful": [2, 3],
    reviewText - text of the review
    --"reviewText": "I bought this for my husband who plays the piano. ..."
    overall - rating of the product
    --"overall": 5.0,
    summary - summary of the review
    --"summary": "Heavenly Highway Hymns",
    unixReviewTime - time of the review (unix time)
    --"unixReviewTime": 1252800000,
    reviewTime - time of the review (raw)
    --"reviewTime": "09 13, 2009"
    """
    datas = []
    review_mapping = defaultdict(dict)
    # older Amazon
    data_flie = dataset_name
    # latest Amazon
    for inter in parse(data_flie):
        if float(inter["overall"]) <= rating_score:  # 小于一定分数去掉
            continue
        user = inter["reviewerID"]
        item = inter["asin"]
        time = inter["unixReviewTime"]
        review_mapping[inter["reviewerID"]][inter["asin"]] = (
            inter["summary"],
            inter["reviewText"],
        )
        datas.append((user, item, int(time)))
    return datas, review_mapping


def Amazon_meta(dataset_name, data_maps):
    """
    asin - ID of the product, e.g. 0000031852
    --"asin": "0000031852",
    title - name of the product
    --"title": "Girls Ballet Tutu Zebra Hot Pink",
    description
    price - price in US dollars (at time of crawl)
    --"price": 3.17,
    imUrl - url of the product image (str)
    --"imUrl": "http://ecx.images-amazon.com/images/I/51fAmVkTbyL._SY300_.jpg",
    related - related products (also bought, also viewed, bought together, buy after viewing)
    --"related":{
        "also_bought": ["B00JHONN1S"],
        "also_viewed": ["B002BZX8Z6"],
        "bought_together": ["B002BZX8Z6"]
    },
    salesRank - sales rank information
    --"salesRank": {"Toys & Games": 211836}
    brand - brand name
    --"brand": "Coxlures",
    categories - list of categories the product belongs to
    --"categories": [["Sports & Outdoors", "Other Sports", "Dance"]]
    """
    datas = {}
    meta_file = (
        "./ID_generation/preprocessing/raw_data/meta_" + dataset_name + ".json.gz"
    )
    item_asins = set(data_maps["item2id"].keys())
    for info in parse(meta_file):
        if info["asin"] not in item_asins:
            continue
        datas[info["asin"]] = info
    return datas


def Yelp(date_min, date_max, rating_score):
    datas = []
    data_flie = (
        "./ID_generation/preprocessing/raw_data/yelp_academic_dataset_review.json"
    )
    lines = open(data_flie).readlines()

    for line in tqdm.tqdm(lines):
        review = json.loads(line.strip())
        user = review["user_id"]
        item = review["business_id"]
        rating = review["stars"]
        # 2004-10-12 10:13:32 2019-12-13 15:51:19
        date = review["date"]
        # 剔除一些例子
        if date < date_min or date > date_max or float(rating) <= rating_score:
            continue
        time = date.replace("-", "").replace(":", "").replace(" ", "")
        datas.append((user, item, int(time)))
    return datas


def Yelp_meta(datamaps):
    meta_infos = {}
    meta_file = (
        "./ID_generation/preprocessing/raw_data/yelp_academic_dataset_business.json"
    )
    item_ids = set(datamaps["item2id"].keys())
    lines = open(meta_file).readlines()
    for line in tqdm.tqdm(lines):
        info = json.loads(line)
        if info["business_id"] not in item_ids:
            continue
        meta_infos[info["business_id"]] = info
    return meta_infos


def get_yelp_reviews(datamaps, meta_infos, user_seq):
    meta_file = (
        "./ID_generation/preprocessing/raw_data/yelp_academic_dataset_review.json"
    )
    item_ids = set(datamaps["item2id"].keys())
    lines = open(meta_file).readlines()

    review_dict = defaultdict(dict)
    for line in tqdm.tqdm(lines):
        info = json.loads(line)
        if info["business_id"] not in item_ids:
            continue
        try:
            item_id = datamaps["item2id"][info["business_id"]]
            user_id = datamaps["user2id"][info["user_id"]]
        except KeyError:
            continue
        review_dict[user_id][item_id] = info["text"]

    id2review = defaultdict(list)
    for reviewer_id in user_seq.keys():
        user_id = datamaps["user2id"][reviewer_id]

        for item in user_seq[reviewer_id]:
            id = datamaps["item2id"][item]
            item_infos = meta_infos[item]
            info_dict = {}
            for key in ["name", "categories", "attributes", "city", "state"]:
                if key in item_infos and item_infos[key] is not None:
                    if isinstance(item_infos[key], dict):
                        # convert into dict of dicts
                        temp_dict = {
                            k: eval(v) if "{" in v else v
                            for k, v in item_infos[key].items()
                        }
                        flattened = pd.json_normalize(temp_dict, sep=".").to_dict(
                            orient="records"
                        )[0]
                        attr_list = [k for k, v in flattened.items() if v == "True"]
                        info_dict[key] = ", ".join(attr_list)
                    elif isinstance(item_infos[key], str):
                        info_dict[key] = feature_process(
                            item_infos[key], use_prompt=False
                        )
            id2review[user_id].append(
                {
                    "itemid": id,
                    "userid": user_id,
                    "review": review_dict[user_id][id],
                    **info_dict,
                }
            )

    return id2review


def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True  # 已经保证Kcore


# 循环过滤 K-core
def filter_Kcore(user_items, user_core, item_core):  # user 接所有items
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    while not isKcore:
        for user, num in user_count.items():
            if user_count[user] < user_core:  # 直接把user 删除
                user_items.pop(user)
            else:
                for item in user_items[user]:
                    if item_count[item] < item_core:
                        user_items[user].remove(item)
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    return user_items


def get_attribute_Amazon(meta_infos, datamaps, attribute_core):

    attributes = defaultdict(int)
    for iid, info in tqdm.tqdm(meta_infos.items()):
        for cates in info["categories"]:
            for cate in cates[1:]:
                attributes[cate] += 1

    print(f"before delete, attribute num:{len(attributes)}")
    new_meta = {}
    for iid, info in tqdm.tqdm(meta_infos.items()):
        new_meta[iid] = []

        for cates in info["categories"]:
            for cate in cates[1:]:
                if attributes[cate] >= attribute_core:
                    new_meta[iid].append(cate)

    attribute2id = {}
    id2attribute = {}
    attributeid2num = defaultdict(int)
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []

    for iid, attributes in new_meta.items():
        item_id = datamaps["item2id"][iid]
        items2attributes[item_id] = []
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            attributeid2num[attribute2id[attribute]] += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))
    print(f"before delete, attribute num:{len(attribute2id)}")
    print(
        f"attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}"
    )

    datamaps["attribute2id"] = attribute2id
    datamaps["id2attribute"] = id2attribute
    datamaps["attributeid2num"] = attributeid2num
    return len(attribute2id), np.mean(attribute_lens), datamaps, items2attributes


def get_attribute_Yelp(meta_infos, datamaps, attribute_core):
    attributes = defaultdict(int)
    for iid, info in tqdm.tqdm(meta_infos.items()):
        try:
            cates = [cate.strip() for cate in info["categories"].split(",")]
            for cate in cates:
                attributes[cate] += 1
        except:
            pass
    print(f"before delete, attribute num:{len(attributes)}")
    new_meta = {}
    for iid, info in tqdm.tqdm(meta_infos.items()):
        new_meta[iid] = []
        try:
            cates = [cate.strip() for cate in info["categories"].split(",")]
            for cate in cates:
                if attributes[cate] >= attribute_core:
                    new_meta[iid].append(cate)
        except:
            pass

    attribute2id = {}
    id2attribute = {}
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []
    # load id map
    for iid, attributes in new_meta.items():
        item_id = datamaps["item2id"][iid]
        items2attributes[item_id] = []
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))
    print(f"after delete, attribute num:{len(attribute2id)}")
    print(
        f"attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}"
    )

    datamaps["attribute2id"] = attribute2id
    datamaps["id2attribute"] = id2attribute
    return len(attribute2id), np.mean(attribute_lens), datamaps, items2attributes


def meta_map(
    meta_infos,
    item2id,
    features_needed=["title", "price", "brand", "feature", "categories", "description"],
    use_prompt=False,
):
    id2meta = {}
    item2meta = {}
    meta_text = {}

    for item, meta in tqdm.tqdm(meta_infos.items()):

        keys = set(meta.keys())
        for feature in features_needed:
            if feature in keys:
                if meta[feature] is not None:
                    if isinstance(meta[feature], dict):
                        # convert into dict of dicts
                        temp_dict = {
                            k: eval(v) if "{" in v else v
                            for k, v in meta[feature].items()
                        }
                        flattened = pd.json_normalize(temp_dict, sep=".").to_dict(
                            orient="records"
                        )[0]
                        attr_list = [k for k, v in flattened.items() if v == "True"]
                        meta_text[feature] = ", ".join(attr_list)
                    elif isinstance(meta[feature], str):
                        meta_text[feature] = feature_process(meta[feature], use_prompt)

        try:
            id = item2id[item]
        except KeyError:
            # item was filtered and id map was created after filtering
            continue

        if not use_prompt:
            item2meta[item] = ", ".join(
                [f"{k.capitalize()}: {meta_text[k]}" for k in meta_text.keys()]
            )
            id2meta[id] = ", ".join(
                [f"{k.capitalize()}: {meta_text[k]}" for k in meta_text.keys()]
            )
        else:
            text = ""
            if "title" in keys and meta["title"] != "":
                text += f"This item is called '{meta_text['title']}'. "
            if "price" in keys:
                text += f"It is priced at {meta_text['price']}. "
            if "brand" in keys and meta["brand"] != "":
                text += f"It is manufactured by '{meta_text['brand']}'. "
            if "categories" in keys and len(meta_text["categories"]):
                text += f"It belongs to the categories of {meta_text['categories']}. "
            if "description" in features_needed and "description" in keys:
                text += f"The description of this item is: {meta_text['description']}. "
            item2meta[item] = text
            id2meta[id] = text
    return id2meta


def get_item_review_map(user_seq, review_mapping, data_maps, meta_infos):
    id2review = defaultdict(list)

    for reviewer_id in user_seq.keys():
        user_id = data_maps["user2id"][reviewer_id]
        for item in user_seq[reviewer_id]:
            id = data_maps["item2id"][item]
            title = "" if "title" not in meta_infos[item] else meta_infos[item]["title"]
            categories = (
                meta_infos[item]["categories"]
                if "categories" in meta_infos[item]
                else ""
            )
            desc = (
                meta_infos[item]["description"]
                if "description" in meta_infos[item]
                else ""
            )
            brand = meta_infos[item]["brand"] if "brand" in meta_infos[item] else ""
            price = meta_infos[item]["price"] if "price" in meta_infos[item] else ""
            id2review[user_id].append(
                {
                    "itemid": id,
                    "userid": user_id,
                    "title": title,
                    "categories": categories,
                    "brand": brand,
                    "price": price,
                    "description": desc,
                    "review_summary": review_mapping[reviewer_id][item][0],
                    "review": review_mapping[reviewer_id][item][1],
                }
            )
    return id2review


def get_steam_reviews(user_seq, review_mapping, datamaps, meta_infos):
    id2review = defaultdict(list)
    for reviewer_id in user_seq.keys():
        user_id = datamaps["user2id"][reviewer_id]
        for item in user_seq[reviewer_id]:
            id = datamaps["item2id"][item]
            title = "" if "title" not in meta_infos[item] else meta_infos[item]["title"]
            genre = meta_infos[item]["genre"] if "genre" in meta_infos[item] else ""
            tags = meta_infos[item]["tags"] if "tags" in meta_infos[item] else ""
            genre = meta_infos[item]["specs"] if "specs" in meta_infos[item] else ""
            price = meta_infos[item]["price"] if "price" in meta_infos[item] else ""
            publisher = (
                meta_infos[item]["publisher"] if "publisher" in meta_infos[item] else ""
            )
            sentiment = (
                meta_infos[item]["sentiment"] if "sentiment" in meta_infos[item] else ""
            )
            id2review[user_id].append(
                {
                    "itemid": id,
                    "userid": user_id,
                    "title": title,
                    "tags": tags,
                    "genre": genre,
                    "price": price,
                    "publisher": publisher,
                    "sentiment": sentiment,
                    "review": review_mapping[item],
                }
            )
    return id2review


def get_movielens_reviews(user_seq, datamaps, item_meta_infos, user_meta_info):
    id2review = defaultdict(list)
    for reviewer_id in user_seq.keys():
        user_id = datamaps["user2id"][reviewer_id]
        for item in user_seq[reviewer_id]:
            id = datamaps["item2id"][item]
            name = (
                ""
                if "name" not in item_meta_infos[item]
                else item_meta_infos[item]["name"]
            )
            genre = (
                item_meta_infos[item]["genre"]
                if "genre" in item_meta_infos[item]
                else ""
            )
            year = (
                item_meta_infos[item]["year"] if "year" in item_meta_infos[item] else ""
            )
            age = (
                user_meta_info[reviewer_id]["age"]
                if "age" in user_meta_info[reviewer_id]
                else ""
            )
            gender = (
                user_meta_info[reviewer_id]["gender"]
                if "gender" in user_meta_info[reviewer_id]
                else ""
            )
            occupation = (
                user_meta_info[reviewer_id]["occupation"]
                if "occupation" in user_meta_info[reviewer_id]
                else ""
            )
            id2review[user_id].append(
                {
                    "itemid": id,
                    "userid": user_id,
                    "name": name,
                    "year": year,
                    "genre": genre,
                    "user_age": age,
                    "user_gender": gender,
                    "user_occupation": occupation,
                }
            )
    return id2review


def add_comma(num):
    # 1000000 -> 1,000,000
    str_num = str(num)
    res_num = ""
    for i in range(len(str_num)):
        res_num += str_num[i]
        if (len(str_num) - i - 1) % 3 == 0:
            res_num += ","
    return res_num[:-1]


def id_map(user_items):  # user_items dict

    user2id = {}  # raw 2 uid
    item2id = {}  # raw 2 iid
    id2user = {}  # uid 2 raw
    id2item = {}  # iid 2 raw
    user_id = 1
    item_id = 1
    final_data = {}
    for user, items in user_items.items():
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1
        iids = []  # item id lists
        for item in items:
            if item not in item2id:
                item2id[item] = str(item_id)
                id2item[str(item_id)] = item
                item_id += 1
            iids.append(item2id[item])
        uid = user2id[user]
        final_data[uid] = iids
    data_maps = {
        "user2id": user2id,
        "item2id": item2id,
        "id2user": id2user,
        "id2item": id2item,
    }
    return final_data, user_id - 1, item_id - 1, data_maps


def get_interaction(datas):
    user_seq = {}
    for data in datas:
        user, item, time = data
        if user in user_seq:
            user_seq[user].append((item, time))
        else:
            user_seq[user] = []
            user_seq[user].append((item, time))

    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])
        items = []
        for t in item_time:
            items.append(t[0])
        user_seq[user] = items
    return user_seq


def list_to_str(l):
    if isinstance(l, list):
        return list_to_str(", ".join(l))
    else:

        return l


def clean_text(raw_text):
    text = list_to_str(raw_text)
    text = html.unescape(text)
    text = text.strip()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\n\t]", " ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"[^\x00-\x7F]", " ", text)
    return text


def feature_process(feature, use_prompt):
    sentence = ""
    if isinstance(feature, float):
        sentence += str(feature)
        if not use_prompt:
            sentence += "."
    elif len(feature) > 0 and isinstance(feature[0], list):
        for v1 in feature:
            for v in v1:
                sentence += clean_text(v)
                sentence += ", "
        sentence = sentence[:-2]
        if not use_prompt:
            sentence += "."
    elif isinstance(feature, list):
        for v1 in feature:
            sentence += clean_text(v1)
    else:
        sentence = clean_text(feature)
    if not use_prompt:
        sentence += " "
    return sentence


def preprocessing(config, require_attributes=False):
    dataset_name, data_type, features_needed = (
        config["name"],
        config["type"],
        config["features_needed"],
    )
    use_prompt = config["use_prompt"]
    if not use_prompt:
        features_used = "_".join(features_needed)
    else:
        features_used = "title_brand_price_categories"
    data_file = f"./ID_generation/preprocessing/processed/{dataset_name}.txt"
    id2meta_file = f"./ID_generation/preprocessing/processed/{dataset_name}_{features_used}_id2meta.json"
    if use_prompt:
        id2meta_file = id2meta_file.replace(".json", "_prompted.json")
    item2attributes_file = (
        f"./ID_generation/preprocessing/processed/{dataset_name}_item2attributes.json"
    )
    attributesmap_file = (
        f"./ID_generation/preprocessing/processed/{dataset_name}_attributesmap.json"
    )

    # if require_attributes:
    #     if os.path.exists(data_file) and os.path.exists(id2meta_file) and os.path.exists(item2attributes_file) and os.path.exists(attributesmap_file):
    #         print(f'{dataset_name} has been processed!')
    #         return
    # else:
    #     if os.path.exists(data_file) and os.path.exists(id2meta_file):
    #         print(f'{dataset_name} has been processed!')
    #         return

    print(
        f"data_name: {dataset_name}, data_type: {data_type}, require_attributes: {require_attributes}"
    )

    np.random.seed(12345)
    random.seed(12345)
    rating_score = 0.0  # rating score smaller than this score would be deleted
    # user 5-core item 5-core
    user_core = 5
    item_core = 5
    attribute_core = 0

    if data_type == "yelp":
        date_max = "2019-12-31 00:00:00"
        date_min = "2019-01-01 00:00:00"
        datas = Yelp(date_min, date_max, rating_score)
    elif data_type == "Amazon":
        datas, review_mapping = Amazon(
            "./ID_generation/preprocessing/raw_data/reviews_"
            + dataset_name
            + "_5.json.gz",
            rating_score=rating_score,
        )
    elif data_type == "lastfm":
        datas = LastFM("./ID_generation/preprocessing/raw_data/lastfm", features_needed)
        return
    elif data_type == "steam":
        dataset = Steam("./ID_generation/preprocessing/")
        datas = dataset.sequence_raw
    elif dataset_name == "movielens":
        dataset = MovieLens("./ID_generation/preprocessing/", name=data_type)
        datas = dataset.sequence_raw
    else:
        raise NotImplementedError(f"Dataset {data_type} not implemented!")

    user_items = get_interaction(datas)

    print(
        f"{dataset_name} Raw data has been processed! Lower than {rating_score} are deleted!"
    )
    # raw_id user: [item1, item2, item3...]
    user_items = filter_Kcore(user_items, user_core=user_core, item_core=item_core)
    print(f"User {user_core}-core complete! Item {item_core}-core complete!")
    user_items_id, user_num, item_num, data_maps = id_map(user_items)
    user_count, item_count, _ = check_Kcore(
        user_items_id, user_core=user_core, item_core=item_core
    )
    user_count_list = list(user_count.values())
    user_avg, user_min, user_max = (
        np.mean(user_count_list),
        np.min(user_count_list),
        np.max(user_count_list),
    )
    item_count_list = list(item_count.values())
    item_avg, item_min, item_max = (
        np.mean(item_count_list),
        np.min(item_count_list),
        np.max(item_count_list),
    )
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    seqs_length = [len(user_items_id[i]) for i in user_items_id.keys()]
    show_info = (
        f"Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n"
        + f"Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n"
        + f"Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%\n"
        + f"Sequence Length Mean: {(sum(seqs_length) / len(seqs_length)):.2f}, Mediam: {statistics.median(seqs_length)}"
    )
    print(show_info)

    print("Begin extracting meta infos...")

    if data_type == "Amazon":
        meta_infos = Amazon_meta(dataset_name, data_maps)
        if require_attributes:
            attribute_num, avg_attribute, datamaps, item2attributes = (
                get_attribute_Amazon(meta_infos, data_maps, attribute_core)
            )
        id2meta = meta_map(
            meta_infos, data_maps["item2id"], features_needed, use_prompt
        )
        item2review = get_item_review_map(
            user_items, review_mapping, data_maps, meta_infos
        )
    elif data_type == "yelp":
        meta_infos = Yelp_meta(data_maps)
        if require_attributes:
            attribute_num, avg_attribute, datamaps, item2attributes = (
                get_attribute_Yelp(meta_infos, data_maps, attribute_core)
            )
        id2meta = meta_map(meta_infos, data_maps["item2id"], features_needed)
        item2review = get_yelp_reviews(data_maps, meta_infos, user_items)
    elif data_type == "lastfm":
        LastFM("./ID_generation/preprocessing/raw_data/lastfm")
        return
    elif data_type == "steam":
        meta_infos = dataset.process_meta_infos()
        id2meta = meta_map(
            meta_infos, data_maps["item2id"], features_needed=features_needed
        )
        item2review = get_steam_reviews(
            user_items, dataset.item2review, data_maps, meta_infos
        )
    elif dataset_name == "movielens":
        item_meta_infos, user_meta_info = dataset.process_meta_infos()
        id2meta = meta_map(
            item_meta_infos, data_maps["item2id"], features_needed=features_needed
        )
        item2review = get_movielens_reviews(
            user_items, data_maps, item_meta_infos, user_meta_info
        )
    else:
        raise NotImplementedError(f"Dataset {data_type} not implemented!")

    if (
        require_attributes
        and not data_type == "steam"
        and not dataset_name == "movielens"
    ):
        print(
            f"{dataset_name} & {add_comma(user_num)}& {add_comma(item_num)} & {user_avg:.1f}"
            f"& {item_avg:.1f}& {add_comma(interact_num)}& {sparsity:.2f}\%&{add_comma(attribute_num)}&"
            f"{avg_attribute:.1f} \\"
        )
    else:
        print(
            f"{dataset_name} & {add_comma(user_num)}& {add_comma(item_num)} & {user_avg:.1f}"
            f"& {item_avg:.1f}& {add_comma(interact_num)}& {sparsity:.2f}\% \\"
        )

    # -------------- Save Data ---------------

    os.makedirs("./ID_generation/preprocessing/processed/", exist_ok=True)
    with open(data_file, "w") as out:
        for user, items in user_items_id.items():
            out.write(user + " " + " ".join(items) + "\n")

    json_str = json.dumps(id2meta)
    with open(id2meta_file, "w") as out:
        out.write(json_str)

    if data_type != "lastfm":
        os.makedirs("./dataset", exist_ok=True)
        json_str = json.dumps(item2review)
        with open(f"./dataset/item2review_{dataset_name}.json", "w") as out:
            out.write(json_str)

    if data_type == "MovieLen_1M":
        os.makedirs("./dataset", exist_ok=True)
        json_str = json.dumps(user_meta_info)
        with open(f"./dataset/user_meta_info_{dataset_name}.json", "w") as out:
            out.write(json_str)

    if (
        require_attributes
        and not data_type == "steam"
        and not dataset_name == "movielens"
    ):
        json_str = json.dumps(item2attributes)
        with open(item2attributes_file, "w") as out:
            out.write(json_str)

        json_str = json.dumps(datamaps)
        with open(attributesmap_file, "w") as out:
            out.write(json_str)


def LastFM(root, features_needed):
    user_core = 5
    item_core = 5
    datas = []
    data_file = f"{root}/user_taggedartists-timestamps.dat"
    lines = open(data_file).readlines()
    for line in tqdm.tqdm(lines[1:]):
        user, item, attribute, timestamp = line.strip().split("\t")
        datas.append((user, item, int(timestamp)))

    user_seq = {}
    user_seq_notime = {}
    for data in datas:
        user, item, time = data
        if user in user_seq:
            if item not in user_seq_notime[user]:
                user_seq[user].append((item, time))
                user_seq_notime[user].append(item)
            else:
                continue
        else:
            user_seq[user] = []
            user_seq_notime[user] = []

            user_seq[user].append((item, time))
            user_seq_notime[user].append(item)

    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])
        items = []
        for t in item_time:
            items.append(t[0])
        user_seq[user] = items

    user_items = filter_Kcore(user_seq, user_core=user_core, item_core=item_core)
    print(f"User {user_core}-core complete! Item {item_core}-core complete!")

    user_items, user_num, item_num, data_maps = id_map(user_items)  # new_num_id
    user_count, item_count, _ = check_Kcore(
        user_items, user_core=user_core, item_core=item_core
    )
    user_count_list = list(user_count.values())
    user_avg, user_min, user_max = (
        np.mean(user_count_list),
        np.min(user_count_list),
        np.max(user_count_list),
    )
    item_count_list = list(item_count.values())
    item_avg, item_min, item_max = (
        np.mean(item_count_list),
        np.min(item_count_list),
        np.max(item_count_list),
    )
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    show_info = (
        f"Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n"
        + f"Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n"
        + f"Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%"
    )
    print(show_info)

    if not os.path.exists(f"{root}/item2attributes.json"):
        # create artist 2 attribute mapping
        meta_item2attribute = defaultdict(list)
        tagmap = {
            l.decode("latin1")
            .strip()
            .split("\t")[0]: l.decode("latin1")
            .strip()
            .split("\t")[1]
            for l in open(f"{root}/tags.dat", "rb").readlines()[1:]
        }
        for line in open(f"{root}/user_taggedartists.dat", "rb").readlines()[1:]:
            artistid = line.decode("latin1").strip().split("\t")[1]
            tag = line.decode("latin1").strip().split("\t")[2]
            meta_item2attribute[artistid].append(tagmap[tag])
        with open(f"{root}/item2attributes.json", "w") as f:
            json.dump(meta_item2attribute, f)
    else:
        meta_item2attribute = json.load(open(f"{root}/item2attributes.json"))

    attribute2id = {}
    id2attribute = {}
    attribute_id = 1
    item2attributes = {}
    attribute_lens = []
    # load id map
    for iid, attributes in meta_item2attribute.items():
        if iid in list(data_maps["item2id"].keys()):
            item_id = data_maps["item2id"][iid]
            item2attributes[item_id] = []
            for attribute in attributes:
                if attribute not in attribute2id:
                    attribute2id[attribute] = attribute_id
                    id2attribute[attribute_id] = attribute
                    attribute_id += 1
                item2attributes[item_id].append(attribute2id[attribute])
            attribute_lens.append(len(item2attributes[item_id]))
    print(f"after delete, attribute num:{len(attribute2id)}")
    print(
        f"attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}"
    )

    data_maps["attribute2id"] = attribute2id
    data_maps["id2attribute"] = id2attribute

    data_name = "lastfm"
    print(
        f"{data_name} & {add_comma(user_num)}& {add_comma(item_num)} & {user_avg:.1f}"
        f"& {item_avg:.1f}& {add_comma(interact_num)}& {sparsity:.2f}\%&{add_comma(len(attribute2id))}&"
        f"{np.mean(attribute_lens):.1f} \\"
    )

    # -------------- Save Data ---------------
    # one user one line
    data_file = os.path.join(*root.split("/")[:-2], "processed", f"{data_name}.txt")
    item2attributes_file = os.path.join(
        *root.split("/")[:-2],
        "processed",
        f"{data_name}_{features_needed[0]}_id2meta.json",
    )

    with open(data_file, "w") as out:
        for user, items in user_items.items():
            out.write(user + " " + " ".join(items) + "\n")

    json_str = json.dumps(meta_item2attribute)
    with open(item2attributes_file, "w") as out:
        out.write(json_str)


class Steam:
    def __init__(self, root) -> None:
        self.root = os.path.abspath(root)
        self.urls = {
            "reviews": "http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz",
            "games": "http://cseweb.ucsd.edu/~wckang/steam_games.json.gz",
        }
        self.download()
        self.process()

    def download(self):
        path = os.path.join(self.root, "raw_data", "steam")

        if os.path.exists(path):
            print(f"{path} exists, download is not needed.")
            return

        os.makedirs(path)
        for d in ["games", "reviews"]:
            print(f"downloading steam from {self.urls[d]}")
            file_name = wget.download(self.urls[d], out=path)
            content = gzip.open(file_name, "rb")
            content = content.read().decode("utf-8").split("\n")
            content = [
                json.loads(json.dumps(ast.literal_eval(line)))
                for idx, line in enumerate(content)
                if line
            ]
            with open(file_name[:-3], "w") as f:
                json.dump(content, f)

    def process(self):
        path = os.path.join(self.root, "processed")
        os.makedirs(path, exist_ok=True)

        print(f"preprocessing steam ...")
        review_file = glob(
            f"{os.path.join(self.root, 'raw_data', 'steam')}/steam_reviews.json"
        )[0]

        with open(review_file, "r") as f:
            raw_data = json.load(f)

        user_counts = Counter([entry["username"] for entry in raw_data])
        raw_data = [entry for entry in raw_data if user_counts[entry["username"]] >= 5]

        user_id, item_id = 0, 1
        self.user2id, self.item2id, self.id2user, self.id2item = {}, {}, {}, {}
        self.item2id["<pad>"], self.id2item[0] = 0, "<pad>"
        self.item2review = {}
        for entry in tqdm.tqdm(raw_data, desc="Mapping unique users and items ..."):
            if entry["username"] not in self.user2id:
                self.user2id[entry["username"]] = user_id
                self.id2user[user_id] = entry["username"]
                user_id += 1
            if entry["product_id"] not in self.item2id:
                self.item2id[entry["product_id"]] = item_id
                self.id2item[item_id] = entry["product_id"]
                self.item2review[item_id] = entry["text"]
                item_id += 1

        self.sequence_raw = []
        for entry in tqdm.tqdm(raw_data, desc="Constructing sequence and graph ..."):
            self.sequence_raw.append(
                (
                    self.user2id[entry["username"]],
                    self.item2id[entry["product_id"]],
                    int(datetime.fromisoformat(entry["date"]).timestamp()),
                )
            )

    def process_meta_infos(self):
        meta_file = glob(
            f"{os.path.join(self.root, 'raw_data', 'steam')}/steam_games.json"
        )[0]
        with open(meta_file, "r") as f:
            raw_data = json.load(f)

        items = {}
        for entry in tqdm.tqdm(raw_data, desc="Creating item content features...."):
            if "id" in entry and entry["id"] in self.item2id:
                meta_dict = {}
                meta_dict["title"] = f"{entry['title']}"
                meta_dict["genre"] = (
                    f"{' '.join(entry['genres']) if 'genres' in entry else 'Unknown'}"
                )
                meta_dict["tags"] = (
                    f"{' '.join(entry['tags']) if 'tags' in entry else 'Unknown'}"
                )
                meta_dict["specs"] = (
                    f"{' '.join(entry['specs']) if 'specs' in entry else 'Unknown'}"
                )
                meta_dict["price"] = f"{entry.get('price', 0)}"
                meta_dict["publisher"] = f"{entry.get('publisher', 'Unknown')}"
                meta_dict["sentiment"] = f"{entry.get('sentiment', 'Unknown')}"
                items[self.item2id[entry["id"]]] = meta_dict

        return items


class MovieLens:
    def __init__(self, root, name) -> None:
        self.root = os.path.abspath(root)
        self.name = name
        self.urls = {
            "MovieLen_1M": "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
            "MovieLen_20M": "http://files.grouplens.org/datasets/movielens/ml-20m.zip",
        }
        self.download()
        self.process()

    def download(self):

        path = os.path.join(self.root, "raw_data", self.name)

        if os.path.exists(path):
            print(f"{path} exists, download is not needed.")
            return

        print(f"downloading {self.name} from {self.urls[self.name]}")
        os.makedirs(path)
        file_name = wget.download(self.urls[self.name], out=path)
        with zipfile.ZipFile(file_name, "r") as z:
            z.extractall(path)
            z.close()

    def process(self):
        path = os.path.join(self.root, "processed")
        os.makedirs(path, exist_ok=True)

        extension = "dat" if self.name == "MovieLen_1M" else "csv"
        split = "::" if self.name == "MovieLen_1M" else ","

        print(f"preprocessing {self.name} ...")
        ratings_file = glob(
            f"{os.path.join(self.root, 'raw_data', self.name)}/*/ratings.{extension}"
        )[0]

        self.sequence_raw = []
        with open(ratings_file, "r") as f:
            raw_data = [line.strip("\n").split(split) for line in f]
            if self.name == "MovieLen_20M":
                raw_data = raw_data[1:]

        user_id, item_id = 0, 1
        self.user2id, self.item2id, self.id2user, self.id2item = {}, {}, {}, {}
        self.item2id["<pad>"], self.id2item[0] = 0, "<pad>"
        for entry in tqdm.tqdm(raw_data, desc="Mapping unique users and items ..."):
            if int(entry[0]) not in self.user2id:
                self.user2id[int(entry[0])] = user_id
                self.id2user[user_id] = int(entry[0])
                user_id += 1
            if int(entry[1]) not in self.item2id:
                self.item2id[int(entry[1])] = item_id
                self.id2item[item_id] = int(entry[1])
                item_id += 1

        for entry in tqdm.tqdm(raw_data, desc="Constructing sequence and graph ..."):
            self.sequence_raw.append(
                (
                    self.user2id[int(entry[0])],
                    self.item2id[int(entry[1])],
                    int(entry[3]),
                )
            )

    def process_meta_infos(self):
        extension = "dat" if self.name == "MovieLen_1M" else "csv"
        split = "::" if self.name == "MovieLen_1M" else ","
        movies_file = glob(
            f"{os.path.join(self.root, 'raw_data', self.name)}/*/movies.{extension}"
        )[0]

        with open(movies_file, "r", encoding="ISO-8859-1") as f:
            raw_data = [line.strip("\n").split(split) for line in f]
            if self.name == "MovieLen_20M":
                raw_data = raw_data[1:]

        items = {}
        for entry in tqdm.tqdm(raw_data, desc="Creating item content features...."):
            if int(entry[0]) in self.item2id:
                meta_dict = {}
                year = re.findall(r"\b\d{4}\b", " ".join(entry))
                year = int(year[0]) if year else "unknown"
                meta_dict["year"] = "<year> " + f"{year}" + " </year> "
                meta_dict["name"] = (
                    "<name> " + f"{entry[1].split('(')[0].strip()}" + " </name> "
                )
                meta_dict["genre"] = (
                    "<genre> " + f"{' '.join(entry[2].split('|'))}" + " </genre>"
                )
                items[self.item2id[int(entry[0])]] = meta_dict

        if self.name == "MovieLen_1M":
            user_jobs = {
                0: "not specified",
                1: "academic or educator",
                2: "artist",
                3: "clerical or admin",
                4: "college or grad student",
                5: "customer service",
                6: "doctor or health care",
                7: "executive or managerial",
                8: "farmer",
                9: "homemaker",
                10: "K-12 student",
                11: "lawyer",
                12: "programmer",
                13: "retired",
                14: "sales or marketing",
                15: "scientist",
                16: "self-employed",
                17: "technician or engineer",
                18: "tradesman or craftsman",
                19: "unemployed",
                20: "writer",
            }

            users_file = glob(
                f"{os.path.join(self.root, 'raw_data', self.name)}/*/users.{extension}"
            )[0]
            with open(users_file, "r") as f:
                raw_data = [line.strip("\n").split(split) for line in f]

            users = {}
            for entry in tqdm.tqdm(raw_data, desc="Computing user embeddings..."):
                if int(entry[0]) in self.user2id:
                    meta_dict = {}
                    meta_dict["gender"] = (
                        f"<gender> {'Female' if entry[1]=='F' else 'Male'} </gender> "
                    )
                    meta_dict["age"] = f"<age> {entry[2]} <\age> "
                    meta_dict["occupation"] = (
                        f"<occupation> {user_jobs[int(entry[3])]} <\occupation>"
                    )
                users[self.user2id[int(entry[0])]] = meta_dict
        else:
            users = {}

        return items, users


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--dataset_name",
        type=str,
        required=True,
        nargs="+",
        help="The names of the datasets",
    )
    parser.add_argument(
        "-t",
        "--data_type",
        type=str,
        choices=["Amazon", "yelp"],
        default="Amazon",
        help="The type of the data (Amazon or yelp)",
    )
    parser.add_argument(
        "-a",
        "--require_attributes",
        action="store_true",
        help="If set, require to extract attributes mappings",
    )
    args = parser.parse_args()

    kwargs = {}

    if args.data_type is not None:
        kwargs["data_type"] = args.data_type
    if args.require_attributes:
        kwargs["require_attributes"] = args.require_attributes

    for name in args.dataset_name:
        kwargs["dataset_name"] = name
        preprocessing(**kwargs)
