"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import os

import requests


def download_file(url, path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {os.path.basename(path)}")
    else:
        print(f"Failed to download {os.path.basename(path)}")


if __name__ == "__main__":
    urls = [
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz",
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz",
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz",
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Sports_and_Outdoors.json.gz",
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Toys_and_Games.json.gz",
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz",
    ]
    directory = "./ID_generation/preprocessing/raw_data/"
    os.makedirs(directory, exist_ok=True)

    for url in urls:
        # Extract the filename from the URL
        filename = url.split("/")[-1]
        filepath = os.path.join(directory, filename)
        if not os.path.exists(filepath):
            print(f"{filename} not found, downloading...")
            download_file(url, filepath)
