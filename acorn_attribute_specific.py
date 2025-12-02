# ============================================================
# ACORN-1 Hybrid Search Implementation for ABO Dataset
# ============================================================

import os
import sys
import json
import numpy as np
import hnswlib
import torch
import time
from PIL import Image
import psutil
from torchvision import models, transforms
import acorn

if __name__ == "__main__":
    path_to_ids = {}
    for k in [0,1,2,3,4]:
        for i in [0,1,2,3,4,5,6,7,8,9,'a','b','c','d','e','f']:
            if k == 0 and i == 'f':
                continue
            with open(f"map{k}{i}.csv", "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    path_to_ids[parts[0]] = parts[3][3:]

    print(f"Loaded {len(path_to_ids)} image ID mappings.")

    path_to_meta = {}
    with open("metadata-small.py", "r") as f:
        for line in f:
            curr_line = line[1:-1]
            curr_line = curr_line.strip().split(",")
            image_id = curr_line[0]
            metadata = ",".join(curr_line[1:])[:-1]
            path_to_meta[path_to_ids[image_id]] = json.loads(metadata)

    embeddings_data = np.load("embeddings_query_full.npy")
    filename_data = np.load("filenames_query_full.npy")
    query_embeddings_data = np.load("embeddings_query_full_query.npy")
    query_filename_data = np.load("filenames_query_full_query.npy")

    # Here we do the extra step for getting only the images and their respective metadata where we find model_year to be one of the attributes
    attribute_specific_embeddings_data = []
    attribute_specific_filename_data = []
    count = 0
    for k in path_to_meta:
        if "model_year" in path_to_meta[k]:
            index = np.where(filename_data == k)[0][0]
            attribute_specific_embeddings_data.append(embeddings_data[index])
            attribute_specific_filename_data.append(k)

    
    ids = np.arange(len(attribute_specific_embeddings_data))
    print(len(ids))


    # # # # --------------------------
    # # # # 3. Build ACORN-1 index
    # # # # --------------------------
    time_start = time.time()
    hnsw_attribute = acorn.HNSWSearch(dim=2048, metadata=path_to_meta)
    hnsw_attribute.init_index(max_elements=len(ids))
    hnsw_attribute.add_items(attribute_specific_embeddings_data, ids)
    print("HNSW index built successfully!")
    time_end = time.time()
    print(f"Indexing completed in {time_end - time_start:.7f} seconds.")

    # Use specific query with model_year, so that the attribute-specific graph can be used for much better performance
    query_vector = query_embeddings_data[5].reshape(1, -1)
    print(query_filename_data[5])
    query_metadata_class_3 = {"model_year": ["leq", 2018], "color": ["substring", "Multicolor"]}

    time_start = time.time()
    results = hnsw_attribute.acorn_search(query_vector, query_metadata_class_3, attribute_specific_filename_data, k=3, meta_search=100)
    time_end = time.time()
    print(f"Search completed in {time_end - time_start:.7f} seconds.")
    if len(results) == 0:
        print("No results found.")
    for r in results:
        print(r)
        print(attribute_specific_filename_data[r[0]])

