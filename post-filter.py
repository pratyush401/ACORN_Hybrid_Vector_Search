import os
import sys
import psutil
from pathlib import Path
import numpy as np
from PIL import Image
import time
import acorn

import torch
from torchvision import models, transforms
import json

# --------------------------------------------------------------
# Post-Filtering Search Script
# --------------------------------------------------------------
# This script demonstrates how to:
# 1. Load image → ID mappings from CSV files
# 2. Load pre-generated metadata
# 3. Load embedding vectors and filenames
# 4. Build an HNSW index using ACORN's HNSWSearch class
# 5. Perform standard post-filter hybrid search
#
# Unlike acorn_search(), which integrates metadata constraints
# during graph traversal, *post-filtering* applies metadata
# only AFTER vector search completes.
# --------------------------------------------------------------
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
            
    # ----------------------------------------------------------
    # Load precomputed embedding vectors and filename arrays
    # ----------------------------------------------------------
    embeddings_data = np.load("embeddings_query_full.npy")
    filename_data = np.load("filenames_query_full.npy")
    query_embeddings_data = np.load("embeddings_query_full_query.npy")
    query_filename_data = np.load("filenames_query_full_query.npy")
    # Create a simple numeric ID array for HNSWlib indexing
    ids = np.arange(len(embeddings_data))
    print(len(ids))

    # ----------------------------------------------------------
    # Build HNSW index using the ACORN HNSWSearch class
    # ----------------------------------------------------------
    time_start = time.time()
    hnsw = acorn.HNSWSearch(dim=2048, metadata=path_to_meta)
    hnsw.init_index(max_elements=len(ids))
    hnsw.add_items(embeddings_data, ids)
    print("HNSW index built successfully!")
    time_end = time.time()
    print(f"Indexing completed in {time_end - time_start:.7f} seconds.")

    # ----------------------------------------------------------
    # Prepare the query vector
    # Here we take the embedding for query index 4 (please change to test other queries)
    # ----------------------------------------------------------
    query_vector = query_embeddings_data[4].reshape(1, -1)
    print(query_filename_data[4])
    # Example metadata constraints for post-filtering:
    # The metadata is provided using the following template
    # {"attribute": [operation, value]}
    # "attribute": color, brand, item_weight, model_year, country
    # "operation": "exact", "<", ">", "leq (<=)", "geq (>=)", "substring"
    # "value": string type, int type, float type
    query_meta_class_2 = {"item_weight": ["<", 2], "brand": ["substring", "Amazon"]}
    query_meta_class_3 = {"country": ["exact", "US"]}

    time_start = time.time()
    # Actual call to post_filter_search that is implemented in acorn.py
    results = hnsw.post_filter_search(query_vector, query_meta_class_2, filename_data, k=3)
    time_end = time.time()

    print(f"Search completed in {time_end - time_start:.7f} seconds.")
     # ----------------------------------------------------------
    # Display the final filtered results
    # ----------------------------------------------------------
    if len(results) == 0:
        print("No results found.")
    for r in results:
        print(r)                     # the index of the neighbor
        print(filename_data[r])      # the corresponding filename
        # print(path_to_meta[filename_data[r]])  # optional metadata display
