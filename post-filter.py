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
     # ----------------------------------------------------------
    # Load mapping between image filenames and internal IDs
    # The map*k*.csv files link original image names to ABO IDs.
    # ----------------------------------------------------------
    path_to_ids = {}
    for k in [0,1,2,3,4]:
        for i in [0,1,2,3,4,5,6,7,8,9,'a','b','c','d','e','f']:
            # Skip invalid file combination found in dataset structure
            if k == 0 and i == 'f':
                continue
            # Open each mapping file and populate id dictionary
            with open(f"map{k}{i}.csv", "r") as f:
                for line in f:
                    # parts[0] = image filename
                    # parts[3] = something like img:xxxxxx → extract ID portion
                    parts = line.strip().split(",")
                    path_to_ids[parts[0]] = parts[3][3:]

    print(f"Loaded {len(path_to_ids)} image ID mappings.")

    # ----------------------------------------------------------
    # Load metadata for each image ID from metadata-small.py
    # The file stores metadata in a 2-column tuple-like format.
    # ----------------------------------------------------------
    path_to_meta = {}
    with open("metadata-small.py", "r") as f:
        for line in f:
            # Strip enclosing brackets
            curr_line = line[1:-1]
            curr_line = curr_line.strip().split(",")
            image_id = curr_line[0] # First column → image ID
            metadata = ",".join(curr_line[1:])[:-1]  # Remaining columns represent metadata encoded as JSON
            path_to_meta[path_to_ids[image_id]] = json.loads(metadata)  # Store metadata mapped to ABO internal ID

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
    # Here we take the embedding for query index 4
    # ----------------------------------------------------------
    query_vector = query_embeddings_data[4].reshape(1, -1)
    print(query_filename_data[4])
    # Example metadata constraints for post-filtering:
    query_meta_class_2 = {"item_weight": ["<", 2], "brand": ["substring", "Amazon"]}
    query_meta_class_3 = {"country": ["exact", "US"]}

    # ----------------------------------------------------------
    # Perform POST-FILTER vector + metadata hybrid search
    #
    # Step 1: ANN vector search retrieves ~50 items
    # Step 2: Metadata is applied AFTER search to filter results
    # ----------------------------------------------------------
    time_start = time.time()
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
