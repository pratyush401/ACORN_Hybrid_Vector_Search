import os
import sys
import psutil
from pathlib import Path
import numpy as np
from PIL import Image
import time
import torch
from torchvision import models, transforms
import json

# --------------------------------------------------------------
# Load all embedding vectors and filenames
# embeddings: shape (N, D) where D = embedding dimension
# filenames: parallel array containing image filenames
# --------------------------------------------------------------
embeddings = np.load('embeddings_query_full.npy')   # shape: (N, D)
assert embeddings.ndim == 2
N, D = embeddings.shape

filenames = np.load('filenames_query_full.npy')

# --------------------------------------------------------------
# Load mapping between original filenames and internal ABO IDs.
# The dataset uses many map*k*.csv files that must be merged.
# --------------------------------------------------------------
path_to_ids = {}
for k in [0,1,2,3,4]:
    for i in [0,1,2,3,4,5,6,7,8,9,'a','b','c','d','e','f']:
        # Skip known empty/invalid file combination for the dataset
        if k == 0 and i == 'f':
            continue
        # Open mapping file and extract (filename → ABO ID)
        with open(f"map{k}{i}.csv", "r") as f:
            for line in f:
                parts = line.strip().split(",")
                # parts[3] contains something like "id:xxxxxx"
                # [3:] removes "id:" prefix
                path_to_ids[parts[0]] = parts[3][3:]

print(f"Loaded {len(path_to_ids)} image ID mappings.")

# --------------------------------------------------------------
# Load metadata and associate it with ABO ID → metadata dict
# The metadata-small.py file contains raw JSON lines encoded
# in a list-like structure: [image_id, metadata_dict].
# --------------------------------------------------------------
path_to_meta = {}
with open("metadata-small.py", "r") as f:
    for line in f:
        curr_line = line[1:-1]  # remove brackets
        curr_line = curr_line.strip().split(",")
        image_id = curr_line[0]
        # Join the rest back into JSON and parse it
        metadata = ",".join(curr_line[1:])[:-1]
        path_to_meta[path_to_ids[image_id]] = json.loads(metadata)

# --------------------------------------------------------------
# Naive k-NN implementation for L2 distance
# This is intentionally slow but simple. Used here only on the
# filtered subset of embeddings after metadata filtering.
# --------------------------------------------------------------
def ann_naive(emb: np.ndarray, query_vec: np.ndarray, k: int):
    """
    Naive L2 k-NN search using explicit Python looping.
    emb: (N, D) numpy array
    query_vec: (D,) numpy array
    Returns (indices, distances)
    """
    N, D = emb.shape
    k = min(k, N) # cannot return more neighbors than available
    q = query_vec.astype(np.float32)

    # --- compute squared L2 distance for each embedding ---
    dists = []
    # Compute squared L2 distance from query to each embedding
    for i in range(N):
        diff = emb[i] - q
        d2 = float(np.dot(diff, diff))  # same as np.sum(diff**2)
        dists.append((i, d2))

    # --- sort by distance (ascending) ---
    dists.sort(key=lambda x: x[1])

    # Monitor memory usage (for debugging/performance analysis)
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    print(f"Resident Set Size (RSS): {memory_info.rss / (1024 * 1024):.2f} MB")
    # --- take top-k ---
    topk = dists[:k]
    idx = [i for i, _ in topk]
    scores = [d for _, d in topk]

    return np.array(idx, dtype=int), np.array(scores, dtype=float)

# --------------------------------------------------------------
# Metadata filtering logic used by the pre-filter stage.
# The function checks if a metadata dictionary satisfies the
# query's metadata constraints before vector search runs.
# --------------------------------------------------------------
def metadata_matches(node_meta: dict, query_metadata: dict) -> bool:
    # No metadata constraints → always match
    if not query_metadata:
        return True
    
    for query_key in query_metadata.keys():  # Evaluate each query constraint
        # If metadata field is missing, immediately reject
        if query_key not in node_meta.keys():
            return False

        op, target = query_metadata[query_key]

        # ------------------------
        # Exact match operations
        # ------------------------
        if op == "exact":
            if query_key == "item_weight" and node_meta[query_key][0]["normalized_value"]["value"] != target:
                return False
            elif query_key == "model_year" and node_meta[query_key][0]["value"] != target:
                return False
            elif query_key == "color" and node_meta[query_key][0]["value"] != target:
                return False
            elif query_key == "country" and node_meta[query_key] != target:
                return False
            elif query_key == "brand" and node_meta[query_key][0]["value"] != target:
                return False

        # ------------------------
        # <= operator
        # ------------------------
        elif op == "leq":
            if query_key == "item_weight" and node_meta[query_key][0]["normalized_value"]["value"] > target:
                return False
            elif query_key == "model_year" and node_meta[query_key][0]["value"] > target:
                return False

        # ------------------------
        # >= operator
        # ------------------------
        elif op == "geq":
            if query_key == "item_weight" and node_meta[query_key][0]["normalized_value"]["value"] < target:
                return False
            elif query_key == "model_year" and node_meta[query_key][0]["value"] < target:
                return False

        # ------------------------
        # < operator
        # ------------------------   
        elif op == "<":
            if (query_key == "item_weight" and node_meta[query_key][0]["normalized_value"]["value"] >= target):
                return False
            elif (query_key == "model_year" and node_meta[query_key][0]["value"] >= target):
                return False
        
        # ------------------------
        # > operator
        # ------------------------
        elif op == ">":
            if (query_key == "item_weight" and node_meta[query_key][0]["normalized_value"]["value"] <= target):
                return False
            elif (query_key == "model_year" and node_meta[query_key][0]["value"] <= target):
                return False

        # ------------------------
        # Substring match for text features
        # ------------------------    
        elif (op == "substring"):
            if (query_key == "color" and target not in node_meta[query_key][0]["value"]):
                return False
            elif (query_key == "country" and target not in node_meta[query_key][0]["value"]):
                return False
            elif (query_key == "brand" and target not in node_meta[query_key][0]["value"]):
                return False

    return True

# --------------------------------------------------------------
# Pre-filter search pipeline:
# 1) Use metadata to filter down to a small candidate pool
# 2) Run simple ANN (L2) ONLY on that filtered subset
# --------------------------------------------------------------
def prefilter_search(query_vec: np.ndarray,
                     query_meta: dict,
                     embeddings: np.ndarray,
                     filenames: np.ndarray,
                     path_to_meta: dict,
                     top_k: int = 10):
    """
    1) Use metadata to restrict to a subset of indices.
    2) Run k-NN ONLY within that subset.

    Good when metadata is selective (subset is small).
    This is efficient when metadata constraints are selective
    and reduce the search pool significantly.
    """

    # 1) Collect indices that match the metadata
    candidate_indices = []
    for i, fname in enumerate(filenames):
        fname = str(fname)
        node_meta = path_to_meta.get(fname, {})
        if metadata_matches(node_meta, query_meta):
            candidate_indices.append(i)

    print(f"[Pre-filter] Matched {len(candidate_indices)} of {len(filenames)} items.")

    # If no items match metadata, return empty list
    if not candidate_indices:
        return []

    # 2) Run k-NN only on those candidates
    sub_emb = embeddings[candidate_indices]
    local_idx, scores = ann_naive(sub_emb, query_vec, k=top_k)

    # Combine results into a structured list
    results = []
    for rank, (li, score) in enumerate(zip(local_idx, scores)):
        global_idx = candidate_indices[li]
        fname      = str(filenames[global_idx])
        results.append({
            "rank":   rank,
            "index":  int(global_idx),
            "file":   fname,
            "score":  float(score),
            "meta":   path_to_meta.get(fname, {})
        })

    return results

# --------------------------------------------------------------
# Main script execution:
# Prepare a query, run pre-filter search, print results.
# --------------------------------------------------------------
if __name__ == "__main__":
    # Load example query embedding + corresponding filename
    query_embeddings_data = np.load("embeddings_query_full_query.npy")
    query_filename_data = np.load("filenames_query_full_query.npy")
    # This is one of the test runs, where the print shows which image is being queried
    # The metadata is provided using the following template
    # {"attribute": [operation, value]}
    # "attribute": color, brand, item_weight, model_year, country
    # "operation": "exact", "<", ">", "leq (<=)", "geq (>=)", "substring"
    # "value": string type, int type, float type
    query_vec = query_embeddings_data[4].reshape(embeddings.shape[1],)
    print(query_filename_data[4])
    print(query_vec.shape)
    # Example metadata filters
    query_meta_class_3 = {"country": ["exact", "US"]}
    query_meta_class_2 = {"item_weight": ["<", 2], "brand": ["substring", "Amazon"]}
    # Run pre-filter search
    time_start = time.time()
    pre_results = prefilter_search(
        query_vec=query_vec,
        query_meta=query_meta_class_2,
        embeddings=embeddings,
        filenames=filenames,
        path_to_meta=path_to_meta,
        top_k=3
    )
    time_end = time.time()
    print(f"Pre-filter search completed in {time_end - time_start:.7f} seconds.")
    # Print only index + filename for each match
    for r in pre_results:
        print(r["index"], r["file"])
