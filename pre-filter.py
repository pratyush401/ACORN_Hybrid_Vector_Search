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

embeddings = np.load('embeddings_query_full.npy')   # shape: (N, D)
assert embeddings.ndim == 2
N, D = embeddings.shape

filenames = np.load('filenames_query_full.npy')

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

def ann_naive(emb: np.ndarray, query_vec: np.ndarray, k: int):
    """
    Naive L2 k-NN search using explicit Python looping.
    emb: (N, D) numpy array
    query_vec: (D,) numpy array
    Returns (indices, distances)
    """
    N, D = emb.shape
    k = min(k, N)
    q = query_vec.astype(np.float32)

    # --- compute squared L2 distance for each embedding ---
    dists = []
    for i in range(N):
        diff = emb[i] - q
        d2 = float(np.dot(diff, diff))  # same as np.sum(diff**2)
        dists.append((i, d2))

    # --- sort by distance (ascending) ---
    dists.sort(key=lambda x: x[1])

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    print(f"Resident Set Size (RSS): {memory_info.rss / (1024 * 1024):.2f} MB")
    # --- take top-k ---
    topk = dists[:k]
    idx = [i for i, _ in topk]
    scores = [d for _, d in topk]

    return np.array(idx, dtype=int), np.array(scores, dtype=float)

def metadata_matches(node_meta: dict, query_metadata: dict) -> bool:
    if not query_metadata:
        return True

    for query_key in query_metadata.keys():
        if query_key not in node_meta.keys():
            return False

        op, target = query_metadata[query_key]

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

        elif op == "leq":
            if query_key == "item_weight" and node_meta[query_key][0]["normalized_value"]["value"] > target:
                return False
            elif query_key == "model_year" and node_meta[query_key][0]["value"] > target:
                return False

        elif op == "geq":
            if query_key == "item_weight" and node_meta[query_key][0]["normalized_value"]["value"] < target:
                return False
            elif query_key == "model_year" and node_meta[query_key][0]["value"] < target:
                return False
            
        elif op == "<":
            if (query_key == "item_weight" and node_meta[query_key][0]["normalized_value"]["value"] >= target):
                return False
            elif (query_key == "model_year" and node_meta[query_key][0]["value"] >= target):
                return False
            
        elif op == ">":
            if (query_key == "item_weight" and node_meta[query_key][0]["normalized_value"]["value"] <= target):
                return False
            elif (query_key == "model_year" and node_meta[query_key][0]["value"] <= target):
                return False
            
        elif (op == "substring"):
            if (query_key == "color" and target not in node_meta[query_key][0]["value"]):
                return False
            elif (query_key == "country" and target not in node_meta[query_key][0]["value"]):
                return False
            elif (query_key == "brand" and target not in node_meta[query_key][0]["value"]):
                return False

    return True

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
    """

    # 1) Collect indices that match the metadata
    candidate_indices = []
    for i, fname in enumerate(filenames):
        fname = str(fname)
        node_meta = path_to_meta.get(fname, {})
        if metadata_matches(node_meta, query_meta):
            candidate_indices.append(i)

    print(f"[Pre-filter] Matched {len(candidate_indices)} of {len(filenames)} items.")

    if not candidate_indices:
        return []

    # 2) Run k-NN only on those candidates
    sub_emb = embeddings[candidate_indices]
    local_idx, scores = ann_naive(sub_emb, query_vec, k=top_k)

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

if __name__ == "__main__":
    query_embeddings_data = np.load("embeddings_query_full_query.npy")
    query_filename_data = np.load("filenames_query_full_query.npy")
    query_vec = query_embeddings_data[4].reshape(embeddings.shape[1],)
    print(query_filename_data[4])
    print(query_vec.shape)
    query_meta_class_3 = {"country": ["exact", "US"]}
    query_meta_class_2 = {"item_weight": ["<", 2], "brand": ["substring", "Amazon"]}
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
    for r in pre_results:
        print(r["index"], r["file"])