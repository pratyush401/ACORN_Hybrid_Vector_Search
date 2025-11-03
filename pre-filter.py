import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import time

import torch
from torchvision import models, transforms
import json

embeddings = np.load('embeddings.npy')   # shape: (N, D)
assert embeddings.ndim == 2
N, D = embeddings.shape

filenames = np.load('filenames.npy')

path_to_ids = {}
for i in [0,1,2,3,4,5,6,7,8,9,'a','b','c','d','e']:
    with open(f"map0{i}.csv", "r") as f:
        for line in f:
            parts = line.strip().split(",")
            path_to_ids[parts[0]] = parts[3][3:]
print(f"Loaded {len(path_to_ids)} image ID mappings.")

path_to_meta = {}
with open("metadata0.py", "r") as f:
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
            continue

        op, target = query_metadata[query_key]

        if op == "exact":
            if query_key == "item_weight" and node_meta[query_key][0]["value"] != target:
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
            if query_key == "item_weight" and node_meta[query_key][0]["value"] > target:
                return False
            elif query_key == "model_year" and node_meta[query_key][0]["value"] > target:
                return False

        elif op == "geq":
            if query_key == "item_weight" and node_meta[query_key][0]["value"] < target:
                return False
            elif query_key == "model_year" and node_meta[query_key][0]["value"] < target:
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
    print(sub_emb.shape)
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

def postfilter_search(query_vec: np.ndarray,
                      query_meta: dict,
                      embeddings: np.ndarray,
                      filenames: np.ndarray,
                      path_to_meta: dict,
                      top_k: int = 10,
                      large_k: int = 200):
    """
    1) Compute k-NN over the WHOLE collection (or ANN results).
    2) Walk results in similarity order and keep only those that match metadata.

    Good when you want strong semantic ranking and metadata is not super selective.
    """

    # Get a larger candidate set than final top_k so metadata filtering has room
    k_all = min(large_k, embeddings.shape[0])
    cand_idx, cand_scores = ann_naive(embeddings, query_vec, k=k_all)

    results = []
    for rank_all, (idx, score) in enumerate(zip(cand_idx, cand_scores)):
        fname = str(filenames[idx])
        node_meta = path_to_meta.get(fname, {})
        if metadata_matches(node_meta, query_meta):
            results.append({
                "rank":   len(results),   # rank after metadata filtering
                "index":  int(idx),
                "file":   fname,
                "score":  float(score),
                "meta":   node_meta
            })
            if len(results) == top_k:
                break

    print(f"[Post-filter] Returned {len(results)} results (from {k_all} ANN candidates).")
    return results

query_embeddings_data = np.load("embeddings_query.npy")
query_filename_data = np.load("filenames_query.npy")
query_vec = query_embeddings_data[1].reshape(embeddings.shape[1],)
print(query_vec.shape)
query_meta = {}
time_start = time.time()
pre_results = prefilter_search(
    query_vec=query_vec,
    query_meta=query_meta,
    embeddings=embeddings,
    filenames=filenames,
    path_to_meta=path_to_meta,
    top_k=3,
)
time_end = time.time()
print(f"Pre-filter search completed in {time_end - time_start:.7f} seconds.")
for r in pre_results:
    print(r["index"], r["file"])

print("\n=== POSTFILTER SEARCH ===")
time_start = time.time()
post_results = postfilter_search(
    query_vec=query_vec,
    query_meta=query_meta,
    embeddings=embeddings,
    filenames=filenames,
    path_to_meta=path_to_meta,
    top_k=3,
    large_k=200,
)
time_end = time.time()
print(f"Post-filter search completed in {time_end - time_start:.7f} seconds")
for r in post_results:
    print(r["index"], r["file"])