# ============================================================
# ACORN-1 Hybrid Search Implementation for ABO Dataset
# ============================================================

import os
import json
import numpy as np
import hnswlib
import torch
import time
from PIL import Image
from torchvision import models, transforms

# ============================================================
# ACORN-1 Hybrid Class Definition
# ============================================================

class ACORN1HybridSearch:
    """
    Lightweight ACORN-1 hybrid search implementation.
    Integrates metadata filtering directly into HNSW traversal.
    """

    def __init__(self, dim, metadata, space='l2', ef_search_default=10):
        self.dim = dim
        self.space = space
        self.index = hnswlib.Index(space=space, dim=dim)
        self.metadata = metadata  # node_id -> dict of metadata
        self.ef_search_default = ef_search_default
        self.initialized = False

    def init_index(self, max_elements, M=64, ef_construction=200, random_seed=42):
        """Initialize HNSW index."""
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M, random_seed=random_seed)
        self.initialized = True

    def add_items(self, vectors, ids):
        """
        Add vectors, their IDs, and metadata to the index.
        :param vectors: np.ndarray of shape (N, dim)
        :param ids: list of node IDs
        :param metadata_list: list of dicts with metadata for each node
        """
        if not self.initialized:
            raise RuntimeError("Call init_index() before adding items.")
        self.index.add_items(vectors, ids)
        # for i, mid in enumerate(ids):
        #     self.metadata[mid] = metadata_list[i]

    def acorn_search(self, query_vector, query_metadata, filenames, k, meta_search):
        """
        ACORN-1 hybrid search:
        - progressively increases ef_search
        - filters nodes dynamically by metadata
        """
        ef_search = 300
        self.index.set_ef(ef_search)
        final_results = []
        visits = 2
        large_k = 200

        while visits <= meta_search:
            # Perform HNSW ANN search
            # for _ in range(6):
                labels, _ = self.index.knn_query(query_vector, max_visits=visits, k=large_k)
                filtered_results = []
                updated_labels = []
                print(labels.shape)

                for label in labels[0]:
                    node_meta = self.metadata.get(filenames[label], {})
                    if (query_metadata.keys() == {}):
                        continue
                    for query_key in query_metadata.keys():
                        if query_key in node_meta.keys():
                            query_value = query_metadata[query_key]
                            # print(node_meta[query_key][0]["value"])
                            # print(query_value[1])]
                            if (query_value[0] == "exact"):
                                if (query_key == "item_weight" and node_meta[query_key][0]["value"] != query_value[1]):
                                    filtered_results.append(label)
                                elif (query_key == "model_year" and node_meta[query_key][0]["value"] != query_value[1]):
                                    filtered_results.append(label)
                                elif (query_key == "color" and node_meta[query_key][0]["value"] != query_value[1]):
                                    filtered_results.append(label)
                                elif (query_key == "country" and node_meta[query_key] != query_value[1]):
                                    filtered_results.append(label)
                                elif (query_key == "brand" and node_meta[query_key][0]["value"] != query_value[1]):
                                    filtered_results.append(label)
                            elif (query_value[0] == "leq"):
                                if (query_key == "item_weight" and node_meta[query_key][0]["value"] > query_value[1]):
                                    filtered_results.append(label)
                                elif (query_key == "model_year" and node_meta[query_key][0]["value"] > query_value[1]):
                                    filtered_results.append(label)
                            elif (query_value[0] == "geq"):
                                if (query_key == "item_weight" and node_meta[query_key][0]["value"] < query_value[1]):
                                    filtered_results.append(label)
                                elif (query_key == "model_year" and node_meta[query_key][0]["value"] < query_value[1]):
                                    filtered_results.append(label)
                            # print("----")

                # print(f"Labels from HNSW: {labels[0]}")
                print(f"Filtered results: {filtered_results}")
                updated_labels = list(labels[0])
                for i in set(filtered_results):
                    self.index.mark_deleted(i)
                    updated_labels.remove(i)
                # print(f"Updated labels after filtering: {updated_labels}")
                
                visits = visits + 1
                final_results = updated_labels[:k]

        return final_results


if __name__ == "__main__":
    # --------------------------
    # 1. Paths (edit these)
    # --------------------------
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
            # id_to_meta[image_id] = metadata
            path_to_meta[path_to_ids[image_id]] = json.loads(metadata)

    image_dir = "./0*"              # Path to ABO images
    limit = 5000                       # limit images for faster demo

    embeddings_data = np.load("embeddings.npy")
    filename_data = np.load("filenames.npy")
    query_embeddings_data = np.load("embeddings_query.npy")
    query_filename_data = np.load("filenames_query.npy")
    ids = np.arange(len(embeddings_data))
    print(len(ids))
    # print(query_embeddings_data.shape)
    # print(filename_data)
    # print(np.where(filename_data == "02a0de12.jpg")[0])
    # print(path_to_meta["0974ad06.jpg"])


    # # # --------------------------
    # # # 3. Build ACORN-1 index
    # # # --------------------------
    acorn = ACORN1HybridSearch(dim=2048, metadata=path_to_meta)
    acorn.init_index(max_elements=len(ids))
    acorn.add_items(embeddings_data, ids)
    print("HNSW index built successfully!")

    query_vector = query_embeddings_data[1].reshape(1, -1)
    print(query_filename_data[1])
    query_metadata = {"country": ["exact", "US"], "item_weight": ["exact", 1.25], "brand": ["exact", "365 Everyday Value"]}

    time_start = time.time()
    results = acorn.acorn_search(query_vector, query_metadata, filename_data, k=3, meta_search=6)
    time_end = time.time()
    print(f"Search completed in {time_end - time_start:.7f} seconds.")
    if len(results) == 0:
        print("No results found.")
    for r in results:
        print(r)
        print(filename_data[r])
        print(path_to_meta[filename_data[r]])

