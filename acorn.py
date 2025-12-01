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
import psutil
from torchvision import models, transforms

# ============================================================
# ACORN-1 Hybrid Class Definition
# ============================================================

class HNSWSearch:
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

    def post_filter_search(self, query_vector, query_metadata, filenames, k):
        ef_search = 50
        self.index.set_ef(ef_search)
        large_k = 50

        labels, _ = self.index.knn_query(query_vector, max_visits=100000, blocked=set(), k=large_k)
        filtered_results = []
        updated_labels = []
        for label in labels[0]:
            node_meta = self.metadata.get(filenames[label], {})
            if (query_metadata.keys() == {}):
                continue
            for query_key in query_metadata.keys():
                if query_key in node_meta.keys():
                    query_value = query_metadata[query_key]
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
                    elif (query_value[0] == "<"):
                            if (query_key == "item_weight" and node_meta[query_key][0]["value"] >= query_value[1]):
                                filtered_results.append(label)
                            elif (query_key == "model_year" and node_meta[query_key][0]["value"] >= query_value[1]):
                                filtered_results.append(label)
                    elif (query_value[0] == ">"):
                            if (query_key == "item_weight" and node_meta[query_key][0]["value"] <= query_value[1]):
                                filtered_results.append(label)
                            elif (query_key == "model_year" and node_meta[query_key][0]["value"] <= query_value[1]):
                                filtered_results.append(label)
                    elif (query_value[0] == "substring"):
                            if (query_key == "color" and query_value[1] not in node_meta[query_key][0]["value"]):
                                filtered_results.append(label)
                            elif (query_key == "country" and query_value[1] not in node_meta[query_key][0]["value"]):
                                filtered_results.append(label)
                            elif (query_key == "brand" and query_value[1] not in node_meta[query_key][0]["value"]):
                                filtered_results.append(label)
                            # print("----")
                else:
                    filtered_results.append(label)

            # print(f"Labels from HNSW: {labels[0]}")
        # print(f"Filtered results: {filtered_results}")
        updated_labels = list(labels[0])
        for i in set(filtered_results):
            updated_labels.remove(i)
        # print(f"Updated labels after filtering: {updated_labels}")

        return updated_labels[:k]
        



    def acorn_search(self, query_vector, query_metadata, filenames, k, meta_search):
        ef_search = 200
        self.index.set_ef(ef_search)
        final_results = []
        filtered_set = set()
        unfiltered_set = set()
        visits = 2
        large_k = 200

        while visits <= meta_search:
                labels, dist = self.index.knn_query(query_vector, max_visits=visits, blocked=filtered_set, k=large_k)
                filtered_results = []
                updated_labels = []

                for label, d in zip(labels[0], dist[0]):
                    node_meta = self.metadata.get(filenames[label], {})
                    if (query_metadata.keys() == {}):
                        continue
                    for query_key in query_metadata.keys():
                        if query_key in node_meta.keys():
                            query_value = query_metadata[query_key]
                            if (query_value[0] == "exact"):
                                if (query_key == "item_weight" and node_meta[query_key][0]["normalized_value"]["value"] != query_value[1]):
                                    filtered_results.append((label, d))
                                elif (query_key == "model_year" and node_meta[query_key][0]["value"] != query_value[1]):
                                    filtered_results.append((label, d))
                                elif (query_key == "color" and node_meta[query_key][0]["value"] != query_value[1]):
                                    filtered_results.append((label, d))
                                elif (query_key == "country" and node_meta[query_key] != query_value[1]):
                                    filtered_results.append((label, d))
                                elif (query_key == "brand" and node_meta[query_key][0]["value"] != query_value[1]):
                                    filtered_results.append((label, d))
                            elif (query_value[0] == "leq"):
                                if (query_key == "item_weight" and node_meta[query_key][0]["normalized_value"]["value"] > query_value[1]):
                                    filtered_results.append((label, d))
                                elif (query_key == "model_year" and node_meta[query_key][0]["value"] > query_value[1]):
                                    filtered_results.append((label, d))
                            elif (query_value[0] == "geq"):
                                if (query_key == "item_weight" and node_meta[query_key][0]["normalized_value"]["value"] < query_value[1]):
                                    filtered_results.append((label, d))
                                elif (query_key == "model_year" and node_meta[query_key][0]["value"] < query_value[1]):
                                    filtered_results.append((label, d))
                            elif (query_value[0] == "<"):
                                if (query_key == "item_weight" and node_meta[query_key][0]["normalized_value"]["value"] >= query_value[1]):
                                    filtered_results.append((label, d))
                                elif (query_key == "model_year" and node_meta[query_key][0]["value"] >= query_value[1]):
                                    filtered_results.append((label, d))
                            elif (query_value[0] == ">"):
                                if (query_key == "item_weight" and node_meta[query_key][0]["normalized_value"]["value"] <= query_value[1]):
                                    filtered_results.append((label, d))
                                elif (query_key == "model_year" and node_meta[query_key][0]["value"] <= query_value[1]):
                                    filtered_results.append((label, d))
                            elif (query_value[0] == "substring"):
                                if (query_key == "color" and query_value[1] not in node_meta[query_key][0]["value"]):
                                    filtered_results.append((label, d))
                                elif (query_key == "country" and query_value[1] not in node_meta[query_key][0]["value"]):
                                    filtered_results.append((label, d))
                                elif (query_key == "brand" and query_value[1] not in node_meta[query_key][0]["value"]):
                                    filtered_results.append((label, d))
                            # print("----")
                        else:
                            filtered_results.append((label, d))

                # print(f"Labels from HNSW: {len(labels[0])}")
                # print(f"Filtered results: {len(set(filtered_results))}")
                updated_labels = list(zip(labels[0], dist[0]))
                # print(f"Updated Labels: {len(updated_labels)}")
                if len(labels[0]) != len(set(filtered_results)):
                    for i,d in set(filtered_results):
                        if i not in unfiltered_set:
                            filtered_set.add(i)
                        updated_labels.remove((i,d))
                else:
                    for i,d in set(filtered_results):
                        unfiltered_set.add(i)
                        updated_labels.remove((i,d))
                    # print(f"{visits} Not blocking")
                # print(f"Updated labels after filtering from visit {visits}: {updated_labels}")
        
                visits = visits + 1
                final_results.extend(updated_labels)
        
        final_results = list(set(final_results))
        final_results.sort(key=lambda x: x[1])
        return final_results[:k]


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
    ids = np.arange(len(embeddings_data))
    print(len(ids))
    # print(query_embeddings_data.shape)
    # print(filename_data)
    # print(np.where(filename_data == "02a0de12.jpg")[0])
    # print(path_to_meta["0974ad06.jpg"])


    # # # # --------------------------
    # # # # 3. Build ACORN-1 index
    # # # # --------------------------
    time_start = time.time()
    hnsw = HNSWSearch(dim=2048, metadata=path_to_meta)
    hnsw.init_index(max_elements=len(ids))
    hnsw.add_items(embeddings_data, ids)
    print("HNSW index built successfully!")
    time_end = time.time()
    print(f"Indexing completed in {time_end - time_start:.7f} seconds.")

    # # process = psutil.Process(os.getpid())
    # # memory_info = process.memory_info()

    # # print(f"Resident Set Size (RSS): {memory_info.rss / (1024 * 1024):.2f} MB")

    query_vector = query_embeddings_data[4].reshape(1, -1)
    print(query_filename_data[4])
    query_metadata_class_2_2 = {"item_weight": ["<", 2], "brand": ["substring", "Amazon"]}
    query_metadata_class_2_1 = {"country": ["exact", "IN"], "brand": ["substring", "Amazon"]}
    query_metadata_class_3 = {"country": ["exact", "US"]}

    time_start = time.time()
    results = hnsw.acorn_search(query_vector, query_metadata_class_2_2, filename_data, k=3, meta_search=10)
    time_end = time.time()
    print(f"Search completed in {time_end - time_start:.7f} seconds.")
    if len(results) == 0:
        print("No results found.")
    for r in results:
        print(r)
        print(filename_data[r[0]])
        # print(path_to_meta[filename_data[r]])