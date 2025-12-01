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

    This class wraps:
    - Initialization and loading of HNSWlib vector index
    - Hybrid vector + metadata filtering
    - Post-filtering function
    - Main ACORN traversal loop that blocks nodes failing metadata criteria
    """
    def __init__(self, dim, metadata, space='l2', ef_search_default=10):
        # Store configuration for vector dimension and similarity space
        self.dim = dim
        self.space = space
        self.index = hnswlib.Index(space=space, dim=dim) # Create an empty HNSW index with the given dimensionality
        self.metadata = metadata  # Metadata dictionary mapping filename → metadata dict
        self.ef_search_default = ef_search_default # Default ef_search value for HNSWlib
        self.initialized = False  # Tracks whether index has been initialized

    def init_index(self, max_elements, M=64, ef_construction=200, random_seed=42):
        """Initialize HNSW index with construction parameters."""
        # M controls graph connectivity, ef_construction controls build accuracy/speed
        self.index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M, random_seed=random_seed)
        self.initialized = True

    def add_items(self, vectors, ids):
        """
        Add vectors and their associated IDs to the index.

        :param vectors: np.ndarray of shape (N, dim)
        :param ids: list or array of integer IDs corresponding to vector rows
        """
        if not self.initialized:
            raise RuntimeError("Call init_index() before adding items.")
        self.index.add_items(vectors, ids) # Insert items into HNSW graph

    def post_filter_search(self, query_vector, query_metadata, filenames, k):
        """
        Runs a standard HNSW search, then filters results using metadata rules.

        This method:
        1. Retrieves a larger candidate set (large_k)
        2. For each candidate, checks metadata constraints
        3. Removes candidates that fail metadata filters
        4. Returns top-k remaining candidates

        This is classic post-filtering (vector-first strategy).
        """

        # Increase ef_search to get higher-quality neighbors for post-filtering
        ef_search = 50
        self.index.set_ef(ef_search)
        large_k = 50 # Retrieve larger list so filtering doesn't empty top-k

        # Perform ANN search with a large candidate pool
        labels, _ = self.index.knn_query(query_vector, max_visits=100000, blocked=set(), k=large_k)
        filtered_results = []
        updated_labels = []
        # Iterate through candidate labels returned by HNSW
        for label in labels[0]:
            node_meta = self.metadata.get(filenames[label], {}) # Fetch metadata for this result via filename_data mapping
            # If no metadata requested, skip filtering
            if (query_metadata.keys() == {}):
                continue
            # Check each metadata constraint provided in query
            for query_key in query_metadata.keys():
                # If the node does not contain that metadata field, filter out the node
                if query_key in node_meta.keys():
                    query_value = query_metadata[query_key]
                    # -------------------------
                    # Exact match filtering
                    # -------------------------
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
                    # -------------------------
                    # Less-or-equal filtering
                    # -------------------------
                    elif (query_value[0] == "leq"):
                            if (query_key == "item_weight" and node_meta[query_key][0]["value"] > query_value[1]):
                                filtered_results.append(label)
                            elif (query_key == "model_year" and node_meta[query_key][0]["value"] > query_value[1]):
                                filtered_results.append(label)
                    # -------------------------
                    # Greater-or-equal filtering
                    # -------------------------
                    elif (query_value[0] == "geq"):
                            if (query_key == "item_weight" and node_meta[query_key][0]["value"] < query_value[1]):
                                filtered_results.append(label)
                            elif (query_key == "model_year" and node_meta[query_key][0]["value"] < query_value[1]):
                                filtered_results.append(label)
                    # -------------------------
                    # Less-than filtering
                    # -------------------------
                    elif (query_value[0] == "<"):
                            if (query_key == "item_weight" and node_meta[query_key][0]["value"] >= query_value[1]):
                                filtered_results.append(label)
                            elif (query_key == "model_year" and node_meta[query_key][0]["value"] >= query_value[1]):
                                filtered_results.append(label)
                    # -------------------------
                    # Greater-than filtering
                    # -------------------------
                    elif (query_value[0] == ">"):
                            if (query_key == "item_weight" and node_meta[query_key][0]["value"] <= query_value[1]):
                                filtered_results.append(label)
                            elif (query_key == "model_year" and node_meta[query_key][0]["value"] <= query_value[1]):
                                filtered_results.append(label)
                    # -------------------------
                    # Substring search (text-based features)
                    # -------------------------
                    elif (query_value[0] == "substring"):
                            if (query_key == "color" and query_value[1] not in node_meta[query_key][0]["value"]):
                                filtered_results.append(label)
                            elif (query_key == "country" and query_value[1] not in node_meta[query_key][0]["value"]):
                                filtered_results.append(label)
                            elif (query_key == "brand" and query_value[1] not in node_meta[query_key][0]["value"]):
                                filtered_results.append(label)
                            # print("----")
                else:
                    filtered_results.append(label)  # Missing metadata field — automatically filter out

        # print(f"Labels from HNSW: {labels[0]}")
        # print(f"Filtered results: {filtered_results}")
        # Remove all filtered labels from HNSW output list
        updated_labels = list(labels[0])
        for i in set(filtered_results):
            updated_labels.remove(i)
        # print(f"Updated labels after filtering: {updated_labels}")

        return updated_labels[:k]
        



    def acorn_search(self, query_vector, query_metadata, filenames, k, meta_search):
        """
        Main ACORN-1 hybrid search method.

        This algorithm:
        - Performs repeated HNSW searches with increasing visit limits.
        - Maintains a set of "filtered" nodes that fail metadata checks.
        - Blocks those nodes in future knn_query calls.
        - Continues until meta_search iterations have occurred.
        - Collects and sorts surviving vector matches.

        This approximates ACORN rejection sampling by progressively
        eliminating nodes that violate metadata constraints.
        """

        # High ef_search for stronger recall
        ef_search = 200
        self.index.set_ef(ef_search)
        final_results = []       # Store all accepted (label, dist) pairs
        filtered_set = set()      # Nodes filtered due to failed metadata
        unfiltered_set = set()    # Nodes encountered but not filtered
        visits = 2                # Initial visit cap for HNSW traversal
        large_k = 200             # Retrieve many nodes each iteration

        # ACORN iterative search loop
        while visits <= meta_search:
                # Perform HNSW search with a visit budget, blocking filtered nodes
                labels, dist = self.index.knn_query(query_vector, max_visits=visits, blocked=filtered_set, k=large_k)
                filtered_results = []
                updated_labels = []

                # Evaluate metadata constraints for each retrieved candidate
                for label, d in zip(labels[0], dist[0]):
                    node_meta = self.metadata.get(filenames[label], {})
                    # If no metadata filters specified, skip filtering
                    if (query_metadata.keys() == {}):
                        continue
                    # Check all metadata constraints
                    for query_key in query_metadata.keys():
                        # Node must contain metadata field to be valid
                        if query_key in node_meta.keys():
                            query_value = query_metadata[query_key]
                            # Repeat same filtering logic as post-filter function,
                            # but storing (label, distance) pairs instead of just labels
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
                            # Missing metadata → automatic rejection
                            filtered_results.append((label, d))

                # print(f"Labels from HNSW: {len(labels[0])}")
                # print(f"Filtered results: {len(set(filtered_results))}")
                # Build an updated list of label-distance results
                updated_labels = list(zip(labels[0], dist[0]))
                # print(f"Updated Labels: {len(updated_labels)}")
                # If some results failed filtering, block them for future iterations
                if len(labels[0]) != len(set(filtered_results)):
                    for i,d in set(filtered_results):
                        if i not in unfiltered_set:
                            filtered_set.add(i)
                        updated_labels.remove((i,d))
                else:
                    # If no metadata filters applied this iteration,
                    # treat these nodes as "unfiltered" and do not block them
                    for i,d in set(filtered_results):
                        unfiltered_set.add(i)
                        updated_labels.remove((i,d))
                    # print(f"{visits} Not blocking")
                # print(f"Updated labels after filtering from visit {visits}: {updated_labels}")
                # Increase traversal visits for next ACORN step
                visits = visits + 1
                # Append surviving candidates to final results
                final_results.extend(updated_labels)
        
        # Deduplicate and sort final results by distance
        final_results = list(set(final_results))
        final_results.sort(key=lambda x: x[1])
        # Return top-k results
        return final_results[:k]


if __name__ == "__main__":
    # Load mapping of dataset file names to internal IDs
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

    # Load metadata for each image into path_to_meta dict
    path_to_meta = {}
    with open("metadata-small.py", "r") as f:
        for line in f:
            curr_line = line[1:-1]
            curr_line = curr_line.strip().split(",")
            image_id = curr_line[0]
            metadata = ",".join(curr_line[1:])[:-1]
            path_to_meta[path_to_ids[image_id]] = json.loads(metadata)

    # Load full vector embeddings, filenames, and query data
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


    # ----------------------------------------------------------
    # Build ACORN-1 HNSW index
    # ----------------------------------------------------------
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
    # Prepare a query vector from precomputed embedding file
    query_vector = query_embeddings_data[4].reshape(1, -1)
    print(query_filename_data[4])
    
    # Example metadata query filters
    query_metadata_class_2_2 = {"item_weight": ["<", 2], "brand": ["substring", "Amazon"]}
    query_metadata_class_2_1 = {"country": ["exact", "IN"], "brand": ["substring", "Amazon"]}
    query_metadata_class_3 = {"country": ["exact", "US"]}

    # Run ACORN hybrid search
    time_start = time.time()
    results = hnsw.acorn_search(query_vector, query_metadata_class_2_2, filename_data, k=3, meta_search=10)
    time_end = time.time()
    print(f"Search completed in {time_end - time_start:.7f} seconds.")
    # Output results
    if len(results) == 0:
        print("No results found.")
    for r in results:
        print(r)
        print(filename_data[r[0]])
        # print(path_to_meta[filename_data[r]])
