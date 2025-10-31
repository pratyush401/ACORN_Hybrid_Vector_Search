# ============================================================
# ACORN-1 Hybrid Search Implementation for ABO Dataset
# ============================================================

import os
import json
import numpy as np
import hnswlib
import torch
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

    def __init__(self, dim, space='l2', ef_search_default=10):
        self.dim = dim
        self.space = space
        self.index = hnswlib.Index(space=space, dim=dim)
        self.metadata = {}  # node_id -> dict of metadata
        self.ef_search_default = ef_search_default
        self.initialized = False

    def init_index(self, max_elements, M=16, ef_construction=200, random_seed=42):
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

    def _passes_metadata_filter(self, node_meta, query_meta):
        """
        Check whether a node's metadata passes query conditions.
        Supports categorical equality and numeric range filters.
        """
        for key, value in query_meta.items():
            if key not in node_meta:
                return False
            node_value = node_meta[key]

            # If query specifies a tuple => range
            if isinstance(value, tuple) and len(value) == 2:
                low, high = value
                if not (low <= node_value <= high):
                    return False
            else:
                # categorical match
                if node_value != value:
                    return False
        return True

    def acorn_search(self, query_vector, query_metadata, k=10, meta_search=50):
        """
        ACORN-1 hybrid search:
        - progressively increases ef_search
        - filters nodes dynamically by metadata
        """
        ef_search = 1
        final_results = []

        while ef_search <= meta_search:
            # Perform HNSW ANN search
            labels, distances = self.index.knn_query(query_vector, k)
            filtered_results = []

            for label, dist in zip(labels[0], distances[0]):
                node_meta = self.metadata.get(label, {})
                if self._passes_metadata_filter(node_meta, query_metadata):
                    filtered_results.append((label, dist))

            if len(filtered_results) >= k:
                # Enough valid results found
                filtered_results.sort(key=lambda x: x[1])
                final_results = filtered_results[:k]
                break
            else:
                # Expand search frontier
                ef_search += 5
                self.index.set_ef(ef_search)

        return final_results


# # ============================================================
# # ABO Dataset Embedding and Index Building
# # ============================================================

# def build_embeddings_from_abo(image_dir, metadata_path, limit=10000):
#     """
#     Load images + metadata from ABO dataset and generate embeddings.
#     :param image_dir: path to image folder
#     :param metadata_path: path to metadata JSON/CSV file
#     :param limit: number of samples to process
#     :return: (vectors, ids, meta_list)
#     """
#     # 1. Load metadata
#     with open(metadata_path, "r") as f:
#         metadata = json.load(f)

#     # 2. Setup ResNet-50 encoder
#     resnet = models.resnet50(pretrained=True)
#     resnet.fc = torch.nn.Identity()  # remove classification head
#     resnet.eval()

#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])

#     def embed_image(img_path):
#         img = Image.open(img_path).convert("RGB")
#         x = transform(img).unsqueeze(0)
#         with torch.no_grad():
#             vec = resnet(x).numpy().flatten().astype(np.float32)
#         return vec

#     # 3. Build arrays
#     ids, vectors, meta_list = [], [], []
#     for i, (pid, meta) in enumerate(metadata.items()):
#         img_path = os.path.join(image_dir, f"{pid}.jpg")
#         if not os.path.exists(img_path):
#             continue
#         try:
#             vec = embed_image(img_path)
#         except Exception:
#             continue
#         ids.append(i)
#         vectors.append(vec)
#         meta_list.append(meta)
#         if len(ids) >= limit:
#             break

#     vectors = np.stack(vectors).astype(np.float32)
#     return vectors, ids, meta_list


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    # --------------------------
    # 1. Paths (edit these)
    # --------------------------
    path_to_ids = {}
    for i in [0,1,2,3,4,5,6,7,8,9,'a','b','c','d','e']:
        with open(f"map0{i}.csv", "r") as f:
            for line in f:
                parts = line.strip().split(",")
                path_to_ids[parts[0]] = parts[3]
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

    # # # --------------------------
    # # # 2. Build embeddings
    # # # --------------------------
    # # print("Building embeddings from ABO dataset...")
    # # vectors, ids, meta_list = build_embeddings_from_abo(image_dir, metadata_path, limit)
    # # dim = vectors.shape[1]
    # # print(f"Loaded {len(ids)} items with {dim}-dimensional embeddings")

    embeddings_data = np.load("embeddings.npy")
    filename_data = np.load("filenames.npy")
    ids = np.arange(len(embeddings_data))
    print(len(ids))

    # # # --------------------------
    # # # 3. Build ACORN-1 index
    # # # --------------------------
    acorn = ACORN1HybridSearch(dim=2048)
    acorn.init_index(max_elements=len(ids))
    acorn.add_items(embeddings_data, ids)
    print("HNSW index built successfully!")

    # # --------------------------
    # # 4. Example query
    # # --------------------------
    # # Example: "Find black items made after 2015"
    # query_img_path = os.path.join(image_dir, f"{ids[0]}.jpg")
    # query_vector = vectors[0].reshape(1, -1)  # using first image as query
    # query_metadata = {"color": "black", "model_year": (2015, 2022)}

    # results = acorn.acorn_search(query_vector, query_metadata, k=5)
    # print("\n=== Search Results ===")
    # for rid, dist in results:
    #     print(f"ID={rid}, dist={dist:.3f}, metadata={acorn.metadata[rid]}")
