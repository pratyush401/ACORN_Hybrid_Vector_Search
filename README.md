
# ACORN Hybrid Vector Search
A hybrid vector search system combining **image embeddings**, **HNSW-based ANN search**, and **structured metadata filtering** on the **Amazon Berkeley Objects (ABO)** dataset.  
Implements an approximate version of **ACORN-1** with:

- Metadata-aware hybrid traversal  
- Pre-filtering and post-filtering baselines  
- ResNet-50 embeddings  
- Full evaluation pipeline (latency, recall, query selectivity classes)

---

# Project Overview

Hybrid search answers queries like:

> *“Find items visually similar to this image **AND** matching metadata filters (brand, weight, country…)”*

This project compares three approaches:

| Technique | Pros | Cons |
|----------|------|------|
| **Pre-filtering** | Perfect accuracy | Very slow when filter selects many items |
| **Post-filtering** | Fast | Accuracy collapses when metadata is highly selective |
| **Hybrid ACORN-style** | Fast + accurate | More complex; requires custom graph traversal |

Our implementation modifies HNSWlib’s Python bindings to approximate ACORN-1 via:

- **max_visits** (limits last-layer traversal depth) 
- **blocked_set** (tracks nodes failing metadata filters)  
- Iterative refinement with metadata-aware pruning

---

# Mapping Report Sections → Code Files 

This section maps each major report section to the corresponding implementation files.

### **Section 2 — Design & Implementation**

#### **2.1 Dataset Ingestion & Embeddings**
| Report Section | Code File | Description |
|----------------|-----------|-------------|
| 2.1 | `parse-json.py` | Extracts and aligns ABO metadata with image filenames |
| 2.1 | `survey_metadata.py` | Surveys metadata distributions; used for query class design |
| 2.1 | `vector_embeddings.py` | Generates 2048-dim visual embeddings using ResNet-50; builds HNSW index |

---

#### **2.2 Metadata Filtering**
| Report Section | Code File | Description |
|----------------|-----------|-------------|
| 2.2 | `pre-filter.py` | Exact pre-filter → brute-force L2 search on filtered subset |
| 2.2 | `post-filter.py` | ANN search (large-k) → metadata pruning |
| 2.2 | `survey_metadata.py` | Utilities for metadata selection and filtering |

---

#### **2.3 Hybrid Search (ACORN-1 (ACORN) Approximation)**
| Report Section | Code File | Description |
|----------------|-----------|-------------|
| 2.3 | `acorn.py` | Primary hybrid search engine; implements max_visits, blocked_set, metadata-aware traversal |

---

### **Section 3 — Experimental Setup**
| Report Section | Code File | Description |
|----------------|-----------|-------------|
| 3.1 | `acorn.py` | Latency + recall measurements for hybrid approach |
| 3.1 | `pre-filter.py` / `post-filter.py` | Baseline latency/accuracy measurements |
| 3.2 | All scripts | Query metadata classes implemented and executed across all methods |

---

### **Section 4 — Evaluation**
| Report Section | Code File | Description |
|----------------|-----------|-------------|
| 4 | `6400Project.ipynb`, `query_images` | Runs full evaluation suite; generates plots included in the final report |

---

### **Attribution**
| Component | Source | Modified? |
|----------|--------|-----------|
| HNSWlib | https://github.com/nmslib/hnswlib | Yes — extended with max_visits + selective blocking |
| ResNet-50 | TorchVision (PyTorch) | No |
| ABO Dataset JSON | Berkeley / Amazon | No |
| All Python logic (ACORN, baselines, ingestion, evaluation) | Original work | No |

---

## **System Architecture**

```
                ┌──────────────────────────┐
                │     Query Input Image     │
                └───────────────┬──────────┘
                                │
                                ▼
                ┌──────────────────────────┐
                │  ResNet-50 Embeddings     │
                │ (vector_embeddings.py)    │
                └───────────────┬──────────┘
                        2048-dim │ vector
                                ▼
  ┌─────────────────────────────────────────────────────────┐
  │                     ACORN Hybrid Search                  │
  │                          (acorn.py)                     │
  │                                                         │
  │   ┌─────────────────┬─────────────────┬────────────────┐ │
  │   │   Pre-filter    │  HNSW Vector    │   Post-filter   │ │
  │   │ (metadata→ANN)  │    Search       │  (ANN→metadata) │ │
  │   └─────────────────┴─────────────────┴────────────────┘ │
  └─────────────────────────────────────────────────────────┘
                                │
                                ▼
                ┌──────────────────────────┐
                │   Final Ranked Results    │
                └──────────────────────────┘
```

---

## **Repository Structure**

```
ACORN_Hybrid_Vector_Search-main/
│
├── acorn.py                 # Main ACORN hybrid search engine
├── vector_embeddings.py     # Generates image embeddings with ResNet‑50
├── survey_metadata.py       # Loads + normalizes metadata
├── pre-filter.py            # Pre-filter implementation
├── post-filter.py           # Post-filter implementation
├── parse-json.py            # Parses ABO metadata JSON
│
├── filenames.npy            # Mapping index ↔ image filename
├── hnswlib.zip              # Serialized ANN index
├── mappings.zip             # Metadata mappings
├── metadata.zip             # Processed metadata dicts
│
├── query_images/            # Sample query images
├── query1Blanket.jpg
├── query1Handbag.jpg
├── query2.jpg
... etc
│
└── 6400Project.ipynb        # Experiments, evaluation & notes
```

---

## **Hybrid Search Pipeline**

### **1. Embedding Generation (ResNet‑50)**
Implemented in `vector_embeddings.py`:
- Loads each image  
- Uses TorchVision transforms  
- Passes through pretrained ResNet‑50  
- Extracts and saves 2048‑dim embeddings  
- Builds HNSWlib ANN index  

**Diagram:**

```
Image → Preprocessing → ResNet‑50 → 2048‑dim Vector → Stored in Index
```

---

## **2. Metadata Processing**
Handled by `survey_metadata.py` and `parse-json.py`.

Metadata fields include:
- Brand  
- Item weight  
- Country  
- Product category  
- Text descriptions  

Metadata queries follow the format:

```python
{"brand": ["substring", "Amazon"], "item_weight": ["<", 2]}
```

---

## **3. Hybrid Search (ACORN‑1)**

Defined in `acorn.py` as:

```
ACORN Search
 ├── Pre-filter 
 ├── Vector search via HNSWlib
 └── Post-filter 
```

Example filters from the code:

```python
query_metadata_class_2_2 = {"item_weight": ["<", 2], "brand": ["substring", "Amazon"]}
query_metadata_class_2_1 = {"country": ["exact", "IN"], "brand": ["substring", "Amazon"]}
query_metadata_class_3   = {"country": ["exact", "US"]}
```

---

## **Example Usage**

### **Run the ACORN hybrid search:**
```bash
python acorn.py
```

### **Run the pre-filter baseline:**
```bash
python pre-filter.py
```

### **Run the post-filter baseline:**
```bash
python post-filter.py
```

### **Replace with your own query image**
Modify in `acorn.py`:

```python
query_filename = "query_images/my_image.jpg"
```

---

## **Example Query Metadata**
```
query_metadata_class_2_1 = {
    "country": ["exact", "IN"],
    "brand": ["substring", "Amazon"]
}

query_metadata_class_3 = {
    "country": ["exact", "US"]
}
```

## **Example Search Output**
```
Search completed in 0.0038412 seconds.
(1342, 0.1421)
product_1342.jpg
Brand: AmazonBasics
Country: IN
Weight: 1.3

(875, 0.1514)
product_875.jpg
Brand: AmazonBasics
Country: IN
Weight: 0.9
```

---

## **Installation**

### **1. Install Dependencies**
```bash
pip install torch torchvision numpy hnswlib pillow psutil
```

### **2. Extract pretrained assets**
```bash
unzip hnswlib.zip     -d hnswlib_index/
unzip metadata.zip    -d metadata/
unzip mappings.zip    -d mappings/
```

---

## **Extending the System**

### Improve Embedding Quality
- Replace ResNet‑50 with **ViT**, **CLIP**, **EfficientNet**, or a fine‑tuned model.

### Faster / Larger Vector Index
- Replace HNSWlib with **FAISS HMSW**, **IVF-PQ**, **Milvus/Qdrant**.

### Deploy as an API
- Wrap into **FastAPI** or **Flask** microservice.

### Learned Re-ranking
- ML model combining vector & metadata scores.

---

## **Performance Notes**
- Hybrid ACORN improves latency by 9–10× vs pre-filter
- Post-filter only works when metadata strongly correlates with visual similarity
- Attribute-aware ACORN boosts accuracy for extremely selective predicates

```

---
## **Acknowledgements**
- **ABO Dataset**  
- **HNSWlib** (Yury Malkov)  
- **PyTorch / TorchVision**
---

