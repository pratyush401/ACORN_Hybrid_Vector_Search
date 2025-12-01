
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

This project shows:

| Technique | Pros | Cons |
|----------|------|------|
| **Pre-filtering** | Perfect accuracy | Slow for low-selectivity filters |
| **Post-filtering** | Fast | Fails when metadata is highly selective |
| **Hybrid ACORN-style** | Fast + accurate | More complex; requires custom traversal |

Our implementation modifies HNSW search behavior in Python to approximate the ACORN-1 predicate-pushdown approach.

---

# Mapping Report Sections → Code Files 

This section directly links report sections to implementation files.

### **Section 2 — Design & Implementation**

#### **2.1 Dataset Ingestion & Embeddings**
| Report Section | Code File | Description |
|----------------|-----------|-------------|
| 2.1 | `parse-json.py` | Extracts metadata from ABO JSON, aligns with images |
| 2.1 | `survey_metadata.py` | Surveys attribute distributions, used for query selection |
| 2.1 | `vector_embeddings.py` | ResNet-50 inference → saves 2048-dim image vectors |

---

#### **2.2 Metadata Filtering**
| Report Section | Code File | Description |
|----------------|-----------|-------------|
| 2.2 | `pre-filter.py` | Exact pre-filter baseline (metadata → ANN over subset) |
| 2.2 | `post-filter.py` | ANN first → metadata filtering baseline |
| 2.2 | `survey_metadata.py` | Attribute statistics for filter design |

---

#### **2.3 Hybrid Search (ACORN-1 Approximation)**
| Report Section | Code File | Description |
|----------------|-----------|-------------|
| 2.3 | `acorn.py` | Main hybrid search engine: iterative max-visits, metadata-aware traversal, `mark_deleted` pruning |

---

### **Section 3 — Experimental Setup**
| Report Section | Code File | Description |
|----------------|-----------|-------------|
| 3.1 | `acorn.py` | Latency + recall measurement for hybrid search |
| 3.1 | `pre-filter.py` / `post-filter.py` | Latency baselines |
| 3.1 | `vector_embeddings.py` | Embedding generation environment |
| 3.2 | All scripts above | Query metadata classes injected into experiments |

---

### **Section 4 — Evaluation**
| Report Section | Code File | Description |
|----------------|-----------|-------------|
| 4 | `6400Project.ipynb` | Generates figures (Class 1/2/3 latency, accuracy curves), runs experiments |

---

### **Attribution**
| Component | Source | Modified? |
|----------|--------|-----------|
| HNSWlib | https://github.com/nmslib/hnswlib | Yes |
| ResNet-50 (TorchVision) | PyTorch | No |
| ABO Dataset JSON | Berkeley / Amazon | No |

Everything else is original work by our team.

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
- Loads each product image  
- Uses TorchVision transforms  
- Passes through pretrained ResNet‑50  
- Saves 2048‑dim embeddings  
- Builds HNSWlib ANN index  

**Diagram:**

```
Image → Preprocessing → ResNet‑50 → 2048‑dim Vector → Stored in Index
```

---

## **2. Metadata Processing**
Handled by `survey_metadata.py` & `parse-json.py`.

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
query_metadata_class_2_2 = {
    "item_weight": ["<", 2],
    "brand": ["substring", "Amazon"]
}

query_metadata_class_3 = {
    "country": ["exact", "US"],
    "item_weight": ["exact", 1.25],
    "brand": ["exact", "365 Everyday Value"]
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
- Replace HNSWlib with **FAISS**, **Milvus**, **Qdrant**, or **Weaviate**.

### Deploy as an API
- Wrap into **FastAPI** or **Flask** service.

### Learned Re-ranking
- ML model combining vector & metadata scores.

---

## **Performance Notes**
- HNSWlib enables sub‑millisecond search on thousands of embeddings.
- Pre-filtering reduces candidate set dramatically.
- Metadata + vector fusion improves accuracy significantly over pure ANN search.

---

## **Diagrams**

### **Hybrid Scoring Flow**

```
          ┌───────────────┐
          │ Query Vector   │
          └───────┬───────┘
                  ▼
        ┌──────────────────────┐
        │  HNSW Vector Search  │
        └───────┬──────────────┘
                ▼
     ┌─────────────────────┐
     │ Metadata Postfilter │
     └─────────┬───────────┘
               ▼
       ┌─────────────────┐
       │ Final Results    │
       └─────────────────┘
```

## **Acknowledgements**
- **ABO Dataset**  
- **HNSWlib** (Yury Malkov)  
- **PyTorch / TorchVision**  

