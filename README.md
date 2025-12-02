
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
| 2.1 | `vector_embeddings.py` | Generates 2048-dim visual embeddings using ResNet-50 |

---

---

#### **2.3 Hybrid Search (ACORN-1 (ACORN) Approximation)**
| Report Section | Code File | Description |
|----------------|-----------|-------------|
| 2.3 | `acorn.py` | Primary hybrid search engine |

---

### **Section 3 — Experimental Setup**
| Report Section | Code File | Description |
|----------------|-----------|-------------|
| 3.1 | `acorn.py` | Latency + recall measurements for hybrid approach |
| 3.1 | `pre-filter.py` / `post-filter.py` | Baseline latency/accuracy measurements |
| 3.2 | `survey_metadata.py` | Surveys metadata distributions; used for query class design |
| 3.2 | All scripts | Query metadata classes implemented and executed across all methods |

---

---

#### **4.3 Attribute aware (ACORN-1 (ACORN) Approximation)**
| Report Section | Code File | Description |
|----------------|-----------|-------------|
| 4.3 | `acorn_attribute_specific.py` | Implements the new graph construction with model year attribute |

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
  ┌───────────────────────────────────────────────────────── ┐
  │                     ACORN Hybrid Search                  │
  │                          (acorn.py)                      │
  │                                                          │
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
ACORN_Hybrid_Vector_Search-main/ (only relevant files shown)
│
├── acorn.py                 # Main ACORN hybrid search engine
├── acorn_attribute_specific.py # Main ACORN hybrid search engine
├── vector_embeddings.py     # Generates image embeddings with ResNet‑50
├── survey_metadata.py       # Loads + normalizes metadata
├── pre-filter.py            # Pre-filter implementation
├── post-filter.py           # Post-filter implementation
├── parse-json.py            # Parses ABO metadata JSON
│
├── hnswlib.zip              # Modified ANN index zip file
├── mappings.zip             # Metadata mappings
├── metadata-small.zip       # Processed metadata dicts
│
├── query_images/            # Sample query images
├── query1Blanket.jpg
├── query1Handbag.jpg
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

## **Example Usage**

### Unzip the metadata-small.zip to get a file called metadata-small.py 
```bash
unzip metadata-small.zip
```

### Get the vector embeddings for all images (need to download 0*{0-e}/, 1*/, 2*/, 3*/, 4*/ from ABO website)
```bash
python3 vector-embeddings.py 
python3 vector-embeddings.py <query image directory> (for embeddings of query image)
```
There are some image directories included here for reference, but it is not the entire dataset (please note). In order to get all of it, it is required to download it from the ABO website
linked here https://amazon-berkeley-objects.s3.amazonaws.com/index.html#download
Please note for the above, the name of the files are hardcoded and need to be changed to replicate results
Use the same names when running the below files

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

## **Example Search Output (example contains acorn.py, others look similar)**
```
Loaded 123099 image ID mappings.
123098
HNSW index built successfully!
Indexing completed in 51.3545160 seconds.
Resident Set Size (RSS): 2405.91 MB
query4Bagpack-AmazonBasics,1.95,IT.jpg
Search completed in 0.0035188 seconds.
(122391, 0.0)
4f8e1a66.jpg
(44167, 19.771643)
1de9d9db.jpg
(3959, 22.932066)
0254c86a.jpg
```

---

## **Installation**

### **1. Install Dependencies**
```bash
pip install torch torchvision numpy hnswlib pillow psutil
```
---

## **Extending the System**

### Improve Embedding Quality
- Replace ResNet‑50 with **EfficientNet**, or a fine‑tuned model.

### Faster / Larger Vector Index
- Replace HNSWlib with **FAISS HMSW**, **IVF-PQ**, **Milvus/Qdrant**.

### Deploy as an API
- Wrap into **FastAPI** or **Flask** microservice.

### Learned Re-ranking
- ML model combining vector & metadata scores.

---

```

Acknowledgements
- ABO Dataset
- HNSWlib (Yury Malkov)
- PyTorch / TorchVision


