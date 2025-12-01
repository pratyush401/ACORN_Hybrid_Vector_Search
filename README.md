
# **ACORN Hybrid Vector Search**

## **Overview**
ACORN Hybrid Vector Search is an end‑to‑end hybrid retrieval system combining **image embeddings**, **vector similarity**, and **structured metadata filtering** to search the **Amazon Berkeley Objects (ABO)** dataset.  
This project implements the ACORN‑1 method:  
- Vector search via **HNSWlib**  
- Metadata filtering (pre & post)  
- ResNet‑50 feature extraction  
- Full hybrid orchestration with real query examples  

It includes scripts for embedding generation, metadata parsing, hybrid search, performance evaluation, and sample test images.

---

## **System Architecture**

```
                ┌─────────────────────────────┐
                │        Input Query Image     │
                └───────────────┬─────────────┘
                                │
                                ▼
                ┌─────────────────────────────┐
                │    ResNet‑50 Embedding       │
                │   (vector_embeddings.py)     │
                └───────────────┬─────────────┘
                                │ 2048‑dim vector
                                ▼
        ┌────────────────────────────────────────────────────┐
        │                ACORN Hybrid Search                  │
        │                     (acorn.py)                      │
        │                                                     │
        │   ┌───────────────┬────────────────┬────────────┐  │
        │   │ Pre‑Filtering │ Vector Search  │ Post‑Filter │  │
        │   │  (metadata)   │  (HNSWlib)     │  (metadata) │  │
        │   └───────────────┴────────────────┴────────────┘  │
        └────────────────────────────────────────────────────┘
                                │
                                ▼
                ┌─────────────────────────────┐
                │     Final Ranked Results     │
                └─────────────────────────────┘
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
 ├── Pre-filter (optional)
 ├── Vector search via HNSWlib
 └── Post-filter (optional)
```

Example filters from the code:

```python
query_metadata_class_2_2 = {"item_weight": ["<", 2], "brand": ["substring", "Amazon"]}
query_metadata_class_2_1 = {"country": ["exact", "IN"], "brand": ["substring", "Amazon"]}
query_metadata_class_3   = {"country": ["exact", "US"]}
```

---

## **Example Usage**

### **Run the hybrid search:**
```bash
python acorn.py
```

### **Replace with your own query image**
Modify in `acorn.py`:

```python
query_filename = "query_images/my_image.jpg"
```

---

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

### 🔧 Improve Embedding Quality
- Replace ResNet‑50 with **ViT**, **CLIP**, **EfficientNet**, or a fine‑tuned model.

### ⚡ Faster / Larger Vector Index
- Replace HNSWlib with **FAISS**, **Milvus**, **Qdrant**, or **Weaviate**.

### 🌐 Deploy as an API
- Wrap into **FastAPI** or **Flask** service.

### 🎯 Learned Re-ranking
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

---

## **License**
Specify your license of choice (MIT recommended).

---

## **Acknowledgements**
- **ABO Dataset**  
- **HNSWlib** (Yury Malkov)  
- **PyTorch / TorchVision**  

---

If you want **badges**, **GIF demos**, or a **visual architecture PNG**, I can generate those too.

