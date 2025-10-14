# 08_rag_end_to_end_results.md

## Experiment: RAG End-to-End Pipeline Test (CPU)

### 1. Setup Summary

| Component | Model / Structure   | Path / Config                            |
| --------- | ------------------- | ---------------------------------------- |
| Embedder  | intfloat/e5-large   | Prebuilt embeddings (context + question) |
| Retriever | FAISS (HNSW Index)  | `/data/processed/hnsw_index.faiss`       |
| Generator | google/flan-t5-base | CPU mode                                 |


### 2. Execution Log

```
[INFO] FAISS index loaded from: /content/drive/MyDrive/RAG Research/data/processed/hnsw_index.faiss
[INFO] Loading model: google/flan-t5-base
[INFO] Model loaded on CPU

============================
Q: What is the capital of France?
A: Paris
============================
```

### 3. Verification Result

- Retriever successfully loaded **HNSW FAISS index**
- Generator successfully loaded **FLAN-T5-base**
- Pipeline correctly produced the answer **"Paris"**
- Confirms **end-to-end RAG system works as expected** on CPU

### 4. Notes

- No hallucination detected in this test.
- Embedding retrieval correctly aligned with context set.
- Ready for extended evaluation (multi-query + accuracy metrics).

> 📅 **Date:** 2025-10-14  
> 👤 **Author:** Amelie Kihm

> 📁 **Notebook Reference:** `notebooks/08_rag_end_to_end.ipynb`
