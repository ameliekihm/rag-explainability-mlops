# Generator Model Evaluation Results

Evaluation Dataset: **SQuAD v2** (validation split, 100 samples)

Metric: **Exact-Match Accuracy**  
Device: **Google Colab CPU**

## 1️⃣ Models Compared

| Model                | Accuracy  | Runtime (approx.) | Notes                                 |
| -------------------- | --------- | ----------------- | ------------------------------------- |
| google/flan-t5-small | 0.370     | ~1 min            | Fastest, lower accuracy               |
| google/flan-t5-base  | **0.420** | ~2.5 min          | ✅ Best balance of accuracy and speed |
| facebook/bart-base   | 0.350     | ~16 min           | Slowest, lowest accuracy              |

## 2️⃣ Summary

- **Best model:** `google/flan-t5-base`
- **Accuracy:** 0.42
- **Rationale:** Achieved highest accuracy among tested models with reasonable runtime.
- **Decision:** `flan-t5-base` will serve as the **default Generator** for the RAG pipeline.
- **Retriever:** `intfloat/e5-large` with FAISS + HNSW remains the embedding model.

---

📅 _Evaluation completed on 2025-10-14 using Colab (CPU runtime)._
