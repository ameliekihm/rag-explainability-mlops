# 07. FAISS Index Tuning Results

This notebook evaluates **Flat**, **IVFFlat**, and **HNSW** indexes on precomputed embeddings.  
Metrics include **Recall@k**, **MRR (Mean Reciprocal Rank)**, and **Latency (ms/query)**.

---

## 1. Flat (Exact Search Baseline)

| k   | Recall@k | MRR    | Latency (ms/query) |
| --- | -------- | ------ | ------------------ |
| 5   | 0.4064   | 0.1957 | 2.80               |
| 10  | 0.5888   | 0.2200 | 2.76               |
| 20  | 0.6948   | 0.2274 | 2.77               |

**Summary:**

- Provides the highest recall.
- Latency is ~2.8 ms/query, relatively slow compared to ANN methods.

## 2. IVFFlat (Approximate Search)

Evaluated with `nlist=100` clusters and varying `nprobe`.

| nprobe | k   | Recall@k | MRR    | Latency (ms/query) |
| ------ | --- | -------- | ------ | ------------------ |
| 1      | 5   | 0.2674   | 0.1302 | 0.37               |
| 1      | 10  | 0.3837   | 0.1456 | 0.36               |
| 1      | 20  | 0.4483   | 0.1502 | 0.36               |
| 5      | 5   | 0.3700   | 0.1790 | 2.14               |
| 5      | 10  | 0.5338   | 0.2007 | 2.14               |
| 5      | 20  | 0.6273   | 0.2073 | 2.16               |
| 10     | 5   | 0.3879   | 0.1872 | 4.57               |
| 10     | 10  | 0.5606   | 0.2101 | 4.57               |
| 10     | 20  | 0.6599   | 0.2171 | 4.56               |

**Summary:**

- Increasing `nprobe` improves recall but increases latency.
- `nprobe=5` achieves a reasonable balance (Recall@20 ≈ 0.63, Latency ≈ 2.1 ms).
- Still slower than HNSW when targeting high recall.

## 3. HNSW (Graph-based ANN)

Evaluated with `M=32`, `efConstruction=200`, and varying `efSearch`.

| efSearch | k   | Recall@k | MRR    | Latency (ms/query) |
| -------- | --- | -------- | ------ | ------------------ |
| 16       | 5   | 0.3077   | 0.1525 | 0.11               |
| 16       | 10  | 0.4658   | 0.1773 | 0.13               |
| 16       | 20  | 0.5522   | 0.1850 | 0.12               |
| 32       | 5   | 0.3437   | 0.1696 | 0.17               |
| 32       | 10  | 0.5202   | 0.1966 | 0.17               |
| 32       | 20  | 0.6120   | 0.2033 | 0.17               |
| 64       | 5   | 0.3648   | 0.1795 | 0.26               |
| 64       | 10  | 0.5528   | 0.2080 | 0.26               |
| 64       | 20  | 0.6521   | 0.2151 | 0.28               |

**Summary:**

- HNSW achieves the best balance of speed and accuracy.
- At `efSearch=64`, Recall@20 ≈ 0.65 with latency ≈ 0.28 ms/query (10× faster than Flat with only minor accuracy drop).
- Recommended configuration for deployment.

## Key Takeaways

- **Flat** = highest accuracy, but slow.
- **IVFFlat** = tunable trade-off via `nprobe`, but latency grows quickly.
- **HNSW** = best speed–accuracy balance; `efSearch=32~64` recommended.
