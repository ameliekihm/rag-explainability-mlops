# FAISS Index Evaluation Results

This document summarizes the retrieval performance of different FAISS index types
on the full **SQuAD v2** dataset (context/question embeddings generated with e5-large).  
We report **Recall@10** and **MRR (Mean Reciprocal Rank)** as evaluation metrics.

## Results Overview

| Index Type               | Recall@10 | MRR    | Notes                                       |
| ------------------------ | --------- | ------ | ------------------------------------------- |
| **Flat (brute force)**   | 0.5888    | 0.2200 | Highest accuracy, but slow and memory-heavy |
| **IVFFlat (clustering)** | 0.3837    | 0.1456 | Fast & memory-efficient, but lower accuracy |
| **HNSW (graph-based)**   | 0.5528    | 0.2081 | Strong balance: high recall, fast retrieval |

## Metric Explanation

- **Recall@10**  
  Percentage of questions where the correct context was found within the top-10 retrieved documents.

  - Example: Recall@10 = 0.5888 → About **59% of questions** had the correct context in the top-10.

- **MRR (Mean Reciprocal Rank)**  
  Measures how high the correct context appears in the ranking.
  - Rank 1 → 1.0
  - Rank 2 → 0.5
  - Rank 5 → 0.2
  - Average of these values across all questions.
  - Example: MRR = 0.2200 → Correct answers usually appear **around 4th–5th place on average**.

## Interpretation

- **Flat** serves as the **baseline** (best accuracy, exhaustive search).
- **IVFFlat** shows the trade-off: much faster, less memory, but accuracy drops.
- **HNSW** achieves near-Flat accuracy while being much faster, making it the **practical choice for production**.

## Next Steps

- Extend experiments with different `k` values (e.g., 5, 20).
- Tune FAISS hyperparameters:
  - **IVFFlat** → adjust `nlist` (clusters) and `nprobe` (search breadth).
  - **HNSW** → adjust `M` (graph connectivity) and `efSearch`.
- Compare retrieval time (latency) in addition to accuracy.
