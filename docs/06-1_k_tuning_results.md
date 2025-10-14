### 🔍 FAISS Retrieval Evaluation (Effect of Increasing k)

| Index Type  | Metric | k=10   | k=20   | Δ Change |
| ----------- | ------ | ------ | ------ | -------- |
| **Flat**    | Recall | 0.5888 | 0.6948 | +0.1060  |
|             | MRR    | 0.2200 | 0.2274 | +0.0074  |
| **IVFFlat** | Recall | 0.3837 | 0.4483 | +0.0646  |
|             | MRR    | 0.1456 | 0.1502 | +0.0046  |
| **HNSW**    | Recall | 0.5528 | 0.6524 | +0.0996  |
|             | MRR    | 0.2081 | 0.2153 | +0.0072  |

> Increasing k from 10 to 20 improved recall consistently across all index types,  
> with **Flat index achieving the best overall performance** (Recall@20 ≈ 0.69).
