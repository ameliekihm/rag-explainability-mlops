# Embedding Model Candidates

| Model                    | Key Features                                             | Strengths                                                                                                                                 | Weaknesses                                                                                       |
| ------------------------ | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **e5-large** (Microsoft) | Encoder-only model trained for retrieval and ranking     | - High accuracy on retrieval benchmarks (BEIR, MTEB)<br>- Good for dense retrieval and QA tasks<br>- Optimized for semantic search        | - Large model size (more memory needed)<br>- Mainly English, weaker for multilingual use         |
| **bge-m3** (BAAI)        | Multilingual embedding model (English + other languages) | - State-of-the-art multilingual performance<br>- Works for retrieval, reranking, and classification<br>- Open-source and actively updated | - Slightly newer, less documented than e5<br>- May require fine-tuning for domain-specific tasks |
| **mpnet** (Microsoft)    | Pre-trained on Masked and Permuted Language Modeling     | - Strong sentence-level embeddings<br>- Widely used in semantic similarity and NLI<br>- Efficient for sentence embeddings                 | - Slightly weaker in retrieval than e5/bge<br>- Older model, not the latest benchmarks           |

---

## General Insights

- **e5-large** → Best for **retrieval tasks in English**, strong baseline for QA.
- **bge-m3** → Best for **multilingual settings**, future-proof, but newer.
- **mpnet** → Reliable for **semantic similarity**, efficient, but retrieval performance is weaker than newer models.

---

## Recommendation for RAG Pipeline

- Start with **e5-large** as the main baseline.
- Keep **bge-m3** as a candidate for multilingual or extended experiments.
- Use **mpnet** as a lighter alternative for quick comparison.
