## Embedding Model Candidates

| Model                    | Key Features                                             | Strengths                                                                                                                                 | Weaknesses                                                                                       |
| ------------------------ | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **e5-large** (Microsoft) | Encoder-only model trained for retrieval and ranking     | - High accuracy on retrieval benchmarks (BEIR, MTEB)<br>- Good for dense retrieval and QA tasks<br>- Optimized for semantic search        | - Large model size (more memory needed)<br>- Mainly English, weaker for multilingual use         |
| **bge-m3** (BAAI)        | Multilingual embedding model (English + other languages) | - State-of-the-art multilingual performance<br>- Works for retrieval, reranking, and classification<br>- Open-source and actively updated | - Slightly newer, less documented than e5<br>- May require fine-tuning for domain-specific tasks |
| **mpnet** (Microsoft)    | Pre-trained on Masked and Permuted Language Modeling     | - Strong sentence-level embeddings<br>- Widely used in semantic similarity and NLI<br>- Efficient for sentence embeddings                 | - Slightly weaker in retrieval than e5/bge<br>- Older model, not the latest benchmarks           |

### General Insights

- **e5-large** → Best for **retrieval tasks in English**, strong baseline for QA.
- **bge-m3** → Best for **multilingual settings**, future-proof, but newer.
- **mpnet** → Reliable for **semantic similarity**, efficient, but retrieval performance is weaker than newer models.

### Recommendation for RAG Pipeline

- Start with **e5-large** as the main baseline.
- Keep **bge-m3** as a candidate for multilingual or extended experiments.
- Use **mpnet** as a lighter alternative for quick comparison.

### Model Comparison Plan

We will compare candidate embedding models (e5-large, bge-m3, mpnet) based on:

1. **Memory Usage**

   - GPU memory usage during embedding generation
   - CPU memory usage if GPU is not available

2. **Performance**

   - Embedding generation speed (sentences per second)
   - Throughput when processing a batch of 1000 samples

3. **Retrieval Quality** (to be tested later)
   - Accuracy on a small subset of SQuAD v2 (top-k retrieval)
   - Will be added after initial memory and speed tests

#### Experimental Setup

- Use a fixed sample (1000 question-context pairs) from SQuAD v2
- Run each model under the same environment (batch size, device)
- Record memory and time usage for fair comparison

#### Expected Outcome

- Identify the most memory-efficient model
- Identify the fastest model for embedding generation
- Prepare for later evaluation on retrieval quality
