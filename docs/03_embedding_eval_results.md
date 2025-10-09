# Embedding Model Evaluation Results

I compared three embedding models on 200 SQuAD v2 samples.  
Evaluation metric: **Precision@5, Recall@5** (k = 5).

> All experiments were run on Google Colab (GPU runtime).

## 1. e5-large

- **Embedding dim**: 1024
- **Runtime**: ~8.4s (200 samples on Colab GPU)
- **Precision@5 = 0.695**  
  → About 70% of retrieved passages in top-5 were relevant.
- **Recall@5 = 0.695**  
  → About 70% of all relevant passages were found in top-5.

**Interpretation:** Strong retrieval accuracy. Best balance between speed and quality.

## 2. bge-m3

- **Embedding dim**: 1024
- **Runtime**: ~8.5s
- **Precision@5 = 0.058**
- **Recall@5 = 0.29**

**Interpretation:** Poor precision. Retrieved many irrelevant passages.

## 3. all-mpnet-base-v2

- **Embedding dim**: 768
- **Runtime**: ~2.5s (fastest)
- **Precision@5 = 0.061**
- **Recall@5 = 0.305**

**Interpretation:** Fast runtime, but retrieval accuracy is weak compared to e5-large.

## Conclusion

- **Best model: e5-large**  
  It gives the highest retrieval performance (~70% precision/recall).
- Next step: I will modularize the embedding model in `src/embeddings/embedder.py` and vectorize the full dataset for FAISS indexing.
