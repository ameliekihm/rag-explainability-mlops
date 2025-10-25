import faiss
import numpy as np
import os

def build_hnsw_index(emb_path, save_path, m=32, ef=128):
    print(f"📦 Loading embeddings from {emb_path}")
    embeddings = np.load(emb_path).astype(np.float32)
    dim = embeddings.shape[1]

    index = faiss.IndexHNSWFlat(dim, m)
    index.hnsw.efConstruction = ef
    index.add(embeddings)
    faiss.write_index(index, save_path)

    print(f"✅ HNSW index saved at {save_path}")

if __name__ == "__main__":
    emb_path = "data/processed/context_embeddings_mini.npy"
    save_path = "data/processed/hnsw_index.faiss"
    build_hnsw_index(emb_path, save_path)
