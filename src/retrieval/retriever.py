import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

class Retriever:
    def __init__(self, model_name="intfloat/e5-large", index_path=None, k=5):
        self.model = SentenceTransformer(model_name)
        self.k = k
        self.index = None

        if index_path:
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
                print(f"[INFO] FAISS index loaded from: {index_path}")
            else:
                raise FileNotFoundError(f"Index file not found: {index_path}")
        else:
            print("[WARN] No index path provided. Call build_index() before search.")

    def embed_query(self, query):
        if isinstance(query, str):
            query = [query]
        embeddings = self.model.encode(["query: " + q for q in query], convert_to_numpy=True, show_progress_bar=False)
        # L2 normalize for stable inner product or cosine distance
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-10, None)
        return embeddings.astype(np.float32)

    def build_index(self, context_embeddings):
        if not isinstance(context_embeddings, np.ndarray):
            raise TypeError("context_embeddings must be a NumPy array")
        dim = context_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(context_embeddings)
        print(f"[INFO] Built new FAISS index with {self.index.ntotal} context vectors")

    def search(self, query):
        if self.index is None:
            raise ValueError("FAISS index is not loaded or built. Please initialize or call build_index().")
        query_emb = self.embed_query(query)
        D, I = self.index.search(query_emb, self.k)
        return I[0], D[0]
