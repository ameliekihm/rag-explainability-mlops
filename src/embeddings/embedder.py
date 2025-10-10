from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import os
from typing import List, Union


class Embedder:
    def __init__(self, model_name: str = "intfloat/e5-large", device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = True,
        to_numpy: bool = True,
        dtype=np.float32
    ):
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=to_numpy
        )

        if to_numpy:
            embeddings = embeddings.astype(dtype)

        return embeddings

    def save_embeddings(self, embeddings: np.ndarray, filename: str):
        # Always save inside data/processed/
        save_dir = os.path.join("data", "processed")
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)

        np.save(filepath, embeddings)
        print(f"✅ Embeddings saved to {filepath}")
