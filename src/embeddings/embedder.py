from sentence_transformers import SentenceTransformer
import torch

class Embedder:
    def __init__(self, model_name: str = "intfloat/e5-large", device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts, batch_size: int = 32):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
