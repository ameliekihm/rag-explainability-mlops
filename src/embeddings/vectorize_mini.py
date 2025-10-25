import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
from src.embeddings.embedder import Embedder
import json
from datasets import load_dataset
import random
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

def main():
    print("📚 Loading SQuAD v2 dataset...")
    ds = load_dataset("squad_v2", split="train[:5000]")  
    contexts = list(set(ds["context"]))
    print(f"Loaded {len(contexts)} unique contexts.")

    
    random.seed(42)
    random.shuffle(contexts)
    sampled = contexts[:1500]  

    sentences = []
    for c in sampled:
        for s in nltk.sent_tokenize(c):
            if len(s.split()) > 3:  
                sentences.append(s.strip())

    print(f"✅ Total sentences after filtering: {len(sentences)}")

    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/contexts.json", "w") as f:
        json.dump(sentences, f, indent=2)
    print("💾 contexts.json saved.")

    embedder = Embedder(model_name="intfloat/e5-large")
    passages = [f"passage: {c}" for c in sentences]
    embeddings = embedder.encode(passages, batch_size=64)
    np.save("data/processed/context_embeddings_mini.npy", embeddings)
    print("✅ context_embeddings_mini.npy rebuilt successfully!")

if __name__ == "__main__":
    main()
