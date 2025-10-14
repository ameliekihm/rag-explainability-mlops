import os
import numpy as np
from datasets import load_dataset
from src.embeddings.embedder import Embedder

def main():
    # 1. Load SQuAD v2 dataset
    print("📚 Loading SQuAD v2 dataset...")
    dataset = load_dataset("squad_v2")

    # Train & Validation splits
    train_contexts = dataset["train"]["context"]
    train_questions = dataset["train"]["question"]
    val_contexts = dataset["validation"]["context"]
    val_questions = dataset["validation"]["question"]

    # 2. Initialize Embedder
    embedder = Embedder(model_name="intfloat/e5-large")
    print(f"✅ Embedder initialized with model: {embedder.model_name}")

    # 3. Encode train split
    print("\n🚀 Encoding train split...")
    train_context_embeddings = embedder.encode(train_contexts, batch_size=64)
    train_question_embeddings = embedder.encode(train_questions, batch_size=64)

    # 4. Encode validation split
    print("\n🚀 Encoding validation split...")
    val_context_embeddings = embedder.encode(val_contexts, batch_size=64)
    val_question_embeddings = embedder.encode(val_questions, batch_size=64)

    # 5. Normalize embeddings (stability for FAISS inner-product search)
    def normalize(x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(norms, 1e-10, None)

    train_context_embeddings = normalize(train_context_embeddings)
    train_question_embeddings = normalize(train_question_embeddings)
    val_context_embeddings = normalize(val_context_embeddings)
    val_question_embeddings = normalize(val_question_embeddings)

    # 6. Save embeddings to processed folder
    save_dir = os.path.join("data", "processed")
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "context_embeddings_train.npy"), train_context_embeddings)
    np.save(os.path.join(save_dir, "question_embeddings_train.npy"), train_question_embeddings)
    np.save(os.path.join(save_dir, "context_embeddings_val.npy"), val_context_embeddings)
    np.save(os.path.join(save_dir, "question_embeddings_val.npy"), val_question_embeddings)

    print("\n✅ All embeddings normalized and saved to data/processed/")

if __name__ == "__main__":
    main()
