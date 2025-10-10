import os
from datasets import load_dataset
from src.embeddings.embedder import Embedder

def main():
    # 1. Load SQuAD v2 dataset
    dataset = load_dataset("squad_v2")

    # Train & Validation splits
    train_contexts = dataset["train"]["context"]
    train_questions = dataset["train"]["question"]

    val_contexts = dataset["validation"]["context"]
    val_questions = dataset["validation"]["question"]

    # 2. Initialize Embedder
    embedder = Embedder(model_name="intfloat/e5-large")

    # 3. Encode train split
    print("Encoding train contexts...")
    train_context_embeddings = embedder.encode(train_contexts)
    print("Encoding train questions...")
    train_question_embeddings = embedder.encode(train_questions)

    # 4. Encode validation split
    print("Encoding validation contexts...")
    val_context_embeddings = embedder.encode(val_contexts)
    print("Encoding validation questions...")
    val_question_embeddings = embedder.encode(val_questions)

    # 5. Save embeddings to processed folder
    save_dir = os.path.join("data", "processed")
    os.makedirs(save_dir, exist_ok=True)

    embedder.save_embeddings(train_context_embeddings, os.path.join(save_dir, "context_embeddings_train.npy"))
    embedder.save_embeddings(train_question_embeddings, os.path.join(save_dir, "question_embeddings_train.npy"))
    embedder.save_embeddings(val_context_embeddings, os.path.join(save_dir, "context_embeddings_val.npy"))
    embedder.save_embeddings(val_question_embeddings, os.path.join(save_dir, "question_embeddings_val.npy"))

    print("✅ All embeddings saved in data/processed/")

if __name__ == "__main__":
    main()
