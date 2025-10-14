from datasets import load_dataset
import json, os

print("📚 Loading SQuAD v2 dataset...")
dataset = load_dataset("squad_v2")

train_contexts = list(dataset["train"]["context"])
save_path = "data/processed/contexts.json"

os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, "w") as f:
    json.dump(train_contexts, f)

print(f"✅ Contexts saved at {save_path}")
