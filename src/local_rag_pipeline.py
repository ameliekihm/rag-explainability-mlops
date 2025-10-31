import sys, os, json, torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(1)

from src.retrieval.retriever import Retriever
from src.generation.generator import Generator


# Load retriever
retriever = Retriever(
    model_name="intfloat/e5-large",
    index_path="data/processed/hnsw_index.faiss",
    k=1
)

# Load contexts
with open("data/processed/contexts.json") as f:
    contexts = json.load(f)

# Load generator
generator = Generator(model_name="google/flan-t5-base")

query = input("Enter your question: ")

# Retrieve context
indices, distances = retriever.search(query)
context = contexts[indices[0]]

# Generate answer
answer, logits, attention = generator.generate_answer(query, context, return_details=True)

print("\nRetrieved context:")
print(context)
print("\nAnswer:")
print(answer)
