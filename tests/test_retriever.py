import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.retrieval.retriever import Retriever


retriever = Retriever(
    model_name="intfloat/e5-large",
    index_path="data/processed/hnsw_index.faiss",
    k=5
)

query = "What is the capital of France?"
indices, distances = retriever.search(query)

print(f"🔍 Query: {query}")
print(f"📄 Top-5 context indices: {indices}")
print(f"📏 Distances: {distances}")
