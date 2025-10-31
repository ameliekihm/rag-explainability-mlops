import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.retriever import Retriever

retriever = Retriever(
    model_name="intfloat/e5-large",
    index_path="data/processed/hnsw_index.faiss",
    k=1
)

with open("data/processed/contexts.json") as f:
    contexts = json.load(f)

def dummy_generate_answer(query, context):
    return f"Based on the context, this question is likely related to: {query}"

def handler(event, context):
    query = event.get("query", "")

    if not query:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "query is required"})
        }

    indices, distances = retriever.search(query)
    top_index = int(indices[0])
    top_context = contexts[top_index]

    answer = dummy_generate_answer(query, top_context)

    response = {
        "query": query,
        "retrieved_index": top_index,
        "context": top_context,
        "answer": answer
    }

    return {
        "statusCode": 200,
        "body": json.dumps(response)
    }
