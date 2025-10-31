import sys
import os
import json
import torch

from transformers import T5ForConditionalGeneration, T5Tokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.retriever import Retriever

retriever = Retriever(
    model_name="intfloat/e5-large",
    index_path="data/processed/hnsw_index.faiss",
    k=1
)

with open("data/processed/contexts.json") as f:
    contexts = json.load(f)

model_path = "lambda/model/t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
device = torch.device("cpu")
model = model.to(device)

def generate_answer(query, context):
    prompt = f"question: {query} context: {context} answer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(inputs.input_ids, max_length=120)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

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

    answer = generate_answer(query, top_context)

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
