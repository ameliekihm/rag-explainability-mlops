import requests

API_URL = "https://sdcgj4g5ad.execute-api.us-east-1.amazonaws.com/dev/rag"

def call_rag_api(query: str):
    payload = {"query": query}
    response = requests.post(API_URL, json=payload)

    if response.status_code != 200:
        return {"answer": "Error from API", "context_preview": ""}

    return response.json()