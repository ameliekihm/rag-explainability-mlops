import requests

API_URL = "https://sdcgj4g5ad.execute-api.us-east-1.amazonaws.com/dev/rag"

def call_rag_api(query: str):
    payload = {"query": query}
    response = requests.post(API_URL, json=payload)

    print("DEBUG STATUS:", response.status_code)
    print("DEBUG BODY:", response.text)

    if response.status_code != 200:
        return {"answer": f"Error from API: {response.status_code}", 
                "context_preview": response.text}

    return response.json()
