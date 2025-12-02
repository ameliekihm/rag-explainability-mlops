import requests
import time
import boto3

API_URL = "https://sdcgj4g5ad.execute-api.us-east-1.amazonaws.com/dev/rag"

cloudwatch = boto3.client("cloudwatch")


def put_metric(name, value):
    cloudwatch.put_metric_data(
        Namespace="RAG_MLOps",
        MetricData=[
            {
                "MetricName": name,
                "Value": value,
                "Unit": "Milliseconds" if name == "Latency" else "Count"
            }
        ]
    )


def call_rag_api(query: str):
    start = time.time()

    payload = {"query": query}
    response = requests.post(API_URL, json=payload)

    latency = (time.time() - start) * 1000
    put_metric("Latency", latency)

    if response.status_code != 200:
        put_metric("ErrorCount", 1)
        return {
            "answer": f"Error from API: {response.status_code}",
            "context_preview": response.text
        }

    return response.json()
