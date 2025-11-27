import json
import boto3

S3_BUCKET = "rag-pipeline-data-prod"
CONTEXT_KEY = "contexts.json"

E5_ENDPOINT = "rag-e5-embed-endpoint"
FLAN_ENDPOINT = "rag-flan-serverless-endpoint"

s3 = boto3.client("s3")
sm = boto3.client("sagemaker-runtime")

# Load contexts once
tmp_context = "/tmp/contexts.json"
s3.download_file(S3_BUCKET, CONTEXT_KEY, tmp_context)

with open(tmp_context) as f:
    contexts = json.load(f)


def embed(query):
    payload = json.dumps({"inputs": query})
    response = sm.invoke_endpoint(
        EndpointName=E5_ENDPOINT,
        ContentType="application/json",
        Body=payload
    )
    return json.loads(response["Body"].read().decode())


def get_best_context(query):
    # TODO: Replace with vector search later
    return contexts[0]


def gen_answer(query, ctx):
    prompt = f"question: {query} context: {ctx} answer:"
    payload = json.dumps({"inputs": prompt})
    response = sm.invoke_endpoint(
        EndpointName=FLAN_ENDPOINT,
        ContentType="application/json",
        Body=payload
    )
    return response["Body"].read().decode()


def handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
    except:
        body = {}

    query = body.get("query")

    if not query:
        return {
            "statusCode": 400,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*"
            },
            "body": json.dumps({"error": "No query"})
        }

    # ---- RAG pipeline ----
    ctx = get_best_context(query)
    answer = gen_answer(query, ctx)

    return {
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*"
        },
        "body": json.dumps({
            "query": query,
            "context_preview": ctx[:200],
            "answer": answer
        })
    }