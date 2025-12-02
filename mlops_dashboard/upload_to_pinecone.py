import json
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os
from tqdm import tqdm

# Load contexts
with open("data/processed/contexts.json") as f:
    contexts = json.load(f)

# Load embedding model
model = SentenceTransformer("intfloat/e5-large")

# Init Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("rag-index")

print("Uploading vectors to Pinecone...")

batch = []
for i, text in tqdm(enumerate(contexts), total=len(contexts)):
    emb = model.encode(text).tolist()

    batch.append({
        "id": str(i),
        "values": emb,
        "metadata": {"text": text}
    })

    # Upsert every 100
    if len(batch) == 100:
        index.upsert(vectors=batch)
        batch = []

# Final batch
if batch:
    index.upsert(vectors=batch)

print("Upload complete.")
