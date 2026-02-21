import uuid
import os
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# client_qdrant = QdrantClient("localhost", port=6333)
client_qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

collection_name = "knowledge_base"

# Recreate collection safely
if client_qdrant.collection_exists(collection_name):
    client_qdrant.delete_collection(collection_name)

client_qdrant.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

def embed_text(text):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# 🔥 UTF-8 FIX HERE
with open("data/documents.txt", "r", encoding="utf-8") as f:
    text = f.read()

docs = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

points = []

for doc in docs:
    vector = embed_text(doc)
    points.append(
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text": doc}
        )
    )

client_qdrant.upsert(collection_name=collection_name, points=points)

print("Documents indexed successfully.")