import uuid
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from dotenv import load_dotenv
import os

load_dotenv()

client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client_qdrant = QdrantClient("localhost", port=6333)

collection_name = "knowledge_base"

client_qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

def embed_text(text):
    response = client_openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

with open("data/documents.txt", "r") as f:
    docs = f.readlines()

points = []

for doc in docs:
    vector = embed_text(doc)
    points.append(
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text":doc}
        )
    )

client_qdrant.upsert(collection_name=collection_name, points=points)

print("Documents indexed successfully.")