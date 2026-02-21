import os
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

# ===== Configuration =====

collection_name = "knowledge_base"
SIMILARITY_THRESHOLD = 0.55

# ===== Environment Variables =====

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")

if not QDRANT_URL:
    raise ValueError("QDRANT_URL is not set")

# Initialize OpenAI client safely
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ===== Helper Functions =====

def get_qdrant_client():
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )


def embed_query(query: str):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return response.data[0].embedding


def retrieve_context(query: str):
    try:
        qdrant_client = get_qdrant_client()
        query_vector = embed_query(query)

        results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=3,
            with_payload=True,
            with_vectors=False
        ).points

        filtered_results = [
            hit for hit in results if hit.score >= SIMILARITY_THRESHOLD
        ]

        if not filtered_results:
            return None

        context = "\n".join(
            [hit.payload.get("text", "") for hit in filtered_results]
        )

        return context

    except UnexpectedResponse as e:
        print("Qdrant query failed:", str(e))
        return None

    except Exception as e:
        print("Unexpected retrieval error:", str(e))
        return None


def generate_answer(query: str):
    context = retrieve_context(query)

    if context is None:
        return "Apologies, I do not currently have information about that in my knowledge base."

    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """
You are a respectful and professional document-based assistant.

Rules:
1. Answer ONLY using the provided context.
2. If the answer is not explicitly found in the context, respond politely with:
   "Apologies, I do not currently have information about that in my knowledge base."
3. Do NOT use prior knowledge.
4. Do NOT guess.
5. Do NOT add information not present in the context.
6. Keep responses concise and professional.
"""
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{query}
"""
            }
        ]
    )

    return completion.choices[0].message.content.strip()