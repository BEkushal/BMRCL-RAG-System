import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import os
from dotenv import load_dotenv

load_dotenv()

EURON_API_KEY = os.getenv("EURON_API_KEY")
EURON_API_URL = "https://api.euron.one/api/v1/euri/chat/completions"

# Load sentence-transformers model once
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str) -> np.ndarray:
    """
    Generate embedding vector for the given text using SentenceTransformer.
    """
    return embedder.encode(text)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two embedding vectors.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_relevant_chunks(query: str, chunks: list, chunk_embeddings: list, top_k: int = 3) -> list:
    query_emb = get_embedding(query)
    similarities = [cosine_similarity(query_emb, emb) for emb in chunk_embeddings]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def generate_answer_with_euron(context_chunks: list, question: str) -> str:
    context = "\n\n".join(context_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURON_API_KEY}"
    }

    json_data = {
        "model": "gemini-2.0-flash",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(EURON_API_URL, headers=headers, json=json_data)
    if response.status_code == 200:
        data = response.json()
        return data['choices'][0]['message']['content']
    else:
        raise Exception(f"API error {response.status_code}: {response.text}")