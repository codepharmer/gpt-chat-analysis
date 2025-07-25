from __future__ import annotations

from typing import List
import openai
import os
try:
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # Add async client for concurrent operations
    async_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except AttributeError:  # fall back for older library versions
    client = openai
    async_client = None


def get_embeddings(texts: List[str], model: str = "text-embedding-3-large") -> List[List[float]]:
    if not texts:
        return []

    MAX_TOKENS_PER_TEXT = 7500
    MAX_TOKENS_PER_REQUEST = 100000
    CHARS_PER_TOKEN = 4

    truncated = []
    for text in texts:
        estimated = len(text) // CHARS_PER_TOKEN
        if estimated > MAX_TOKENS_PER_TEXT:
            max_chars = MAX_TOKENS_PER_TEXT * CHARS_PER_TOKEN
            truncated.append(text[:max_chars])
        else:
            truncated.append(text)

    all_embeddings = []
    current_batch = []
    current_tokens = 0

    for text in truncated:
        t_tokens = len(text) // CHARS_PER_TOKEN
        if current_tokens + t_tokens > MAX_TOKENS_PER_REQUEST and current_batch:
            try:
                resp = client.embeddings.create(input=current_batch, model=model)
                all_embeddings.extend([d.embedding for d in resp.data])
            except Exception:
                embedding_dim = 1536
                for _ in current_batch:
                    all_embeddings.append([0.0] * embedding_dim)
            current_batch = []
            current_tokens = 0

        current_batch.append(text)
        current_tokens += t_tokens

    if current_batch:
        try:
            resp = client.embeddings.create(input=current_batch, model=model)
            all_embeddings.extend([d.embedding for d in resp.data])
        except Exception:
            embedding_dim = 1536
            for _ in current_batch:
                all_embeddings.append([0.0] * embedding_dim)

    return all_embeddings
