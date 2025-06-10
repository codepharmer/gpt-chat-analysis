from __future__ import annotations

from typing import List
from sklearn.metrics.pairwise import cosine_similarity

from .io import ChatTurn
from .openai_utils import get_embeddings

TESTING_MODE = False
TEST_CHUNK_LIMIT = 5


def filter_user_turns(turns: List[ChatTurn]) -> str:
    """Concatenate all user messages into one string."""
    user_texts = [t.text for t in turns if t.speaker.lower() == "user"]
    return "\n\n".join(user_texts)


def sliding_window_chunks(text: str, max_tokens: int = 400, overlap: int = 50) -> List[str]:
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(" ".join(tokens[start:end]))
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


def semantic_cut_points(chunks: List[str], threshold: float = 0.75) -> List[str]:
    if not chunks:
        return []
    if len(chunks) == 1:
        return chunks

    if TESTING_MODE:
        test_chunks = chunks[:TEST_CHUNK_LIMIT]
    else:
        test_chunks = chunks

    try:
        embs = get_embeddings(test_chunks)
        if not embs or len(embs) != len(test_chunks):
            return test_chunks

        keep = [0]
        for i in range(1, len(test_chunks)):
            if sum(embs[i-1]) == 0 or sum(embs[i]) == 0:
                keep.append(i)
                continue
            sim = cosine_similarity([embs[i-1]], [embs[i]])[0][0]
            if sim < threshold:
                keep.append(i)

        return [test_chunks[idx] for idx in keep]
    except Exception:
        return test_chunks


def segment_text(text: str) -> List[str]:
    raw_chunks = sliding_window_chunks(text)
    return semantic_cut_points(raw_chunks)
