 
# persona_pipeline.py

import os
import re
import json
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any

import openai
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import coreferee

# ─── Config & setup ────────────────────────────────────────────────────────────

openai.api_key = os.getenv("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")  # or set directly

# Load spaCy model + coreference
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("coreferee")

# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ChatTurn:
    speaker: str
    text: str
    timestamp: datetime

# ─── Phase 0: Load & clean ────────────────────────────────────────────────────

def load_and_clean_log(path: str) -> List[ChatTurn]:
    """
    Assumes each line in `path` is a JSON object:
      {"speaker": "user"/"assistant", "text": "...", "utc_time": "..."}
    """
    turns: List[ChatTurn] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            text = raw["text"]
            # strip markdown & HTML tags
            text = re.sub(r"<[^>]+>", "", text)
            text = re.sub(r"```[\s\S]*?```", "", text)
            text = text.strip()
            ts = datetime.fromisoformat(raw["utc_time"])
            turns.append(ChatTurn(raw["speaker"], text, ts))
    return turns

# ─── Phase 1: Speaker filtering ───────────────────────────────────────────────

def filter_user_turns(turns: List[ChatTurn]) -> str:
    """Concatenate all user messages into one long string."""
    user_texts = [t.text for t in turns if t.speaker.lower() == "user"]
    return "\n\n".join(user_texts)

# ─── Phase 2: Segmentation ────────────────────────────────────────────────────

def sliding_window_chunks(text: str, max_tokens: int = 1000, overlap: int = 150) -> List[str]:
    """
    Very rough tokenizer based on whitespace:
      - Splits into tokens
      - Yields overlapping windows
    """
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        if end == len(tokens):
            break
        start = end - overlap
    return chunks

def semantic_cut_points(chunks: List[str], threshold: float = 0.75) -> List[str]:
    """
    Merge adjacent chunks unless their embeddings cosine-similarity
    drops below threshold (i.e. a topic shift).
    """
    embs = _get_embeddings(chunks)
    keep = [0]
    for i in range(1, len(chunks)):
        sim = cosine_similarity([embs[i-1]], [embs[i]])[0][0]
        if sim < threshold:
            keep.append(i)
    # re-chunk at each keep index
    out = []
    for idx in keep:
        out.append(chunks[idx])
    return out

def segment_text(text: str) -> List[str]:
    raw_chunks = sliding_window_chunks(text)
    return semantic_cut_points(raw_chunks)

# ─── Helpers: OpenAI Embeddings ───────────────────────────────────────────────

def _get_embeddings(texts: List[str], model: str="text-embedding-3-small") -> List[List[float]]:
    resp = openai.Embedding.create(input=texts, model=model)
    return [data["embedding"] for data in resp["data"]]

# ─── Phase 3: Chunk-wise extraction ───────────────────────────────────────────

GOALS_PATTERNS = [
    r"\bI (?:am )?(?:working on|planning to|hoping to|want to|will)\b([^.\n]+)",
]
INTEREST_PATTERNS = [
    r"\bI (?:like|love|enjoy|am interested in)\b([^.\n]+)",
]
LIFESTYLE_PATTERNS = [
    r"\bI (?:wake up|go to bed|exercise|eat|travel)\b([^.\n]+)",
]

def extract_chunk_info(chunk: str) -> Dict[str, List[str]]:
    doc = nlp(chunk)
    info = {"background": [], "style": [], "goals": [], "lifestyle": [], "interests": []}
    # Background: named entities
    for ent in doc.ents:
        if ent.label_ in ("PERSON","GPE","ORG","DATE","NORP"):
            info["background"].append(ent.text)
    # Goals, interests, lifestyle via regex
    for cat, patterns in [("goals", GOALS_PATTERNS),
                          ("interests", INTEREST_PATTERNS),
                          ("lifestyle", LIFESTYLE_PATTERNS)]:
        for pat in patterns:
            for m in re.finditer(pat, chunk, re.IGNORECASE):
                info[cat].append(m.group(1).strip())
    # Style: simple proxies (length, punctuation)
    avg_sent = sum(len(sent) for sent in doc.sents) / max(len(list(doc.sents)),1)
    formality = "formal" if avg_sent>80 else "casual"
    info["style"].append(f"{formality}, {len(doc.sents)} sentences")
    # dedupe
    for k in info:
        info[k] = list(dict.fromkeys(info[k]))
    return info

# ─── Phase 4: Local summarisation ────────────────────────────────────────────

CHUNK_SUMMARY_PROMPT = """
You are an analytic extractor. Given ONLY the text below, fill the JSON template.
Return JSON only.

TEMPLATE:
{{
  "background": [...],
  "style": [...],
  "goals": [...],
  "lifestyle": [...],
  "interests": [...]
}}

If a field is missing write an empty array.

TEXT: 
{chunk} 

"""

def summarize_chunk(chunk: str) -> Dict[str, List[str]]:
    resp = openai.ChatCompletion.create(
    model="gpt-4o-mini",
    messages=[
    {"role":"system", "content": CHUNK_SUMMARY_PROMPT.format(chunk=chunk)}
    ],
    temperature=0.0,
    max_tokens=300
    )
    return json.loads(resp.choices[0].message.content)
 
def aggregate_facts(all_facts: List[Dict[str, List[str]]],
                    sim_thresh: float = 0.85) -> Dict[str, List[str]]:
    """
    Aggregate multiple chunk-level fact dictionaries into a global summary.
    Uses embedding similarity to deduplicate and cluster similar facts.

    Args:
        all_facts: List of dictionaries with fields like "background", "style", etc.
        sim_thresh: Cosine similarity threshold for considering facts similar.

    Returns:
        A dictionary with deduplicated, representative facts for each category.
    """
    flat: List[Dict[str, Any]] = []
    for fact_dict in all_facts:
        for category, items in fact_dict.items():
            for item in items:
                flat.append({"cat": category, "fact": item})

    # Embed each fact string with category prefix to preserve semantics
    texts = [f"{f['cat']}::{f['fact']}" for f in flat]
    embs = _get_embeddings(texts)  # Assumes access to OpenAI or local embedding model

    # Build graph where similar facts are connected
    G = nx.Graph()
    G.add_nodes_from(range(len(texts)))
    sims = cosine_similarity(embs)

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if sims[i, j] > sim_thresh:
                G.add_edge(i, j)

    # Cluster similar facts
    clusters = nx.connected_components(G)
    summary: Dict[str, List[str]] = {
        "background": [],
        "style": [],
        "goals": [],
        "lifestyle": [],
        "interests": []
    }

    for comp in clusters:
        indices = list(comp)
        best_idx = max(indices, key=lambda idx: len(flat[idx]["fact"]))
        cat = flat[best_idx]["cat"]
        summary[cat].append(flat[best_idx]["fact"])

    return summary

# ─── Phase 6: Global synthesis ───────────────────────────────────────────────

GLOBAL_SYNTH_PROMPT = """
Role: Expert biographer.
Input: facts grouped by category below as JSON.
Task: Write 5–10 paragraphs, one per category in the order:
Background, Communication Style, Current Goals & Projects, Lifestyle & Habits, Personal Interests.
Tone: Neutral third-person narrative.
≤ 700 words total.

FACTS:
{facts_json}
"""

def synthesize_summary(facts: Dict[str, List[str]]) -> str:
    resp = openai.ChatCompletion.create(
    model="gpt-4o-mini",
    messages=[
    {"role":"system","content": GLOBAL_SYNTH_PROMPT.format(facts_json=json.dumps(facts))}
    ],
    temperature=0.7,
    max_tokens=800
    )
    return resp.choices[0].message.content.strip()

# ─── Phase 7: Packaging ──────────────────────────────────────────────────────

def run_pipeline(log_path: str) -> Dict[str, Any]:
    turns = load_and_clean_log(log_path)
    user_text = filter_user_turns(turns)
    chunks = segment_text(user_text)
    # extract & summarise each chunk
    local_summaries = []
    for chunk in chunks:
    struct = extract_chunk_info(chunk)
    # you can choose either rule-based struct OR LLM-based summarisation
    # here we show LLM-based:
    struct = summarize_chunk(chunk)
    local_summaries.append(struct)
    # aggregate
    facts = aggregate_facts(local_summaries)
    # final narrative
    narrative = synthesize_summary(facts)
    # machine-readable persona
    persona_json = {"narrative": narrative, "facts": facts}
    return persona_json

# ─── Example usage ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("log_path", help="path to JSON-line chat log")
    p.add_argument("out_path", help="where to write persona.json")
    args = p.parse_args()

    
    persona = run_pipeline(args.log_path)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(persona, f, indent=2, ensure_ascii=False)
    print(f"Wrote persona to {args.out_path}")
 

 