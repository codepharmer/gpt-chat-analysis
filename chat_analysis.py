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

# Testing mode - set to True for development/testing with limited data
TESTING_MODE = False
TEST_CHUNK_LIMIT = 5
TEST_FACT_LIMIT = 10

client = openai.OpenAI(api_key="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")  # Initialize OpenAI client

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
    Load chat log from various JSON formats:
    - Line-by-line JSON objects: {"speaker": "user"/"assistant", "text": "...", "utc_time": "..."}
    - Single JSON array containing message objects
    - OpenAI chat completion format with messages array
    """
    turns: List[ChatTurn] = []
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        
    # Try to parse as single JSON first
    try:
        data = json.loads(content)
          # Handle different JSON structures
        if isinstance(data, list):
            # Check if this is a ChatGPT conversation export (array of conversations)
            for item in data:
                if isinstance(item, dict) and "mapping" in item:
                    # ChatGPT export format - extract messages from mapping
                    mapping = item["mapping"]
                    if isinstance(mapping, dict):
                        for node_id, node in mapping.items():
                            if isinstance(node, dict) and "message" in node and node["message"]:
                                turn = _parse_message_object(node["message"])
                                if turn:
                                    turns.append(turn)
                else:
                    # Regular message object
                    turn = _parse_message_object(item)
                    if turn:
                        turns.append(turn)
        elif isinstance(data, dict):
            # Single object, check for different structures
            if "mapping" in data:
                # Single ChatGPT conversation
                mapping = data["mapping"]
                if isinstance(mapping, dict):
                    for node_id, node in mapping.items():
                        if isinstance(node, dict) and "message" in node and node["message"]:
                            turn = _parse_message_object(node["message"])
                            if turn:
                                turns.append(turn)
            elif "messages" in data and isinstance(data["messages"], list):
                # OpenAI chat completion format
                for msg in data["messages"]:
                    turn = _parse_message_object(msg)
                    if turn:
                        turns.append(turn)
            else:
                # Single message object
                turn = _parse_message_object(data)
                if turn:
                    turns.append(turn)
    except json.JSONDecodeError:
        # Fall back to line-by-line parsing
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                turn = _parse_message_object(raw)
                if turn:
                    turns.append(turn)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Warning: Skipping malformed line {line_num}: {e}")
                continue
    
    return turns

def _parse_message_object(raw: Any) -> ChatTurn:
    """
    Parse a message object from various formats and return a ChatTurn.
    Handles different field names and structures.
    """
    if not isinstance(raw, dict):
        return None
        
    # Extract text content, including ChatGPT export format with content.parts
    text = ""
    # Handle content dict with parts list
    if isinstance(raw.get("content"), dict) and "parts" in raw["content"]:
        parts = raw["content"]["parts"]
        if isinstance(parts, list):
            text = "".join(str(p) for p in parts)
        else:
            text = str(parts)
    elif "text" in raw:
        text = raw["text"]
    elif isinstance(raw.get("content"), str):
        text = raw["content"]
    elif "message" in raw:
        text = raw["message"]
    else:
        return None
      # Extract speaker/role
    speaker = ""
    if "speaker" in raw:
        speaker = raw["speaker"]
    elif "role" in raw:
        speaker = raw["role"]
    elif "author" in raw:
        # Handle ChatGPT format where author is an object with role
        author = raw["author"]
        if isinstance(author, dict) and "role" in author:
            speaker = author["role"]
        else:
            speaker = str(author)
    else:
        speaker = "unknown"
      # Extract timestamp
    timestamp = datetime.now()  # Default fallback
    for time_field in ["create_time", "utc_time", "timestamp", "created_at", "time"]:
        if time_field in raw:
            try:
                if isinstance(raw[time_field], str):
                    timestamp = datetime.fromisoformat(raw[time_field].replace('Z', '+00:00'))
                elif isinstance(raw[time_field], (int, float)):
                    timestamp = datetime.fromtimestamp(raw[time_field])
                break
            except (ValueError, TypeError):
                continue
    
    # Clean text
    if isinstance(text, str):
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = text.strip()
    else:
        text = str(text)
    
    return ChatTurn(speaker, text, timestamp)
# ─── Phase 1: Speaker filtering ───────────────────────────────────────────────

def filter_user_turns(turns: List[ChatTurn]) -> str:
    """Concatenate all user messages into one long string."""
    user_texts = [t.text for t in turns if t.speaker.lower() == "user"]
    return "\n\n".join(user_texts)

# ─── Phase 2: Segmentation ────────────────────────────────────────────────────

def sliding_window_chunks(text: str, max_tokens: int = 400, overlap: int = 50) -> List[str]:
    """
    Very rough tokenizer based on whitespace:
      - Splits into tokens
      - Yields overlapping windows
      - Reduced max_tokens significantly to stay well under embedding model limits
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
    if not chunks:
        return []
    if len(chunks) == 1:
        return chunks
    
    # Apply testing mode if enabled
    if TESTING_MODE:
        test_chunks = chunks[:TEST_CHUNK_LIMIT]
        print(f"Testing mode: Processing {len(test_chunks)} chunks (limited for testing)")
    else:
        test_chunks = chunks
        print(f"Processing {len(test_chunks)} chunks for semantic analysis")
    
    try:
        embs = _get_embeddings(test_chunks)
        if not embs or len(embs) != len(test_chunks):
            print("Warning: Failed to get embeddings, returning original chunks")
            return test_chunks
            
        keep = [0]
        for i in range(1, len(test_chunks)):
            # Check if embeddings are valid (not all zeros)
            if (sum(embs[i-1]) == 0 or sum(embs[i]) == 0):
                keep.append(i)  # Keep boundary if embeddings failed
                continue
                
            sim = cosine_similarity([embs[i-1]], [embs[i]])[0][0]
            if sim < threshold:
                keep.append(i)
                
        # re-chunk at each keep index
        out = []
        for idx in keep:
            out.append(test_chunks[idx])
        return out
    except Exception as e:
        print(f"Warning: Semantic cut points failed: {e}")
        return test_chunks  # Return test chunks if processing fails

def segment_text(text: str) -> List[str]:
    raw_chunks = sliding_window_chunks(text)
    return semantic_cut_points(raw_chunks)

# ─── Helpers: OpenAI Embeddings ───────────────────────────────────────────────

def _get_embeddings(texts: List[str], model: str="text-embedding-3-large") -> List[List[float]]:
    """
    Get embeddings for texts, handling token limits by batching requests and truncating individual texts.
    """
    if not texts:
        return []
    
    # Constants for the text-embedding-3-large model
    MAX_TOKENS_PER_TEXT = 7500  # Stay well under 8192 limit
    MAX_TOKENS_PER_REQUEST = 100000  # Conservative batch limit
    CHARS_PER_TOKEN = 4  # Rough estimation
    
    # Truncate individual texts if they're too long
    truncated_texts = []
    for text in texts:
        estimated_tokens = len(text) // CHARS_PER_TOKEN
        if estimated_tokens > MAX_TOKENS_PER_TEXT:
            max_chars = MAX_TOKENS_PER_TEXT * CHARS_PER_TOKEN
            truncated_text = text[:max_chars]
            truncated_texts.append(truncated_text)
        else:
            truncated_texts.append(text)
    
    # Batch the requests
    all_embeddings = []
    current_batch = []
    current_token_count = 0
    
    for text in truncated_texts:
        text_tokens = len(text) // CHARS_PER_TOKEN
        
        # If adding this text would exceed the batch limit, process current batch
        if current_token_count + text_tokens > MAX_TOKENS_PER_REQUEST and current_batch:
            try:
                resp = client.embeddings.create(input=current_batch, model=model)
                all_embeddings.extend([data.embedding for data in resp.data])
            except Exception as e:
                print(f"Warning: Batch embedding failed: {e}")
                # Add zero embeddings for this batch
                embedding_dim = 1536  # dimension for text-embedding-3-large
                for _ in current_batch:
                    all_embeddings.append([0.0] * embedding_dim)
            
            current_batch = []
            current_token_count = 0
        
        current_batch.append(text)
        current_token_count += text_tokens
    
    # Process remaining batch
    if current_batch:
        try:
            resp = client.embeddings.create(input=current_batch, model=model)
            all_embeddings.extend([data.embedding for data in resp.data])
        except Exception as e:
            print(f"Warning: Final batch embedding failed: {e}")
            # Add zero embeddings for this batch
            embedding_dim = 1536  # dimension for text-embedding-3-large
            for _ in current_batch:
                all_embeddings.append([0.0] * embedding_dim)
    
    return all_embeddings

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
                info[cat].append(m.group(1).strip())    # Style: simple proxies (length, punctuation)
    sentences = list(doc.sents)
    avg_sent = sum(len(sent.text) for sent in sentences) / max(len(sentences), 1)
    formality = "formal" if avg_sent > 80 else "casual"
    info["style"].append(f"{formality}, {len(sentences)} sentences")
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
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system", "content": CHUNK_SUMMARY_PROMPT.format(chunk=chunk)}
            ],
            temperature=0.0,
            max_tokens=4000,
            timeout=30  # Add timeout to prevent hanging
        )
        result = json.loads(resp.choices[0].message.content)
        return result
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON response: {e}")
        return {
            "background": [],
            "style": [],
            "goals": [],
            "lifestyle": [],
            "interests": []
        }
    except Exception as e:
        print(f"Warning: Failed to summarize chunk: {e}")
        return {
            "background": [],
            "style": [],
            "goals": [],
            "lifestyle": [],
            "interests": []
        }
 
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
    # Initialize empty summary structure
    summary: Dict[str, List[str]] = {
        "background": [],
        "style": [],
        "goals": [],
        "lifestyle": [],
        "interests": []
    }
    
    if not all_facts:
        return summary
    
    flat: List[Dict[str, Any]] = []
    for fact_dict in all_facts:
        for category, items in fact_dict.items():
            for item in items:
                flat.append({"cat": category, "fact": item})

    if not flat:
        return summary
        
    # Embed each fact string with category prefix to preserve semantics
    texts = [f"{f['cat']}::{f['fact']}" for f in flat]
    
    # Apply testing mode if enabled
    if TESTING_MODE:
        test_texts = texts[:TEST_FACT_LIMIT]
        test_flat = flat[:TEST_FACT_LIMIT]
        print(f"Testing mode: Processing {len(test_texts)} facts for similarity analysis (out of {len(texts)} total)")
    else:
        test_texts = texts
        test_flat = flat
        print(f"Processing {len(test_texts)} facts for similarity analysis")
    
    embs = _get_embeddings(test_texts)  # Assumes access to OpenAI or local embedding model

    # Build graph where similar facts are connected
    G = nx.Graph()
    G.add_nodes_from(range(len(test_texts)))
    sims = cosine_similarity(embs)

    for i in range(len(test_texts)):
        for j in range(i + 1, len(test_texts)):
            if sims[i, j] > sim_thresh:
                G.add_edge(i, j)

    # Cluster similar facts
    clusters = nx.connected_components(G)

    for comp in clusters:
        indices = list(comp)
        best_idx = max(indices, key=lambda idx: len(test_flat[idx]["fact"]))
        cat = test_flat[best_idx]["cat"]
        summary[cat].append(test_flat[best_idx]["fact"])
    
    # Add remaining facts that weren't clustered
    processed_indices = set()
    for comp in nx.connected_components(G):
        processed_indices.update(comp)
    
    for i, fact_item in enumerate(test_flat):
        if i not in processed_indices:
            summary[fact_item["cat"]].append(fact_item["fact"])

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
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system","content": GLOBAL_SYNTH_PROMPT.format(facts_json=json.dumps(facts))}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Warning: Failed to synthesize summary: {e}")
        return "Unable to generate narrative summary due to processing error."

# ─── Phase 7: Packaging ──────────────────────────────────────────────────────

def run_pipeline(log_path: str) -> Dict[str, Any]:
    print(f"Loading chat log from: {log_path}")
    turns = load_and_clean_log(log_path)
    print(f"Found {len(turns)} chat turns")
    
    user_text = filter_user_turns(turns)
    print(f"User text length: {len(user_text)} characters")
    
    # Handle empty user text
    if not user_text.strip():
        return {
            "narrative": "No user messages found in the chat log.",
            "facts": {
                "background": [],
                "style": [],
                "goals": [],
                "lifestyle": [],
                "interests": []            }
        }
    
    print("Segmenting text into chunks...")
    chunks = segment_text(user_text)
    print(f"Created {len(chunks)} chunks for processing")
      # Handle empty chunks
    if not chunks:
        return {
            "narrative": "No meaningful content found in user messages.",
            "facts": {
                "background": [],
                "style": [],
                "goals": [],
                "lifestyle": [],
                "interests": []
            }
        }
    
    # Apply testing mode if enabled
    if TESTING_MODE:
        test_chunks = chunks[:TEST_CHUNK_LIMIT]
        print(f"Testing mode: Processing only first {len(test_chunks)} chunks out of {len(chunks)} total")
    else:
        test_chunks = chunks
        print(f"Processing all {len(test_chunks)} chunks")
    
    # extract & summarise each chunk
    print("Processing chunks...")
    local_summaries = []
    for i, chunk in enumerate(test_chunks):
        print(f"Processing chunk {i+1}/{len(test_chunks)}")
        struct = extract_chunk_info(chunk)
        # you can choose either rule-based struct OR LLM-based summarisation
        # here we show LLM-based:
        struct = summarize_chunk(chunk)
        local_summaries.append(struct)
    
    # aggregate all summaries
    print("Aggregating facts...")
    facts = aggregate_facts(local_summaries)
    
    # final narrative
    print("Generating final narrative...")
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


