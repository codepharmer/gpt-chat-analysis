from __future__ import annotations

import asyncio
import json
import re
from typing import Dict, List, Any

import spacy
import coreferee
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

from .openai_utils import client, get_embeddings

nlp = spacy.load("en_core_web_trf")
nlp.add_pipe("coreferee")

GOALS_PATTERNS = [r"\bI (?:am )?(?:working on|planning to|hoping to|want to|will)\b([^.\n]+)"]
INTEREST_PATTERNS = [r"\bI (?:like|love|enjoy|am interested in)\b([^.\n]+)"]
LIFESTYLE_PATTERNS = [r"\bI (?:wake up|go to bed|exercise|eat|travel)\b([^.\n]+)"]
PRIORITY_PATTERNS = [r"\bMy top priority is\b([^.\n]+)", r"\bI prioritize\b([^.\n]+)"]
DISLIKE_PATTERNS  = [r"\bI (?:don't like|dislike|hate)\b([^.\n]+)"]
PROJECT_PATTERNS  = [r"\bI (?:am working on|current project is)\b([^.\n]+)"]

CLASSIFY_PROMPT = """
Classify the following sentence into one or more of these categories: 
Background, Goals, Lifestyle, Interests, Priorities, Projects, Likes, Dislikes.
Return JSON: {{ "categories": [...], "text": "..." }}.
Sentence:
{sentence}
"""


async def classify_sentence_async(sentence: str, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
    """Classify a single sentence asynchronously with rate limiting."""
    async with semaphore:
        try:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "system", "content": CLASSIFY_PROMPT.format(sentence=sentence)}],
                    temperature=0.0,
                    max_tokens=200
                )
            )
            data = json.loads(resp.choices[0].message.content)
            return {"sentence": sentence, "categories": data.get("categories", []), "success": True}
        except Exception as e:
            return {"sentence": sentence, "categories": [], "success": False, "error": str(e)}


async def classify_sentences_batch(sentences: List[str], max_concurrent: int = 100) -> List[Dict[str, Any]]:
    """Classify multiple sentences concurrently with rate limiting (up to 10k/min)."""
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [classify_sentence_async(sentence, semaphore) for sentence in sentences]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions that occurred
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            final_results.append({
                "sentence": sentences[i],
                "categories": [],
                "success": False,
                "error": str(result)
            })
        else:
            final_results.append(result)
    
    return final_results


def classify_sentence(sent: str) -> List[str]:
    """Classify a single sentence synchronously."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": CLASSIFY_PROMPT.format(sentence=sent)}],
            temperature=0.0,
            max_tokens=200
        )
        data = json.loads(resp.choices[0].message.content)
        return data.get("categories", [])
    except Exception:
        return []


def classify_sentences(sentences: List[str], max_concurrent: int = 100) -> List[Dict[str, Any]]:
    """Classify multiple sentences concurrently (up to 10k requests/minute)."""
    return asyncio.run(classify_sentences_batch(sentences, max_concurrent))


def extract_chunk_info(chunk: str) -> Dict[str, List[str]]:
    doc = nlp(chunk)
    info = {
      "background": [], "style": [], "goals": [], "lifestyle": [],
      "interests": [], "priorities": [], "likes": [], "dislikes": [], "projects": []
    }    
    
    # Extract named entities for background information
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "GPE", "ORG", "DATE", "NORP"):
            info["background"].append(ent.text)
    
    # Use regex patterns for initial extraction
    for cat, patterns in [("goals", GOALS_PATTERNS), 
                          ("interests", INTEREST_PATTERNS),
                            ("lifestyle", LIFESTYLE_PATTERNS),
                            ("priorities", PRIORITY_PATTERNS),
                            ("dislikes", DISLIKE_PATTERNS),
                            ("projects", PROJECT_PATTERNS)]:
        for pat in patterns:
            for m in re.finditer(pat, chunk, re.IGNORECASE):
                info[cat].append(m.group(1).strip())

    sentences = list(doc.sents)
    
    # Extract individual sentences for AI classification
    sentence_texts = [sent.text.strip() for sent in sentences if len(sent.text.strip()) > 10]
    
    # Classify sentences using the async batch classifier
    if sentence_texts:
        try:
            classified_results = classify_sentences(sentence_texts, max_concurrent=50)
            for result in classified_results:
                if result.get("success", False) and result.get("categories"):
                    sentence_text = result["sentence"]
                    categories = result["categories"]
                    
                    # Map categories to our info structure and add classified sentences
                    for category in categories:
                        category_lower = category.lower()
                        if category_lower in info:
                            # Add the full sentence for context
                            info[category_lower].append(sentence_text)
        except Exception as e:
            # If classification fails, fall back to regex-only extraction
            print(f"Warning: Sentence classification failed: {e}")
    
    # Calculate style information
    avg_sent = sum(len(s.text) for s in sentences) / max(len(sentences), 1)
    formality = "formal" if avg_sent > 80 else "casual"
    info["style"].append(f"{formality}, {len(sentences)} sentences")
    
    # Remove duplicates while preserving order
    for k in info:
        info[k] = list(dict.fromkeys(info[k]))
    
    return info


CHUNK_SUMMARY_PROMPT = """You are an analytic extractor. Given ONLY the text below, fill the JSON template.\nReturn JSON only.\n\nTEMPLATE:\n{{\n  \"background\": [...],\n  \"style\": [...],\n  \"goals\": [...],\n  \"lifestyle\": [...],\n  \"interests\": [...]\n}}\n\nIf a field is missing write an empty array.\n\nTEXT:\n{chunk}\n"""


def summarize_chunk(chunk: str) -> Dict[str, List[str]]:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": CHUNK_SUMMARY_PROMPT.format(chunk=chunk)}],
            temperature=0.0,
            max_tokens=4000,
            timeout=30,
        )
        return json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        return {"background": [], "style": [], "goals": [], "lifestyle": [], "interests": [],"priorities": [], "likes": [], "dislikes": [], "projects": []}
    except Exception:
        return {"background": [], "style": [], "goals": [], "lifestyle": [], "interests": [],"priorities": [], "likes": [], "dislikes": [], "projects": []}


def aggregate_facts(all_facts: List[Dict[str, List[str]]], sim_thresh: float = 0.85) -> Dict[str, List[str]]:
    summary: Dict[str, List[str]] = {
        "background": [],
        "style": [],
        "goals": [],
        "lifestyle": [],
        "interests": [],
        "priorities": [], 
        "likes": [], 
        "dislikes": [], 
        "projects": []
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

    texts = [f"{f['cat']}::{f['fact']}" for f in flat]
    embs = get_embeddings(texts)

    G = nx.Graph()
    G.add_nodes_from(range(len(texts)))
    sims = cosine_similarity(embs)
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if sims[i, j] > sim_thresh:
                G.add_edge(i, j)

    clusters = nx.connected_components(G)
    for comp in clusters:
        indices = list(comp)
        best_idx = max(indices, key=lambda idx: len(flat[idx]["fact"]))
        cat = flat[best_idx]["cat"]
        summary[cat].append(flat[best_idx]["fact"])

    processed_indices = set()
    for comp in nx.connected_components(G):
        processed_indices.update(comp)
    for i, fact_item in enumerate(flat):
        if i not in processed_indices:
            summary[fact_item["cat"].strip()].append(fact_item["fact"])

    return summary


GLOBAL_SYNTH_PROMPT = """Role: Expert biographer.\nInput: facts grouped by category below as JSON.\nTask: Write 5–10 paragraphs, one per category in the order: Background, Communication Style, Current Goals & Projects, Lifestyle & Habits, Personal Interests.\nTone: Neutral third-person narrative.\n≤ 2000 words total.\n\nFACTS:\n{facts_json}\n"""


def synthesize_summary(facts: Dict[str, List[str]]) -> str:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": GLOBAL_SYNTH_PROMPT.format(facts_json=json.dumps(facts))}],
            temperature=0.3,
            max_tokens=4000,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return "Unable to generate narrative summary due to processing error."


def extract_chunk_info_ai_only(chunk: str) -> Dict[str, List[str]]:
    """Extract information from chunk using AI sentence classification only."""
    doc = nlp(chunk)
    info = {
      "background": [], "style": [], "goals": [], "lifestyle": [],
      "interests": [], "priorities": [], "likes": [], "dislikes": [], "projects": []
    }
    
    sentences = list(doc.sents)
    sentence_texts = [sent.text.strip() for sent in sentences if len(sent.text.strip()) > 10]
    
    # Use AI classification for all categorization
    if sentence_texts:
        try:
            classified_results = classify_sentences(sentence_texts, max_concurrent=50)
            for result in classified_results:
                if result.get("success", False) and result.get("categories"):
                    sentence_text = result["sentence"]
                    categories = result["categories"]
                    
                    # Map categories to our info structure
                    for category in categories:
                        category_lower = category.lower()
                        if category_lower in info:
                            info[category_lower].append(sentence_text)
        except Exception as e:
            print(f"Warning: AI sentence classification failed: {e}")
            return info
    
    # Calculate style information
    avg_sent = sum(len(s.text) for s in sentences) / max(len(sentences), 1)
    formality = "formal" if avg_sent > 80 else "casual"
    info["style"].append(f"{formality}, {len(sentences)} sentences")
    
    # Remove duplicates while preserving order
    for k in info:
        info[k] = list(dict.fromkeys(info[k]))
    
    return info


def extract_chunk_info_traditional(chunk: str) -> Dict[str, List[str]]:
    """Extract information from chunk using traditional regex and NER methods only."""
    doc = nlp(chunk)
    info = {
      "background": [], "style": [], "goals": [], "lifestyle": [],
      "interests": [], "priorities": [], "likes": [], "dislikes": [], "projects": []
    }    
    
    # Extract named entities for background information
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "GPE", "ORG", "DATE", "NORP"):
            info["background"].append(ent.text)
    
    # Use regex patterns for extraction
    for cat, patterns in [("goals", GOALS_PATTERNS), 
                          ("interests", INTEREST_PATTERNS),
                            ("lifestyle", LIFESTYLE_PATTERNS),
                            ("priorities", PRIORITY_PATTERNS),
                            ("dislikes", DISLIKE_PATTERNS),
                            ("projects", PROJECT_PATTERNS)]:
        for pat in patterns:
            for m in re.finditer(pat, chunk, re.IGNORECASE):
                info[cat].append(m.group(1).strip())

    sentences = list(doc.sents)
    
    # Calculate style information
    avg_sent = sum(len(s.text) for s in sentences) / max(len(sentences), 1)
    formality = "formal" if avg_sent > 80 else "casual"
    info["style"].append(f"{formality}, {len(sentences)} sentences")
    
    # Remove duplicates while preserving order
    for k in info:
        info[k] = list(dict.fromkeys(info[k]))
    
    return info
