from __future__ import annotations

import json
from typing import Any, Dict, Optional

from .io import load_and_clean_log
from .segments import (
    filter_user_turns,
    segment_text,
    TESTING_MODE,
    TEST_CHUNK_LIMIT,
)
from .analysis import (
    extract_chunk_info,
    extract_chunk_info_ai_only,
    extract_chunk_info_traditional,
    summarize_chunk,
    aggregate_facts,
    synthesize_summary,
)
from .cache_utils import get_cache_dir, load_chunk, save_chunk


def run_pipeline(log_path: str, cache_dir: Optional[str] = None, use_ai_classification: bool = True) -> Dict[str, Any]:
    """
    Run the chat analysis pipeline.
    
    Args:
        log_path: Path to the chat log file
        cache_dir: Optional cache directory
        use_ai_classification: If True, uses hybrid regex+AI classification. If False, uses traditional regex-only method.
    """
    print(f"Loading chat log from: {log_path}")
    turns = load_and_clean_log(log_path)
    print(f"Found {len(turns)} chat turns")

    user_text = filter_user_turns(turns)
    print(f"User text length: {len(user_text)} characters")

    if not user_text.strip():
        return {"narrative": "No user messages found in the chat log.", "facts": {"background": [], "style": [], "goals": [], "lifestyle": [], "interests": []}}

    print("Segmenting text into chunks...")
    chunks = segment_text(user_text)
    print(f"Created {len(chunks)} chunks for processing")

    cache_path = get_cache_dir(log_path, cache_dir)

    if not chunks:
        return {"narrative": "No meaningful content found in user messages.", "facts": {"background": [], "style": [], "goals": [], "lifestyle": [], "interests": []}}

    if TESTING_MODE:
        test_chunks = chunks[:TEST_CHUNK_LIMIT]
        print(f"Testing mode: Processing only first {len(test_chunks)} chunks out of {len(chunks)} total")
    else:
        test_chunks = chunks
        print(f"Processing all {len(test_chunks)} chunks")

    print("Processing chunks...")
    local_summaries = []
    
    # Choose extraction method based on parameter
    extraction_method = "AI-enhanced" if use_ai_classification else "traditional regex"
    print(f"Using {extraction_method} extraction method")
    
    for i, chunk in enumerate(test_chunks):
        print(f"Processing chunk {i+1}/{len(test_chunks)}")
        cached = load_chunk(cache_path, chunk)
        if cached is not None:
            print("  loaded from cache")
            local_summaries.append(cached)
            continue

        # Use chosen extraction method
        if use_ai_classification:
            struct = extract_chunk_info(chunk)  # Uses hybrid regex + AI classification
        else:
            # For backwards compatibility, create a simplified version without AI classification
            struct = extract_chunk_info_traditional(chunk)
            
        summary = summarize_chunk(chunk)
        for key in struct:
            struct[key].extend(summary.get(key, []))
            struct[key] = list(dict.fromkeys(struct[key]))
        save_chunk(cache_path, chunk, struct)
        local_summaries.append(struct)

    print("Aggregating facts...")
    facts = aggregate_facts(local_summaries)

    print("Generating final narrative...")
    narrative = synthesize_summary(facts)

    return {"narrative": narrative, "facts": facts}
