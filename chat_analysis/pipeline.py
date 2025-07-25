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
from .template_analysis_enhanced import analyze_conversations_with_queued_batching
from .cache_utils import get_cache_dir, load_chunk, save_chunk


def run_pipeline(
    log_path: str, 
    cache_dir: Optional[str] = None, 
    use_ai_classification: bool = True,
    analysis_mode: str = "template"
) -> Dict[str, Any]:
    """
    Run the chat analysis pipeline with queued batch processing.
    
    Args:
        log_path: Path to the chat log file
        cache_dir: Optional cache directory (only used for legacy modes)
        use_ai_classification: For backwards compatibility with legacy modes
        analysis_mode: Analysis approach to use:
                      - 'template': Uses queued batch processing (recommended, default)
                      - 'hybrid': Legacy regex + AI classification + summarization 
                      - 'traditional': Legacy regex + NER only (no AI classification)
                      
                      Note: Legacy modes (hybrid/traditional) are deprecated and will be removed.
                      Use 'template' mode for best performance and accuracy.
    
    Returns:
        Dict with analysis results:
        - 'template': {"template": dict, "human_readable": str, "metadata": dict}
        - 'hybrid'/'traditional': {"narrative": str, "facts": dict} (deprecated)
    """
    print(f"Loading chat log from: {log_path}")
    turns = load_and_clean_log(log_path)
    print(f"Found {len(turns)} chat turns")

    user_text = filter_user_turns(turns)
    print(f"User text length: {len(user_text)} characters")

    if not user_text.strip():
        if analysis_mode == 'template':
            return {
                "template": {},
                "human_readable": "No user messages found in the chat log.",
                "metadata": {"conversations_processed": 0, "total_conversations": 0}
            }
        else:
            return {
                "narrative": "No user messages found in the chat log.", 
                "facts": {"background": [], "style": [], "goals": [], "lifestyle": [], "interests": []}
            }

    # Template-based analysis (recommended path)
    if analysis_mode == 'template':
        print("Running queued batch analysis (recommended)")
        
        # For template analysis, we work with the full conversation as a single unit
        conversations = [{"content": user_text, "id": "full_conversation"}]
        
        try:
            result = analyze_conversations_with_queued_batching(
                conversations=conversations,
                max_concurrent=20,
                max_retries=3,
                validate_result=True,
                progress_callback=lambda msg: print(f"  {msg}")
            )
            
            print(f"Queued batch analysis completed in {result['metadata']['processing_time']:.2f}s")
            return result
            
        except Exception as e:
            print(f"Template analysis failed: {e}")
            return {
                "template": {},
                "human_readable": f"Template analysis failed: {str(e)}",
                "metadata": {"error": str(e)}
            }

    # Legacy pipeline (deprecated - maintained for backwards compatibility only)
    print(f"WARNING: Using deprecated analysis mode '{analysis_mode}'. Please switch to 'template' mode.")
    print("Legacy modes will be removed in a future version.")
    
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
    
    # Choose extraction method based on parameters
    if analysis_mode == "traditional":
        extraction_method = "traditional regex-only"
        use_ai = False
    else:  # hybrid mode (default)
        extraction_method = "AI-enhanced hybrid" if use_ai_classification else "traditional regex"
        use_ai = use_ai_classification
    
    print(f"Using {extraction_method} extraction method")
    
    for i, chunk in enumerate(test_chunks):
        print(f"Processing chunk {i+1}/{len(test_chunks)}")
        cached = load_chunk(cache_path, chunk)
        if cached is not None:
            print("  loaded from cache")
            local_summaries.append(cached)
            continue

        # Use chosen extraction method
        if analysis_mode == "traditional":
            struct = extract_chunk_info_traditional(chunk)
        elif use_ai:
            struct = extract_chunk_info(chunk)  # Uses hybrid regex + AI classification
        else:
            # Backwards compatibility: hybrid mode but no AI classification
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


def run_template_analysis(log_path: str, quick: bool = False) -> Dict[str, Any]:
    """
    Convenience function to run template-based analysis with queued batch processing.
    
    Args:
        log_path: Path to the chat log file
        quick: Deprecated parameter (kept for backwards compatibility)
        
    Returns:
        Template analysis results using queued batch processing
    """
    if quick:
        print("WARNING: 'quick' parameter is deprecated. Queued batch processing is always fast.")
    
    return run_pipeline(log_path, analysis_mode="template")


def run_comparative_analysis(log_path: str, cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Run both traditional and template-based analysis for comparison.
    
    Note: This function is deprecated as legacy modes will be removed.
    Use run_template_analysis() for best results.
    
    Args:
        log_path: Path to the chat log file
        cache_dir: Optional cache directory
        
    Returns:
        Dict containing results from both analysis methods
    """
    print("WARNING: run_comparative_analysis() is deprecated. Use run_template_analysis() instead.")
    print("Running comparative analysis...")
    
    print("\n=== Legacy Traditional Analysis ===")
    traditional_result = run_pipeline(
        log_path, 
        cache_dir=cache_dir, 
        analysis_mode="hybrid",
        use_ai_classification=True
    )
    
    print("\n=== Queued Batch Template Analysis ===")
    template_result = run_pipeline(
        log_path,
        analysis_mode="template"
    )
    
    return {
        "traditional": traditional_result,
        "template": template_result,
        "comparison_metadata": {
            "traditional_facts_count": sum(len(v) if isinstance(v, list) else 0 for v in traditional_result.get("facts", {}).values()),
            "template_confidence": template_result.get("metadata", {}).get("average_confidence", 0),
        }
    }


# Backwards compatibility - keep the original function signature
def run_analysis_pipeline(log_path: str, cache_dir: Optional[str] = None, use_ai_classification: bool = True) -> Dict[str, Any]:
    """
    Backwards compatible wrapper for the original pipeline function.
    
    DEPRECATED: Use run_template_analysis() for best performance and accuracy.
    
    Args:
        log_path: Path to the chat log file
        cache_dir: Optional cache directory
        use_ai_classification: Whether to use AI classification
        
    Returns:
        Traditional analysis results (deprecated)
    """
    print("WARNING: run_analysis_pipeline() is deprecated. Use run_template_analysis() instead.")
    return run_pipeline(log_path, cache_dir, use_ai_classification, analysis_mode="hybrid")
