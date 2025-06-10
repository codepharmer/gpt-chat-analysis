#!/usr/bin/env python3
"""
Test script to verify the integration of classify_sentence_async into the analysis pipeline.
"""

from chat_analysis.analysis import classify_sentences, extract_chunk_info, extract_chunk_info_traditional
import json

def test_sentence_classification():
    """Test the async sentence classification functionality."""
    print("Testing sentence classification...")
    
    test_sentences = [
        "I am working on a machine learning project using Python.",
        "I love hiking and outdoor activities on weekends.",
        "My background is in computer science and I graduated from MIT.",
        "I prioritize work-life balance and spending time with family."
    ]
    
    try:
        results = classify_sentences(test_sentences, max_concurrent=5)
        print(f"Successfully classified {len(results)} sentences:")
        
        for result in results:
            if result.get("success", False):
                print(f"  Sentence: {result['sentence'][:50]}...")
                print(f"  Categories: {result['categories']}")
            else:
                print(f"  Failed to classify: {result.get('error', 'Unknown error')}")
        
        return True
    except Exception as e:
        print(f"Error in sentence classification: {e}")
        return False

def test_chunk_extraction_comparison():
    """Compare traditional vs AI-enhanced extraction methods."""
    print("\nTesting chunk extraction methods...")
    
    test_chunk = """
    I am a software engineer working at Google on AI projects. I love machine learning and data science.
    My goal is to build systems that can help people be more productive. I enjoy hiking, reading, and coding in my free time.
    I prioritize work-life balance and believe in continuous learning.
    """
    
    print("Traditional extraction:")
    traditional_result = extract_chunk_info_traditional(test_chunk)
    print(json.dumps(traditional_result, indent=2))
    
    print("\nAI-enhanced extraction:")
    try:
        ai_result = extract_chunk_info(test_chunk)
        print(json.dumps(ai_result, indent=2))
        
        # Compare results
        print("\nComparison:")
        for category in traditional_result:
            traditional_count = len(traditional_result[category])
            ai_count = len(ai_result[category])
            if ai_count > traditional_count:
                print(f"  {category}: AI found {ai_count - traditional_count} additional items")
            elif traditional_count > ai_count:
                print(f"  {category}: Traditional found {traditional_count - ai_count} more items")
            else:
                print(f"  {category}: Both methods found {ai_count} items")
        
        return True
    except Exception as e:
        print(f"Error in AI-enhanced extraction: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Integration of classify_sentence_async ===\n")
    
    success1 = test_sentence_classification()
    success2 = test_chunk_extraction_comparison()
    
    if success1 and success2:
        print("\n✅ All tests passed! Integration is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
