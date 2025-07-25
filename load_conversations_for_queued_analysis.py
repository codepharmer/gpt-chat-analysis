#!/usr/bin/env python3
"""
Script to load actual conversations and run queued batch analysis.
Modify the load_conversations() function to load your real data.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any

from chat_analysis.template_analysis_enhanced import analyze_conversations_with_queued_batching

def load_conversations_from_json(file_path: str) -> List[Dict[str, Any]]:
    """
    Load conversations from a JSON file.
    
    Expected format:
    [
        {"id": "conv_1", "content": "conversation text..."},
        {"id": "conv_2", "content": "conversation text..."},
        ...
    ]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conversations = []
    for i, item in enumerate(data):
        if isinstance(item, dict):
            conversations.append({
                "id": item.get("id", f"conversation_{i}"),
                "content": item.get("content", "")
            })
        elif isinstance(item, str):
            # If it's just a list of strings
            conversations.append({
                "id": f"conversation_{i}",
                "content": item
            })
    
    return conversations

def load_conversations_from_txt_files(directory: str) -> List[Dict[str, Any]]:
    """
    Load conversations from individual text files in a directory.
    Each .txt file becomes one conversation.
    """
    conversations = []
    txt_files = Path(directory).glob("*.txt")
    
    for file_path in txt_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        conversations.append({
            "id": file_path.stem,  # Use filename without extension as ID
            "content": content
        })
    
    return conversations

def load_conversations_from_processed_metadata(file_path: str = "processed_conversations_metadata.json") -> List[Dict[str, Any]]:
    """
    Load conversations from your processed_conversations_metadata.json file.
    This seems to be the format you're already using.
    """
    if not Path(file_path).exists():
        print(f"❌ File {file_path} not found!")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conversations = []
    
    # Handle different possible formats
    if isinstance(data, list):
        # Direct list of conversations
        for i, item in enumerate(data):
            if isinstance(item, dict) and "content" in item:
                conversations.append({
                    "id": item.get("id", f"conversation_{i}"),
                    "content": item["content"]
                })
    elif isinstance(data, dict):
        # Nested structure - look for conversations
        if "conversations" in data:
            for i, conv in enumerate(data["conversations"]):
                conversations.append({
                    "id": conv.get("id", f"conversation_{i}"),
                    "content": conv.get("content", "")
                })
        # Or if it's a dict of conversations
        else:
            for key, value in data.items():
                if isinstance(value, dict) and "content" in value:
                    conversations.append({
                        "id": key,
                        "content": value["content"]
                    })
                elif isinstance(value, str):
                    conversations.append({
                        "id": key,
                        "content": value
                    })
    
    return conversations

def main():
    print("🚀 Queued Batch Analysis with Real Conversations")
    print("=" * 60)
    
    # Choose your data loading method:
    
    # Option 1: Load from processed_conversations_metadata.json (if you have this)
    conversations = load_conversations_from_processed_metadata()
    
    # Option 2: Load from a JSON file
    # conversations = load_conversations_from_json("your_conversations.json")
    
    # Option 3: Load from individual text files
    # conversations = load_conversations_from_txt_files("conversations_directory")
    
    if not conversations:
        print("❌ No conversations loaded! Please check your data source.")
        print("\nTo use this script:")
        print("1. Modify the load_conversations() calls above")
        print("2. Point to your actual conversation data")
        print("3. Ensure the format matches: [{'id': '...', 'content': '...'}]")
        return
    
    print(f"✅ Loaded {len(conversations)} conversations")
    
    # Show sample conversation
    if conversations:
        sample = conversations[0]
        print(f"\n📝 Sample conversation (ID: {sample['id']}):")
        print(f"   Content preview: {sample['content'][:100]}...")
    
    # Ask user for confirmation
    print(f"\n🤔 Ready to process {len(conversations)} conversations with queued batching?")
    if len(conversations) > 100:
        print(f"⚠️  WARNING: {len(conversations)} conversations is a lot! This will make many API calls.")
        print("   Consider starting with a smaller subset for testing.")
    
    response = input("Continue? (y/N): ").lower().strip()
    if response not in ['y', 'yes']:
        print("❌ Cancelled by user")
        return
    
    # Run the analysis
    start_time = time.time()
    
    try:
        print(f"\n🔥 Running QUEUED BATCH analysis on {len(conversations)} conversations...")
        result = analyze_conversations_with_queued_batching(
            conversations=conversations,
            max_concurrent=20,  # Adjust based on your API rate limits
            max_retries=3,
            validate_result=False,  # Skip validation for speed
            progress_callback=lambda msg: print(f"📢 {msg}")
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print(f"📊 Rate: {len(conversations)/total_time:.1f} conversations/second")
        
        # Save results
        output_file = f"queued_batch_result_{int(time.time())}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        print(f"💾 Results saved to: {output_file}")
        
        # Show summary
        metadata = result.get('metadata', {})
        print(f"\n📈 Performance Summary:")
        print(f"  Total API requests: {metadata.get('total_api_requests', 0)}")
        print(f"  Conversations processed: {metadata.get('conversations_processed', 0)}")
        print(f"  Queued batching: {metadata.get('queued_batching', False)}")
        
        phase_times = metadata.get('phase_times', {})
        if phase_times:
            print(f"\n⏱️  Phase Breakdown:")
            for phase, time_taken in phase_times.items():
                print(f"    {phase}: {time_taken:.2f}s")
        
        # Show human readable result preview
        human_readable = result.get('human_readable', '')
        if human_readable:
            print(f"\n📝 Result Preview:")
            print(f"  {human_readable[:300]}...")
            
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 