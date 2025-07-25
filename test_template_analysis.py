#!/usr/bin/env python3
"""
Test script to run template analysis enhanced on 10 conversations from merged_conversations.json
"""

import json
import random
import sys
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from chat_analysis.template_analysis_enhanced import analyze_conversations_with_queued_batching


def extract_conversation_content(conversation_data):
    """Extract text content from ChatGPT conversation format"""
    if not isinstance(conversation_data, dict):
        return ""
    
    mapping = conversation_data.get('mapping', {})
    if not mapping:
        return ""
    
    messages = []
    
    # Extract messages from the mapping structure
    for node_id, node_data in mapping.items():
        if not isinstance(node_data, dict):
            continue
            
        message = node_data.get('message')
        if not message:
            continue
            
        # Get message content
        content = message.get('content')
        if not content:
            continue
            
        # Handle different content formats
        if isinstance(content, dict):
            parts = content.get('parts', [])
            if parts and isinstance(parts, list):
                text_parts = [str(part) for part in parts if part]
                if text_parts:
                    role = message.get('author', {}).get('role', 'unknown')
                    message_text = '\n'.join(text_parts)
                    messages.append(f"{role.upper()}: {message_text}")
        elif isinstance(content, str) and content.strip():
            role = message.get('author', {}).get('role', 'unknown')
            messages.append(f"{role.upper()}: {content}")
    
    return '\n\n'.join(messages) if messages else ""


def load_sample_conversations(filename: str, count: int = 10):
    """Load and randomly select N conversations from merged_conversations.json"""
    print(f"Loading all conversations from {filename} to randomly select {count}...")
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different possible structures - load ALL conversations first
        raw_conversations = []
        if isinstance(data, list):
            raw_conversations = data  # Load all, not just first N
        elif isinstance(data, dict):
            if 'conversations' in data:
                raw_conversations = data['conversations']  # Load all
            else:
                # Try to find conversation-like objects
                for key, value in data.items():
                    if isinstance(value, dict) and ('mapping' in value or 'content' in value):
                        raw_conversations.append(value)
        
        print(f"Found {len(raw_conversations)} raw conversations in dataset")
        
        # Extract conversation content from ALL conversations
        all_conversations = []
        for i, conv_data in enumerate(raw_conversations):
            if isinstance(conv_data, dict):
                # Try to extract content from ChatGPT format
                content = extract_conversation_content(conv_data)
                
                if content.strip():
                    conversation = {
                        'id': conv_data.get('id', conv_data.get('conversation_id', f'conv_{i}')),
                        'content': content,
                        'title': conv_data.get('title', f'Conversation {i+1}'),
                        'create_time': conv_data.get('create_time')
                    }
                    all_conversations.append(conversation)
                    if (i + 1) % 100 == 0:  # Show progress every 100 conversations
                        print(f"  Processed {i + 1}/{len(raw_conversations)} conversations...")
        
        print(f"\nSuccessfully extracted {len(all_conversations)} valid conversations with content")
        
        # Randomly sample the requested number of conversations
        if len(all_conversations) <= count:
            print(f"Requested {count} conversations, but only {len(all_conversations)} valid conversations available. Using all.")
            selected_conversations = all_conversations
        else:
            print(f"Randomly selecting {count} conversations from {len(all_conversations)} valid conversations...")
            selected_conversations = random.sample(all_conversations, count)
        
        print(f"Final selection: {len(selected_conversations)} conversations")
        
        # Show sample conversation structure
        if selected_conversations:
            print(f"\nSample conversation structure:")
            sample = selected_conversations[0]
            print(f"ID: {sample['id']}")
            print(f"Title: {sample.get('title', 'No title')}")
            content_preview = sample['content'][:300] if len(sample['content']) > 300 else sample['content']
            print(f"Content preview (first 300 chars): {content_preview}...")
        
        return selected_conversations
        
    except Exception as e:
        print(f"Error loading conversations: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    """Main test function"""
    print("=" * 60)
    print("TESTING ENHANCED TEMPLATE ANALYSIS ON 10 CONVERSATIONS")
    print("=" * 60)
    
    # Load 10 sample conversations
    conversations = load_sample_conversations('merged_conversations.json', 1000)
    
    if not conversations:
        print("❌ Failed to load conversations. Exiting.")
        return
    
    print(f"\n✅ Loaded {len(conversations)} conversations for analysis")
    
    # Run the enhanced template analysis
    print("\n🚀 Starting enhanced template analysis...")
    start_time = time.time()
    
    try:
        result = analyze_conversations_with_queued_batching(
            conversations=conversations,
            max_concurrent=100,  # Slightly higher concurrency for 10 conversations
            max_retries=3,
            validate_result=True,
            progress_callback=lambda msg: print(f"📊 Progress: {msg}")
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n🎉 Analysis completed in {total_time:.2f} seconds!")
        
        # Display results summary
        print("\n" + "=" * 60)
        print("ANALYSIS RESULTS SUMMARY")
        print("=" * 60)
        
        metadata = result.get('metadata', {})
        print(f"Conversations processed: {metadata.get('conversations_processed', 'Unknown')}")
        print(f"Total processing time: {metadata.get('processing_time', 'Unknown'):.2f}s")
        print(f"Phase times: {metadata.get('phase_times', {})}")
        print(f"API requests made: {metadata.get('total_api_requests', 'Unknown')}")
        
        # Show validation results if available
        validation = result.get('validation')
        if validation:
            print(f"\nValidation:")
            print(f"  Overall quality: {validation.get('overall_quality', 'Unknown')}")
            print(f"  Validated: {validation.get('validated', 'Unknown')}")
        
        # Show human readable preview
        human_readable = result.get('human_readable', '')
        if human_readable:
            print(f"\nHuman-readable preview (first 500 chars):")
            print("-" * 40)
            preview = human_readable[:500] + "..." if len(human_readable) > 500 else human_readable
            print(preview)
        
        # Save full results
        output_filename = f"test_results_10_conversations_{int(time.time())}.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Full results saved to: {output_filename}")
        
    except Exception as e:
        print(f"❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 