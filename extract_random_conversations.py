#!/usr/bin/env python3
"""
Script to extract 5 random JSON objects from conversations.json file.
Assumes the root of the file is an array [].
"""

import json
import random
import os
import sys
from pathlib import Path

def extract_random_conversations(input_file, output_file=None, num_samples=5):
    """
    Extract random conversations from a JSON file containing an array of conversations.
    
    Args:
        input_file (str): Path to the input conversations.json file
        output_file (str, optional): Path to save the random conversations. If None, prints to stdout
        num_samples (int): Number of random conversations to extract (default: 5)
    """
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False
    
    try:
        print(f"Loading conversations from {input_file}...")
        
        # Load the JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        # Verify it's a list/array
        if not isinstance(conversations, list):
            print("Error: The JSON file root is not an array.")
            return False
        
        total_conversations = len(conversations)
        print(f"Found {total_conversations} conversations in the file.")
        
        # Check if we have enough conversations
        if total_conversations < num_samples:
            print(f"Warning: Only {total_conversations} conversations available, extracting all of them.")
            num_samples = total_conversations
        
        # Randomly sample conversations
        print(f"Extracting {num_samples} random conversations...")
        random_conversations = random.sample(conversations, num_samples)
        
        # Prepare output data
        output_data = {
            "metadata": {
                "total_conversations": total_conversations,
                "extracted_count": num_samples,
                "extraction_method": "random_sample"
            },
            "conversations": random_conversations
        }
        
        # Output the results
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"Random conversations saved to: {output_file}")
        else:
            print("\n" + "="*50)
            print("EXTRACTED RANDOM CONVERSATIONS:")
            print("="*50)
            print(json.dumps(output_data, indent=2, ensure_ascii=False))
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in file '{input_file}': {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Main function to handle command line arguments."""
    
    # Default paths
    default_input = Path("chatgpt_062025/conversations.json")
    default_output = Path("random_conversations_sample.json")
    
    # Check if we're running from the project root
    current_dir = Path.cwd()
    input_file = current_dir / default_input
    
    # If not found in current directory, try absolute path
    if not input_file.exists():
        input_file = Path(r"c:\Users\Wesso\NossonAI\gpt-chat-analysis\chatgpt_062025\conversations.json")
    
    if not input_file.exists():
        print("Error: Could not find conversations.json file.")
        print("Please ensure you're running this script from the project root directory,")
        print("or that the file exists at: chatgpt_062025/conversations.json")
        return
    
    output_file = current_dir / default_output
    
    print("=" * 60)
    print("Random Conversation Extractor")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Extracting: 5 random conversations")
    print("-" * 60)
    
    success = extract_random_conversations(
        input_file=str(input_file),
        output_file=str(output_file),
        num_samples=5
    )
    
    if success:
        print("\n✅ Extraction completed successfully!")
        print(f"📁 Results saved to: {output_file}")
    else:
        print("\n❌ Extraction failed!")

if __name__ == "__main__":
    main()
