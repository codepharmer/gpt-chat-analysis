#!/usr/bin/env python3
"""
Quick script to examine the structure of merged_conversations.json
"""

import json

def examine_conversations():
    try:
        with open('merged_conversations.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Data type: {type(data)}")
        print(f"Length: {len(data) if isinstance(data, (list, dict)) else 'N/A'}")
        
        if isinstance(data, list) and data:
            print(f"First item type: {type(data[0])}")
            print(f"First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'N/A'}")
            
            # Show a sample conversation structure
            if isinstance(data[0], dict):
                sample = data[0]
                print(f"\nSample conversation structure:")
                for key, value in sample.items():
                    if isinstance(value, str):
                        print(f"  {key}: {type(value)} (length: {len(value)})")
                    elif isinstance(value, dict):
                        print(f"  {key}: {type(value)} (keys: {len(value)})")
                    elif isinstance(value, list):
                        print(f"  {key}: {type(value)} (length: {len(value)})")
                    else:
                        print(f"  {key}: {type(value)}")
                
                # Examine the mapping structure more closely
                mapping = sample.get('mapping', {})
                print(f"\nMapping structure:")
                print(f"  Mapping keys count: {len(mapping)}")
                if mapping:
                    first_mapping_key = list(mapping.keys())[0]
                    first_mapping_value = mapping[first_mapping_key]
                    print(f"  First mapping key: {first_mapping_key}")
                    print(f"  First mapping value type: {type(first_mapping_value)}")
                    if isinstance(first_mapping_value, dict):
                        print(f"  First mapping value keys: {list(first_mapping_value.keys())}")
                        
                        # Look for message content
                        message = first_mapping_value.get('message')
                        if message:
                            print(f"  Message type: {type(message)}")
                            if isinstance(message, dict):
                                print(f"  Message keys: {list(message.keys())}")
                                content = message.get('content')
                                if content:
                                    print(f"  Content type: {type(content)}")
                                    if isinstance(content, dict):
                                        print(f"  Content keys: {list(content.keys())}")
                                        parts = content.get('parts')
                                        if parts:
                                            print(f"  Parts type: {type(parts)} (length: {len(parts)})")
                                            if parts and isinstance(parts[0], str):
                                                print(f"  First part preview: {parts[0][:100]}...")
                        
        print(f"\nTotal conversations: {len(data) if isinstance(data, list) else 'N/A'}")
        
    except Exception as e:
        print(f"Error examining file: {e}")

if __name__ == "__main__":
    examine_conversations() 