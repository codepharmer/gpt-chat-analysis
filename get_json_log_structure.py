#!/usr/bin/env python3
import json
import argparse
import sys

def print_structure(obj, indent=0, current_key=None):
    """
    Recursively print the “shape” of a JSON object:
      - For dicts: lists keys and recurses into each value
      - For lists: shows length and recurses into the first element (if any)
      - For scalars: prints the value and its type
    """
    # Skip rendering if the current key is "parts" then only show the key and the type
    if current_key == "parts":
        print(f"{' ' * indent}parts: ({type(obj).__name__})")
        return
        
    prefix = ' ' * indent
    if isinstance(obj, dict):
        print(f"{prefix}Object {{")
        for key, value in obj.items():
            print(f"{prefix}  \"{key}\": ({type(value).__name__})")
            print_structure(value, indent + 4, current_key=key)
        print(f"{prefix}}}")
    elif isinstance(obj, list):
        print(f"{prefix}Array[{len(obj)}] [")
        if obj:
            # Show the structure of the first element as representative
            print_structure(obj[0], indent + 4, current_key=current_key)
            if len(obj) > 1:
                print(f"{prefix}    … ({len(obj)-1} more elements)")
        print(f"{prefix}]")
    else:
        # Scalar value
        print(f"{prefix}{repr(obj)}  ← {type(obj).__name__}")

def main():
    parser = argparse.ArgumentParser(
        description="Print the structure of a JSON file (keys, array lengths, value types)."
    )
    parser.add_argument(
        'file',
        nargs='?',
        type=argparse.FileType('r'),
        default=sys.stdin,
        help="Path to JSON file (or omit to read from stdin)"
    )
    args = parser.parse_args()

    try:
        data = json.load(args.file)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}", file=sys.stderr)
        sys.exit(1)

    print_structure(data)

if __name__ == '__main__':
    main()
