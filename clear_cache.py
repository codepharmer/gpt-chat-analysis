#!/usr/bin/env python3
"""
Script to clear various types of cache for the chat analysis project.
"""

import argparse
import sys
from pathlib import Path

from chat_analysis.cache_utils import clear_all_caches, clear_python_cache


def clear_all_project_caches(project_root: str = None):
    """Clear all types of cache for the project."""
    if project_root is None:
        project_root = Path(__file__).parent
    else:
        project_root = Path(project_root)
    
    print("=" * 50)
    print("Clearing All Project Caches")
    print("=" * 50)
    
    # Clear Python bytecode cache
    print("\n1. Clearing Python bytecode cache (__pycache__)...")
    clear_python_cache(project_root)
    
    # Clear application-specific caches
    print("\n2. Clearing application-specific caches...")
    cache_cleared = False
    
    # Look for conversation files to clear their associated caches
    conversation_files = []
    for pattern in ["**/conversations.json", "**/*_log.json", "**/*.jsonl"]:
        conversation_files.extend(project_root.glob(pattern))
    
    if conversation_files:
        for conv_file in conversation_files:
            print(f"Checking cache for: {conv_file.name}")
            if clear_all_caches(str(conv_file)):
                cache_cleared = True
    
    # Also look for any existing cache directories
    cache_dirs = list(project_root.glob("**/*_cache"))
    if cache_dirs:
        from chat_analysis.cache_utils import clear_cache
        for cache_dir in cache_dirs:
            if cache_dir.is_dir():
                clear_cache(cache_dir)
                cache_cleared = True
    
    if not cache_cleared:
        print("No application-specific caches found")
    
    print("\n" + "=" * 50)
    print("Cache clearing completed!")
    print("=" * 50)


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Clear cache for the chat analysis project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clear_cache.py                    # Clear all caches
  python clear_cache.py --python-only      # Clear only Python bytecode cache
  python clear_cache.py --app-only         # Clear only application cache
  python clear_cache.py --log-file path    # Clear cache for specific log file
        """
    )
    
    parser.add_argument(
        "--python-only",
        action="store_true",
        help="Clear only Python bytecode cache (__pycache__)"
    )
    
    parser.add_argument(
        "--app-only",
        action="store_true",
        help="Clear only application-specific cache"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Clear cache for a specific log file"
    )
    
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Root directory of the project (default: current script directory)"
    )
    
    args = parser.parse_args()
    
    if args.log_file:
        print(f"Clearing cache for log file: {args.log_file}")
        if clear_all_caches(args.log_file):
            print("Cache cleared successfully!")
        else:
            print("Failed to clear cache or no cache found.")
            sys.exit(1)
    elif args.python_only:
        print("Clearing Python bytecode cache...")
        if clear_python_cache(args.project_root):
            print("Python cache cleared successfully!")
        else:
            print("Failed to clear Python cache.")
            sys.exit(1)
    elif args.app_only:
        print("Clearing application-specific cache...")
        # This is more complex since we need to find all conversation files
        project_root = Path(args.project_root) if args.project_root else Path(__file__).parent
        cache_cleared = False
        
        # Look for conversation files
        conversation_files = []
        for pattern in ["**/conversations.json", "**/*_log.json", "**/*.jsonl"]:
            conversation_files.extend(project_root.glob(pattern))
        
        for conv_file in conversation_files:
            print(f"Clearing cache for: {conv_file.name}")
            if clear_all_caches(str(conv_file)):
                cache_cleared = True
        
        # Also look for any existing cache directories
        cache_dirs = list(project_root.glob("**/*_cache"))
        if cache_dirs:
            from chat_analysis.cache_utils import clear_cache
            for cache_dir in cache_dirs:
                if cache_dir.is_dir():
                    clear_cache(cache_dir)
                    cache_cleared = True
        
        if cache_cleared:
            print("Application cache cleared successfully!")
        else:
            print("No application-specific caches found.")
    else:
        # Clear all caches
        clear_all_project_caches(args.project_root)


if __name__ == "__main__":
    main()
