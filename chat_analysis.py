from chat_analysis import run_pipeline
from chat_analysis.cache_utils import clear_all_caches

import argparse
import json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("log_path", help="path to JSON-line chat log")
    p.add_argument("out_path", help="where to write persona.json")
    p.add_argument("--cache-dir", help="optional directory to store temporary caches")
    p.add_argument("--clear-cache", action="store_true", help="clear cache for this log file before processing")
    p.add_argument("--no-ai-classification", action="store_true", help="disable AI sentence classification and use traditional regex-only extraction")
    args = p.parse_args()

    if args.clear_cache:
        print("Clearing cache before processing...")
        clear_all_caches(args.log_path, args.cache_dir)

    # Use AI classification by default, but allow disabling it
    use_ai_classification = not args.no_ai_classification
    persona = run_pipeline(args.log_path, cache_dir=args.cache_dir, use_ai_classification=use_ai_classification)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(persona, f, indent=2, ensure_ascii=False)
    print(f"Wrote persona to {args.out_path}")
