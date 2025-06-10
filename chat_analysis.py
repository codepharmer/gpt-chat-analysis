from chat_analysis import run_pipeline

import argparse
import json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("log_path", help="path to JSON-line chat log")
    p.add_argument("out_path", help="where to write persona.json")
    args = p.parse_args()

    persona = run_pipeline(args.log_path)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(persona, f, indent=2, ensure_ascii=False)
    print(f"Wrote persona to {args.out_path}")
