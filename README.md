# Chat Analysis Pipeline

`chat_analysis.py` is a command line tool that distills large chat logs into a concise persona summary. It understands several JSON formats including ChatGPT conversation exports and OpenAI API logs.

## Requirements

```bash
pip install -r requirements.txt
```

An OpenAI API key is required for summarisation. Set `OPENAI_API_KEY` in your environment.

## Usage

```bash
python chat_analysis.py <log_path> <output_path>
```

The script reads the chat log at `log_path` and writes a JSON file containing a narrative biography and categorized facts about the user.

## How it works

1. **Log parsing** – messages are loaded from different JSON structures and converted to `ChatTurn` objects.
2. **User text extraction** – only messages from the `user` speaker are concatenated.
3. **Segmentation** – text is split into overlapping chunks and embeddings help locate topic boundaries.
4. **Chunk summarisation** – GPT‑4 is used to extract background, style, goals, lifestyle and interests from each chunk.
5. **Aggregation** – similar facts are deduplicated via embedding similarity.
6. **Narrative generation** – a short biography is produced from the aggregated facts.

`chat_analysis.ipynb` contains an earlier exploratory notebook, but the Python script is the recommended entry point.

## Example

```bash
python chat_analysis.py sample_log.json persona.json
```

The resulting `persona.json` will look like:

```json
{
  "narrative": "...",
  "facts": {
    "background": [...],
    "style": [...],
    "goals": [...],
    "lifestyle": [...],
    "interests": [...]
  }
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
