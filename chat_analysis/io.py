from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Any


@dataclass
class ChatTurn:
    speaker: str
    text: str
    timestamp: datetime


def load_and_clean_log(path: str) -> List[ChatTurn]:
    """Load chat log from various JSON formats."""
    turns: List[ChatTurn] = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    try:
        data = json.loads(content)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "mapping" in item:
                    mapping = item["mapping"]
                    if isinstance(mapping, dict):
                        for node in mapping.values():
                            if isinstance(node, dict) and "message" in node and node["message"]:
                                turn = _parse_message_object(node["message"])
                                if turn:
                                    turns.append(turn)
                else:
                    turn = _parse_message_object(item)
                    if turn:
                        turns.append(turn)
        elif isinstance(data, dict):
            if "mapping" in data:
                mapping = data["mapping"]
                if isinstance(mapping, dict):
                    for node in mapping.values():
                        if isinstance(node, dict) and "message" in node and node["message"]:
                            turn = _parse_message_object(node["message"])
                            if turn:
                                turns.append(turn)
            elif "messages" in data and isinstance(data["messages"], list):
                for msg in data["messages"]:
                    turn = _parse_message_object(msg)
                    if turn:
                        turns.append(turn)
            else:
                turn = _parse_message_object(data)
                if turn:
                    turns.append(turn)
    except json.JSONDecodeError:
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                turn = _parse_message_object(raw)
                if turn:
                    turns.append(turn)
            except (json.JSONDecodeError, KeyError, ValueError):
                print(f"Warning: Skipping malformed line {line_num}")
                continue

    return turns


def _parse_message_object(raw: Any) -> ChatTurn | None:
    """Parse a message object and return a ChatTurn."""
    if not isinstance(raw, dict):
        return None

    text = ""
    if isinstance(raw.get("content"), dict) and "parts" in raw["content"]:
        parts = raw["content"]["parts"]
        if isinstance(parts, list):
            text = "".join(str(p) for p in parts)
        else:
            text = str(parts)
    elif "text" in raw:
        text = raw["text"]
    elif isinstance(raw.get("content"), str):
        text = raw["content"]
    elif "message" in raw:
        text = raw["message"]
    else:
        return None

    speaker = ""
    if "speaker" in raw:
        speaker = raw["speaker"]
    elif "role" in raw:
        speaker = raw["role"]
    elif "author" in raw:
        author = raw["author"]
        if isinstance(author, dict) and "role" in author:
            speaker = author["role"]
        else:
            speaker = str(author)
    else:
        speaker = "unknown"

    timestamp = datetime.now()
    for time_field in ["create_time", "utc_time", "timestamp", "created_at", "time"]:
        if time_field in raw:
            try:
                if isinstance(raw[time_field], str):
                    timestamp = datetime.fromisoformat(raw[time_field].replace("Z", "+00:00"))
                elif isinstance(raw[time_field], (int, float)):
                    timestamp = datetime.fromtimestamp(raw[time_field])
                break
            except (ValueError, TypeError):
                continue

    if isinstance(text, str):
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = text.strip()
    else:
        text = str(text)

    return ChatTurn(speaker, text, timestamp)
