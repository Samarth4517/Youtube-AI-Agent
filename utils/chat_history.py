"""
Simple JSON-file based chat history persistence, keyed by video ID, so a
user's Q&A history for a video survives across app restarts within the
same machine (stored under data/chat_history.json).
"""

import json
import os
from datetime import datetime
from typing import List, Dict

HISTORY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chat_history.json")


def _load_all() -> dict:
    if not os.path.exists(HISTORY_PATH):
        return {}
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_all(data: dict) -> None:
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_history(video_id: str) -> List[Dict]:
    data = _load_all()
    return data.get(video_id, [])


def append_message(video_id: str, role: str, content: str) -> None:
    data = _load_all()
    data.setdefault(video_id, [])
    data[video_id].append(
        {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )
    _save_all(data)


def clear_history(video_id: str) -> None:
    data = _load_all()
    if video_id in data:
        data[video_id] = []
        _save_all(data)
