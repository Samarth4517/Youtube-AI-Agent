"""
Utilities for extracting and cleaning YouTube video transcripts.
"""

import re
from typing import Optional

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)


class TranscriptError(Exception):
    """Raised when a transcript cannot be fetched, with a user-friendly message."""


def extract_video_id(url: str) -> Optional[str]:
    """
    Extract the 11-character YouTube video ID from a variety of URL formats:
    - https://www.youtube.com/watch?v=VIDEOID
    - https://youtu.be/VIDEOID
    - https://www.youtube.com/embed/VIDEOID
    - https://www.youtube.com/shorts/VIDEOID
    """
    if not url:
        return None

    patterns = [
        r"(?:v=|/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be/([0-9A-Za-z_-]{11})",
        r"youtube\.com/shorts/([0-9A-Za-z_-]{11})",
        r"youtube\.com/embed/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_transcript(video_url: str, languages=("en", "en-US", "en-GB")) -> str:
    """
    Fetch and return the full transcript text for a YouTube video URL.
    Raises TranscriptError with a friendly message on any failure.
    """
    video_id = extract_video_id(video_url)
    if not video_id:
        raise TranscriptError(
            "Couldn't recognize that as a valid YouTube URL. "
            "Please paste a link like https://www.youtube.com/watch?v=..."
        )

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id, languages=list(languages)
        )
    except TranscriptsDisabled:
        raise TranscriptError(
            "This video has transcripts disabled by the uploader. "
            "Try a different video."
        )
    except NoTranscriptFound:
        raise TranscriptError(
            "No transcript found in a supported language for this video."
        )
    except VideoUnavailable:
        raise TranscriptError(
            "This video is unavailable (private, deleted, or region-locked)."
        )
    except Exception as exc:  # noqa: BLE001 - surface any other API error cleanly
        raise TranscriptError(f"Could not fetch transcript: {exc}")

    full_text = " ".join(chunk["text"] for chunk in transcript_list if chunk.get("text"))
    full_text = re.sub(r"\s+", " ", full_text).strip()

    if not full_text:
        raise TranscriptError("The transcript for this video was empty.")

    return full_text


def chunk_transcript(text: str, chunk_size: int = 1000, chunk_overlap: int = 150):
    """
    Split transcript text into overlapping chunks for embedding / vector storage.
    Uses LangChain's RecursiveCharacterTextSplitter for sentence-aware splitting.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)
