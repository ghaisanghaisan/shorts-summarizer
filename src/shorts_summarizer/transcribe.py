"""Speech-to-text with faster-whisper (local)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from .schema import TranscriptSegment


@lru_cache(maxsize=2)
def _model(name: str):
    from faster_whisper import WhisperModel
    return WhisperModel(name, device="auto", compute_type="int8")


def transcribe(video: Path, model_name: str | None = None) -> tuple[list[TranscriptSegment], str | None]:
    name = model_name or os.getenv("WHISPER_MODEL", "small")
    try:
        segments, info = _model(name).transcribe(str(video), vad_filter=True)
        segs = [TranscriptSegment(t_start=round(s.start, 2), t_end=round(s.end, 2), text=s.text.strip())
                for s in segments if s.text.strip()]
    except Exception as e:  # no audio stream, decode error, etc.
        print(f"  [transcribe] skipped: {e}")
        return [], None
    return segs, (info.language if segs else None)
