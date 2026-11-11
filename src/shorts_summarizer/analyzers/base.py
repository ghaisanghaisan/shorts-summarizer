from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..schema import Analysis, Meta, OcrLine, SelectedFrame, TranscriptSegment, VisualPass


@dataclass
class Evidence:
    meta: Meta
    frames: list[SelectedFrame]
    ocr: list[OcrLine]
    transcript: list[TranscriptSegment]
    language: str | None


class Analyzer(Protocol):
    model: str

    def describe(self, ev: Evidence) -> VisualPass: ...

    def analyze(self, ev: Evidence, visuals: VisualPass) -> Analysis: ...
