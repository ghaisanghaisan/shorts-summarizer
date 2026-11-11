"""Gemini backend: same two passes and prompts as the Claude backend, same output schema."""

from __future__ import annotations

import os
import time
from pathlib import Path

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from ..schema import Analysis, VisualPass
from .base import Evidence
from .claude import ANALYZE_SYSTEM, DESCRIBE_SYSTEM, _fmt_meta, _fmt_ocr, _fmt_transcript

DEFAULT_MODEL = "gemini-3.8-flash"


class GeminiAnalyzer:
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("set GEMINI_API_KEY in .env (get one at https://aistudio.google.com/apikey)")
        self.client = genai.Client(api_key=key)

    def _parse(self, system: str, parts: list, out: type[BaseModel], attempts: int = 5):
        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=out.model_json_schema(),
        )
        for i in range(attempts):
            try:
                resp = self.client.models.generate_content(model=self.model, contents=parts, config=config)
                break
            except errors.APIError as e:
                retryable = e.code == 429 or (e.code or 0) >= 500
                if not retryable or i == attempts - 1:
                    raise
                wait = 5 * 2 ** i
                print(f"  [gemini] {e.code} {e.status}, retrying in {wait}s ({i + 1}/{attempts - 1})")
                time.sleep(wait)
        if not resp.text:
            raise RuntimeError(f"no output from {self.model}: {resp.prompt_feedback or resp.candidates}")
        return out.model_validate_json(resp.text)

    def describe(self, ev: Evidence) -> VisualPass:
        parts: list = []
        for f in ev.frames:
            parts.append(f"[frame t={f.t:.2f}s]")
            parts.append(types.Part.from_bytes(data=Path(f.path).read_bytes(), mime_type="image/jpeg"))
        parts.append(f"TRANSCRIPT:\n{_fmt_transcript(ev)}\n\nOCR TEXT:\n{_fmt_ocr(ev)}\n\n"
                     f"Describe all {len(ev.frames)} frames.")
        return self._parse(DESCRIBE_SYSTEM, parts, VisualPass)

    def analyze(self, ev: Evidence, visuals: VisualPass) -> Analysis:
        vis = "\n".join(f"[{v.t:.1f}s] ({v.kind}) {v.description}" for v in visuals.frames)
        text = (f"METADATA:\n{_fmt_meta(ev)}\n\nTRANSCRIPT:\n{_fmt_transcript(ev)}\n\n"
                f"ON-SCREEN TEXT (OCR):\n{_fmt_ocr(ev)}\n\nVISUALS:\n{vis}")
        return self._parse(ANALYZE_SYSTEM, [text], Analysis)
