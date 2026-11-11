"""Claude backend: one multi-image call for visuals, one text call for analysis."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import anthropic

from ..schema import Analysis, VisualPass
from .base import Evidence

DEFAULT_MODEL = "claude-opus-5-5"
BETAS = ["server-side-fallback-2026-07-01"]

DESCRIBE_SYSTEM = """You are the visual-evidence stage of a misinformation and propaganda research pipeline.
You receive keyframes from one short-form video, in order, each labelled with its timestamp,
plus the transcript and the OCR text already extracted.

For EVERY frame, return one entry with:
- t: the frame's timestamp label, copied exactly
- kind: the closest visual category
- description: what is visibly shown. Be factual and specific (setting, people, objects, logos,
  graphics, editing such as split-screen, zoom, stitched clip, green-screen, watermark of another
  account). Use the sequence for context, e.g. "same speaker, now with a news screenshot behind".
  Do NOT judge truthfulness or intent here.
- people: visible people, described by role/appearance; give a name only if it is captioned on screen
- missed_text: on-screen text you can read that is NOT already in the OCR list (else empty)

Use kind "ai_generated_suspect" only for concrete visual artifacts (warped hands/text, inconsistent
lighting, morphing), and say which artifact in the description."""

ANALYZE_SYSTEM = """You are the analysis stage of a misinformation and propaganda research pipeline.
You receive the evidence for one short-form video: post metadata, timestamped transcript, OCR
on-screen text, and per-frame visual descriptions.

Produce indicators WITH evidence, not verdicts. Do not decide whether claims are true; that is a
downstream fact-checking step. Rules:
- summary: neutral, 2-4 sentences, what the video says and shows
- claims: every factual assertion (speech, on-screen text, or implied by visuals). checkable=true
  only for concrete, verifiable statements (numbers, events, quotes, attributions)
- propaganda_techniques: use the SemEval-2020 Task 11 labels. Only include a technique when you can
  quote evidence; evidence_quote must be verbatim from transcript/OCR when the evidence is textual.
  Calibrate confidence; an empty list is a valid answer for neutral content.
- sentiment.arc: split the video into a few segments by time
- risk_flags: short snake_case indicator labels (e.g. unsourced_statistic, emotional_manipulation,
  out_of_context_footage_suspect, impersonation_suspect, urgent_call_to_action)
Timestamps are seconds."""


def _img_block(path: str) -> dict:
    data = base64.standard_b64encode(Path(path).read_bytes()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def _fmt_transcript(ev: Evidence) -> str:
    if not ev.transcript:
        return "(no speech detected)"
    return "\n".join(f"[{s.t_start:.1f}-{s.t_end:.1f}s] {s.text}" for s in ev.transcript)


def _fmt_ocr(ev: Evidence) -> str:
    if not ev.ocr:
        return "(no on-screen text detected)"
    return "\n".join(f"[{o.t:.1f}s] {o.text}" for o in ev.ocr)


def _fmt_meta(ev: Evidence) -> str:
    m = ev.meta
    return json.dumps({
        "platform": m.platform, "author": m.author, "posted_at": m.posted_at,
        "caption": m.caption, "hashtags": m.hashtags, "duration_s": m.duration,
        "views": m.view_count, "likes": m.like_count, "language": ev.language,
    }, ensure_ascii=False)


class RefusalError(RuntimeError):
    pass


class ClaudeAnalyzer:
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("SHORTS_MODEL", DEFAULT_MODEL)
        self.client = anthropic.Anthropic()

    def _parse(self, system: str, content: list[dict], output_format):
        resp = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=16000,
            betas=BETAS,
            fallbacks="default",  # re-run on a fallback model if a safety classifier declines
            system=system,
            messages=[{"role": "user", "content": content}],
            output_format=output_format,
        )
        if resp.stop_reason == "refusal":
            raise RefusalError(f"model declined: {resp.stop_details}")
        if resp.stop_reason == "max_tokens" or resp.parsed_output is None:
            raise RuntimeError(f"no parsed output (stop_reason={resp.stop_reason})")
        return resp.parsed_output

    def describe(self, ev: Evidence) -> VisualPass:
        content: list[dict] = []
        for f in ev.frames:
            content.append({"type": "text", "text": f"[frame t={f.t:.2f}s]"})
            content.append(_img_block(f.path))
        content.append({"type": "text", "text":
                        f"TRANSCRIPT:\n{_fmt_transcript(ev)}\n\nOCR TEXT:\n{_fmt_ocr(ev)}\n\n"
                        f"Describe all {len(ev.frames)} frames."})
        return self._parse(DESCRIBE_SYSTEM, content, VisualPass)

    def analyze(self, ev: Evidence, visuals: VisualPass) -> Analysis:
        vis = "\n".join(f"[{v.t:.1f}s] ({v.kind}) {v.description}" for v in visuals.frames)
        text = (f"METADATA:\n{_fmt_meta(ev)}\n\nTRANSCRIPT:\n{_fmt_transcript(ev)}\n\n"
                f"ON-SCREEN TEXT (OCR):\n{_fmt_ocr(ev)}\n\nVISUALS:\n{vis}")
        return self._parse(ANALYZE_SYSTEM, [{"type": "text", "text": text}], Analysis)
