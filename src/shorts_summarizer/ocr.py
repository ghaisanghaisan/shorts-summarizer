"""On-screen text via RapidOCR (ONNX, CPU-friendly)."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from functools import lru_cache

import numpy as np

from .schema import OcrLine

MIN_CONF = 0.6


@lru_cache(maxsize=1)
def _engine():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def read_text(img_bgr: np.ndarray) -> list[tuple[str, float]]:
    """Return [(text, confidence)] top-to-bottom for one frame."""
    result, _ = _engine()(img_bgr)
    if not result:
        return []
    # result rows: [box(4 points), text, score]; sort by top y then x
    rows = sorted(result, key=lambda r: (min(p[1] for p in r[0]), min(p[0] for p in r[0])))
    return [(text.strip(), float(score)) for _, text, score in rows
            if float(score) >= MIN_CONF and len(text.strip()) >= 2]


def norm(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.lower()).strip()


def tokens(lines: list[tuple[str, float]]) -> set[str]:
    """Word tokens used for text-change scoring. Pure numbers are ignored so ticking
    counters / timers / like counts don't look like new captions."""
    return {w for t, _ in lines for w in norm(t).split() if not w.isdigit()}


def _squash(text: str) -> str:
    return norm(text).replace(" ", "")


def _similar(a: str, b: str) -> bool:
    a, b = _squash(a), _squash(b)
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.75


def dedupe(per_frame: list[tuple[float, list[tuple[str, float]]]], window: float = 3.0) -> list[OcrLine]:
    """Collapse the same overlay seen across nearby frames (incl. partial/garbled OCR variants
    like 'WELCOME' / 'WELCOMETO' / 'WELCOME TO'); keep first time, longest text."""
    out: list[OcrLine] = []
    last_seen: dict[int, float] = {}  # index in out -> last time seen
    for t, lines in per_frame:
        for text, conf in lines:
            if sum(ch.isalnum() for ch in text) < 3:
                continue
            match = next((i for i, ts in last_seen.items()
                          if t - ts <= window and _similar(out[i].text, text)), None)
            if match is None:
                out.append(OcrLine(t=round(t, 2), text=text, confidence=round(conf, 3)))
                last_seen[len(out) - 1] = t
            else:
                o = out[match]
                if len(_squash(text)) > len(_squash(o.text)):
                    o.text, o.confidence = text, round(conf, 3)
                last_seen[match] = t
        last_seen = {i: ts for i, ts in last_seen.items() if t - ts <= window}
    return out
