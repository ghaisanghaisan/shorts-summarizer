"""Turn per-frame model descriptions into merged visual spans + on-screen text list."""

from __future__ import annotations

import imagehash

from .ocr import _similar, norm
from .schema import OcrLine, OnScreenText, SelectedFrame, Visual, VisualPass


def _jaccard(a: str, b: str) -> float:
    x, y = set(norm(a).split()), set(norm(b).split())
    return len(x & y) / len(x | y) if x | y else 1.0


def merge_visuals(frames: list[SelectedFrame], vp: VisualPass, duration: float) -> list[Visual]:
    by_t = {round(f.t, 2): f for f in frames}
    descs = sorted(vp.frames, key=lambda d: d.t)
    visuals: list[Visual] = []
    for i, d in enumerate(descs):
        f = by_t.get(round(d.t, 2)) or min(frames, key=lambda f: abs(f.t - d.t))
        t_end = descs[i + 1].t if i + 1 < len(descs) else max(duration, d.t)
        if visuals:
            last = visuals[-1]
            same_img = (imagehash.hex_to_hash(last.phashes[-1]) - imagehash.hex_to_hash(f.phash)) <= 12
            if last.kind == d.kind and (same_img or _jaccard(last.description, d.description) > 0.6):
                last.t_end = round(t_end, 2)
                last.frame_paths.append(f.path)
                last.phashes.append(f.phash)
                last.people = list(dict.fromkeys(last.people + d.people))
                continue
        visuals.append(Visual(t_start=round(d.t, 2), t_end=round(t_end, 2), kind=d.kind,
                              description=d.description, people=d.people,
                              frame_paths=[f.path], phashes=[f.phash]))
    return visuals


def merge_text(ocr_lines: list[OcrLine], vp: VisualPass) -> list[OnScreenText]:
    out = [OnScreenText(t=o.t, text=o.text, source="ocr") for o in ocr_lines]
    seen = [o.text for o in ocr_lines]
    for d in vp.frames:
        for txt in d.missed_text:
            if norm(txt) and not any(_similar(txt, s) for s in seen):
                seen.append(txt)
                out.append(OnScreenText(t=d.t, text=txt, source="model"))
    return sorted(out, key=lambda x: x.t)
