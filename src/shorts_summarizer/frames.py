"""Informative-frame selection.

Candidates are sampled at `fps`. Each gets a base score from
  - scene change   (HSV histogram distance to the previous candidate)
  - text change    (new OCR tokens vs previous candidate; overlays change without cuts)
and must pass a quality gate (not blurry, not black). Greedy selection then adds
pHash novelty, a min time gap, forced first/last frames and gap filling so long
static stretches (talking heads) still get coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import subprocess
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

from . import ocr
from .schema import OcrLine, SelectedFrame


@dataclass
class Candidate:
    t: float
    img: np.ndarray
    phash: imagehash.ImageHash
    sharpness: float
    brightness: float
    ocr_lines: list[tuple[str, float]]
    scene: float = 0.0
    text_change: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def base(self) -> float:
        return self.scene + 1.5 * self.text_change


def _resize(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    return img if s >= 1 else cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def _hist(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def sample(video: Path, fps: float = 2.0, ocr_fps: float = 1.0, long_side: int = 960) -> tuple[list[Candidate], float]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = total / native if total else 0.0
    step = max(1, round(native / fps))
    ocr_every = max(1, round(fps / ocr_fps))  # OCR is the slow part; captions persist > 1s
    cands: list[Candidate] = []
    prev_hist = None
    prev_tokens: set[str] = set()
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                img = _resize(frame, long_side)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                n = len(cands)
                lines = ocr.read_text(img) if n % ocr_every == 0 else (cands[-1].ocr_lines if cands else [])
                c = Candidate(
                    t=idx / native,
                    img=img,
                    phash=imagehash.phash(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))),
                    sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                    brightness=float(gray.mean()),
                    ocr_lines=lines,
                )
                hist = _hist(img)
                if prev_hist is not None:
                    c.scene = float(max(0.0, 1.0 - cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)))
                toks = ocr.tokens(lines)
                if toks:
                    new = toks - prev_tokens
                    c.text_change = len(new) / len(toks)
                prev_hist, prev_tokens = hist, toks
                cands.append(c)
        idx += 1
    cap.release()
    return cands, duration or (cands[-1].t if cands else 0.0)


def _to_h264(video: Path) -> Path:
    out = video.with_name("video_h264.mp4")
    if not out.exists():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-an",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", str(out)], check=True)
    return out


def select(cands: list[Candidate], duration: float, budget: int | None = None,
           min_gap: float = 0.5, max_static_gap: float = 6.0) -> list[Candidate]:
    if not cands:
        return []
    if budget is None:
        budget = int(np.clip(duration / 4, 8, 16))
    med_sharp = float(np.median([c.sharpness for c in cands])) or 1.0
    good = [c for c in cands if c.brightness > 15 and c.sharpness > 0.25 * med_sharp] or cands

    chosen: list[Candidate] = []

    def ok_to_add(c: Candidate, need_novel: bool = True) -> bool:
        if any(abs(c.t - s.t) < min_gap for s in chosen):
            return False
        if need_novel and chosen:
            novelty = min(c.phash - s.phash for s in chosen) / 64
            # near-duplicate image and no new text -> redundant
            if novelty < 0.1 and c.text_change < 0.3:
                return False
        return True

    for c, why in ((good[0], "hook"), (good[-1], "ending")):
        if ok_to_add(c, need_novel=False):
            c.reasons.append(why)
            chosen.append(c)

    for c in sorted(good, key=lambda c: c.base, reverse=True):
        if len(chosen) >= budget or c.base < 0.15:
            break
        if ok_to_add(c):
            if c.scene >= 0.3:
                c.reasons.append("scene_change")
            if c.text_change >= 0.3:
                c.reasons.append("new_text")
            if not c.reasons:
                c.reasons.append("visual_change")
            chosen.append(c)

    # coverage: fill long stretches with no selected frame
    while len(chosen) < budget:
        ts = sorted([0.0] + [s.t for s in chosen] + [duration])
        gaps = [(b - a, a, b) for a, b in zip(ts, ts[1:]) if b - a > max_static_gap]
        if not gaps:
            break
        _, a, b = max(gaps)
        mid = (a + b) / 2
        pool = [c for c in good if a < c.t < b and ok_to_add(c, need_novel=False)]
        if not pool:
            break
        c = min(pool, key=lambda c: abs(c.t - mid))
        c.reasons.append("coverage")
        chosen.append(c)

    return sorted(chosen, key=lambda c: c.t)


def extract(video: Path, out_dir: Path, fps: float = 2.0, budget: int | None = None
            ) -> tuple[list[SelectedFrame], list[OcrLine], float]:
    """Run sampling + selection, write JPEGs. Returns (selected frames, deduped OCR, duration)."""
    cands, duration = sample(video, fps=fps)
    if not cands:  # codec OpenCV can't decode (e.g. AV1) -> transcode and retry
        cands, duration = sample(_to_h264(video), fps=fps)
    if not cands:
        raise RuntimeError(f"could not decode any frames from {video}")
    chosen = select(cands, duration, budget=budget)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for c in chosen:
        p = out_dir / f"t{c.t:07.2f}.jpg"
        cv2.imwrite(str(p), _resize(c.img, 768), [cv2.IMWRITE_JPEG_QUALITY, 85])
        frames.append(SelectedFrame(t=round(c.t, 2), path=str(p), phash=str(c.phash),
                                    score=round(c.base, 3), reasons=c.reasons))
    ocr_lines = ocr.dedupe([(c.t, c.ocr_lines) for c in cands])
    return frames, ocr_lines, duration
