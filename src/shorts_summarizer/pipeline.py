"""fetch -> frames+OCR -> transcribe -> describe -> analyze, each stage cached on disk."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, TypeVar

from pydantic import BaseModel, TypeAdapter

from . import fetch, frames, transcribe
from .analyzers import Evidence, get_analyzer
from .describe import merge_text, merge_visuals
from .schema import (Analysis, Meta, OcrLine, SelectedFrame, SentimentVerdict, TranscriptSegment,
                     VideoReport, VisualPass)

T = TypeVar("T")


class FramesStage(BaseModel):
    frames: list[SelectedFrame]
    ocr: list[OcrLine]
    duration: float


class TranscriptStage(BaseModel):
    language: str | None
    segments: list[TranscriptSegment]


def _cached(path: Path, typ, compute: Callable[[], T], force: bool = False) -> T:
    ta = TypeAdapter(typ)
    if path.exists() and not force:
        return ta.validate_json(path.read_text())
    val = compute()
    path.write_text(ta.dump_json(val, indent=2).decode())
    return val


def run(url: str, out_root: Path = Path("out"), cookies_from_browser: str | None = None,
        analyzer: str = "claude", model: str | None = None, budget: int | None = None,
        reanalyze: bool = False, lang: str = "id", log: Callable[[str], None] = print) -> VideoReport:
    info = fetch.probe(url, cookies_from_browser)
    key = fetch.video_key(info, url)
    wd = out_root / key
    wd.mkdir(parents=True, exist_ok=True)
    report_path = wd / "report.json"
    if report_path.exists() and not reanalyze:
        cached = VideoReport.model_validate_json(report_path.read_text())
        if cached.report_language == lang:
            log(f"[{key}] cached")
            return cached

    def _dl() -> Meta:
        log(f"[{key}] downloading")
        return fetch.download(url, wd, cookies_from_browser)[1]

    meta = _cached(wd / "meta.json", Meta, _dl)
    video = wd / "video.mp4"

    def _frames():
        log(f"[{key}] selecting frames + OCR")
        sel, ocr_lines, dur = frames.extract(video, wd / "frames", budget=budget)
        return FramesStage(frames=sel, ocr=ocr_lines, duration=dur)

    fr = _cached(wd / "frames.json", FramesStage, _frames)
    sel, ocr_lines = fr.frames, fr.ocr
    duration = float(meta.duration or fr.duration)
    log(f"[{key}] {len(sel)} frames selected, {len(ocr_lines)} OCR lines")

    def _tx():
        log(f"[{key}] transcribing")
        segs, lang = transcribe.transcribe(video)
        return TranscriptStage(language=lang, segments=segs)

    tx = _cached(wd / "transcript.json", TranscriptStage, _tx)
    segs = tx.segments

    ev = Evidence(meta=meta, frames=sel, ocr=ocr_lines, transcript=segs, language=tx.language)
    an = get_analyzer(analyzer, model, lang)

    def _describe() -> VisualPass:
        log(f"[{key}] describing visuals ({an.model})")
        return an.describe(ev)

    vp = _cached(wd / f"visuals.{lang}.json", VisualPass, _describe, force=reanalyze)

    def _analyze() -> Analysis:
        log(f"[{key}] analyzing")
        return an.analyze(ev, vp)

    a = _cached(wd / f"analysis.{lang}.json", Analysis, _analyze, force=reanalyze)

    visuals = merge_visuals(sel, vp, duration)
    report = VideoReport(
        meta=meta, summary=a.summary, topics=a.topics,
        visuals=visuals, visuals_used=[v.description for v in visuals],
        on_screen_text=merge_text(ocr_lines, vp),
        transcript=segs, language=tx.language,
        claims=a.claims, sentiment=a.sentiment, propaganda_techniques=a.propaganda_techniques,
        call_to_action=a.call_to_action, target_groups=a.target_groups, risk_flags=a.risk_flags,
        model=an.model, report_language=lang,
    )
    report_path.write_text(report.model_dump_json(indent=2))
    with (out_root / "reports.jsonl").open("a") as f:
        f.write(report.model_dump_json() + "\n")
    log(f"[{key}] done -> {report_path}")
    return report


def sentiment_stage(report_path: Path, mode: str | None = None, force: bool = False,
                    log: Callable[[str], None] = print) -> SentimentVerdict:
    """Whole-video sentiment verdict for an existing report.json, cached as sentiment.json."""
    from . import sentiment

    out = report_path.with_name("sentiment.json")
    if out.exists() and not force:
        cached = SentimentVerdict.model_validate_json(out.read_text())
        if mode is None or cached.mode == mode:
            return cached
    report = VideoReport.model_validate_json(report_path.read_text())
    log(f"[{report.meta.platform}_{report.meta.id}] sentiment ({mode or 'default mode'})")
    verdict = sentiment.analyze(report, mode=mode)
    out.write_text(verdict.model_dump_json(indent=2))
    return verdict
