"""Whole-video sentiment verdict from a VideoReport.

Combines a local Indonesian classifier (default mdhugol/indonesia-bert-sentiment-classification,
picked by benchmarks/sentiment_bench.py: macro-F1 0.929 on clean NusaX vs 0.639 for apriandito)
over the transcript, on-screen text and caption with the LLM's own sentiment score from the report.

The local model is optional (`uv sync --extra sentiment`); without it we fall back to LLM-only.
"""

from __future__ import annotations

import os
from functools import lru_cache

from .schema import SentimentUnit, SentimentVerdict, VideoReport

DEFAULT_MODEL = "mdhugol/indonesia-bert-sentiment-classification"
# models whose config only has LABEL_0.. ids; mapping from their model cards
LABEL_MAPS = {
    "mdhugol/indonesia-bert-sentiment-classification": {0: "positive", 1: "neutral", 2: "negative"},
}
POS_T, NEG_T = 0.2, -0.2
OCR_WINDOW = 3.0
MAX_WORDS = 150  # ~256 IndoBERT tokens incl. context


def _label(score: float) -> str:
    return "positive" if score >= POS_T else "negative" if score <= NEG_T else "neutral"


def context_of(report: VideoReport) -> str:
    if report.topics:
        return report.topics[0]
    return report.summary.split(". ")[0][:200]


def _chunks(text: str) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + MAX_WORDS]) for i in range(0, len(words), MAX_WORDS)] or [text]


def build_units(report: VideoReport) -> list[SentimentUnit]:
    """Text units: transcript segments, OCR lines grouped into ~3s windows, and the caption."""
    units: list[SentimentUnit] = []
    for s in report.transcript:
        for c in _chunks(s.text):
            units.append(SentimentUnit(t=s.t_start, source="transcript", text=c, weight=len(c.split())))

    ocr = sorted((o for o in report.on_screen_text if o.source == "ocr"), key=lambda o: o.t)
    group: list = []
    for o in ocr + [None]:
        if group and (o is None or o.t - group[0].t > OCR_WINDOW):
            text = " ".join(g.text for g in group)
            units.append(SentimentUnit(t=group[0].t, source="ocr", text=text, weight=len(text.split())))
            group = []
        if o is not None:
            group.append(o)

    if report.meta.caption:
        for c in _chunks(report.meta.caption):
            units.append(SentimentUnit(t=None, source="caption", text=c, weight=len(c.split())))
    return [u for u in units if u.text.strip()]


class Scorer:
    """Context-pair or single-text 3-class classifier -> {negative, neutral, positive} probs."""

    def __init__(self, name: str = DEFAULT_MODEL, pair: bool | None = None, label_map: dict | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        self.pair = ("apriandito" in name) if pair is None else pair
        raw = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        canon = {"negatif": "negative", "netral": "neutral", "positif": "positive"}
        self.id2label = label_map or LABEL_MAPS.get(name) or {i: canon.get(v, v) for i, v in raw.items()}
        assert set(self.id2label.values()) == {"negative", "neutral", "positive"}, self.id2label

    def probs(self, texts: list[str], context: str | list[str] | None = None, batch: int = 16) -> list[dict[str, float]]:
        out = []
        for i in range(0, len(texts), batch):
            tb = texts[i:i + batch]
            if self.pair:
                cb = context[i:i + batch] if isinstance(context, list) else [context or "Topik umum"] * len(tb)
                enc = self.tok(cb, tb, truncation="only_second", max_length=256, padding=True, return_tensors="pt")
            else:
                enc = self.tok(tb, truncation=True, max_length=256, padding=True, return_tensors="pt")
            with self.torch.no_grad():
                p = self.torch.softmax(self.model(**enc).logits, dim=-1)
            for row in p.tolist():
                out.append({self.id2label[j]: v for j, v in enumerate(row)})
        return out


@lru_cache(maxsize=2)
def get_scorer(name: str = DEFAULT_MODEL) -> Scorer:
    return Scorer(name)


def analyze(report: VideoReport, mode: str | None = None, model_name: str | None = None) -> SentimentVerdict:
    mode = mode or os.getenv("SENTIMENT_MODE", "ensemble")
    model_name = model_name or os.getenv("SENTIMENT_MODEL", DEFAULT_MODEL)
    llm_score = float(report.sentiment.score)
    context = context_of(report)
    units = build_units(report)

    model_score = None
    if mode == "ensemble" and units:
        try:
            scorer = get_scorer(model_name)
        except ImportError:
            print("  [sentiment] transformers/torch not installed (uv sync --extra sentiment); using LLM only")
            mode = "llm"
        else:
            for u, p in zip(units, scorer.probs([u.text for u in units], context)):
                u.probs = {k: round(v, 4) for k, v in p.items()}
                u.score = round(p["positive"] - p["negative"], 4)
            total = sum(u.weight for u in units) or 1.0
            model_score = sum(u.weight * u.score for u in units) / total
    if mode == "ensemble" and model_score is None:
        mode = "llm"

    if mode == "ensemble":
        score = (model_score + llm_score) / 2
        agree = _label(model_score) == _label(llm_score)
    else:
        score, agree = llm_score, None

    return SentimentVerdict(
        label=_label(score), score=round(score, 4), mode=mode,
        model=model_name if mode == "ensemble" else None,
        model_score=None if model_score is None else round(model_score, 4),
        llm_score=llm_score, llm_label=report.sentiment.overall, agree=agree,
        needs_review=(agree is False) or report.sentiment.overall == "mixed",
        context=context, emotions=report.sentiment.emotions, arc=report.sentiment.arc, units=units,
    )
