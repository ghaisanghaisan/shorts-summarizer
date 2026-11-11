"""Data contracts. `VideoReport` is what downstream modules consume."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

VisualKind = Literal[
    "talking_head", "b_roll", "screenshot", "meme", "chart", "news_clip",
    "text_card", "product", "crowd_event", "ai_generated_suspect", "other",
]


# ---------- deterministic evidence ----------

class Meta(BaseModel):
    platform: str
    id: str
    url: str
    author: str | None = None
    posted_at: str | None = None
    caption: str | None = None
    hashtags: list[str] = []
    duration: float | None = None
    view_count: int | None = None
    like_count: int | None = None
    comment_count: int | None = None
    share_count: int | None = None


class SelectedFrame(BaseModel):
    t: float
    path: str
    phash: str
    score: float
    reasons: list[str]


class TranscriptSegment(BaseModel):
    t_start: float
    t_end: float
    text: str


class OcrLine(BaseModel):
    t: float
    text: str
    confidence: float


# ---------- LLM output: visual description pass ----------

class FrameDescription(BaseModel):
    t: float = Field(description="Timestamp label of the frame, copied exactly")
    kind: VisualKind
    description: str = Field(description="Factual description of what is visible. No judgement.")
    people: list[str] = Field(description="Visible people, by role/appearance or name only if captioned")
    missed_text: list[str] = Field(description="On-screen text visible in this frame that is NOT in the OCR list")


class VisualPass(BaseModel):
    frames: list[FrameDescription]


# ---------- LLM output: analysis pass ----------

class Claim(BaseModel):
    text: str
    t: float | None
    modality: Literal["speech", "text", "visual"]
    checkable: bool = Field(description="True if it is a factual claim that could be fact-checked")
    entities: list[str]


class SentimentSegment(BaseModel):
    t_start: float
    t_end: float
    polarity: Literal["negative", "neutral", "positive"]
    emotions: list[str]


class Sentiment(BaseModel):
    overall: Literal["negative", "neutral", "positive", "mixed"]
    score: float = Field(description="-1.0 (very negative) to 1.0 (very positive)")
    emotions: list[str] = Field(description="Dominant emotional tones, e.g. fear, anger, outrage, hope")
    arc: list[SentimentSegment]


class PropagandaTechnique(BaseModel):
    technique: Literal[
        "loaded_language", "name_calling_labeling", "repetition", "exaggeration_minimisation",
        "doubt", "appeal_to_fear_prejudice", "flag_waving", "causal_oversimplification",
        "slogans", "appeal_to_authority", "black_and_white_fallacy", "thought_terminating_cliche",
        "whataboutism_strawman_red_herring", "bandwagon_reductio_ad_hitlerum",
    ]
    evidence_quote: str = Field(description="Verbatim quote from transcript/OCR, or a visual description")
    t: float | None
    confidence: float = Field(description="0.0 to 1.0")


class Analysis(BaseModel):
    summary: str = Field(description="2-4 sentence neutral summary of the video")
    topics: list[str]
    claims: list[Claim]
    sentiment: Sentiment
    propaganda_techniques: list[PropagandaTechnique]
    call_to_action: str | None
    target_groups: list[str] = Field(description="Groups the video addresses or targets")
    risk_flags: list[str] = Field(
        description="Short indicator labels, e.g. 'unsourced_statistic', 'out_of_context_footage_suspect'"
    )


# ---------- final report ----------

class Visual(BaseModel):
    t_start: float
    t_end: float
    kind: VisualKind
    description: str
    people: list[str]
    frame_paths: list[str]
    phashes: list[str]


class OnScreenText(BaseModel):
    t: float
    text: str
    source: Literal["ocr", "model"]


class VideoReport(BaseModel):
    meta: Meta
    summary: str
    topics: list[str]
    visuals: list[Visual]
    visuals_used: list[str]
    on_screen_text: list[OnScreenText]
    transcript: list[TranscriptSegment]
    language: str | None
    claims: list[Claim]
    sentiment: Sentiment
    propaganda_techniques: list[PropagandaTechnique]
    call_to_action: str | None
    target_groups: list[str]
    risk_flags: list[str]
    model: str
