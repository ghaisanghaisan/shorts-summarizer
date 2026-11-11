# shorts-summarizer

Makes an AI "watch" TikToks / Instagram Reels and emits a structured, timestamped `VideoReport`
(summary, visuals, on-screen text, transcript, claims, sentiment, propaganda-technique indicators)
for downstream misinformation / sentiment modules.

## Setup
Requires Python 3.11+, [uv](https://docs.astral.sh/uv/) and `ffmpeg` on PATH.
```bash
uv sync
cp .env.example .env   # set ANTHROPIC_API_KEY and/or GEMINI_API_KEY
```

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude backend |
| `GEMINI_API_KEY` | Gemini backend ([get a key](https://aistudio.google.com/apikey)) |
| `SHORTS_ANALYZER` | `claude` (default) or `gemini` |
| `SHORTS_MODEL` / `GEMINI_MODEL` | model override per backend |
| `WHISPER_MODEL` | `small` (default); use `medium` for non-English |

## Usage
```bash
uv run shorts analyze https://www.tiktok.com/@user/video/123
uv run shorts analyze urls.txt                       # one URL per line
uv run shorts crawl @nasa --limit 5                  # TikTok profile
uv run shorts crawl "#election" --limit 10 --since 2026-11-01
uv run shorts crawl @someone --platform instagram --cookies-from-browser firefox
uv run shorts analyze <url> --analyzer gemini        # use Gemini instead of Claude
uv run shorts analyze <url> --reanalyze              # re-run LLM stages only (after prompt edits)
```

Output: `out/<platform>_<id>/report.json` (+ `video.mp4`, `frames/`, stage caches) and
`out/reports.jsonl` (one report per line).

## Pipeline
1. **fetch** (yt-dlp): mp4 + metadata
2. **frames** – candidates at 2 fps, OCR at 1 fps. Score = scene change (HSV hist) + 1.5 × new
   on-screen text; blur/black gate; greedy pick with pHash novelty, 0.5 s min gap, forced
   hook/ending frames, gap-fill for static stretches. Budget 8–16 frames.
3. **ocr** (RapidOCR): verbatim text, fuzzy-deduped across frames
4. **transcribe** (faster-whisper, `WHISPER_MODEL`, default `small`)
5. **describe** – one multi-image Claude call over all selected frames in order → per-frame
   descriptions, merged into spans → `visuals` / `visuals_used`
6. **analyze** – text-only Claude call → claims, sentiment arc, SemEval-2020 propaganda
   techniques with verbatim evidence, risk flags (indicators, not verdicts)

Backends (`--analyzer`): Claude (`claude-opus-5-5`, with server-side refusal fallbacks) or
Gemini (`gemini-3.8-flash`, retries on 429/5xx). Both use the same prompts and output schema.
Works with any yt-dlp-supported URL (TikTok, Instagram Reels, YouTube Shorts, ...).
