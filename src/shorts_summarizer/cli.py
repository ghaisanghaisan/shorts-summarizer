from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(add_completion=False, help="Make an AI 'watch' TikToks / IG Reels.")

CookiesOpt = typer.Option(None, "--cookies-from-browser", help="firefox|chrome|... (needed for most IG listing)")
OutOpt = typer.Option(Path("out"), "--out", help="Output directory")
LangOpt = typer.Option(os.getenv("REPORT_LANG", "id"), "--lang",
                       help="Report language: id (Bahasa Indonesia, default), en, ... ($REPORT_LANG)")
SentimentOpt = typer.Option(False, "--sentiment", help="Also write sentiment.json (whole-video verdict)")
AnalyzerOpt = typer.Option(os.getenv("SHORTS_ANALYZER", "claude"), "--analyzer",
                           help="claude|gemini (default $SHORTS_ANALYZER or claude)")


@app.command()
def analyze(
    urls: list[str] = typer.Argument(..., help="Video URL(s), or a path to a .txt file of URLs"),
    out: Path = OutOpt,
    cookies_from_browser: Optional[str] = CookiesOpt,
    analyzer: str = AnalyzerOpt,
    lang: str = LangOpt,
    model: Optional[str] = typer.Option(None, help="Override model (else $SHORTS_MODEL / $GEMINI_MODEL)"),
    frames: Optional[int] = typer.Option(None, help="Frame budget (default scales 8-16 with length)"),
    reanalyze: bool = typer.Option(False, help="Re-run the LLM stages, keep download/OCR/transcript cache"),
    sentiment: bool = SentimentOpt,
):
    """Analyze one or more videos."""
    from .pipeline import run

    expanded: list[str] = []
    for u in urls:
        p = Path(u)
        if not u.startswith("http") and p.is_file():
            expanded += [l.strip() for l in p.read_text().splitlines() if l.strip() and not l.startswith("#")]
        else:
            expanded.append(u)
    _run_all(expanded, out, cookies_from_browser, analyzer, lang, model, frames, reanalyze, sentiment=sentiment)


@app.command()
def crawl(
    target: str = typer.Argument(..., help="@user, #hashtag, or a profile/tag URL"),
    platform: str = typer.Option("tiktok", help="tiktok|instagram (ignored if target is a URL)"),
    limit: int = typer.Option(10, help="Max videos"),
    since: Optional[str] = typer.Option(None, help="Only videos on/after YYYY-MM-DD"),
    out: Path = OutOpt,
    cookies_from_browser: Optional[str] = CookiesOpt,
    analyzer: str = AnalyzerOpt,
    lang: str = LangOpt,
    model: Optional[str] = typer.Option(None),
    sleep: float = typer.Option(2.0, help="Seconds between videos (be polite)"),
    sentiment: bool = SentimentOpt,
):
    """List a profile/hashtag's videos and analyze each."""
    from .crawl import list_videos

    if target.startswith("http"):
        platform = "instagram" if "instagram" in target else "tiktok"
    urls = list_videos(target, platform, limit, cookies_from_browser, since)
    typer.echo(f"found {len(urls)} videos")
    _run_all(urls, out, cookies_from_browser, analyzer, lang, model, None, False, sleep, sentiment=sentiment)


def _run_all(urls, out, cookies, analyzer, lang, model, frames, reanalyze, sleep: float = 0.0,
             sentiment: bool = False):
    from .pipeline import run

    out.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    for i, u in enumerate(urls):
        try:
            r = run(u, out, cookies, analyzer=analyzer, lang=lang, model=model, budget=frames, reanalyze=reanalyze)
            ok += 1
            typer.echo(f"  summary: {r.summary}")
            typer.echo(f"  visuals_used: {len(r.visuals_used)} | text: {len(r.on_screen_text)} | "
                       f"claims: {len(r.claims)} | techniques: {[t.technique for t in r.propaganda_techniques]}")
            if sentiment:
                from .pipeline import sentiment_stage
                _echo_sentiment(sentiment_stage(out / f"{r.meta.platform}_{r.meta.id}" / "report.json",
                                                force=reanalyze))
        except Exception as e:  # keep crawling past one bad video
            fail += 1
            typer.secho(f"[error] {u}: {type(e).__name__}: {e}", fg="red", err=True)
        if sleep and i < len(urls) - 1:
            time.sleep(sleep)
    typer.echo(f"done: {ok} ok, {fail} failed -> {out}/reports.jsonl")


def _echo_sentiment(v):
    extra = f" (model {v.model_score:+.2f}, llm {v.llm_score:+.2f}, {'agree' if v.agree else 'DISAGREE'})" \
        if v.mode == "ensemble" else f" (llm only)"
    flag = "  [needs review]" if v.needs_review else ""
    typer.echo(f"  sentiment: {v.label.upper()} {v.score:+.2f}{extra} | emotions: {', '.join(v.emotions)}{flag}")


@app.command("sentiment")
def sentiment_cmd(
    paths: list[Path] = typer.Argument(..., help="report.json file(s), a video dir, or an out/ dir"),
    mode: Optional[str] = typer.Option(None, help="ensemble (IndoBERT + LLM) | llm  (default $SENTIMENT_MODE or ensemble)"),
    force: bool = typer.Option(False, help="Recompute even if sentiment.json exists"),
):
    """Whole-video sentiment verdict (positive / neutral / negative) for existing reports."""
    from .pipeline import sentiment_stage

    reports: list[Path] = []
    for p in paths:
        if p.is_file():
            reports.append(p)
        elif (p / "report.json").exists():
            reports.append(p / "report.json")
        else:
            reports += sorted(p.glob("*/report.json"))
    if not reports:
        raise typer.BadParameter("no report.json found")
    for rp in reports:
        typer.echo(str(rp.parent))
        _echo_sentiment(sentiment_stage(rp, mode=mode, force=force))


if __name__ == "__main__":
    app()
