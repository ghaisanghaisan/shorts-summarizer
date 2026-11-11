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
AnalyzerOpt = typer.Option(os.getenv("SHORTS_ANALYZER", "claude"), "--analyzer",
                           help="claude|gemini (default $SHORTS_ANALYZER or claude)")


@app.command()
def analyze(
    urls: list[str] = typer.Argument(..., help="Video URL(s), or a path to a .txt file of URLs"),
    out: Path = OutOpt,
    cookies_from_browser: Optional[str] = CookiesOpt,
    analyzer: str = AnalyzerOpt,
    model: Optional[str] = typer.Option(None, help="Override model (else $SHORTS_MODEL / $GEMINI_MODEL)"),
    frames: Optional[int] = typer.Option(None, help="Frame budget (default scales 8-16 with length)"),
    reanalyze: bool = typer.Option(False, help="Re-run the LLM stages, keep download/OCR/transcript cache"),
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
    _run_all(expanded, out, cookies_from_browser, analyzer, model, frames, reanalyze)


@app.command()
def crawl(
    target: str = typer.Argument(..., help="@user, #hashtag, or a profile/tag URL"),
    platform: str = typer.Option("tiktok", help="tiktok|instagram (ignored if target is a URL)"),
    limit: int = typer.Option(10, help="Max videos"),
    since: Optional[str] = typer.Option(None, help="Only videos on/after YYYY-MM-DD"),
    out: Path = OutOpt,
    cookies_from_browser: Optional[str] = CookiesOpt,
    analyzer: str = AnalyzerOpt,
    model: Optional[str] = typer.Option(None),
    sleep: float = typer.Option(2.0, help="Seconds between videos (be polite)"),
):
    """List a profile/hashtag's videos and analyze each."""
    from .crawl import list_videos

    if target.startswith("http"):
        platform = "instagram" if "instagram" in target else "tiktok"
    urls = list_videos(target, platform, limit, cookies_from_browser, since)
    typer.echo(f"found {len(urls)} videos")
    _run_all(urls, out, cookies_from_browser, analyzer, model, None, False, sleep)


def _run_all(urls, out, cookies, analyzer, model, frames, reanalyze, sleep: float = 0.0):
    from .pipeline import run

    out.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    for i, u in enumerate(urls):
        try:
            r = run(u, out, cookies, analyzer=analyzer, model=model, budget=frames, reanalyze=reanalyze)
            ok += 1
            typer.echo(f"  summary: {r.summary}")
            typer.echo(f"  visuals_used: {len(r.visuals_used)} | text: {len(r.on_screen_text)} | "
                       f"claims: {len(r.claims)} | techniques: {[t.technique for t in r.propaganda_techniques]}")
        except Exception as e:  # keep crawling past one bad video
            fail += 1
            typer.secho(f"[error] {u}: {type(e).__name__}: {e}", fg="red", err=True)
        if sleep and i < len(urls) - 1:
            time.sleep(sleep)
    typer.echo(f"done: {ok} ok, {fail} failed -> {out}/reports.jsonl")


if __name__ == "__main__":
    app()
