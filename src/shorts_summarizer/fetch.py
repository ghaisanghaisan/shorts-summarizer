"""Download a single video + metadata with yt-dlp."""

from __future__ import annotations

import re
from pathlib import Path

import yt_dlp

from .schema import Meta


def ydl_base_opts(cookies_from_browser: str | None = None) -> dict:
    opts: dict = {"quiet": True, "no_warnings": True, "noprogress": True}
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser,)
    return opts


def platform_of(url: str, extractor: str | None = None) -> str:
    s = (extractor or url).lower()
    if "tiktok" in s:
        return "tiktok"
    if "instagram" in s:
        return "instagram"
    return re.sub(r"\W+", "_", extractor or "web").strip("_").lower() or "web"


def probe(url: str, cookies_from_browser: str | None = None) -> dict:
    with yt_dlp.YoutubeDL(ydl_base_opts(cookies_from_browser)) as ydl:
        return ydl.extract_info(url, download=False)


def video_key(info: dict, url: str) -> str:
    return f"{platform_of(url, info.get('extractor_key'))}_{info['id']}"


def download(url: str, workdir: Path, cookies_from_browser: str | None = None) -> tuple[Path, Meta]:
    workdir.mkdir(parents=True, exist_ok=True)
    video_path = workdir / "video.mp4"
    opts = ydl_base_opts(cookies_from_browser) | {
        "outtmpl": str(workdir / "video.%(ext)s"),
        # prefer H.264 (OpenCV can't decode AV1/VP9 on many builds); fall back to anything
        "format": ("bv*[vcodec^=avc1][height<=1280]+ba[ext=m4a]/b[ext=mp4][vcodec^=avc1]/"
                   "bv*[vcodec^=avc1]+ba/bv*+ba/b"),
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    if not video_path.exists():
        # some formats land with another extension; normalise
        cand = next((p for p in workdir.glob("video.*") if p.suffix != ".part"), None)
        if cand is None:
            raise RuntimeError(f"yt-dlp produced no file for {url}")
        cand.rename(video_path)

    desc = info.get("description") or info.get("title") or ""
    tags = info.get("tags") or re.findall(r"#(\w+)", desc)
    meta = Meta(
        platform=platform_of(url, info.get("extractor_key")),
        id=str(info["id"]),
        url=info.get("webpage_url") or url,
        author=info.get("uploader") or info.get("channel") or info.get("uploader_id"),
        posted_at=info.get("upload_date"),
        caption=desc or None,
        hashtags=list(dict.fromkeys(tags)),
        duration=info.get("duration"),
        view_count=info.get("view_count"),
        like_count=info.get("like_count"),
        comment_count=info.get("comment_count"),
        share_count=info.get("repost_count"),
    )
    return video_path, meta
