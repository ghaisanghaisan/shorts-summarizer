"""Turn a profile / hashtag target into a list of video URLs."""

from __future__ import annotations

import yt_dlp

from .fetch import ydl_base_opts


def target_to_url(target: str, platform: str) -> str:
    if target.startswith("http"):
        return target
    if platform == "tiktok":
        if target.startswith("#"):
            return f"https://www.tiktok.com/tag/{target[1:]}"
        return f"https://www.tiktok.com/@{target.lstrip('@')}"
    if platform == "instagram":
        if target.startswith("#"):
            return f"https://www.instagram.com/explore/tags/{target[1:]}/"
        return f"https://www.instagram.com/{target.lstrip('@')}/reels/"
    raise ValueError(f"unknown platform {platform!r}")


def list_videos(target: str, platform: str = "tiktok", limit: int = 10,
                cookies_from_browser: str | None = None, since: str | None = None) -> list[str]:
    url = target_to_url(target, platform)
    opts = ydl_base_opts(cookies_from_browser) | {"extract_flat": "in_playlist", "playlistend": limit}
    if since:
        opts["daterange"] = yt_dlp.utils.DateRange(since.replace("-", ""))
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = info.get("entries") or [info]
    urls = []
    for e in entries:
        if not e:
            continue
        u = e.get("url") or e.get("webpage_url")
        if u and not u.startswith("http") and platform == "instagram":
            u = f"https://www.instagram.com/reel/{u}/"
        if u:
            urls.append(u)
    return urls[:limit]
