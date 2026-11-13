# brag plan: shorts-summarizer

**What it is:** a CLI that "watches" TikToks, Reels and Shorts for you and returns an evidence
report: the visuals, on-screen text, transcript, claims, sentiment and propaganda techniques,
each with a timestamp.
**Who it's for:** misinformation researchers, fact-checkers and hackathon teams who can't watch
every viral video themselves.
**What sets it apart:** it picks the informative frames itself (scene cuts plus *new on-screen text*),
keeps OCR text and speech word for word, and returns flags *with receipts*, not just verdicts. The
report comes out in Bahasa Indonesia.
**Hook:** a loud, fake viral hoax video ("PEMERINTAH SEMBUNYIKAN INI!") that nobody has time to check.
**Share caption:** "Nobody can watch every viral video. shorts-summarizer does, and shows its receipts."

## Source material (all real)
- The demo input is a 12-second hoax-style clip **made for this video**: fictional account `@info_viral_88`,
  no real people or brands, Indonesian TTS voiceover. It's a stand-in so no real person or
  copyrighted broadcast appears in a public video.
- Everything downstream is the **real pipeline output** on that clip: the frame candidate scores, the
  picked frames and their reasons, OCR, Whisper transcript, and the Gemini (`gemini-3.8-flash`) report
  in Indonesian. See `work/analysis/`.
- Terminal command and flags are the real CLI.

## Tone
`default`: punchy, playful, clean. A dark UI with the project's own JSON and terminal look
(JetBrains Mono for data, Poppins for headlines). Accent colors: alert red `#ef4444` for the hoax,
signal green `#22c55e` for detections, ink `#0b0f17` background.

## Music
120 BPM, A minor (Am–F–C–G), one beat = 0.5s. A one-beat pickup puts the downbeats at 0.5, 2.5, 4.5, and so on.
Tense, hats-only intro, then a riser, a drop at 2.5s, a final chord hit at 20.5s and a tail to 22s.
SFX are synthesized in the same key: soft whooshes on cuts, chord-tone pops when frames are picked,
quiet typing ticks, and a chord hit when a technique is flagged.

## Storyboard (22.0s, every cut on a downbeat)
| # | Time | Scene | On screen |
|---|---|---|---|
| 1 | 0.0–2.5 | **Hook** | The hoax clip plays in a phone frame. "Nobody watches every video." |
| 2 | 2.5–6.5 | **Reveal** | Drop. Title `shorts-summarizer`, and a terminal types `uv run shorts analyze <url>`. "Paste a TikTok, Reel or Short." |
| 3 | 6.5–10.5 | **Highlight 1: picks frames** | Real candidate filmstrip with the real score curve; picked frames light up with real reasons (`hook`, `new_text`, `scene_change`). "Picks the frames that matter." |
| 4 | 10.5–14.5 | **Highlight 2: reads + hears** | Real OCR lines and transcript lines type in with timestamps. "Reads every caption. Hears every word." |
| 5 | 14.5–18.5 | **Highlight 3: receipts** | Real report: propaganda technique chips, each with a verbatim Indonesian evidence quote, plus risk flags. "Flags the manipulation, with receipts." |
| 6 | 18.5–22.0 | **Outro** | `shorts-summarizer` · "TikTok · Reels · Shorts → evidence report" · `uv run shorts analyze <url>` |
