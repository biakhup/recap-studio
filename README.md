# Recap Studio v3 AUTO — one-click movie recap dubber

**Upload a video → Generate → Download.** The server does everything:

1. **Auto transcribe** — faster-whisper writes down the speech with timestamps
2. **Auto translate** — each line is translated to the dub language
   (Google Translate via deep-translator, MyMemory as fallback)
3. **Auto voice dub** — edge-tts speaks every translated line with a neural
   voice (Burmese: Nilar ♀ / Thiha ♂, English: Aria ♀ / Guy ♂, plus Thai,
   Korean, Japanese, Chinese, Hindi, Indonesian, Vietnamese, Malay)
4. **Auto edit & render** — every voice clip is speed-fitted into its original
   time window, the dub track is assembled on the exact timeline, translated
   subtitles are burned in, and ffmpeg renders the final MP4
   (original size, 9:16 for Reels/TikTok, or 16:9 for YouTube)

This is the *server* version of Recap Studio. The older v2 ran only in the
phone browser, which cannot record an AI voice into a file — v3 fixes that by
doing the work on a server (your PC, a VPS/Docker, or free Google Colab).

## Run locally

```bash
cd recap-studio-v3
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python server.py            # or: uvicorn server:app --host 0.0.0.0 --port 7860
# open http://localhost:7860
```

Requires `ffmpeg` + `ffprobe` on the PATH. For correct burned-in Burmese
subtitles the machine needs a Myanmar font (e.g. Noto Sans Myanmar) — the
Dockerfile and Colab notebook handle this; Colab already ships Noto fonts.

## Run free on Google Colab

Open `recap_v3_colab.ipynb`, **Runtime → Run all**, wait for
**YOUR WEBSITE IS LIVE AT:** and open that link on your phone.
Keep the Colab tab open — closing it stops the website, and each new run
gives a new link (free sessions last up to ~12 hours).

## Settings

| Env var | Default | Meaning |
|---|---|---|
| `WHISPER_MODEL` | `small` | faster-whisper model: `tiny` (fast, rough), `base`, `small` (good balance), `medium` (slow on CPU) |
| `MAX_UPLOAD_MB` | `200` | upload size cap |
| `PORT` | `7860` | server port |

## API

- `GET /` — the web page
- `GET /api/health` — `{ok, app, whisper_model, max_upload_mb}`
- `POST /api/process` — multipart: `file` (video) + `source_lang` (`auto`, `en`, `my`, …),
  `target_lang` (`my`, `en`, …), `voice` (`female`/`male`), `aspect`
  (`original`/`9:16`/`16:9`), `keep_original_audio` (`true`/`false`) → `{job_id}`
- `GET /api/status/{job_id}` — `{stage, percent, message, error?}`
- `GET /api/download/{job_id}` — the rendered MP4
- `GET /api/subtitles/{job_id}` — the translated `.srt`

Jobs are stored under `jobs/<job_id>/` (`input.*`, `subtitles.srt`,
`output.mp4`, `status.json`, plus a `work/` scratch folder). Jobs run one at
a time; a queue forms automatically on a busy server.

## Honest limits

- CPU transcription is slow: roughly real-time or slower with `small`.
  Short clips (under ~3 minutes) work best, especially on free Colab.
- Translation and edge-tts need internet (they are online services).
- The AI voice is synthetic, timed to the original lines — it is **not**
  lip-sync; mouths will not match the new words exactly.
- Automatic translation can word things oddly — check before posting.
- Tedim (Zomi) has no ASR/TTS of its own; choose Burmese for the closest
  supported pipeline.
