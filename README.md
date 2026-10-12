# Recap Studio v3 AUTO — one-click movie recap dubber

**Upload a video → Generate → Download.** The server does everything:

1. **Auto transcribe** — faster-whisper writes down the speech, with word
   timestamps
2. **Atomic units** — the words are re-grouped into short narration units
   (~4.2 s target, ~5.6 s max) that break at sentence/clause ends and pauses
   and tile the timeline (boundaries split inter-unit silence at the
   midpoint). If word timestamps are missing, long segments are split
   proportionally instead.
3. **Auto translate** — each unit is translated to the dub language, in one
   of two modes:
   - **Fast** (default) — Google Translate via deep-translator, MyMemory as
     fallback. Every line passes quality checks (not empty, not a duplicate
     of the previous line, and — for Burmese — no foreign script mixed into
     Burmese words; digits, punctuation and English proper nouns are fine).
     A line that fails a check is retried once via the other provider.
   - **Quality (Gemini)** — optional, needs a Gemini API key (see below).
     Gemini first reads the whole transcript and builds a story memory
     (characters, relationships, places, key terms with consistent
     translations), then translates in context batches of ~20 units with an
     audit/repair pass, in a natural spoken narrator style. If Gemini fails
     for a batch (quota, bad reply, no key), that batch falls back to Fast
     and the job status shows the split, e.g. `gemini 12/20, mt 8/20`.
4. **Auto voice dub** — edge-tts speaks every translated unit with a neural
   voice (Burmese: Nilar ♀ / Thiha ♂, English: Aria ♀ / Guy ♂, plus Thai,
   Korean, Japanese, Chinese, Hindi, Indonesian, Vietnamese, Malay), with
   Google TTS as fallback. Each clip's leading/trailing silence is trimmed
   (**short pauses**: ~0.05 s head / ~0.20 s tail) so narration flows tightly.
5. **Smart Sync edit & render** — the dub leads and the picture follows:
   each unit's video window is re-timed to exactly its dub length.
   Speed-up is capped at 1.25× (the window tail is trimmed beyond that);
   slow-down is capped at ~1.35×, and beyond that short freeze holds
   (`tpad` clone frames) are distributed across the unit instead of slower
   motion. Subtitles are re-timed to the new picture and burned in, and
   ffmpeg renders the final MP4 (original size, 9:16 for Reels/TikTok, or
   16:9 for YouTube). The picture-vs-narration duration difference is
   logged at the end of every job. The old behaviour (voice clips
   speed-fitted into their original windows, picture untouched) is still
   available as `sync_mode=fit`.

> **Original audio + Smart Sync:** the picture is re-timed to follow the
> new voice, so the original audio no longer matches the picture timing.
> With "keep original sound" on, the original track is mixed underneath
> only as low-volume ambience — expect it to drift against the picture.

This is the *server* version of Recap Studio. The older v2 ran only in the
phone browser, which cannot record an AI voice into a file — v3 fixes that by
doing the work on a server (your PC, a VPS/Docker, or free Google Colab).

## Resumable jobs

Every stage checkpoints into the job folder (`transcript.json`,
`units.json`, `translations.json`, `dub_manifest.json`, `render_meta.json`,
plus `options.json` with the submitted settings). If a job stops — server
restart, crash, closed page — it does not have to start over:
`POST /api/resume/{job_id}` (or the **Resume** button the page shows after
an error) re-enters the pipeline at the first stage without a checkpoint.
Job status is also persisted to `status.json`, so after a restart the page
can still show old jobs and resume them.

## Gemini quality mode & your key

- Get a free key from Google AI Studio and paste it into the form when
  "Quality — Gemini" is selected, or set `GEMINI_API_KEY` on the server.
- The key is used only for that job's requests. It is **never written to
  disk, never stored, and never logged** — it is deliberately excluded
  from the saved `options.json`. A quality-mode job that is *resumed*
  later has no key any more, so it continues in Fast mode and says so in
  its status/log.
- Model: `GEMINI_TEXT_MODEL` env var, default `gemini-2.5-flash`. Calls
  are plain HTTPS requests — no extra SDK.

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
| `GEMINI_API_KEY` | _(empty)_ | optional server-side Gemini key for Quality translate mode |
| `GEMINI_TEXT_MODEL` | `gemini-2.5-flash` | Gemini model used by Quality mode |
| `SYNC_MODE` | `smart` | default render sync: `smart` (Smart Sync) or `fit` (legacy) |

## API

- `GET /` — the web page
- `GET /api/health` — `{ok, app, whisper_model, max_upload_mb, languages, …}`
- `POST /api/process` — multipart: `file` (video) + `source_lang` (`auto`, `en`, `my`, …),
  `target_lang` (`my`, `en`, …), `voice` (`female`/`male`), `aspect`
  (`original`/`9:16`/`16:9`), `keep_original_audio` (`true`/`false`),
  `translate_mode` (`fast`/`quality`), `sync_mode` (`smart`/`fit`),
  `gemini_api_key` (optional, never stored) → `{job_id}`
- `GET /api/status/{job_id}` — `{stage, percent, message, error?,
  tts, translate, sync, …}` (read from disk after a restart)
- `POST /api/resume/{job_id}` — resume an unfinished job from its
  checkpoints → `{resumed: true/false, message}` (a finished or running
  job is refused gracefully)
- `GET /api/download/{job_id}` — the rendered MP4
- `GET /api/subtitles/{job_id}` — the translated `.srt`

Jobs are stored under `jobs/<job_id>/` (`input.*`, `options.json`,
`status.json`, `job.log`, checkpoint JSONs, `subtitles.srt`, `output.mp4`,
plus a `work/` scratch folder). Jobs run one at a time; a queue forms
automatically on a busy server.

## Honest limits

- CPU transcription is slow: roughly real-time or slower with `small`.
  Short clips (under ~3 minutes) work best, especially on free Colab.
- Translation, edge-tts and Gemini need internet (they are online services).
- The AI voice is synthetic — it is **not** lip-sync; mouths will not match
  the new words exactly.
- Automatic translation can word things oddly — check before posting.
  Quality (Gemini) mode is noticeably better for Burmese story narration,
  but still check names and numbers.
- Tedim (Zomi) has no ASR/TTS of its own; choose Burmese for the closest
  supported pipeline.
