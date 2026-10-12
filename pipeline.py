"""Recap Studio v3 AUTO — the automatic dubbing pipeline.

Steps (all local, driven by ffmpeg):
  1. Extract the video's audio (16 kHz mono wav)
  2. Transcribe it with faster-whisper -> timed segments
  3. Translate each segment to the target language (deep-translator)
  4. Speak each translated segment with edge-tts, falling back to gTTS
     (per-segment, failure-proof; provider used is logged to tts.log)
  5. Fit every voice clip into its original time window and assemble one
     full-length dub track that lines up with the picture
  6. Burn the translated subtitles into the video and render the final MP4

The pipeline never raises for a single bad segment: a failed translation
keeps the original text, a failed TTS segment becomes silence. Only truly
fatal problems (no audio, unreadable file, ffmpeg missing) fail the job.
"""
import asyncio
import json
import os
import re
import subprocess

# Sanitize proxy env vars (some HTTP libs crash on bracketed IPv6 entries)
for _var in ("no_proxy", "NO_PROXY"):
    if os.environ.get(_var):
        _clean = [e.strip() for e in os.environ[_var].split(",") if "[" not in e and "]" not in e]
        os.environ[_var] = ",".join(_clean)

# If the runtime provides its own CA bundle (e.g. behind a TLS-intercepting
# egress proxy, as in some sandboxes), point requests-based libraries at it.
# Harmless on Colab / normal networks, where SSL_CERT_FILE is usually unset.
_ca_bundle = os.environ.get("SSL_CERT_FILE")
if _ca_bundle and os.path.exists(_ca_bundle):
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _ca_bundle)

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")

# ---------------------------------------------------------------- languages
# ONE central map for every language we support. For each language:
#   google   – code for deep-translator's Google Translate
#   mymemory – locale code for the MyMemory fallback translator
#   whisper  – code faster-whisper accepts for transcription (source side)
#   gtts     – gTTS language code, or None where gTTS has no such voice
#              (that language is then edge-tts only, silence after that)
#   edge     – Microsoft edge-tts neural voices (female + male)
# Codes below were verified against the installed libraries: Google and
# gTTS both call Filipino "tl" (Tagalog); gTTS has NO Lao, so lo is
# edge-only. Tedim/Zomi (ctd) is deliberately NOT here — no translator or
# TTS supports it, so offering it as an auto target would be fake.
def _L(google, mymemory, whisper, gtts, female, male):
    return {"google": google, "mymemory": mymemory, "whisper": whisper,
            "gtts": gtts, "edge": {"female": female, "male": male}}


LANGS = {
    "my":    _L("my", "my-MM", "my", "my", "my-MM-NilarNeural", "my-MM-ThihaNeural"),
    "en":    _L("en", "en-GB", "en", "en", "en-US-AriaNeural", "en-US-GuyNeural"),
    "th":    _L("th", "th-TH", "th", "th", "th-TH-PremwadeeNeural", "th-TH-NiwatNeural"),
    "zh-CN": _L("zh-CN", "zh-CN", "zh", "zh-CN", "zh-CN-XiaoxiaoNeural", "zh-CN-YunyangNeural"),
    "ja":    _L("ja", "ja-JP", "ja", "ja", "ja-JP-NanamiNeural", "ja-JP-KeitaNeural"),
    "ko":    _L("ko", "ko-KR", "ko", "ko", "ko-KR-SunHiNeural", "ko-KR-InJoonNeural"),
    "hi":    _L("hi", "hi-IN", "hi", "hi", "hi-IN-SwaraNeural", "hi-IN-MadhurNeural"),
    "id":    _L("id", "id-ID", "id", "id", "id-ID-GadisNeural", "id-ID-ArdiNeural"),
    "vi":    _L("vi", "vi-VN", "vi", "vi", "vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"),
    "ms":    _L("ms", "ms-MY", "ms", "ms", "ms-MY-YasminNeural", "ms-MY-OsmanNeural"),
    "es":    _L("es", "es-ES", "es", "es", "es-ES-ElviraNeural", "es-ES-AlvaroNeural"),
    "fr":    _L("fr", "fr-FR", "fr", "fr", "fr-FR-DeniseNeural", "fr-FR-HenriNeural"),
    "de":    _L("de", "de-DE", "de", "de", "de-DE-KatjaNeural", "de-DE-ConradNeural"),
    "ar":    _L("ar", "ar-SA", "ar", "ar", "ar-SA-ZariyahNeural", "ar-SA-HamedNeural"),
    "bn":    _L("bn", "bn-IN", "bn", "bn", "bn-IN-TanishaaNeural", "bn-IN-BashkarNeural"),
    "ta":    _L("ta", "ta-IN", "ta", "ta", "ta-IN-PallaviNeural", "ta-IN-ValluvarNeural"),
    "te":    _L("te", "te-IN", "te", "te", "te-IN-ShrutiNeural", "te-IN-MohanNeural"),
    "ur":    _L("ur", "ur-PK", "ur", "ur", "ur-PK-UzmaNeural", "ur-PK-AsadNeural"),
    "pt":    _L("pt", "pt-BR", "pt", "pt", "pt-BR-FranciscaNeural", "pt-BR-AntonioNeural"),
    "ru":    _L("ru", "ru-RU", "ru", "ru", "ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"),
    "it":    _L("it", "it-IT", "it", "it", "it-IT-ElsaNeural", "it-IT-DiegoNeural"),
    "fil":   _L("tl", "fil-PH", "tl", "tl", "fil-PH-BlessicaNeural", "fil-PH-AngeloNeural"),
    "km":    _L("km", "km-KH", "km", "km", "km-KH-SreymomNeural", "km-KH-PisethNeural"),
    "lo":    _L("lo", "lo-LA", "lo", None, "lo-LA-KeomanyNeural", "lo-LA-ChanthavongNeural"),
}
DEFAULT_VOICE = {"female": "en-US-AriaNeural", "male": "en-US-GuyNeural"}

# Normalise whatever the UI / user sends ("my", "Burmese", "burmese", "en"...)
# to one of the LANGS keys above.
LANG_ALIASES = {
    "burmese": "my", "myanmar": "my", "my": "my",
    "english": "en", "en": "en",
    "thai": "th", "th": "th",
    "hindi": "hi", "hi": "hi",
    "korean": "ko", "ko": "ko",
    "japanese": "ja", "ja": "ja",
    "chinese": "zh-CN", "mandarin": "zh-CN", "zh": "zh-CN", "zh-cn": "zh-CN",
    "indonesian": "id", "id": "id",
    "vietnamese": "vi", "vi": "vi",
    "malay": "ms", "ms": "ms",
    "spanish": "es", "es": "es",
    "french": "fr", "fr": "fr",
    "german": "de", "de": "de",
    "arabic": "ar", "ar": "ar",
    "bengali": "bn", "bn": "bn",
    "tamil": "ta", "ta": "ta",
    "telugu": "te", "te": "te",
    "urdu": "ur", "ur": "ur",
    "portuguese": "pt", "pt": "pt",
    "russian": "ru", "ru": "ru",
    "italian": "it", "it": "it",
    "filipino": "fil", "tagalog": "fil", "fil": "fil", "tl": "fil",
    "khmer": "km", "km": "km",
    "lao": "lo", "lo": "lo",
    "tedim": "my", "zomi": "my",  # no Tedim ASR/TTS exists; Burmese is the honest fallback
}


def norm_lang(code):
    if not code:
        return "auto"
    return LANG_ALIASES.get(str(code).strip().lower(), str(code).strip().lower())


def pick_voice(lang, gender):
    entry = LANGS.get(lang)
    if entry:
        return entry["edge"].get(gender, DEFAULT_VOICE["female"])
    return DEFAULT_VOICE.get(gender, DEFAULT_VOICE["female"])


def whisper_code(lang):
    """faster-whisper code for a normalised language, or None for auto."""
    if lang in ("auto", "", None):
        return None
    entry = LANGS.get(lang)
    if entry:
        return entry["whisper"]
    code = str(lang).split("-")[0]
    return code or None


# ---------------------------------------------------------------- ffmpeg helpers
def run(cmd, **kw):
    """Run a command, raising a readable error on failure."""
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kw)
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()[-3:]
        raise RuntimeError("Command failed: " + " ".join(cmd[:2]) + " … " + " | ".join(tail))
    return r


def probe_duration(path):
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path])
    return float(json.loads(r.stdout)["format"]["duration"])


def probe_has_audio(path):
    r = run(["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", path])
    return bool(r.stdout.strip())


def srt_time(sec):
    sec = max(0.0, float(sec))
    h = int(sec // 3600); m = int((sec % 3600) // 60)
    s = int(sec % 60); ms = int(round((sec - int(sec)) * 1000))
    if ms == 1000:
        s += 1; ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{srt_time(seg['start'])} --> {srt_time(seg['end'])}\n"
                    f"{seg['dub_text']}\n\n")


# ---------------------------------------------------------------- step 2: ASR
_whisper_model = None

def transcribe(audio_wav, source_lang):
    """Return (segments, detected_lang). Segments: {start, end, text}."""
    global _whisper_model
    from faster_whisper import WhisperModel
    if _whisper_model is None:
        _whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    lang = whisper_code(source_lang)  # zh-CN -> zh, fil -> tl, auto -> None
    segs, info = _whisper_model.transcribe(audio_wav, language=lang, vad_filter=True)
    out = []
    for s in segs:
        text = (s.text or "").strip()
        if text:
            out.append({"start": float(s.start), "end": float(s.end), "text": text})
    return out, (info.language or source_lang or "en")


# ---------------------------------------------------------------- step 3: translate
def translate_segments(segments, source_lang, target_lang, log):
    """Fill seg['dub_text'] with the translation (or original if skipped/failed)."""
    if source_lang == target_lang:
        for seg in segments:
            seg["dub_text"] = seg["text"]
        return
    from deep_translator import GoogleTranslator, MyMemoryTranslator
    # Codes come from the central LANGS map (e.g. Filipino is "tl" for
    # Google, "fil-PH" for MyMemory). Unknown codes pass through as-is.
    def gcode(c):
        return LANGS.get(c, {}).get("google", c)

    def mmcode(c):
        return LANGS.get(c, {}).get("mymemory", c)

    src = "auto" if source_lang in ("auto", "", None) else source_lang
    mm_src = "en-GB" if src == "auto" else mmcode(src)
    mm_tgt = mmcode(target_lang)
    google = GoogleTranslator(source=gcode(src) if src != "auto" else "auto",
                              target=gcode(target_lang))
    for seg in segments:
        text = seg["text"]
        try:
            seg["dub_text"] = google.translate(text) or text
            continue
        except Exception as e:
            log(f"Google translate failed for one line ({e}); trying MyMemory")
        try:
            seg["dub_text"] = MyMemoryTranslator(source=mm_src, target=mm_tgt).translate(text) or text
        except Exception as e:
            log(f"Translation failed for one line ({e}); keeping original text")
            seg["dub_text"] = text


# ---------------------------------------------------------------- step 4: TTS
# Fallback chain per segment: edge-tts first (best voices), then gTTS
# (Google Translate TTS — simpler, one voice per language, no gender
# choice, but reachable from networks where Microsoft's edge-tts WSS
# endpoint is blocked with a 403). A segment only becomes silence if
# every provider fails.
MIN_TTS_BYTES = 1024  # a real spoken line is never a few hundred bytes

# Languages gTTS can speak, derived from the central LANGS map
# (languages whose map entry has gtts=None — currently only Lao — are
# edge-tts only and fall back to silence if edge is unreachable).
GTTS_LANGS = {code: e["gtts"] for code, e in LANGS.items() if e["gtts"]}


def _mp3_ok(path):
    return os.path.exists(path) and os.path.getsize(path) >= MIN_TTS_BYTES


async def _tts_async(text, voice, out_path):
    import ssl
    import edge_tts
    # edge-tts pins its SSL context to the certifi bundle only, which fails
    # behind TLS-intercepting proxies (self-signed chain). Swap in the system
    # default context, which also honours SSL_CERT_FILE. Safe everywhere.
    try:
        edge_tts.communicate._SSL_CTX = ssl.create_default_context()
    except Exception:
        pass
    proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or None
    await edge_tts.Communicate(text, voice, proxy=proxy).save(out_path)


def tts_segment(text, voice, out_path):
    asyncio.run(_tts_async(text, voice, out_path))


def _tts_gtts(text, target_lang, out_path):
    from gtts import gTTS
    code = GTTS_LANGS.get(target_lang)
    if not code:
        raise ValueError(f"gTTS has no voice for language '{target_lang}'")
    gTTS(text=text, lang=code).save(out_path)


def tts_segment_fallback(text, voice, target_lang, out_path, log):
    """Voice one segment. Returns 'edge-tts', 'gtts' or None (caller uses
    silence). Never raises."""
    try:
        if os.path.exists(out_path):
            os.remove(out_path)
        tts_segment(text, voice, out_path)
        if _mp3_ok(out_path):
            return "edge-tts"
        log(f"edge-tts produced an empty/tiny file ({os.path.getsize(out_path) if os.path.exists(out_path) else 0} bytes); trying gTTS")
    except Exception as e:
        log(f"edge-tts failed ({e}); trying gTTS")
    try:
        if os.path.exists(out_path):
            os.remove(out_path)
        _tts_gtts(text, target_lang, out_path)
        if _mp3_ok(out_path):
            return "gtts"
        log(f"gTTS produced an empty/tiny file; line will be silent")
    except Exception as e:
        log(f"gTTS failed ({e}); line will be silent")
    return None


def atempo_chain(speed):
    """ffmpeg atempo only takes 0.5–2.0 per instance; chain for bigger values."""
    parts = []
    s = speed
    while s > 2.0:
        parts.append("atempo=2.0"); s /= 2.0
    while s < 0.5:
        parts.append("atempo=0.5"); s /= 0.5
    parts.append(f"atempo={s:.4f}")
    return ",".join(parts)


def silence_wav(path, seconds):
    run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
         "-t", f"{seconds:.3f}", "-c:a", "pcm_s16le", path])


def fit_clip(src_mp3, window, out_wav):
    """Speed-fit / pad a TTS clip to exactly `window` seconds of 44.1k mono wav."""
    dur = probe_duration(src_mp3)
    filters = []
    if dur > window and window > 0:
        filters.append(atempo_chain(min(dur / window, 4.0)))
    filt = (",".join(filters) + ",") if filters else ""
    filt += f"aresample=44100,apad,atrim=0:{window:.3f},asetpts=N/SR/TB"
    run(["ffmpeg", "-y", "-i", src_mp3, "-af", filt, "-c:a", "pcm_s16le", out_wav])


# ---------------------------------------------------------------- main entry
def run_job(job_dir, params, status_cb):
    """Run the whole pipeline for one job. status_cb(stage, percent, message)."""
    def log(msg):
        print(f"[job] {msg}", flush=True)

    video_path = params["video_path"]
    source_lang = norm_lang(params.get("source_lang") or "auto")
    target_lang = norm_lang(params.get("target_lang") or "my")
    gender = "male" if params.get("voice") == "male" else "female"
    aspect = params.get("aspect") or "original"
    keep_original = bool(params.get("keep_original_audio"))

    work = os.path.join(job_dir, "work")
    os.makedirs(work, exist_ok=True)

    # ---- 1. extract audio
    status_cb("extract", 5, "Getting the sound out of your video…")
    if not probe_has_audio(video_path):
        raise RuntimeError("This video has no sound track, so there is nothing to dub.")
    total_dur = probe_duration(video_path)
    audio_wav = os.path.join(work, "audio.wav")
    run(["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", audio_wav])

    # ---- 2. transcribe
    status_cb("transcribe", 15, f"Listening and writing down the speech (Whisper {WHISPER_MODEL})…")
    segments, detected = transcribe(audio_wav, source_lang)
    if not segments:
        raise RuntimeError("No speech was found in this video.")
    src_eff = norm_lang(detected)
    log(f"{len(segments)} segments, detected language: {detected}")
    status_cb("transcribe", 40, f"Found {len(segments)} lines of speech ({detected}).")

    # ---- 3. translate
    status_cb("translate", 45, f"Translating to {target_lang}…")
    translate_segments(segments, src_eff, target_lang, log)
    status_cb("translate", 58, "Translation done.")

    # ---- subtitles file (translated lines, original timing)
    srt_path = os.path.join(job_dir, "subtitles.srt")
    write_srt(segments, srt_path)

    # ---- 4+5. TTS each segment, fit to its window, assemble the dub track
    voice = pick_voice(target_lang, gender)
    log(f"TTS voice: {voice} (fallback: gTTS where supported)")
    tts_log_path = os.path.join(job_dir, "tts.log")
    tts_counts = {"edge-tts": 0, "gtts": 0, "silence": 0}

    def tlog(msg):
        log(msg)
        try:
            with open(tts_log_path, "a", encoding="utf-8") as fh:
                fh.write(msg + "\n")
        except Exception:
            pass

    parts, list_file = [], os.path.join(work, "concat.txt")
    cursor = 0.0
    n = len(segments)
    for i, seg in enumerate(segments):
        status_cb("tts", 60 + int(20 * (i + 1) / n),
                  f"Creating the new voice… line {i + 1} of {n}")
        start, end = seg["start"], max(seg["end"], seg["start"] + 0.4)
        window = end - start
        if start - cursor > 0.05:  # gap before this line -> silence
            gap = os.path.join(work, f"gap_{i}.wav")
            silence_wav(gap, start - cursor)
            parts.append(gap)
        seg_wav = os.path.join(work, f"seg_{i}.wav")
        mp3 = os.path.join(work, f"seg_{i}.mp3")
        try:
            if not seg["dub_text"].strip():
                raise ValueError("empty line")
            provider = tts_segment_fallback(seg["dub_text"], voice, target_lang,
                                            mp3, tlog)
            if provider is None:
                raise RuntimeError("all TTS providers failed")
            tts_counts[provider] += 1
            seg["tts_provider"] = provider
            tlog(f"line {i + 1}/{n}: voiced by {provider}")
            fit_clip(mp3, window, seg_wav)
        except Exception as e:  # one bad line must not kill the job
            tts_counts["silence"] += 1
            seg["tts_provider"] = "silence"
            tlog(f"line {i + 1}/{n}: SILENT ({e})")
            log(f"TTS failed for line {i + 1} ({e}); using silence")
            silence_wav(seg_wav, window)
        parts.append(seg_wav)
        cursor = end
    if total_dur - cursor > 0.05:
        tail = os.path.join(work, "tail.wav")
        silence_wav(tail, total_dur - cursor)
        parts.append(tail)
    with open(list_file, "w", encoding="utf-8") as f:
        for p in parts:
            f.write(f"file '{p}'\n")
    dub_wav = os.path.join(work, "dub.wav")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
         "-c:a", "pcm_s16le", dub_wav])

    # ---- 6. render final MP4
    status_cb("render", 85, "Putting picture, new voice and subtitles together…")
    out_path = os.path.join(job_dir, "output.mp4")
    vf = []
    if aspect == "9:16":
        vf.append("scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280")
    elif aspect == "16:9":
        vf.append("scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720")
    style = ("FontName=Noto Sans Myanmar,FontSize=18,PrimaryColour=&H00FFFFFF,"
             "OutlineColour=&H90000000,BorderStyle=1,Outline=2,Shadow=0,"
             "Alignment=2,MarginV=36")
    vf.append(f"subtitles={srt_path}:force_style='{style}'")
    cmd = ["ffmpeg", "-y", "-i", video_path, "-i", dub_wav]
    if keep_original:
        cmd += ["-filter_complex",
                f"[0:v]{','.join(vf)}[v];[0:a]volume=0.12[orig];"
                f"[1:a][orig]amix=inputs=2:duration=first:normalize=0[a]",
                "-map", "[v]", "-map", "[a]"]
    else:
        cmd += ["-vf", ",".join(vf), "-map", "0:v", "-map", "1:a"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
            "-c:a", "aac", "-b:a", "160k", "-shortest",
            "-movflags", "+faststart", out_path]
    run(cmd)

    tts_summary = (f"edge-tts {tts_counts['edge-tts']}/{n}, "
                   f"gTTS {tts_counts['gtts']}/{n}, "
                   f"silent {tts_counts['silence']}/{n}")
    tlog(f"TTS summary: {tts_summary}")
    log(f"TTS summary: {tts_summary}")
    status_cb("done", 100, "Done! Your dubbed video is ready to download.")
    return {"output": out_path, "srt": srt_path, "segments": len(segments),
            "detected_lang": detected, "voice": voice,
            "tts": tts_summary, "tts_counts": tts_counts}
