"""Recap Studio v3 AUTO — the automatic dubbing pipeline.

Steps (all local, driven by ffmpeg):
  1. Extract the video's audio (16 kHz mono wav)
  2. Transcribe it with faster-whisper -> timed segments with word stamps
  3. Re-group the words into short "atomic" units (~4.2 s, max ~5.6 s) that
     tile the whole timeline — these units drive everything after this
  4. Translate each unit (fast MT with quality checks, or Gemini quality
     mode with story memory + audit), never crashing on a single bad line
  5. Speak each unit with edge-tts, falling back to gTTS; trim the built-in
     head/tail silence of every clip ("short pauses") so lines join tightly
  6. Render: in Smart Sync mode (default) the dub leads and the picture is
     re-timed to it; in "fit" mode the voice is speed-fitted into the
     original windows as before. Translated subtitles are burned in,
     retimed to whichever timeline the render produced.

Every stage writes a checkpoint into the job folder (transcript.json,
units.json, translations.json, dub_manifest.json), so a job can be resumed
after a crash/restart without redoing finished work (see /api/resume).

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
GEMINI_MODEL = os.environ.get("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
DEFAULT_SYNC_MODE = os.environ.get("SYNC_MODE", "smart")

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

# Human-readable names, used in Gemini prompts.
LANG_NAMES = {
    "my": "Burmese (Myanmar)", "en": "English", "th": "Thai",
    "zh-CN": "Chinese (Simplified)", "ja": "Japanese", "ko": "Korean",
    "hi": "Hindi", "id": "Indonesian", "vi": "Vietnamese", "ms": "Malay",
    "es": "Spanish", "fr": "French", "de": "German", "ar": "Arabic",
    "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "ur": "Urdu",
    "pt": "Portuguese", "ru": "Russian", "it": "Italian",
    "fil": "Filipino", "km": "Khmer", "lo": "Lao", "auto": "the detected language",
}

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


def write_srt(units, path):
    """units need start/end (whichever timeline is being rendered) + dub_text."""
    with open(path, "w", encoding="utf-8") as f:
        for i, u in enumerate(units, 1):
            f.write(f"{i}\n{srt_time(u['start'])} --> {srt_time(u['end'])}\n"
                    f"{u['dub_text']}\n\n")


def _save_json(path, obj):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
    except Exception:
        pass


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------- step 2: ASR
_whisper_model = None

def transcribe(audio_wav, source_lang):
    """Return (segments, detected_lang).

    Segments: {start, end, text, words:[{start, end, word}]} — word stamps
    feed the atomic re-segmentation below; words may be [] if the model
    could not provide them.
    """
    global _whisper_model
    from faster_whisper import WhisperModel
    if _whisper_model is None:
        _whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    lang = whisper_code(source_lang)  # zh-CN -> zh, fil -> tl, auto -> None
    segs, info = _whisper_model.transcribe(audio_wav, language=lang,
                                           vad_filter=True, word_timestamps=True)
    out = []
    for s in segs:
        text = (s.text or "").strip()
        if not text:
            continue
        words = []
        for w in (getattr(s, "words", None) or []):
            wt = (w.word or "").strip()
            if wt:
                words.append({"start": float(w.start), "end": float(w.end), "word": wt})
        out.append({"start": float(s.start), "end": float(s.end),
                    "text": text, "words": words})
    return out, (info.language or source_lang or "en")


# ------------------------------------------------- step 3: atomic re-segmentation
UNIT_TARGET = 4.2   # aim for units of about this many seconds
UNIT_MAX = 5.6      # never let a unit run longer than this (if we can help it)
_SENT_END = set(".!?…")
_CLAUSE_END = set(",;:—)]\"'")


def _word_ends(word, chars):
    w = (word or "").strip()
    return bool(w) and w[-1] in chars


def _units_from_words(words):
    """Group a flat word list into short units at sentence/clause/pause breaks."""
    groups = []
    cur = [words[0]]
    for i in range(1, len(words)):
        w, prev = words[i], words[i - 1]
        if not cur:
            cur = [w]
            continue
        gap = w["start"] - prev["end"]
        cur_dur = prev["end"] - cur[0]["start"]
        if gap > 1.0 and cur_dur >= 1.5:
            # a real pause with a reasonably full unit before it: break here
            groups.append(cur)
            cur = [w]
            continue
        cur.append(w)
        d = w["end"] - cur[0]["start"]
        next_gap = (words[i + 1]["start"] - w["end"]) if i + 1 < len(words) else 99.0
        if (d >= UNIT_MAX
                or (d >= UNIT_TARGET and (_word_ends(w["word"], _SENT_END)
                                          or _word_ends(w["word"], _CLAUSE_END)))
                or (d >= UNIT_TARGET and next_gap >= 0.4)):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    # merge crumbs (<1 s) into the previous unit rather than leaving slivers
    merged = []
    for g in groups:
        dur = g[-1]["end"] - g[0]["start"]
        if merged and dur < 1.0:
            merged[-1].extend(g)
        else:
            merged.append(list(g))
    if len(merged) > 1:
        first_dur = merged[0][-1]["end"] - merged[0][0]["start"]
        if first_dur < 0.8:
            merged[1] = merged[0] + merged[1]
            merged.pop(0)
    units = []
    for g in merged:
        text = " ".join(w["word"] for w in g)
        text = re.sub(r"\s+([,.!?;:…])", r"\1", text)
        units.append({"start": g[0]["start"], "end": g[-1]["end"], "text": text})
    return units


def _units_proportional(segments):
    """No word stamps: split long segments' text into ~UNIT_TARGET-sized
    chunks and give each chunk an equal slice of the segment's window."""
    units = []
    for s in segments:
        dur = s["end"] - s["start"]
        text = (s["text"] or "").strip()
        if not text:
            continue
        if dur <= UNIT_MAX:
            units.append({"start": s["start"], "end": s["end"], "text": text})
            continue
        n = max(2, int(round(dur / UNIT_TARGET)))
        tokens = text.split()
        if len(tokens) < n:  # spaceless script: split by characters
            tokens = list(text)
        if len(tokens) < n:
            units.append({"start": s["start"], "end": s["end"], "text": text})
            continue
        per = len(tokens) / n
        step = dur / n
        for k in range(n):
            chunk = tokens[int(k * per):int((k + 1) * per)]
            if chunk:
                units.append({"start": s["start"] + k * step,
                              "end": s["start"] + (k + 1) * step,
                              "text": " ".join(chunk)})
    return units


def make_units(segments, total_dur):
    """Return (units, source) where source is 'words' | 'proportional' | 'raw'.

    Every unit gains tile_start/tile_end: the silence between neighbours is
    split at its midpoint, the first unit starts at 0 and the last ends at
    total_dur, so the units tile the whole timeline with no gaps.
    """
    words = []
    for s in segments:
        for w in (s.get("words") or []):
            t = (w.get("word") or "").strip()
            if t:
                words.append({"start": float(w["start"]), "end": float(w["end"]),
                              "word": t})
    if words:
        units, source = _units_from_words(words), "words"
    else:
        units = _units_proportional(segments)
        source = "proportional" if units else "raw"
        if not units:
            units = [{"start": s["start"], "end": s["end"], "text": s["text"]}
                     for s in segments]
    units.sort(key=lambda u: u["start"])
    n = len(units)
    for i, u in enumerate(units):
        u["tile_start"] = 0.0 if i == 0 else (units[i - 1]["end"] + u["start"]) / 2.0
        u["tile_end"] = (float(total_dur) if i == n - 1
                         else (u["end"] + units[i + 1]["start"]) / 2.0)
        if u["tile_end"] - u["tile_start"] < 0.1:  # degenerate window guard
            u["tile_end"] = u["tile_start"] + 0.1
    return units, source


# ---------------------------------------------------------------- step 4: translate
def _foreign_clean(text):
    """Burmese quality guard: in a token that contains Myanmar script, strip
    letters from other scripts mixed into the same word (a known MT glitch).
    Pure-foreign tokens (proper nouns), digits and punctuation are kept.
    Returns (cleaned_text, changed)."""
    def is_my(ch):
        return "\u1000" <= ch <= "\u109f"

    out, changed = [], False
    for tok in (text or "").split(" "):
        if any(is_my(c) for c in tok):
            kept = []
            for c in tok:
                if c.isalpha() and not is_my(c):
                    changed = True
                    continue
                kept.append(c)
            out.append("".join(kept))
        else:
            out.append(tok)
    return " ".join(out), changed


def _check_translation(text, prev_text, target_lang):
    """Return (issues, cleaned_text). Issues: empty / duplicate /
    foreign-script / empty-after-clean."""
    issues = []
    if not text or not text.strip():
        issues.append("empty")
    if prev_text and text and text.strip() and text.strip() == prev_text.strip():
        issues.append("duplicate")
    cleaned = text or ""
    if target_lang == "my" and text:
        cleaned, changed = _foreign_clean(text)
        if changed:
            issues.append("foreign-script")
        if not cleaned.strip() and text.strip():
            issues.append("empty-after-clean")
    return issues, cleaned


def _make_mt(source_lang, target_lang):
    """Return {'google': fn, 'mymemory': fn} — fn(text)->str, raises on failure.
    Codes come from the central LANGS map (Filipino is 'tl' for Google,
    'fil-PH' for MyMemory)."""
    from deep_translator import GoogleTranslator, MyMemoryTranslator

    def gcode(c):
        return LANGS.get(c, {}).get("google", c)

    def mmcode(c):
        return LANGS.get(c, {}).get("mymemory", c)

    src = "auto" if source_lang in ("auto", "", None) else source_lang
    google = GoogleTranslator(source=gcode(src) if src != "auto" else "auto",
                              target=gcode(target_lang))
    mm_src = "en-GB" if src == "auto" else mmcode(src)

    def via_google(text):
        return google.translate(text)

    def via_mymemory(text):
        return MyMemoryTranslator(source=mm_src,
                                  target=mmcode(target_lang)).translate(text)

    return {"google": via_google, "mymemory": via_mymemory}


def _translate_fast_one(text, fns, prev_dub, target_lang, log):
    """Translate one unit with quality checks. Providers are tried in order;
    a check failure retries with the alternate provider. Returns
    (text_or_None, provider_or_None)."""
    best = None
    for prov in ("google", "mymemory"):
        try:
            t = fns[prov](text)
        except Exception as e:
            log(f"translate via {prov} failed ({e})")
            continue
        issues, cleaned = _check_translation(t, prev_dub, target_lang)
        if "foreign-script" in issues and "empty-after-clean" not in issues:
            log(f"translate via {prov}: stripped foreign letters mixed into Burmese")
            t, issues = cleaned, [i for i in issues if i != "foreign-script"]
        if not issues:
            return t, prov
        log(f"translate check failed via {prov} ({', '.join(issues)}); "
            "retrying with alternate provider" if prov == "google" else
            f"translate check failed via {prov} ({', '.join(issues)})")
        if cleaned and cleaned.strip():
            best = cleaned
        elif t and t.strip():
            best = t
    return best, None


def _translate_fast(units, source_lang, target_lang, log, counts):
    fns = _make_mt(source_lang, target_lang)
    prev_dub = None
    for u in units:
        text, prov = _translate_fast_one(u["text"], fns, prev_dub, target_lang, log)
        if prov:
            counts[prov] += 1
            u["dub_text"] = text
        elif text:
            # passed a provider but failed checks on both: usable, unattributed
            u["dub_text"] = text
            counts["mt_best_effort"] = counts.get("mt_best_effort", 0) + 1
        else:
            counts["kept_original"] += 1
            u["dub_text"] = u["text"]
        prev_dub = u["dub_text"]


# ---- Gemini quality mode -------------------------------------------------
def _scrub(text, key):
    """Never let the API key reach a log line (some HTTP errors echo URLs)."""
    return str(text).replace(key, "***") if key else str(text)


def _gemini_json(api_key, prompt, log, timeout=90):
    """One Gemini call expecting a JSON answer. Retries once on bad
    JSON/transport, then raises. The key is only ever sent as the API
    query param — it is never logged (errors are scrubbed)."""
    import requests
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent")
    payload = {"contents": [{"parts": [{"text": prompt}]}],
               "generationConfig": {"temperature": 0.3,
                                    "responseMimeType": "application/json"}}
    last_err = None
    for attempt in range(2):
        try:
            r = requests.post(url, params={"key": api_key}, json=payload,
                              timeout=timeout)
            if r.status_code != 200:
                raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:160]}")
            txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(txt)
        except Exception as e:
            last_err = e
            log("Gemini call failed (attempt %d): %s"
                % (attempt + 1, _scrub(e, api_key)[:200]))
    raise RuntimeError(f"Gemini failed: {_scrub(last_err, api_key)[:200]}")


def _validate_id_map(obj, ids):
    """obj must map every unit id (as string) to a non-empty string."""
    if not isinstance(obj, dict):
        return False
    for i in ids:
        v = obj.get(str(i))
        if not isinstance(v, str) or not v.strip():
            return False
    return True


def _gemini_translate_batch(api_key, batch, memory, prev_text, next_text,
                            source_lang, target_lang, log):
    """Translate one batch (list of (idx, unit)) with context; audit-repair;
    return {idx: text}. Raises on failure (caller falls back to fast MT)."""
    ids = [i for i, _ in batch]
    src_name = LANG_NAMES.get(source_lang, source_lang)
    tgt_name = LANG_NAMES.get(target_lang, target_lang)
    units_json = json.dumps([{"id": i, "text": u["text"]} for i, u in batch],
                            ensure_ascii=False)
    prompt = (
        f"Translate video narration units from {src_name} to {tgt_name}, in a "
        "natural spoken narrator style.\n"
        "Rules: preserve the meaning of each unit FULLY — do not condense, "
        "omit, summarise or invent anything. Keep names and terms consistent "
        "with the story memory. Return ONLY a JSON object mapping each unit "
        "id (as a string) to its translation.\n"
        f"Story memory: {json.dumps(memory or {}, ensure_ascii=False)}\n"
        f"Previous unit (context only, do not translate): {prev_text or ''}\n"
        f"Next unit (context only, do not translate): {next_text or ''}\n"
        f"Units: {units_json}")
    result = None
    for attempt in range(2):
        obj = _gemini_json(api_key, prompt, log)
        if _validate_id_map(obj, ids):
            result = {i: obj[str(i)].strip() for i in ids}
            break
        log(f"Gemini batch output failed validation (attempt {attempt + 1})")
        prompt += ("\nYour previous answer was invalid: return a JSON object "
                   "with exactly the same ids, each mapped to a non-empty "
                   "translation string.")
    if result is None:
        raise RuntimeError("Gemini batch failed validation twice")
    # audit pass: repair omissions / duplicates / misplaced content
    try:
        audit_units = json.dumps(
            [{"id": i, "source": u["text"], "translation": result[i]}
             for i, u in batch], ensure_ascii=False)
        audit_prompt = (
            f"You are auditing {tgt_name} translations of video narration. "
            "For each unit, compare source and translation and fix omissions, "
            "duplicated content, or content that belongs to a neighbouring "
            "unit. Preserve meaning fully; do not condense or invent. Return "
            "ONLY a JSON object mapping every id (as a string) to its "
            f"corrected translation.\nUnits: {audit_units}")
        obj = _gemini_json(api_key, audit_prompt, log)
        if _validate_id_map(obj, ids):
            result = {i: obj[str(i)].strip() for i in ids}
        else:
            log("Audit output failed validation; keeping pre-audit translation")
    except Exception as e:
        log(f"Audit pass failed ({_scrub(e, api_key)[:120]}); "
            "keeping pre-audit translation")
    return result


def _translate_quality(units, source_lang, target_lang, api_key, log, counts):
    tgt_name = LANG_NAMES.get(target_lang, target_lang)
    # story-context pass: one compact memory for the whole transcript
    memory = {}
    try:
        transcript = "\n".join(f"{i + 1}. {u['text']}" for i, u in enumerate(units))
        prompt = (
            f"You are helping translate a video narration into {tgt_name}. "
            "Read the full transcript and return ONLY a compact JSON object: "
            '{"characters": ["..."], "relationships": ["..."], '
            '"places": ["..."], '
            f'"terms": {{"source term": "consistent {tgt_name} rendering"}}, '
            '"plot_summary": "2-3 sentences"}. Keep it short.\n'
            f"Transcript:\n{transcript[:15000]}")
        obj = _gemini_json(api_key, prompt, log)
        if isinstance(obj, dict):
            memory = obj
            log("Gemini story-context pass done")
    except Exception as e:
        log(f"Gemini story-context pass failed ({_scrub(e, api_key)[:120]}); "
            "continuing without story memory")
    fns = None  # fast-MT fallback translators, built lazily if a batch fails
    batch_size = 20
    for start in range(0, len(units), batch_size):
        batch = list(enumerate(units))[start:start + batch_size]
        prev_text = units[start - 1]["text"] if start > 0 else ""
        nxt = start + batch_size
        next_text = units[nxt]["text"] if nxt < len(units) else ""
        try:
            result = _gemini_translate_batch(api_key, batch, memory, prev_text,
                                             next_text, source_lang,
                                             target_lang, log)
            for i, u in batch:
                u["dub_text"] = result[i]
                counts["gemini"] += 1
        except Exception as e:
            log(f"Gemini batch at unit {start} failed "
                f"({_scrub(e, api_key)[:120]}); falling back to fast MT for it")
            if fns is None:
                fns = _make_mt(source_lang, target_lang)
            prev_dub = units[start - 1].get("dub_text") if start > 0 else None
            for i, u in batch:
                text, prov = _translate_fast_one(u["text"], fns, prev_dub,
                                                 target_lang, log)
                if prov:
                    counts[prov] += 1
                    u["dub_text"] = text
                elif text:
                    counts["mt_best_effort"] = counts.get("mt_best_effort", 0) + 1
                    u["dub_text"] = text
                else:
                    counts["kept_original"] += 1
                    u["dub_text"] = u["text"]
                prev_dub = u["dub_text"]


def translate_units(units, source_lang, target_lang, log, mode="fast",
                    gemini_key=None):
    """Fill unit['dub_text']. Returns (counts, summary).

    counts keys: gemini / google / mymemory / mt_best_effort / kept_original.
    Any failure degrades gracefully — this function does not raise.
    """
    counts = {"gemini": 0, "google": 0, "mymemory": 0,
              "mt_best_effort": 0, "kept_original": 0}
    n = len(units)
    if source_lang == target_lang:
        for u in units:
            u["dub_text"] = u["text"]
        counts["kept_original"] = n
        return counts, f"not needed (same language) {n}/{n}"
    mode = (mode or "fast").lower()
    note = ""
    if mode == "quality":
        key = gemini_key or os.environ.get("GEMINI_API_KEY")
        if key:
            _translate_quality(units, source_lang, target_lang, key, log, counts)
        else:
            note = " (quality requested but no Gemini key — fell back to fast MT)"
            log("Quality mode requested but no Gemini key available; "
                "falling back to fast MT")
            _translate_fast(units, source_lang, target_lang, log, counts)
    else:
        _translate_fast(units, source_lang, target_lang, log, counts)
    mt = counts["google"] + counts["mymemory"] + counts["mt_best_effort"]
    if counts["gemini"]:
        summary = f"gemini {counts['gemini']}/{n}, mt {mt}/{n}"
    else:
        summary = f"mt {mt}/{n} (google {counts['google']}, mymemory {counts['mymemory']})"
        if mode == "quality":
            summary = f"gemini 0/{n}, " + summary
    if counts["mt_best_effort"]:
        summary += f", best-effort {counts['mt_best_effort']}"
    if counts["kept_original"]:
        summary += f", kept original {counts['kept_original']}/{n}"
    return counts, summary + note


# ---------------------------------------------------------------- step 5: TTS
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


# ---- "short pauses": trim the silence TTS clips carry at both ends -------
# edge-tts clips typically carry ~0.2 s of lead-in and up to ~0.8 s of tail
# silence; concatenated raw, lines join with ~1 s gaps. Trimming to ~0.05 s
# head / ~0.20 s tail makes narration flow. A min-duration guard protects
# very short clips: if trimming would leave < 0.25 s, keep the raw clip.
TRIM_HEAD = 0.05
TRIM_TAIL = 0.20
TRIM_MIN = 0.25


def trim_silence(src_path, out_wav, log):
    """Convert src to 44.1k mono wav with head/tail silence trimmed.
    Returns the resulting duration in seconds."""
    filt = (f"silenceremove=start_periods=1:start_threshold=-45dB:"
            f"start_silence={TRIM_HEAD},areverse,"
            f"silenceremove=start_periods=1:start_threshold=-45dB:"
            f"start_silence={TRIM_TAIL},areverse")
    try:
        run(["ffmpeg", "-y", "-i", src_path, "-af", filt,
             "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le", out_wav])
        dur = probe_duration(out_wav)
        if dur >= TRIM_MIN:
            return dur
        log(f"trim guard: trimmed clip would be only {dur:.2f}s; using raw clip")
    except Exception as e:
        log(f"silence trim failed ({e}); using raw clip")
    run(["ffmpeg", "-y", "-i", src_path, "-ar", "44100", "-ac", "1",
         "-c:a", "pcm_s16le", out_wav])
    return probe_duration(out_wav)


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
         "-t", f"{max(seconds, 0.01):.3f}", "-c:a", "pcm_s16le", path])


def fit_clip(src, window, out_wav):
    """Speed-fit / pad an audio clip to exactly `window` seconds (44.1k mono)."""
    dur = probe_duration(src)
    filters = []
    if dur > window and window > 0:
        filters.append(atempo_chain(min(dur / window, 4.0)))
    filt = (",".join(filters) + ",") if filters else ""
    filt += f"aresample=44100,apad,atrim=0:{window:.3f},asetpts=N/SR/TB"
    run(["ffmpeg", "-y", "-i", src, "-af", filt, "-c:a", "pcm_s16le", out_wav])


def _concat_wavs(parts, list_file, out_wav):
    with open(list_file, "w", encoding="utf-8") as f:
        for p in parts:
            f.write(f"file '{p}'\n")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
         "-c:a", "pcm_s16le", out_wav])


# ------------------------------------------------------- step 6a: fit assembly
def assemble_fit(units, work):
    """OLD behaviour: fit each (trimmed) dub clip into its original tile
    window and lay the clips on the original timeline with silence gaps."""
    parts = []
    cursor = 0.0
    for i, u in enumerate(units):
        start, end = u["tile_start"], max(u["tile_end"], u["tile_start"] + 0.4)
        window = end - start
        if start - cursor > 0.05:
            gap = os.path.join(work, f"fit_gap_{i}.wav")
            silence_wav(gap, start - cursor)
            parts.append(gap)
        seg_wav = os.path.join(work, f"fit_seg_{i}.wav")
        if u.get("_wav") and u.get("tts_provider") not in (None, "silence"):
            fit_clip(u["_wav"], window, seg_wav)
        else:
            silence_wav(seg_wav, window)
        parts.append(seg_wav)
        cursor = end
    total = units[-1]["tile_end"] if units else 0.0
    return parts, total


# -------------------------------------------------- step 6b: Smart Sync render
SPEED_UP_CAP = 1.25    # picture may be sped up at most this much
SLOW_DOWN_CAP = 1.35   # ... or slowed down at most this much; beyond that,
                       # short freeze holds are distributed across the unit
SYNC_FPS = 30


def _smart_pieces(units, durs):
    """Plan the retimed picture as a list of pieces:
      ("play", src_start, src_end, out_dur)   — window played at out_dur length
      ("freeze", src_time, out_dur)           — hold the frame at src_time
    The dub leads: every unit's pieces sum to exactly its dub duration."""
    pieces = []
    for u, d in zip(units, durs):
        a, b = u["tile_start"], u["tile_end"]
        W = max(b - a, 0.05)
        D = max(float(d), 0.05)
        s = D / W  # timestamp stretch needed ( >1 = slow the picture down )
        if s < 1.0 / SPEED_UP_CAP:
            # dub is much shorter: play the head of the window at the cap
            used = D * SPEED_UP_CAP
            pieces.append(("play", a, a + used, D))
        elif s <= SLOW_DOWN_CAP:
            pieces.append(("play", a, b, D))
        else:
            # dub is much longer: slow to the cap and distribute the rest
            # as freeze holds after each third of the window
            rest = D - W * SLOW_DOWN_CAP
            k = 3
            for j in range(k):
                ca = a + W * j / k
                cb = a + W * (j + 1) / k
                pieces.append(("play", ca, cb, (cb - ca) * SLOW_DOWN_CAP))
                pieces.append(("freeze", max(ca, cb - 0.05), rest / k))
    return pieces


def render_smart_video(video_path, units, durs, out_path, log):
    """Re-time the picture to the dub, piece by piece, at SYNC_FPS.
    Returns the measured duration of the retimed video."""
    pieces = _smart_pieces(units, durs)
    n = len(pieces)
    log(f"Smart Sync: {len(units)} units -> {n} picture pieces")
    split = "".join(f"[s{i}]" for i in range(n))
    fc = [f"[0:v]split={n}{split}"]
    labels = []
    for i, p in enumerate(pieces):
        lab = f"[v{i}]"
        if p[0] == "play":
            _, a, b, out_dur = p
            factor = out_dur / max(b - a, 0.01)
            fc.append(f"[s{i}]trim=start={a:.3f}:end={b:.3f},"
                      f"setpts=(PTS-STARTPTS)*{factor:.5f},"
                      f"fps={SYNC_FPS},setpts=N/({SYNC_FPS}*TB),format=yuv420p{lab}")
        else:
            _, at, out_dur = p
            fc.append(f"[s{i}]trim=start={at:.3f}:end={at + 0.06:.3f},"
                      f"setpts=PTS-STARTPTS,"
                      f"tpad=stop_mode=clone:stop_duration={out_dur:.3f},"
                      f"trim=duration={out_dur:.3f},"
                      f"fps={SYNC_FPS},setpts=N/({SYNC_FPS}*TB),format=yuv420p{lab}")
        labels.append(lab)
    fc.append("".join(labels) + f"concat=n={n}:v=1:a=0,fps={SYNC_FPS},format=yuv420p[outv]")
    run(["ffmpeg", "-y", "-i", video_path, "-filter_complex", ";".join(fc),
         "-map", "[outv]", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "20", "-an", out_path])
    return probe_duration(out_path)


# ------------------------------------------------------- step 6c: final render
_SUB_STYLE = ("FontName=Noto Sans Myanmar,FontSize=18,PrimaryColour=&H00FFFFFF,"
              "OutlineColour=&H90000000,BorderStyle=1,Outline=2,Shadow=0,"
              "Alignment=2,MarginV=36")


def render_final(video_path, srt_path, dub_wav, out_path, aspect,
                 keep_original, original_video_path=None):
    """Burn subtitles + apply aspect + mux the dub (optionally with the
    original audio mixed low). video_path is the picture source (the
    original video in fit mode, the retimed video in smart mode);
    original_video_path supplies the original audio when it differs."""
    vf = []
    if aspect == "9:16":
        vf.append("scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280")
    elif aspect == "16:9":
        vf.append("scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720")
    vf.append(f"subtitles={srt_path}:force_style='{_SUB_STYLE}'")
    orig_src = original_video_path or video_path
    cmd = ["ffmpeg", "-y", "-i", video_path, "-i", dub_wav]
    if keep_original:
        if orig_src != video_path:
            cmd += ["-i", orig_src]
            oi = 2
        else:
            oi = 0
        cmd += ["-filter_complex",
                f"[0:v]{','.join(vf)}[v];[{oi}:a]volume=0.12[orig];"
                f"[1:a][orig]amix=inputs=2:duration=first:normalize=0[a]",
                "-map", "[v]", "-map", "[a]"]
    else:
        cmd += ["-vf", ",".join(vf), "-map", "0:v", "-map", "1:a"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
            "-c:a", "aac", "-b:a", "160k", "-shortest",
            "-movflags", "+faststart", out_path]
    run(cmd)


# ---------------------------------------------------------------- main entry
def run_job(job_dir, params, status_cb):
    """Run the pipeline for one job, reusing any stage checkpoints already
    in job_dir (transcript.json / units.json / translations.json /
    dub_manifest.json). status_cb(stage, percent, message)."""
    job_log = os.path.join(job_dir, "job.log")

    def log(msg):
        print(f"[job] {msg}", flush=True)
        try:
            with open(job_log, "a", encoding="utf-8") as fh:
                fh.write(str(msg) + "\n")
        except Exception:
            pass

    video_path = params["video_path"]
    source_lang = norm_lang(params.get("source_lang") or "auto")
    target_lang = norm_lang(params.get("target_lang") or "my")
    gender = "male" if params.get("voice") == "male" else "female"
    aspect = params.get("aspect") or "original"
    keep_original = bool(params.get("keep_original_audio"))
    translate_mode = str(params.get("translate_mode") or "fast").lower()
    if translate_mode not in ("fast", "quality"):
        translate_mode = "fast"
    sync_mode = str(params.get("sync_mode") or DEFAULT_SYNC_MODE).lower()
    if sync_mode not in ("smart", "fit"):
        sync_mode = "smart"
    # The Gemini key is transient: params (from the form) or env, never
    # written to any checkpoint/options file by this module or the server.
    gemini_key = params.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")

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

    # ---- 2. transcribe (checkpoint: transcript.json)
    tpath = os.path.join(job_dir, "transcript.json")
    saved = _load_json(tpath)
    if saved and saved.get("segments"):
        segments = saved["segments"]
        detected = saved.get("detected", source_lang)
        log(f"checkpoint: reusing transcript.json ({len(segments)} segments)")
        status_cb("transcribe", 35, f"Reusing the saved transcript ({len(segments)} lines).")
    else:
        status_cb("transcribe", 15,
                  f"Listening and writing down the speech (Whisper {WHISPER_MODEL})…")
        segments, detected = transcribe(audio_wav, source_lang)
        if not segments:
            raise RuntimeError("No speech was found in this video.")
        _save_json(tpath, {"segments": segments, "detected": detected,
                           "total_dur": total_dur})
        status_cb("transcribe", 35, f"Found {len(segments)} lines of speech ({detected}).")
    src_eff = norm_lang(detected)
    log(f"{len(segments)} transcript segments, detected language: {detected}")

    # ---- 3. atomic units (checkpoint: units.json)
    upath = os.path.join(job_dir, "units.json")
    saved = _load_json(upath)
    if saved and saved.get("units"):
        units = saved["units"]
        units_source = saved.get("source", "checkpoint")
        log(f"checkpoint: reusing units.json ({len(units)} units)")
    else:
        units, units_source = make_units(segments, total_dur)
        _save_json(upath, {"units": [{k: u[k] for k in
                                     ("start", "end", "text", "tile_start", "tile_end")}
                                    for u in units],
                           "source": units_source, "total_dur": total_dur})
    for u in units:
        u.pop("dub_text", None)
    n = len(units)
    dur_list = [round(u["tile_end"] - u["tile_start"], 2) for u in units]
    log(f"{n} atomic units (source: {units_source}), tile durations: {dur_list}")
    status_cb("units", 40, f"Split the speech into {n} short lines.")

    # ---- 4. translate (checkpoint: translations.json)
    trpath = os.path.join(job_dir, "translations.json")
    saved = _load_json(trpath)
    tr_meta = {"target_lang": target_lang, "src_eff": src_eff, "mode": translate_mode}
    if (saved and saved.get("meta") == tr_meta and saved.get("dub_texts")
            and len(saved["dub_texts"]) == n):
        for u, t in zip(units, saved["dub_texts"]):
            u["dub_text"] = t
        translate_counts = saved.get("counts", {})
        translate_summary = saved.get("summary", "checkpoint")
        log("checkpoint: reusing translations.json")
        status_cb("translate", 58, "Reusing the saved translation.")
    else:
        status_cb("translate", 45, f"Translating to {target_lang}…")
        translate_counts, translate_summary = translate_units(
            units, src_eff, target_lang, log,
            mode=translate_mode, gemini_key=gemini_key)
        _save_json(trpath, {"meta": tr_meta,
                            "dub_texts": [u["dub_text"] for u in units],
                            "counts": translate_counts,
                            "summary": translate_summary})
        status_cb("translate", 58, "Translation done.")
    log(f"translate: {translate_summary}")

    # ---- 5. TTS + trim (checkpoint: dub_manifest.json)
    voice = pick_voice(target_lang, gender)
    tts_log_path = os.path.join(job_dir, "tts.log")

    def tlog(msg):
        log(msg)
        try:
            with open(tts_log_path, "a", encoding="utf-8") as fh:
                fh.write(msg + "\n")
        except Exception:
            pass

    mpath = os.path.join(job_dir, "dub_manifest.json")
    saved = _load_json(mpath)
    tts_meta = {"target_lang": target_lang, "voice": voice}
    manifest_ok = (saved and saved.get("meta") == tts_meta
                   and len(saved.get("units", [])) == n
                   and all(e.get("provider") == "silence"
                           or os.path.exists(os.path.join(work, e.get("wav", "")))
                           for e in saved["units"]))
    tts_counts = {"edge-tts": 0, "gtts": 0, "silence": 0}
    if manifest_ok:
        for u, e in zip(units, saved["units"]):
            u["tts_provider"] = e["provider"]
            u["_dur"] = float(e["dur"])
            u["_wav"] = (os.path.join(work, e["wav"])
                         if e.get("wav") else None)
            key = e["provider"] if e["provider"] in tts_counts else "silence"
            tts_counts[key] += 1
        log("checkpoint: reusing dub_manifest.json (voice clips already made)")
        status_cb("tts", 80, "Reusing the saved voice clips.")
    else:
        entries = []
        for i, u in enumerate(units):
            status_cb("tts", 60 + int(20 * (i + 1) / n),
                      f"Creating the new voice… line {i + 1} of {n}")
            window = max(u["tile_end"] - u["tile_start"], 0.4)
            mp3 = os.path.join(work, f"unit_{i}.mp3")
            wav = os.path.join(work, f"unit_{i}.wav")
            text = (u.get("dub_text") or "").strip()
            provider = None
            if text:
                provider = tts_segment_fallback(text, voice, target_lang, mp3, tlog)
            if provider:
                dur = trim_silence(mp3, wav, tlog)
                u["tts_provider"] = provider
                u["_wav"] = wav
                u["_dur"] = dur
                tts_counts[provider] += 1
                entries.append({"provider": provider,
                                "wav": os.path.basename(wav), "dur": dur})
                tlog(f"line {i + 1}/{n}: voiced by {provider}, "
                     f"{dur:.2f}s after trim (window {window:.2f}s)")
            else:
                u["tts_provider"] = "silence"
                u["_wav"] = None
                u["_dur"] = window
                tts_counts["silence"] += 1
                entries.append({"provider": "silence", "wav": None, "dur": window})
                tlog(f"line {i + 1}/{n}: SILENT (no text or all TTS failed)")
        _save_json(mpath, {"meta": tts_meta, "units": entries})
    tts_summary = (f"edge-tts {tts_counts['edge-tts']}/{n}, "
                   f"gTTS {tts_counts['gtts']}/{n}, "
                   f"silent {tts_counts['silence']}/{n}")
    tlog(f"TTS summary: {tts_summary}")
    log(f"TTS summary: {tts_summary}; voice {voice}")

    # ---- 6. assemble + render
    out_path = os.path.join(job_dir, "output.mp4")
    srt_path = os.path.join(job_dir, "subtitles.srt")
    render_meta = {"sync_mode": sync_mode, "aspect": aspect,
                   "keep_original": keep_original, "target_lang": target_lang}
    rmpath = os.path.join(job_dir, "render_meta.json")
    sync_info = None
    if os.path.exists(out_path) and _load_json(rmpath) == render_meta:
        log("checkpoint: output.mp4 already rendered with these settings; skipping render")
        status_cb("render", 95, "Reusing the finished video.")
    else:
        status_cb("render", 85, "Putting picture, new voice and subtitles together…")
        durs = []
        for u in units:
            if u.get("tts_provider") in (None, "silence"):
                durs.append(max(u["tile_end"] - u["tile_start"], 0.4))
            else:
                durs.append(float(u.get("_dur") or
                                  max(u["tile_end"] - u["tile_start"], 0.4)))
        dub_wav = os.path.join(work, "dub.wav")
        if sync_mode == "fit":
            parts, _total = assemble_fit(units, work)
            _concat_wavs(parts, os.path.join(work, "concat.txt"), dub_wav)
            write_srt([{"start": u["tile_start"], "end": u["tile_end"],
                        "dub_text": u["dub_text"]} for u in units], srt_path)
            render_final(video_path, srt_path, dub_wav, out_path, aspect,
                         keep_original)
            narration = probe_duration(dub_wav)
            sync_info = {"mode": "fit", "picture": round(total_dur, 3),
                         "narration": round(narration, 3),
                         "diff": round(total_dur - narration, 3)}
        else:
            # dub track: trimmed clips back to back, silence for silent units
            parts = []
            for i, u in enumerate(units):
                if u.get("tts_provider") in (None, "silence"):
                    sil = os.path.join(work, f"smart_sil_{i}.wav")
                    silence_wav(sil, durs[i])
                    parts.append(sil)
                else:
                    parts.append(u["_wav"])
            _concat_wavs(parts, os.path.join(work, "concat.txt"), dub_wav)
            narration = probe_duration(dub_wav)
            # retimed subtitles on the new (dub-led) timeline
            srt_units, cursor = [], 0.0
            for u, d in zip(units, durs):
                srt_units.append({"start": cursor, "end": cursor + d,
                                  "dub_text": u["dub_text"]})
                cursor += d
            write_srt(srt_units, srt_path)
            smart_path = os.path.join(work, "smart.mp4")
            picture = render_smart_video(video_path, units, durs, smart_path, log)
            diff = picture - narration
            sync_info = {"mode": "smart", "picture": round(picture, 3),
                         "narration": round(narration, 3),
                         "diff": round(diff, 3)}
            log(f"Smart Sync timing: picture={picture:.2f}s "
                f"narration={narration:.2f}s diff={diff:+.2f}s")
            render_final(smart_path, srt_path, dub_wav, out_path, aspect,
                         keep_original, original_video_path=video_path)
        _save_json(rmpath, render_meta)

    status_cb("done", 100, "Done! Your dubbed video is ready to download.")
    return {"output": out_path, "srt": srt_path, "segments": n, "units": n,
            "units_source": units_source, "detected_lang": detected,
            "voice": voice, "tts": tts_summary, "tts_counts": tts_counts,
            "translate": translate_summary, "translate_counts": translate_counts,
            "sync": sync_info}
