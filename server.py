"""Recap Studio v3 AUTO — server backend.

Upload a video -> the server transcribes it, translates it, dubs it with an
AI voice, burns subtitles and renders a finished MP4 you can download.
Pipeline lives in pipeline.py. Jobs run one at a time in a background thread
(CPU-only friendly); poll /api/status/{job_id} for progress.
"""
import os
import re
import shutil
import threading
import traceback
import uuid

# Sanitize proxy env vars (some HTTP libs crash on bracketed IPv6 entries)
for var in ("no_proxy", "NO_PROXY"):
    if os.environ.get(var):
        cleaned = [e.strip() for e in os.environ[var].split(",") if "[" not in e and "]" not in e]
        os.environ[var] = ",".join(cleaned)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

import pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(HERE, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))
ALLOWED_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

app = FastAPI(title="Recap Studio v3 AUTO")
jobs = {}            # job_id -> status dict (also persisted to status.json)
_jobs_lock = threading.Lock()
_run_lock = threading.Lock()   # one pipeline job at a time (CPU)


def _save_status(job_id):
    import json
    path = os.path.join(JOBS_DIR, job_id, "status.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(jobs[job_id], f, ensure_ascii=False)
    except Exception:
        pass


def _set_status(job_id, **kw):
    with _jobs_lock:
        jobs[job_id].update(kw)
        _save_status(job_id)


def _run_job(job_id, params):
    def cb(stage, percent, message):
        _set_status(job_id, stage=stage, percent=percent, message=message)
    with _run_lock:
        _set_status(job_id, stage="starting", percent=1, message="Starting…")
        try:
            result = pipeline.run_job(os.path.join(JOBS_DIR, job_id), params, cb)
            _set_status(job_id, stage="done", percent=100,
                        message="Done! Your dubbed video is ready to download.",
                        segments=result["segments"],
                        detected_lang=result["detected_lang"], voice=result["voice"],
                        tts=result.get("tts"), tts_counts=result.get("tts_counts"))
        except Exception as e:
            traceback.print_exc()
            _set_status(job_id, stage="error", percent=0,
                        message="Something went wrong while processing this video.",
                        error=str(e)[:500])


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "static", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
def health():
    return {"ok": True, "app": "recap-studio-v3", "whisper_model": pipeline.WHISPER_MODEL,
            "max_upload_mb": MAX_UPLOAD_MB,
            "languages": sorted(pipeline.LANGS.keys()),
            "edge_only_tts": sorted(c for c, e in pipeline.LANGS.items() if not e["gtts"])}


@app.post("/api/process")
async def process(file: UploadFile = File(...),
                  source_lang: str = Form("auto"),
                  target_lang: str = Form("my"),
                  voice: str = Form("female"),
                  aspect: str = Form("original"),
                  keep_original_audio: str = Form("false")):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400,
                            detail=f"Please upload a video file ({', '.join(sorted(ALLOWED_EXT))}).")
    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    # safe stored name: never trust the uploaded filename as a path
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.basename(file.filename))[:80] or "video"
    video_path = os.path.join(job_dir, "input" + ext)
    size = 0
    limit = MAX_UPLOAD_MB * 1024 * 1024
    with open(video_path, "wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > limit:
                out.close()
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(status_code=400,
                                    detail=f"File too big (max {MAX_UPLOAD_MB} MB). "
                                           "Short clips work best.")
            out.write(chunk)
    with _jobs_lock:
        jobs[job_id] = {"job_id": job_id, "stage": "queued", "percent": 0,
                        "message": "Uploaded. Waiting to start…", "source_name": safe_name}
        _save_status(job_id)
    params = {"video_path": video_path, "source_lang": source_lang,
              "target_lang": target_lang, "voice": voice, "aspect": aspect,
              "keep_original_audio": str(keep_original_audio).lower() in ("true", "1", "yes")}
    threading.Thread(target=_run_job, args=(job_id, params), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    with _jobs_lock:
        st = jobs.get(job_id)
    if st is None:  # maybe the server restarted; fall back to disk
        import json
        path = os.path.join(JOBS_DIR, job_id, "status.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                st = json.load(f)
        else:
            raise HTTPException(status_code=404, detail="Job not found.")
    return st


@app.get("/api/download/{job_id}")
def download(job_id: str):
    out = os.path.join(JOBS_DIR, job_id, "output.mp4")
    if not os.path.exists(out):
        raise HTTPException(status_code=404, detail="The video is not ready yet.")
    return FileResponse(out, media_type="video/mp4", filename="recap-dubbed.mp4")


@app.get("/api/subtitles/{job_id}")
def subtitles(job_id: str):
    srt = os.path.join(JOBS_DIR, job_id, "subtitles.srt")
    if not os.path.exists(srt):
        raise HTTPException(status_code=404, detail="Subtitles are not ready yet.")
    return FileResponse(srt, media_type="text/plain", filename="subtitles.srt")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
