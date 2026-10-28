#!/usr/bin/env python3
"""
AutoClipper website with Free and Pro plans.

    pip install -r requirements.txt
    python server.py                      # then open http://localhost:8000

Plans
    Free  - picks the moments themselves on a video player; 10-second clips;
            FREE_CLIPS_PER_DAY clips (default 4) per 24 hours.
    Pro   - the full AutoClipper: finds the best moments automatically,
            20-60 second clips, links or uploads, no daily limit.

Upgrade someone to Pro:
    - visit /admin (set ADMIN_PASSWORD first), or
    - python server.py set-plan someone@example.com pro
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import queue
import re
import secrets
import shutil
import sqlite3
import sys
import threading
import time
import traceback
import zipfile
from html import escape
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

import clipper

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
def env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


DATA_DIR = Path(os.getenv("DATA_DIR", "data")).resolve()
JOBS_DIR = DATA_DIR / "jobs"
DB_PATH = DATA_DIR / "app.db"
WORKERS = int(os.getenv("WORKERS", "1"))              # videos processed at the same time
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "4096"))
MAX_MINUTES = float(os.getenv("MAX_MINUTES", "180"))  # longest video accepted
MAX_CLIPS = int(os.getenv("MAX_CLIPS", "8"))          # Pro: most clips per video
KEEP_HOURS = float(os.getenv("KEEP_HOURS", "24"))     # results are deleted after this
MAX_ACTIVE_JOBS = int(os.getenv("MAX_ACTIVE_JOBS", "2"))  # per person, at the same time
FREE_CLIPS_PER_DAY = int(os.getenv("FREE_CLIPS_PER_DAY", "4"))
FREE_CLIP_SECONDS = float(os.getenv("FREE_CLIP_SECONDS", "10"))
ALLOW_URLS = env_bool("ALLOW_URLS", True)             # Pro only
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
FONT = os.getenv("CAPTION_FONT", "Liberation Sans")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
UPGRADE_URL = os.getenv("UPGRADE_URL", "")            # e.g. a Stripe payment link
SECURE_COOKIES = env_bool("SECURE_COOKIES", False)    # set true once you have https
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
PLANS = ("free", "pro")
SESSION_DAYS = 30

# share of the progress bar each stage takes up
STAGES_AUTO = {"download": (0.0, 0.10), "transcribe": (0.10, 0.50), "pick": (0.60, 0.05),
               "render": (0.65, 0.35), "done": (1.0, 0)}
STAGES_MANUAL = {"download": (0.0, 0.10), "render": (0.10, 0.90), "done": (1.0, 0)}

JOBS_DIR.mkdir(parents=True, exist_ok=True)
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
work_queue: "queue.Queue[str]" = queue.Queue()


# --------------------------------------------------------------------------- #
# Database (users + free-plan usage)
# --------------------------------------------------------------------------- #
db_lock = threading.Lock()


def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    with db_lock, db() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                pw_hash TEXT NOT NULL,
                plan TEXT NOT NULL DEFAULT 'free',
                created REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                job_id TEXT NOT NULL,
                clips INTEGER NOT NULL,
                ts REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS usage_user_ts ON usage(user_id, ts);
        """)


def hash_pw(pw: str) -> str:
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(pw.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${h.hex()}"


def check_pw(pw: str, stored: str) -> bool:
    try:
        _, salt, h = stored.split("$")
        got = hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=2 ** 14, r=8, p=1, dklen=32)
        return hmac.compare_digest(got.hex(), h)
    except ValueError:
        return False


def get_user(user_id: int) -> sqlite3.Row | None:
    with db() as con:
        return con.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def free_usage(user_id: int) -> tuple[int, float | None]:
    """(clips used in the last 24 hours, time the oldest of those expires)."""
    since = time.time() - 86400
    with db() as con:
        rows = con.execute("SELECT clips, ts FROM usage WHERE user_id = ? AND ts > ? ORDER BY ts",
                           (user_id, since)).fetchall()
    used = sum(r["clips"] for r in rows)
    return used, (rows[0]["ts"] + 86400 if rows else None)


def charge(user_id: int, job_id: str, clips: int) -> None:
    with db_lock, db() as con:
        con.execute("INSERT INTO usage (user_id, job_id, clips, ts) VALUES (?, ?, ?, ?)",
                    (user_id, job_id, clips, time.time()))


def refund(job_id: str) -> None:
    with db_lock, db() as con:
        con.execute("DELETE FROM usage WHERE job_id = ?", (job_id,))


# --------------------------------------------------------------------------- #
# Sessions (signed cookie)
# --------------------------------------------------------------------------- #
def secret_key() -> bytes:
    if os.getenv("SECRET_KEY"):
        return os.environ["SECRET_KEY"].encode()
    f = DATA_DIR / "secret.key"
    if not f.exists():
        f.write_text(secrets.token_hex(32))
    return f.read_text().strip().encode()


SECRET = b""


def make_session(user_id: int) -> str:
    payload = f"{user_id}:{int(time.time()) + SESSION_DAYS * 86400}"
    sig = hmac.new(SECRET, payload.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(payload.encode()).decode() + "." + \
        base64.urlsafe_b64encode(sig).decode()


def read_session(token: str | None) -> int | None:
    if not token or "." not in token:
        return None
    try:
        p, s = token.split(".", 1)
        payload = base64.urlsafe_b64decode(p.encode())
        sig = base64.urlsafe_b64decode(s.encode())
    except Exception:
        return None
    if not hmac.compare_digest(sig, hmac.new(SECRET, payload, hashlib.sha256).digest()):
        return None
    uid, exp = payload.decode().split(":")
    return int(uid) if int(exp) > time.time() else None


def current_user(request: Request) -> sqlite3.Row:
    uid = read_session(request.cookies.get("session"))
    user = get_user(uid) if uid else None
    if not user:
        raise HTTPException(401, "Log in to continue.")
    return user


def set_cookie(resp: Response, user_id: int) -> None:
    resp.set_cookie("session", make_session(user_id), max_age=SESSION_DAYS * 86400,
                    httponly=True, samesite="lax", secure=SECURE_COOKIES)


def me_payload(user: sqlite3.Row) -> dict:
    out = {"email": user["email"], "plan": user["plan"], "upgrade_url": UPGRADE_URL}
    if user["plan"] == "free":
        used, next_at = free_usage(user["id"])
        out.update(free_left=max(0, FREE_CLIPS_PER_DAY - used), free_per_day=FREE_CLIPS_PER_DAY,
                   free_seconds=FREE_CLIP_SECONDS, next_free_at=next_at)
    return out


# --------------------------------------------------------------------------- #
# Job storage
# --------------------------------------------------------------------------- #
def job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def save(job: dict) -> None:
    d = job_dir(job["id"])
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "job.json.tmp"
    tmp.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    tmp.replace(d / "job.json")


def update(job_id: str, **fields) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        job.update(fields)
        save(job)


def load_existing() -> None:
    """Reload past jobs after a restart; unfinished ones are re-queued."""
    for f in sorted(JOBS_DIR.glob("*/job.json"), key=lambda p: p.stat().st_mtime):
        try:
            job = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if job.get("status") in ("queued", "running"):
            job.update(status="queued", stage="queued", progress=0.0, message="Waiting in line")
            jobs[job["id"]] = job
            work_queue.put(job["id"])
        else:
            jobs[job["id"]] = job


def public(job: dict) -> dict:
    out = {k: job.get(k) for k in ("id", "status", "stage", "progress", "message", "error",
                                   "title", "clips", "created", "mode")}
    if job["status"] == "queued":
        with jobs_lock:
            waiting = sorted((j for j in jobs.values() if j["status"] == "queued"),
                             key=lambda j: j["created"])
            running = sum(1 for j in jobs.values() if j["status"] == "running")
        ahead = next((i for i, j in enumerate(waiting) if j["id"] == job["id"]), 0)
        out["ahead"] = ahead + running
    return out


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def process(job_id: str) -> None:
    job = jobs[job_id]
    d = job_dir(job_id)
    manual = job["mode"] == "manual"
    stages = STAGES_MANUAL if manual else STAGES_AUTO
    update(job_id, status="running", stage="download", progress=0.0, message="Getting the video")
    last = [0.0]

    def progress(stage: str, frac: float, msg: str) -> None:
        base, span = stages.get(stage, (None, 0))
        if base is None:
            return
        overall = round(min(1.0, base + span * max(0.0, min(1.0, frac))), 3)
        now = time.time()
        if stage != job.get("stage") or now - last[0] > 1 or stage == "done":
            last[0] = now
            update(job_id, stage=stage, progress=overall, message=msg)

    try:
        out = clipper.run_pipeline(
            job["source"], d / "out", clips=job["clips_wanted"], layout=job["layout"],
            captions=job["captions"], model=WHISPER_MODEL, font=FONT,
            allow_local=job["kind"] == "upload", max_minutes=MAX_MINUTES, flat=True,
            manual_starts=job["starts"] if manual else None, manual_len=FREE_CLIP_SECONDS,
            progress=progress)
        meta = json.loads((out / "clips.json").read_text(encoding="utf-8"))
        clips = [{"file": c["file"], "title": c["title"], "duration": c["duration"],
                  "start": c["start"], "end": c["end"]} for c in meta["clips"]]
        title = job.get("title") if job["kind"] == "upload" else (meta.get("title") or "")
        update(job_id, status="done", stage="done", progress=1.0, message="Your clips are ready",
               clips=clips, title=title, finished=time.time())
    except clipper.ClipperError as e:
        refund(job_id)
        update(job_id, status="error", error=str(e), message="Something went wrong")
    except Exception:
        traceback.print_exc()
        refund(job_id)
        update(job_id, status="error", message="Something went wrong",
               error="We couldn't process this video. Try another file, or try again later.")
    finally:
        # free disk: the source video and work files aren't needed any more
        shutil.rmtree(d / "out" / ".work", ignore_errors=True)
        for f in d.glob("input.*"):
            f.unlink(missing_ok=True)


def worker() -> None:
    while True:
        job_id = work_queue.get()
        try:
            if job_id in jobs and jobs[job_id]["status"] == "queued":
                process(job_id)
        finally:
            work_queue.task_done()


def cleaner() -> None:
    while True:
        cutoff = time.time() - KEEP_HOURS * 3600
        with jobs_lock:
            old = [j for j in jobs.values()
                   if j["status"] in ("done", "error") and j.get("created", 0) < cutoff]
            for j in old:
                jobs.pop(j["id"], None)
        for j in old:
            shutil.rmtree(job_dir(j["id"]), ignore_errors=True)
        time.sleep(600)


# --------------------------------------------------------------------------- #
# Web app
# --------------------------------------------------------------------------- #
app = FastAPI(title="AutoClipper", docs_url=None, redoc_url=None, openapi_url=None)
HERE = Path(__file__).parent
# The page and fonts work either in a "static" folder or right next to server.py,
# so they can all be uploaded to the top of the repo.
ASSET_DIRS = [HERE / "static" / "fonts", HERE / "static", HERE / "fonts", HERE]


def find_asset(name: str) -> Path | None:
    return next((d / name for d in ASSET_DIRS if (d / name).is_file()), None)


@app.get("/static/fonts/{name}")
@app.get("/static/{name}")
def asset(name: str):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.(woff2|txt)", name):
        raise HTTPException(404, "Not found.")
    path = find_asset(name)
    if not path:
        raise HTTPException(404, "Not found.")
    return FileResponse(path, headers={"Cache-Control": "public, max-age=604800"})


@app.exception_handler(HTTPException)
async def http_error(_, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code,
                        headers=getattr(exc, "headers", None))


@app.get("/")
def index():
    page = find_asset("index.html")
    if not page:
        raise HTTPException(500, "index.html is missing. Upload it next to server.py.")
    return FileResponse(page)


@app.get("/api/config")
def config():
    return {"allow_urls": ALLOW_URLS, "max_upload_mb": MAX_UPLOAD_MB, "max_minutes": MAX_MINUTES,
            "max_clips": MAX_CLIPS, "keep_hours": KEEP_HOURS,
            "free_per_day": FREE_CLIPS_PER_DAY, "free_seconds": FREE_CLIP_SECONDS,
            "upgrade_url": UPGRADE_URL}


# ----- accounts ------------------------------------------------------------ #
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_attempts: dict[str, list[float]] = {}


def slow_down(key: str) -> None:
    """Basic protection against password guessing: 10 tries per 15 minutes."""
    now = time.time()
    hits = [t for t in _attempts.get(key, []) if now - t < 900]
    if len(hits) >= 10:
        raise HTTPException(429, "Too many attempts. Wait 15 minutes and try again.")
    hits.append(now)
    _attempts[key] = hits


@app.post("/api/signup")
def signup(request: Request, email: str = Form(...), password: str = Form(...)):
    email = email.strip().lower()
    slow_down("signup:" + (request.client.host if request.client else "?"))
    if not EMAIL_RE.match(email) or len(email) > 200:
        raise HTTPException(400, "Enter a valid email address.")
    if len(password) < 8:
        raise HTTPException(400, "Use a password with at least 8 characters.")
    try:
        with db_lock, db() as con:
            cur = con.execute("INSERT INTO users (email, pw_hash, plan, created) VALUES (?,?,?,?)",
                              (email, hash_pw(password), "free", time.time()))
            uid = cur.lastrowid
    except sqlite3.IntegrityError:
        raise HTTPException(409, "There's already an account with that email. Log in instead.")
    resp = JSONResponse(me_payload(get_user(uid)))
    set_cookie(resp, uid)
    return resp


@app.post("/api/login")
def login(email: str = Form(...), password: str = Form(...)):
    email = email.strip().lower()
    slow_down("login:" + email)
    with db() as con:
        user = con.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if not user or not check_pw(password, user["pw_hash"]):
        raise HTTPException(401, "That email and password don't match.")
    resp = JSONResponse(me_payload(user))
    set_cookie(resp, user["id"])
    return resp


@app.post("/api/logout")
def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("session")
    return resp


@app.get("/api/me")
def me(user=Depends(current_user)):
    return me_payload(user)


# ----- jobs ---------------------------------------------------------------- #
def get_job(job_id: str, user) -> dict:
    job = jobs.get(job_id) if re.fullmatch(r"[A-Za-z0-9_-]{8,40}", job_id) else None
    if not job or job.get("user_id") != user["id"]:
        raise HTTPException(404, "We couldn't find that job. Results are kept for "
                                 f"{KEEP_HOURS:g} hours.")
    return job


@app.post("/api/jobs")
async def create_job(file: UploadFile | None = File(None), url: str = Form(""),
                     clips: int = Form(4), layout: str = Form("face"),
                     captions: bool = Form(True), starts: str = Form(""),
                     user=Depends(current_user)):
    plan = user["plan"]
    with jobs_lock:
        active = sum(1 for j in jobs.values()
                     if j.get("user_id") == user["id"] and j["status"] in ("queued", "running"))
    if active >= MAX_ACTIVE_JOBS:
        raise HTTPException(429, f"You already have {active} videos in progress. "
                                 "Wait for one to finish, then try again.")
    if layout not in ("face", "center", "blur"):
        layout = "face"

    picked: list[float] = []
    if plan == "free":
        # Free plan: the person chooses the moments; 10-second clips; daily limit.
        try:
            picked = sorted({round(float(x), 2) for x in json.loads(starts or "[]")})
        except (ValueError, TypeError):
            raise HTTPException(400, "Pick the moments you want to clip.")
        picked = [s for s in picked if s >= 0]
        if not picked:
            raise HTTPException(400, "Pick at least one moment to clip.")
        used, _ = free_usage(user["id"])
        left = FREE_CLIPS_PER_DAY - used
        if left <= 0:
            raise HTTPException(403, f"You've used your {FREE_CLIPS_PER_DAY} free clips for "
                                     "today. Upgrade to Pro for unlimited clips.")
        if len(picked) > left:
            raise HTTPException(403, f"You have {left} free clip{'s' if left != 1 else ''} left "
                                     "today. Remove some moments, or upgrade to Pro.")
        if url.strip() and not (file and file.filename):
            raise HTTPException(403, "Pasting links is a Pro feature. Upload the video file "
                                     "instead.")
        clips = len(picked)
    else:
        clips = max(1, min(MAX_CLIPS, clips))

    job_id = secrets.token_urlsafe(12)
    d = job_dir(job_id)
    d.mkdir(parents=True)

    def reject(code: int, msg: str):
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(code, msg)

    if file is not None and file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in VIDEO_EXTS:
            reject(400, "That file type isn't supported. Upload an MP4, MOV, MKV or WEBM video.")
        dest = d / f"input{ext}"
        size, limit = 0, MAX_UPLOAD_MB * 1024 * 1024
        with dest.open("wb") as fh:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    fh.close()
                    reject(413, f"That file is over {MAX_UPLOAD_MB} MB.")
                fh.write(chunk)
        source, kind, title = str(dest), "upload", Path(file.filename).stem[:120]
    else:
        url = url.strip()
        if not ALLOW_URLS:
            reject(400, "Links are turned off here. Upload a video file instead.")
        if not re.match(r"^https?://[^\s]+$", url, re.I):
            reject(400, "Paste a full link starting with https://, or upload a file.")
        source, kind, title = url, "link", ""

    job = {"id": job_id, "status": "queued", "stage": "queued", "progress": 0.0,
           "message": "Waiting in line", "error": None, "title": title, "clips": [],
           "created": time.time(), "source": source, "kind": kind, "user_id": user["id"],
           "plan": plan, "mode": "manual" if plan == "free" else "auto", "starts": picked,
           "clips_wanted": clips, "layout": layout, "captions": captions}
    if plan == "free":
        # check again after the upload, in case two uploads were sent at the same time
        used, _ = free_usage(user["id"])
        if used + len(picked) > FREE_CLIPS_PER_DAY:
            reject(403, "You've reached today's free clip limit. Upgrade to Pro for unlimited "
                        "clips.")
        charge(user["id"], job_id, len(picked))
    with jobs_lock:
        jobs[job_id] = job
        save(job)
    work_queue.put(job_id)
    return public(job)


@app.get("/api/jobs")
def my_jobs(user=Depends(current_user)):
    with jobs_lock:
        mine = [public(j) for j in jobs.values() if j.get("user_id") == user["id"]]
    return sorted(mine, key=lambda j: j["created"], reverse=True)[:20]


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, user=Depends(current_user)):
    return public(get_job(job_id, user))


@app.get("/api/jobs/{job_id}/files/{name}")
def job_file(job_id: str, name: str, download: bool = False, user=Depends(current_user)):
    job = get_job(job_id, user)
    if not re.fullmatch(r"clip_\d+\.mp4", name):
        raise HTTPException(404, "No such clip.")
    path = job_dir(job["id"]) / "out" / name
    if not path.is_file():
        raise HTTPException(404, "No such clip.")
    base = clipper.slugify(job.get("title") or "clip", 40)
    return FileResponse(path, media_type="video/mp4",
                        filename=f"{base}-{name}" if download else None)


@app.get("/api/jobs/{job_id}/zip")
def job_zip(job_id: str, user=Depends(current_user)):
    job = get_job(job_id, user)
    if job["status"] != "done":
        raise HTTPException(409, "Clips aren't ready yet.")
    out = job_dir(job["id"]) / "out"
    zpath = job_dir(job["id"]) / "clips.zip"
    if not zpath.exists():
        tmp = zpath.with_suffix(".tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
            for f in sorted(out.glob("clip_*.mp4")):
                z.write(f, f.name)
        tmp.replace(zpath)
    base = clipper.slugify(job.get("title") or "clips", 40)
    return FileResponse(zpath, media_type="application/zip", filename=f"{base}-clips.zip")


# ----- admin --------------------------------------------------------------- #
basic = HTTPBasic(auto_error=False)


def admin_only(creds: HTTPBasicCredentials | None = Depends(basic)):
    if not ADMIN_PASSWORD:
        raise HTTPException(404, "Not found.")
    ok = creds and hmac.compare_digest(creds.password.encode(), ADMIN_PASSWORD.encode())
    if not ok:
        raise HTTPException(401, "Admin login required.",
                            headers={"WWW-Authenticate": 'Basic realm="AutoClipper admin"'})


@app.get("/admin", response_class=HTMLResponse, dependencies=[Depends(admin_only)])
def admin(q: str = ""):
    with db() as con:
        rows = con.execute("SELECT * FROM users WHERE email LIKE ? ORDER BY created DESC LIMIT 200",
                           (f"%{q.strip().lower()}%",)).fetchall()
        counts = dict(con.execute("SELECT plan, COUNT(*) FROM users GROUP BY plan").fetchall())
    trs = "".join(
        f"<tr><td>{escape(r['email'])}</td><td>{time.strftime('%Y-%m-%d', time.gmtime(r['created']))}</td>"
        f"<td><form method=post action=/admin/plan><input type=hidden name=user_id value={r['id']}>"
        + "".join(f"<button name=plan value={p} {'class=on' if r['plan'] == p else ''}>{p.title()}</button>"
                  for p in PLANS)
        + "</form></td></tr>" for r in rows)
    return f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Admin - AutoClipper</title>
<style>body{{font:16px system-ui;background:#1E1433;color:#F4F0FF;margin:0;padding:24px}}
table{{border-collapse:collapse;width:100%;max-width:820px}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #4B3B78}}
button{{font:inherit;padding:6px 14px;margin-right:6px;border-radius:99px;border:1px solid #4B3B78;background:none;color:#B9AED8;cursor:pointer}}
button.on{{background:#FFE500;color:#1E1433;border-color:#FFE500}}input[type=search]{{font:inherit;padding:10px;width:280px;border-radius:8px;border:1px solid #4B3B78;background:#291C46;color:#F4F0FF}}
a{{color:#FFE500}}</style>
<h1>Users</h1><p>{counts.get('free', 0)} free, {counts.get('pro', 0)} Pro. Click a plan to change someone's plan.</p>
<form><input type=search name=q value="{escape(q)}" placeholder="Search by email"> <button>Search</button></form>
<table><tr><th>Email</th><th>Joined</th><th>Plan</th></tr>{trs}</table>"""


@app.post("/admin/plan", dependencies=[Depends(admin_only)])
def admin_plan(user_id: int = Form(...), plan: str = Form(...)):
    if plan not in PLANS:
        raise HTTPException(400, "Unknown plan.")
    with db_lock, db() as con:
        con.execute("UPDATE users SET plan = ? WHERE id = ?", (plan, user_id))
    return RedirectResponse("/admin", status_code=303)


# --------------------------------------------------------------------------- #
# Startup + command line
# --------------------------------------------------------------------------- #
def setup() -> None:
    global SECRET
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    SECRET = secret_key()


@app.on_event("startup")
def start_workers() -> None:
    setup()
    load_existing()
    for _ in range(max(1, WORKERS)):
        threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=cleaner, daemon=True).start()


def cli_set_plan(email: str, plan: str) -> None:
    setup()
    if plan not in PLANS:
        sys.exit(f"Plan must be one of: {', '.join(PLANS)}")
    with db_lock, db() as con:
        n = con.execute("UPDATE users SET plan = ? WHERE email = ?",
                        (plan, email.strip().lower())).rowcount
    print(f"{email} is now on the {plan} plan." if n else f"No account found for {email}.")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "set-plan":
        cli_set_plan(sys.argv[2], sys.argv[3])
    else:
        import uvicorn
        uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))
