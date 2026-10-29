#!/usr/bin/env python3
"""
AutoClipper - turn one long video into short, vertical, captioned clips.

    python clipper.py "https://www.youtube.com/watch?v=..."
    python clipper.py my_video.mp4

Pipeline:
    1. Download the video (yt-dlp)            - or use a local file path
    2. Transcribe with word timings (faster-whisper, runs locally)
    3. Score every 20-60s window and pick the best N (default 4)
       - local scoring: hooks, questions, emotion words, energy, pace
       - or an LLM picks (set ANTHROPIC_API_KEY or OPENAI_API_KEY)
    4. Render each clip as 1080x1920 at a steady frame rate:
       smooth face-following crop + word-by-word captions

The same pipeline powers the website (server.py) through run_pipeline().
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import urllib.request
import wave
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable

import numpy as np

OUT_W, OUT_H = 1080, 1920

ProgressFn = Callable[[str, float, str], None]  # (stage, fraction 0-1, message)


class ClipperError(Exception):
    """A problem we can explain to the user (bad link, no speech, etc.)."""


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Sentence:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Clip:
    start: float
    end: float
    score: float
    title: str
    reason: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(f"[autoclipper] {msg}", flush=True)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kw)


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise ClipperError(f"'{tool}' not found. Install ffmpeg and make sure it's on "
                               "your PATH (see README).")


def slugify(text: str, limit: int = 60) -> str:
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    return (re.sub(r"[\s_-]+", "-", text)[:limit].strip("-")) or "video"


def _rate(s: str | None) -> float:
    try:
        num, den = (s or "0/0").split("/")
        return float(num) / float(den) if float(den) else 0.0
    except ValueError:
        return 0.0


def probe(path: Path) -> dict:
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate:format=duration",
            "-of", "json", str(path)])
        info = json.loads(out)
        st = info["streams"][0]
    except (subprocess.CalledProcessError, KeyError, IndexError, json.JSONDecodeError):
        raise ClipperError("That file doesn't look like a video we can read.")
    # avg_frame_rate is the real rate; r_frame_rate can be a huge "timebase" number on
    # variable-frame-rate files (phone recordings, streams), which caused choppy clips.
    fps = _rate(st.get("avg_frame_rate"))
    if not (1 <= fps <= 120):
        fps = _rate(st.get("r_frame_rate"))
    if not (1 <= fps <= 120):
        fps = 30.0
    return {"width": int(st["width"]), "height": int(st["height"]),
            "fps": fps, "duration": float(info["format"].get("duration", 0) or 0)}


# --------------------------------------------------------------------------- #
# 1. Download
# --------------------------------------------------------------------------- #
def download(source: str, workdir: Path, cookies_browser: str | None = None,
             progress: ProgressFn | None = None, allow_local: bool = True) -> tuple[Path, str]:
    if allow_local and Path(source).is_file():
        p = Path(source)
        return p, p.stem
    if not re.match(r"^https?://", source, re.I):
        raise ClipperError("Please give a full link starting with http:// or https://")
    try:
        import yt_dlp
    except ImportError:
        raise ClipperError("yt-dlp is not installed: pip install -r requirements.txt")

    log("Downloading video...")

    def hook(d):
        if progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                progress("download", min(0.99, d.get("downloaded_bytes", 0) / total),
                         "Downloading video")

    opts = {
        "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(workdir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
    }
    if cookies_browser:
        opts["cookiesfrombrowser"] = (cookies_browser,)
    if os.getenv("YTDLP_COOKIES_FILE"):
        opts["cookiefile"] = os.environ["YTDLP_COOKIES_FILE"]
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(source, download=True)
            title = info.get("title") or "video"
    except Exception as e:
        msg = str(e)
        if "confirm you" in msg and "bot" in msg:
            raise ClipperError("YouTube blocked the download from our server. Please download "
                               "the video and upload the file instead.")
        log(f"Download failed: {msg}")
        if "Unsupported URL" in msg:
            raise ClipperError("That link doesn't point to a video we can download. "
                               "Upload the video file instead.")
        low = msg.lower()
        if "private video" in low or "members-only" in low or "confirm your age" in low:
            raise ClipperError("That video is private or restricted. Upload the file instead.")
        raise ClipperError("We couldn't download that link. Check that it opens in your "
                           "browser, or upload the video file instead.")
    files = sorted(workdir.glob("source.*"), key=lambda p: p.stat().st_size, reverse=True)
    if not files:
        raise ClipperError("Download failed - no file was produced.")
    return files[0], title


# --------------------------------------------------------------------------- #
# 2. Transcribe
# --------------------------------------------------------------------------- #
def extract_audio(video: Path, wav: Path) -> None:
    try:
        run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1",
             "-ar", "16000", "-c:a", "pcm_s16le", str(wav)])
    except subprocess.CalledProcessError:
        raise ClipperError("This video has no audio track we can read.")


_models: dict[str, object] = {}
_model_lock = threading.Lock()


def get_whisper(model_size: str):
    """Load each Whisper model once and reuse it (big speed-up for the website)."""
    with _model_lock:
        if model_size not in _models:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise ClipperError("faster-whisper is not installed: "
                                   "pip install -r requirements.txt")
            log(f"Loading Whisper '{model_size}' (first run downloads the model)...")
            _models[model_size] = WhisperModel(model_size, device="auto", compute_type="auto")
        return _models[model_size]


def transcribe(wav: Path, model_size: str, language: str | None, cache: Path,
               duration: float = 0, progress: ProgressFn | None = None) -> list[Word]:
    if cache.exists():
        log("Using cached transcript.")
        return [Word(**w) for w in json.loads(cache.read_text(encoding="utf-8"))]

    model = get_whisper(model_size)
    log("Transcribing...")
    words: list[Word] = []
    # Feed the audio 10 minutes at a time so long videos don't fill the memory.
    with wave.open(str(wav), "rb") as wf:
        sr = wf.getframerate()
        total = duration or wf.getnframes() / sr or 1
        chunk = sr * 600
        offset = 0.0
        while True:
            raw = wf.readframes(chunk)
            if not raw:
                break
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            segments, info = model.transcribe(audio, language=language, word_timestamps=True,
                                              vad_filter=True)
            language = language or getattr(info, "language", None)  # keep it consistent
            for seg in segments:
                for w in seg.words or []:
                    t = w.word.strip()
                    if t:
                        words.append(Word(round(offset + w.start, 3), round(offset + w.end, 3), t))
                done = offset + seg.end
                if progress:
                    progress("transcribe", min(0.99, done / total), "Listening to the video")
                else:
                    print(f"\r  transcribed {done:7.1f}s", end="", flush=True)
            offset += len(audio) / sr
            del audio
    if not progress:
        print()
    cache.write_text(json.dumps([asdict(w) for w in words]), encoding="utf-8")
    return words


def transcribe_segment(video: Path, start: float, dur: float, model_size: str,
                       language: str | None, wav: Path) -> list[Word]:
    """Transcribe only one short part of the video (for captions on picked clips)."""
    try:
        run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
             "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)])
    except subprocess.CalledProcessError:
        return []  # no audio -> no captions, still make the clip
    model = get_whisper(model_size)
    segments, _ = model.transcribe(str(wav), language=language, word_timestamps=True,
                                   vad_filter=True)
    words = []
    for seg in segments:
        for w in seg.words or []:
            t = w.word.strip()
            if t:
                words.append(Word(round(start + w.start, 3), round(start + w.end, 3), t))
    return words


def build_sentences(words: list[Word], max_gap: float = 0.9, max_len: float = 15.0) -> list[Sentence]:
    """Group words into sentence-ish units (punctuation, pauses, or length)."""
    sentences: list[Sentence] = []
    cur: list[Word] = []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        ends = w.text[-1:] in ".?!"
        gap = (nxt.start - w.end) if nxt else 99
        if ends or gap > max_gap or (w.end - cur[0].start) > max_len or nxt is None:
            sentences.append(Sentence(cur[0].start, cur[-1].end,
                                      " ".join(x.text for x in cur), cur))
            cur = []
    return sentences


# --------------------------------------------------------------------------- #
# 3a. Local (free) scoring
# --------------------------------------------------------------------------- #
HOOK_STARTERS = (
    "here's", "here is", "this is why", "the secret", "the truth", "the problem", "the reason",
    "nobody", "no one", "never", "stop", "why", "how", "what if", "imagine", "i used to",
    "you need", "you have to", "you should", "the biggest", "the best", "the worst",
    "most people", "if you", "let me tell", "listen", "the one thing", "the only",
    "did you know", "i can't believe", "this changed", "i was", "i quit", "i lost", "i made",
)
POWER_WORDS = set("""
secret truth mistake mistakes never always crazy insane shocking shocked amazing incredible
money million billion thousand rich broke free fail failed failure success successful
love hate fear afraid scared angry happy sad die died death dead kill killed best worst
biggest huge massive problem wrong right important actually honestly literally seriously
crazy wild unbelievable story happened realized learned lesson change changed changes
nobody everyone everybody hack trick tip tips rule rules why how mind brain life
""".split())
FILLERS = {"um", "uh", "erm", "hmm", "like", "basically", "you know"}


def loudness_profile(wav: Path, step: float = 0.5) -> np.ndarray:
    """Loudness per half-second. Reads the audio a minute at a time to keep memory low."""
    parts = []
    with wave.open(str(wav), "rb") as wf:
        sr = wf.getframerate()
        hop = int(sr * step)
        block = hop * 120  # one minute
        while True:
            raw = wf.readframes(block)
            if not raw:
                break
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            n = len(data) // hop
            if n == 0:
                break
            parts.append(np.sqrt(np.mean(data[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-9))
    rms = np.concatenate(parts) if parts else np.array([1.0])
    db = 20 * np.log10(rms + 1e-9)
    speech = db[db > np.percentile(db, 20)]
    mu, sd = (speech.mean(), speech.std() + 1e-6) if len(speech) else (db.mean(), 1.0)
    return (db - mu) / sd  # z-score per half-second


def score_window(sents: list[Sentence], energy: np.ndarray, target: float) -> tuple[float, str]:
    start, end = sents[0].start, sents[-1].end
    dur = end - start
    words = [w for s in sents for w in s.words]
    toks = [re.sub(r"[^\w']", "", w.text.lower()) for w in words]
    n = max(1, len(toks))
    first = sents[0].text.lower().strip()
    reasons = []

    hook = 0.0
    if first.startswith(HOOK_STARTERS):
        hook += 1.0
        reasons.append("strong hook")
    if first.endswith("?"):
        hook += 0.6
        reasons.append("opens with a question")
    if re.search(r"\d", " ".join(s.text for s in sents[:2])):
        hook += 0.3
    hook = min(hook, 1.5)

    power = sum(t in POWER_WORDS for t in toks)
    power_s = min(1.0, power / max(3.0, n / 25))
    if power_s > 0.6:
        reasons.append("emotional/high-interest words")

    q = sum(s.text.endswith("?") for s in sents)
    ex = sum(s.text.endswith("!") for s in sents)
    punct_s = min(1.0, (q + ex) / 3)

    i0, i1 = int(start / 0.5), max(int(start / 0.5) + 1, int(end / 0.5))
    seg = energy[i0:i1] if i1 <= len(energy) else energy[i0:]
    energy_s = float(np.clip((seg.mean() + 0.5 * seg.std()) / 1.5, -1, 1)) if len(seg) else 0.0
    if energy_s > 0.4:
        reasons.append("high energy")

    wps = n / max(dur, 1e-3)
    pace_s = 1 - min(1.0, abs(wps - 3.0) / 2.0)

    filler = sum(t in FILLERS for t in toks) / n
    gaps = [b.start - a.end for a, b in zip(sents, sents[1:])]
    dead_air = sum(g for g in gaps if g > 1.2)

    length_s = 1 - min(1.0, abs(dur - target) / target)
    clean_end = 0.5 if sents[-1].text[-1:] in ".?!" else 0.0

    score = (3.0 * hook + 2.0 * power_s + 1.0 * punct_s + 1.5 * energy_s + 1.0 * pace_s
             + 1.0 * length_s + clean_end - 4.0 * filler - 0.4 * dead_air)
    return score, ", ".join(reasons) or "solid self-contained segment"


def make_title(sents: list[Sentence], limit: int = 70) -> str:
    t = sents[0].text.strip()
    return (t[: limit - 1] + "…") if len(t) > limit else t


def pick_local(sentences: list[Sentence], energy: np.ndarray, n: int,
               min_len: float, max_len: float) -> list[Clip]:
    target = (min_len + max_len) / 2
    cands: list[Clip] = []
    for i in range(len(sentences)):
        for j in range(i, len(sentences)):
            dur = sentences[j].end - sentences[i].start
            if dur > max_len:
                break
            if dur >= min_len:
                window = sentences[i:j + 1]
                s, why = score_window(window, energy, target)
                cands.append(Clip(window[0].start, window[-1].end, s, make_title(window), why))
    cands.sort(key=lambda c: c.score, reverse=True)
    return select_non_overlapping(cands, n)


def select_non_overlapping(cands: list[Clip], n: int, buffer: float = 3.0,
                           taken: list[Clip] | None = None) -> list[Clip]:
    picked = list(taken or [])
    for c in cands:
        if len(picked) >= n:
            break
        if all(c.end + buffer <= p.start or c.start >= p.end + buffer for p in picked):
            picked.append(c)
    return picked


# --------------------------------------------------------------------------- #
# 3b. LLM picker (optional, smarter)
# --------------------------------------------------------------------------- #
LLM_PROMPT = """You are a viral short-form video editor (TikTok, YouTube Shorts, Reels).
Below is a timestamped transcript of a long video. Choose the {n} BEST moments to turn into
standalone vertical clips.

Rules:
- Each clip must be between {min_len:.0f} and {max_len:.0f} seconds long.
- Clips must not overlap.
- Each must make sense on its own with no prior context, open with a strong hook in the
  first 3 seconds, and end on a complete thought (a punchline, insight, or payoff).
- Prefer: surprising claims, strong opinions, emotional stories, useful tips, funny moments.
- Use the timestamps exactly as given (seconds).

Reply with ONLY JSON, no other text:
{{"clips": [{{"start": 123.4, "end": 165.2, "title": "short catchy title", "reason": "why it works"}}]}}

Transcript:
{transcript}"""


def _http_json(url: str, headers: dict, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"content-type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def ask_llm(prompt: str) -> str:
    if os.getenv("ANTHROPIC_API_KEY"):
        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        log(f"Asking Claude ({model}) to pick the best moments...")
        data = _http_json("https://api.anthropic.com/v1/messages",
                          {"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                           "anthropic-version": "2023-06-01"},
                          {"model": model, "max_tokens": 2000,
                           "messages": [{"role": "user", "content": prompt}]})
        return "".join(b.get("text", "") for b in data.get("content", []))
    if os.getenv("OPENAI_API_KEY"):
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        log(f"Asking OpenAI ({model}) to pick the best moments...")
        base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        data = _http_json(f"{base}/chat/completions",
                          {"authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                          {"model": model,
                           "messages": [{"role": "user", "content": prompt}]})
        return data["choices"][0]["message"]["content"]
    raise RuntimeError("No ANTHROPIC_API_KEY or OPENAI_API_KEY set")


def pick_llm(sentences: list[Sentence], n: int, min_len: float, max_len: float) -> list[Clip]:
    transcript = "\n".join(f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in sentences)
    transcript = transcript[:400_000]
    raw = ask_llm(LLM_PROMPT.format(n=n, min_len=min_len, max_len=max_len, transcript=transcript))
    m = re.search(r"\{.*\}", raw, re.S)
    items = json.loads(m.group(0))["clips"] if m else []

    clips: list[Clip] = []
    for k, it in enumerate(items):
        try:
            s, e = float(it["start"]), float(it["end"])
        except (KeyError, TypeError, ValueError):
            continue
        # snap to sentence boundaries so we never cut mid-word
        starts = [x for x in sentences if x.start <= s + 1.0]
        st = (starts[-1] if starts else sentences[0]).start
        ends = [x for x in sentences if x.end >= e - 1.0]
        en = (ends[0] if ends else sentences[-1]).end
        if en - st > max_len + 5:
            en = max((x.end for x in sentences if x.end <= st + max_len), default=st + max_len)
        if en - st < min_len * 0.75:
            continue
        clips.append(Clip(st, en, float(n - k), str(it.get("title", ""))[:80],
                          str(it.get("reason", ""))))
    return select_non_overlapping(clips, n, buffer=0)


# --------------------------------------------------------------------------- #
# 4a. Face detection + smooth camera
# --------------------------------------------------------------------------- #
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
_warned_no_detector = False
_yunet_tried = False


def _yunet_model() -> Path | None:
    global _yunet_tried
    path = Path.home() / ".cache" / "autoclipper" / "face_detection_yunet_2023mar.onnx"
    if path.exists() and path.stat().st_size > 100_000:
        return path
    if _yunet_tried:
        return None
    _yunet_tried = True
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        log("Downloading face detector model (one time, ~230 KB)...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(YUNET_URL, tmp)
        if tmp.stat().st_size < 100_000:
            tmp.unlink()
            return None
        tmp.replace(path)
        return path
    except Exception:
        return None


def make_face_detector(w: int, h: int):
    """Return detect(frame_bgr) -> list[(x, y, w, h)], or None if no detector is available."""
    global _warned_no_detector
    import cv2

    if hasattr(cv2, "FaceDetectorYN"):  # modern neural detector, handles angles/lighting well
        model = _yunet_model()
        if model:
            det = cv2.FaceDetectorYN.create(str(model), "", (w, h), 0.6, 0.3, 5000)
            det.setInputSize((w, h))

            def detect_yunet(frame):
                _, faces = det.detect(frame)
                return [] if faces is None else [tuple(f[:4]) for f in faces]
            return detect_yunet

    if hasattr(cv2, "CascadeClassifier"):  # classic fallback (OpenCV 4.x)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                        "haarcascade_frontalface_default.xml")

        def detect_haar(frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return list(cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                 minSize=(int(w * 0.03), int(w * 0.03))))
        return detect_haar

    if not _warned_no_detector:
        log("No face detector available - using center crop.")
        _warned_no_detector = True
    return None


def face_track(video: Path, start: float, dur: float, src_w: int, src_h: int,
               sample_fps: float = 5.0) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (sample_times, face_center_x) for the clip, or None if no faces were found."""
    aw = min(960, src_w)
    ah = int(round(src_h * aw / src_w / 2) * 2)
    detect = make_face_detector(aw, ah)
    if detect is None:
        return None
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(video),
           "-vf", f"fps={sample_fps},scale={aw}:{ah}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    frame_bytes = aw * ah * 3
    scale = src_w / aw
    centers: list[float | None] = []
    prev: float | None = None
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(ah, aw, 3)
        faces = [f for f in detect(frame) if f[2] * f[3] > 0]
        if faces:
            # keep only reasonably big faces (ignore the crowd), then stick with the
            # person we're already on instead of jumping to whoever is slightly bigger
            biggest = max(f[2] * f[3] for f in faces)
            main = [f for f in faces if f[2] * f[3] >= 0.45 * biggest]
            if prev is None:
                f = max(main, key=lambda f: f[2] * f[3])
                pick = (f[0] + f[2] / 2) * scale
            else:
                pick = min(((f[0] + f[2] / 2) * scale for f in main), key=lambda c: abs(c - prev))
            centers.append(pick)
            prev = pick
        else:
            centers.append(None)
    proc.stdout.close()
    proc.wait()

    found = [c for c in centers if c is not None]
    if len(found) < max(2, 0.15 * len(centers)):
        return None

    # fill gaps with the last known position
    arr, last = [], found[0]
    for c in centers:
        last = c if c is not None else last
        arr.append(last)
    arr = np.array(arr, dtype=float)

    # median filter removes one-off false detections
    k = 5
    padded = np.pad(arr, (k // 2, k // 2), mode="edge")
    arr = np.array([np.median(padded[i:i + k]) for i in range(len(arr))])
    return np.arange(len(arr)) / sample_fps, arr


def camera_path(times: np.ndarray, centers: np.ndarray, n_frames: int, fps: float,
                crop_w: int, src_w: int) -> np.ndarray:
    """Per-frame crop left-x for a calm, professional-looking camera.

    - small movements inside a dead zone are ignored (no jitter)
    - medium moves glide with easing and a speed limit (no whip pans)
    - if the subject moves far away and stays there, do a clean hard cut
    """
    t = np.arange(n_frames) / fps
    target = np.interp(t, times, centers)
    dead = crop_w * 0.10
    cut_dist = crop_w * 0.55
    max_step = crop_w * 0.9 / fps  # px per frame
    cam = float(target[0])
    vel = 0.0
    far_frames = 0
    out = np.empty(n_frames)
    for i, tg in enumerate(target):
        d = tg - cam
        if abs(d) > cut_dist:
            far_frames += 1
            if far_frames > 0.6 * fps:  # it's a real move, not a glitch -> cut
                cam, vel, far_frames = float(tg), 0.0, 0
                out[i] = cam
                continue
        else:
            far_frames = 0
        want = 0.0 if abs(d) <= dead else (d - np.sign(d) * dead) * 0.06
        vel += (want - vel) * 0.15  # ease in/out
        vel = float(np.clip(vel, -max_step, max_step))
        cam += vel
        out[i] = cam
    return np.clip(out - crop_w / 2, 0, src_w - crop_w)


# --------------------------------------------------------------------------- #
# 4b. Captions (ASS, word-by-word highlight)
# --------------------------------------------------------------------------- #
def ass_time(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def ass_escape(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")")


# Caption fonts bundled with AutoClipper (files sit next to clipper.py).
# key: (font family name, file name or None for a system font, use bold)
CAPTION_FONTS = {
    "classic": (None, None, True),  # uses the --font / CAPTION_FONT system font
    "montserrat": ("Montserrat Black", "Montserrat_900Black.ttf", False),
    "anton": ("Anton", "Anton_400Regular.ttf", False),
    "bangers": ("Bangers", "Bangers_400Regular.ttf", False),
    "poppins": ("Poppins ExtraBold", "Poppins_800ExtraBold.ttf", False),
    "bebas": ("Bebas Neue", "BebasNeue_400Regular.ttf", False),
    "marker": ("Permanent Marker", "PermanentMarker_400Regular.ttf", False),
}
CAPTION_SIZES = {"s": 72, "m": 92, "l": 116}
DEFAULT_STYLE = {"font": "classic", "color": "#FFFFFF", "highlight": "#FFE500",
                 "background": None, "size": "m", "position": "bottom", "uppercase": True}
HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def clean_style(style: dict | None) -> dict:
    """Merge with defaults and drop anything invalid."""
    s = dict(DEFAULT_STYLE)
    for k, v in (style or {}).items():
        if k == "font" and v in CAPTION_FONTS:
            s[k] = v
        elif k in ("color", "highlight") and isinstance(v, str) and HEX_RE.match(v):
            s[k] = v.upper()
        elif k == "background" and (v is None or (isinstance(v, str) and HEX_RE.match(v))):
            s[k] = v.upper() if v else None
        elif k == "size" and v in CAPTION_SIZES:
            s[k] = v
        elif k == "position" and v in ("bottom", "middle"):
            s[k] = v
        elif k == "uppercase":
            s[k] = bool(v)
    return s


def font_search_dirs() -> list[Path]:
    here = Path(__file__).resolve().parent
    return [here, here / "fonts", here / "static" / "fonts"]


def stage_font(key: str, workdir: Path) -> None:
    """Copy a bundled caption font next to the subtitle file so ffmpeg can find it."""
    fname = CAPTION_FONTS.get(key, (None, None, True))[1]
    if not fname or (workdir / fname).exists():
        return
    for d in font_search_dirs():
        if (d / fname).is_file():
            shutil.copy(d / fname, workdir / fname)
            return
    log(f"  caption font {fname} not found - using the default font")


def ass_color(hex_rgb: str, alpha: int = 0) -> str:
    """'#RRGGBB' -> ASS '&HAABBGGRR' (alpha 0 = solid, 255 = invisible)."""
    r, g, b = hex_rgb[1:3], hex_rgb[3:5], hex_rgb[5:7]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def ass_header(styles: list[str]) -> str:
    return (f"[Script Info]\nScriptType: v4.00+\nPlayResX: {OUT_W}\nPlayResY: {OUT_H}\n"
            "WrapStyle: 0\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
            "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
            "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            + "\n".join(styles) +
            "\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
            "Effect, Text\n")


def write_captions(words: list[Word], clip: Clip, path: Path, font: str,
                   max_words: int = 3, uppercase: bool | None = None,
                   style: dict | None = None, captions: bool = True,
                   watermark: str | None = None) -> None:
    st = clean_style(style)
    if uppercase is not None:
        st["uppercase"] = uppercase
    family, fname, bold = CAPTION_FONTS[st["font"]]
    if fname:
        stage_font(st["font"], path.parent)
    family = family or font
    size = CAPTION_SIZES[st["size"]]
    text_c = ass_color(st["color"])
    hi_c = ass_color(st["highlight"])
    if st["background"]:  # solid box behind the words
        border_style, outline_c, outline, shadow = 3, ass_color(st["background"]), 16, 0
    else:                 # classic black outline
        border_style, outline_c, outline, shadow = 1, "&H00000000", 7, 3
    align, margin_v = (2, 560) if st["position"] == "bottom" else (5, 0)
    styles = [f"Style: Cap,{family},{size},{text_c},{text_c},{outline_c},&H96000000,"
              f"{-1 if bold else 0},0,0,0,100,100,1,0,{border_style},{outline},{shadow},"
              f"{align},80,80,{margin_v},1"]
    if watermark:
        styles.append(f"Style: Mark,{font},44,&H55FFFFFF,&H55FFFFFF,&H99000000,&H00000000,"
                      "-1,0,0,0,100,100,1,0,1,2,0,9,0,50,70,1")

    lines = []
    if captions:
        ws = [Word(w.start - clip.start, w.end - clip.start, w.text) for w in words
              if w.end > clip.start and w.start < clip.end]
        chunks: list[list[Word]] = []
        cur: list[Word] = []
        for w in ws:
            cur.append(w)
            if len(cur) >= max_words or w.text[-1:] in ".,?!;:":
                chunks.append(cur)
                cur = []
        if cur:
            chunks.append(cur)
        dur = clip.duration
        hi = hi_c.replace("&H00", "&H")
        base = text_c.replace("&H00", "&H")
        for ci, chunk in enumerate(chunks):
            chunk_end = chunks[ci + 1][0].start if ci + 1 < len(chunks) else chunk[-1].end + 0.3
            chunk_end = min(chunk_end, chunk[-1].end + 0.6, dur)
            for wi, w in enumerate(chunk):
                start = max(0.0, w.start)
                end = chunk[wi + 1].start if wi + 1 < len(chunk) else chunk_end
                if end <= start:
                    continue
                parts = []
                for k, x in enumerate(chunk):
                    t = ass_escape(x.text.upper() if st["uppercase"] else x.text)
                    if k == wi:  # current word: highlight colour + slight pop
                        parts.append(rf"{{\c{hi}&\fscx112\fscy112}}{t}{{\c{base}&\fscx100\fscy100}}")
                    else:
                        parts.append(t)
                lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Cap,,0,0,0,,"
                             f"{' '.join(parts)}")
    if watermark:
        lines.append(f"Dialogue: 1,{ass_time(0)},{ass_time(clip.duration + 1)},Mark,,0,0,0,,"
                     f"{ass_escape(watermark)}")
    path.write_text(ass_header(styles) + "\n".join(lines) + "\n", encoding="utf-8")


def has_audio(path: Path) -> bool:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                          "stream=index", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout
    return bool(out.strip())


def add_end_card(clip_path: Path, workdir: Path, title: str, subtitle: str = "",
                 seconds: float = 2.0, crf: int = 20, preset: str = "veryfast") -> None:
    """Append a short branded card ("Made with AutoClipper") to the end of a clip."""
    stage_font("anton", workdir)
    ass = workdir / "endcard.ass"
    styles = [
        "Style: Small,Liberation Sans,54,&H00D8AEB9,&H00D8AEB9,&H00000000,&H00000000,"
        "-1,0,0,0,100,100,1,0,1,0,0,5,80,80,0,1",
        "Style: Big,Anton,170,&H0000E5FF,&H0000E5FF,&H00000000,&H00000000,"
        "0,0,0,0,100,100,2,0,1,0,0,5,60,60,0,1",
    ]
    end = ass_time(seconds + 1)
    lines = [
        rf"Dialogue: 0,0:00:00.00,{end},Small,,0,0,0,,{{\pos({OUT_W // 2},{OUT_H // 2 - 170})\fad(150,0)}}Made with",
        rf"Dialogue: 0,0:00:00.00,{end},Big,,0,0,0,,{{\pos({OUT_W // 2},{OUT_H // 2})\fad(150,0)}}{ass_escape(title.upper())}",
    ]
    if subtitle:
        lines.append(rf"Dialogue: 0,0:00:00.00,{end},Small,,0,0,0,,{{\pos({OUT_W // 2},{OUT_H // 2 + 170})\fad(150,0)}}{ass_escape(subtitle)}")
    ass.write_text(ass_header(styles) + "\n".join(lines) + "\n", encoding="utf-8")

    fps = probe(clip_path)["fps"]
    tmp = clip_path.with_name(clip_path.stem + "_end.mp4")
    audio = has_audio(clip_path)
    card = (f"color=c=0x1E1433:s={OUT_W}x{OUT_H}:r={fps:g}:d={seconds}")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(clip_path.resolve()),
           "-f", "lavfi", "-i", card]
    if audio:
        cmd += ["-f", "lavfi", "-t", f"{seconds}", "-i", "anullsrc=r=48000:cl=stereo"]
        graph = (f"[1:v]subtitles={ass.name}:fontsdir=.,setsar=1,format=yuv420p[e];"
                 "[0:v]setsar=1,format=yuv420p[m];[0:a]aresample=48000,"
                 "aformat=channel_layouts=stereo[ma];[m][ma][e][2:a]concat=n=2:v=1:a=1[v][a]")
        maps = ["-map", "[v]", "-map", "[a]", "-c:a", "aac", "-b:a", "160k"]
    else:
        graph = (f"[1:v]subtitles={ass.name}:fontsdir=.,setsar=1,format=yuv420p[e];"
                 "[0:v]setsar=1,format=yuv420p[m];[m][e]concat=n=2:v=1:a=0[v]")
        maps = ["-map", "[v]"]
    run(cmd + ["-filter_complex", graph, *maps, "-c:v", "libx264", "-threads", FFMPEG_THREADS,
               "-preset", preset,
               "-crf", str(crf), "-pix_fmt", "yuv420p", "-r", f"{fps:g}",
               "-movflags", "+faststart", str(tmp.resolve())], cwd=str(workdir))
    tmp.replace(clip_path)


# --------------------------------------------------------------------------- #
# 4c. Render
# --------------------------------------------------------------------------- #
FFMPEG_THREADS = os.getenv("FFMPEG_THREADS", "2")  # fewer threads = less memory


def encode_args(out: Path, fps: float, crf: int, preset: str) -> list[str]:
    return ["-r", f"{fps:g}", "-c:v", "libx264", "-threads", FFMPEG_THREADS,
            "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-g", str(int(round(fps * 2))),
            "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
            "-movflags", "+faststart", "-shortest", str(out)]


def render_clip(video: Path, info: dict, clip: Clip, ass: Path, out: Path,
                layout: str, crf: int, preset: str, max_fps: float = 30,
                on_frame: Callable[[float], None] | None = None) -> None:
    src_w, src_h = info["width"], info["height"]
    # steady, phone-friendly frame rate: never above the source or max_fps
    fps = min(info["fps"], max_fps)
    fps = 30.0 if fps < 10 else round(fps, 3)
    start, dur = clip.start, clip.duration
    # run with cwd=ass.parent -> no path-escaping issues; caption fonts are staged there
    subs = f"subtitles={ass.name}:fontsdir=."
    seek = ["-ss", f"{start:.3f}", "-t", f"{dur:.3f}"]
    cwd = str(ass.parent)
    src = str(video.resolve())

    already_vertical = src_w / src_h <= 9 / 16 + 0.01
    crop_w = int(src_h * 9 / 16) // 2 * 2
    track = None
    if not already_vertical and layout == "face":
        track = face_track(video, start, dur, src_w, src_h)
        if track is None and not _warned_no_detector:
            log("  no face found - using center crop")

    if track is None:
        if already_vertical:
            vf = (f"fps={fps},scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
                  f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2,setsar=1,{subs}")
        elif layout == "blur":  # fit full frame, blurred copy behind it
            vf = (f"fps={fps},split[a][b];[a]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
                  f"crop={OUT_W}:{OUT_H},boxblur=30:3,eq=brightness=-0.08[bg];"
                  f"[b]scale={OUT_W}:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,{subs}")
        else:
            vf = (f"fps={fps},crop={crop_w}:{src_h}:(iw-{crop_w})/2:0,"
                  f"scale={OUT_W}:{OUT_H},setsar=1,{subs}")
        run(["ffmpeg", "-y", "-v", "error", *seek, "-i", src, "-filter_complex", vf,
             *encode_args(out.resolve(), fps, crf, preset)], cwd=cwd)
        if on_frame:
            on_frame(1.0)
        return

    # Face-tracked: decode at a constant frame rate -> crop in Python -> one encode
    n_frames = int(round(dur * fps))
    lefts = camera_path(*track, n_frames + int(fps), fps, crop_w, src_w)
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-threads", FFMPEG_THREADS, *seek, "-i", src,
                            "-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                           stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error",
                            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{crop_w}x{src_h}",
                            "-framerate", f"{fps:g}", "-i", "-",
                            *seek, "-i", src,
                            "-map", "0:v", "-map", "1:a?",
                            "-vf", f"scale={OUT_W}:{OUT_H}:flags=bicubic,setsar=1,{subs}",
                            *encode_args(out.resolve(), fps, crf, preset)],
                           stdin=subprocess.PIPE, cwd=cwd)
    fb = src_w * src_h * 3
    i = 0
    try:
        while True:
            buf = dec.stdout.read(fb)
            if len(buf) < fb:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(src_h, src_w, 3)
            x = int(round(lefts[min(i, len(lefts) - 1)]))
            enc.stdin.write(np.ascontiguousarray(frame[:, x:x + crop_w]).tobytes())
            i += 1
            if on_frame and i % 30 == 0:
                on_frame(min(1.0, i / max(1, n_frames)))
    finally:
        enc.stdin.close()
        dec.stdout.close()
        dec.wait()
        if enc.wait() != 0:
            raise ClipperError("Video encoding failed.")


# --------------------------------------------------------------------------- #
# Pipeline (used by both the command line and the website)
# --------------------------------------------------------------------------- #
def fmt(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def run_pipeline(source: str, out_root: Path, *, clips: int = 4, min_len: float = 20,
                 max_len: float = 60, picker: str = "auto", layout: str = "face",
                 model: str = "small", language: str | None = None, font: str = "Arial",
                 uppercase: bool = True, captions: bool = True, crf: int = 20,
                 preset: str = "veryfast", max_fps: float = 30,
                 cookies_browser: str | None = None, allow_local: bool = True,
                 max_minutes: float | None = None, flat: bool = False,
                 manual_starts: list[float] | None = None, manual_len: float = 10,
                 caption_style: dict | None = None, watermark: str | None = None,
                 end_card: tuple[str, ...] | None = None,
                 progress: ProgressFn | None = None) -> Path:
    """Run everything. Returns the folder holding clip_N.mp4 files and clips.json.

    manual_starts: if given, skip the automatic picker and make one clip of
    manual_len seconds starting at each of these times (used by the free plan).
    caption_style: font/colours/background/size/position (see DEFAULT_STYLE).
    watermark: small text in the top corner of every clip (free plan).
    end_card: (title, subtitle) for a 2-second branded card at the end (free plan).
    """
    def report(stage: str, frac: float, msg: str) -> None:
        if progress:
            progress(stage, frac, msg)

    require_ffmpeg()
    out_root = Path(out_root)
    work = out_root / ".work"
    work.mkdir(parents=True, exist_ok=True)

    report("download", 0.0, "Getting the video")
    video, title = download(source, work, cookies_browser, progress, allow_local)
    info = probe(video)
    if max_minutes and info["duration"] > max_minutes * 60:
        raise ClipperError(f"Video is {fmt(info['duration'])} long - the limit is "
                           f"{max_minutes:g} minutes.")
    log(f"Video: '{title}' {info['width']}x{info['height']} {info['fps']:.2f}fps "
        f"{fmt(info['duration'])}")
    report("download", 1.0, "Video ready")
    dest = out_root if flat else out_root / slugify(title)
    dest.mkdir(parents=True, exist_ok=True)

    words: list[Word] = []
    if manual_starts is not None:
        # The person chose the moments: no need to listen to the whole video.
        if info["duration"] < 1:
            raise ClipperError("This video is too short to clip.")
        picked = []
        for s in manual_starts:
            s = max(0.0, min(float(s), max(0.0, info["duration"] - manual_len)))
            picked.append(Clip(s, min(s + manual_len, info["duration"]), 0.0,
                               f"Clip at {fmt(s)}", "picked by you"))
        if not picked:
            raise ClipperError("Pick at least one moment to clip.")
    else:
        report("transcribe", 0.0, "Listening to the video")
        wav = work / f"{slugify(title)}.wav"
        extract_audio(video, wav)
        words = transcribe(wav, model, language, work / f"{slugify(title)}.words.json",
                           info["duration"], progress)
        if not words:
            raise ClipperError("No speech found in this video - nothing to clip.")
        sentences = build_sentences(words)
        energy = loudness_profile(wav)

        report("pick", 0.0, "Finding the best moments")
        picked = []
        use_llm = picker == "llm" or (
            picker == "auto" and (os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")))
        if use_llm:
            try:
                picked = pick_llm(sentences, clips, min_len, max_len)
            except Exception as e:  # never fail the whole run because of the LLM
                log(f"LLM picker failed ({e}); falling back to local scoring.")
        if len(picked) < clips:
            local = pick_local(sentences, energy, clips * 5, min_len, max_len)
            picked = select_non_overlapping(local, clips, taken=picked)
        if not picked:
            raise ClipperError("Couldn't find any moments long enough - the video may be too "
                               "short or have too little talking.")
        if len(picked) < clips:
            log(f"Only found {len(picked)} good non-overlapping moments (video may be short).")
        report("pick", 1.0, f"Found {len(picked)} moments")

    meta = []
    for k, clip in enumerate(picked, 1):
        name = f"clip_{k}"
        log(f"Rendering {name}: {fmt(clip.start)}-{fmt(clip.end)} ({clip.duration:.0f}s) "
            f"- {clip.title}")
        base = (k - 1) / len(picked)
        report("render", base, f"Making clip {k} of {len(picked)}")
        ass = work / f"{name}.ass"
        if captions and manual_starts is not None:
            words += transcribe_segment(video, clip.start, clip.duration, model, language,
                                        work / f"{name}.wav")
        style = dict(caption_style or {})
        style.setdefault("uppercase", uppercase)
        write_captions(words, clip, ass, font, style=style, captions=captions,
                       watermark=watermark)
        out = dest / f"{name}.mp4"
        render_clip(video, info, clip, ass, out, layout, crf, preset, max_fps,
                    on_frame=lambda f, k=k, base=base: report(
                        "render", base + f / len(picked), f"Making clip {k} of {len(picked)}"))
        length = clip.duration
        if end_card:
            add_end_card(out, work, end_card[0], end_card[1] if len(end_card) > 1 else "",
                         crf=crf, preset=preset)
            length += 2.0
        text = " ".join(w.text for w in words if clip.start <= w.start < clip.end)
        meta.append({"file": out.name, "start": round(clip.start, 2), "end": round(clip.end, 2),
                     "duration": round(length, 1), "title": clip.title,
                     "why": clip.reason, "transcript": text})

    (dest / "clips.json").write_text(json.dumps({"source": source, "title": title,
                                                 "clips": meta}, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    report("done", 1.0, "Done")
    log(f"Done! {len(meta)} clip(s) saved to {dest.resolve()}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description="Turn a long video into short vertical clips.")
    ap.add_argument("source", help="Video URL (YouTube, etc.) or a local video file")
    ap.add_argument("-n", "--clips", type=int, default=4, help="number of clips (default 4)")
    ap.add_argument("--min", dest="min_len", type=float, default=20, help="min clip seconds")
    ap.add_argument("--max", dest="max_len", type=float, default=60, help="max clip seconds")
    ap.add_argument("-o", "--out", default="clips", help="output folder (default ./clips)")
    ap.add_argument("--picker", choices=["auto", "local", "llm"], default="auto",
                    help="auto = LLM if an API key is set, otherwise local scoring")
    ap.add_argument("--layout", choices=["face", "center", "blur"], default="face",
                    help="face = follow the speaker; center = center crop; "
                         "blur = whole frame on a blurred background (good for screen/gameplay)")
    ap.add_argument("--model", default="small",
                    help="Whisper model: tiny, base, small, medium, large-v3 (default small)")
    ap.add_argument("--language", default=None, help="force language code, e.g. en, tl, es")
    ap.add_argument("--font", default="Arial", help="caption font (default Arial)")
    ap.add_argument("--no-upper", action="store_true", help="don't uppercase captions")
    ap.add_argument("--no-captions", action="store_true", help="skip burned-in captions")
    ap.add_argument("--fps", type=float, default=30, help="max output frame rate (default 30)")
    ap.add_argument("--crf", type=int, default=20, help="quality (lower = better, default 20)")
    ap.add_argument("--preset", default="veryfast", help="x264 preset (default veryfast)")
    ap.add_argument("--cookies-from-browser", default=None,
                    help="pass browser cookies to yt-dlp (chrome, firefox, edge...)")
    ap.add_argument("--at", action="append", default=None, metavar="TIME",
                    help="make a clip starting here instead of auto-picking, e.g. --at 1:23 "
                         "(repeat for more clips)")
    ap.add_argument("--length", type=float, default=10,
                    help="clip length in seconds when using --at (default 10)")
    a = ap.parse_args()

    def secs(t: str) -> float:
        parts = [float(x) for x in t.split(":")]
        return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))

    try:
        starts = [secs(t) for t in a.at] if a.at else None
    except ValueError:
        sys.exit("[autoclipper] --at times look like 1:23 or 83")
    try:
        run_pipeline(a.source, Path(a.out), clips=a.clips, min_len=a.min_len,
                     manual_starts=starts, manual_len=a.length,
                     max_len=a.max_len, picker=a.picker, layout=a.layout, model=a.model,
                     language=a.language, font=a.font, uppercase=not a.no_upper,
                     captions=not a.no_captions, crf=a.crf, preset=a.preset, max_fps=a.fps,
                     cookies_browser=a.cookies_from_browser)
    except ClipperError as e:
        sys.exit(f"[autoclipper] {e}")


if __name__ == "__main__":
    main()
