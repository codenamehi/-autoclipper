#!/usr/bin/env python3
"""
AutoClipper - turn one long video link into short, vertical, captioned clips.

    python clipper.py "https://www.youtube.com/watch?v=..."

Pipeline:
    1. Download the video (yt-dlp)            - or use a local file path
    2. Transcribe with word timings (faster-whisper, runs locally)
    3. Score every 20-60s window and pick the best N (default 4)
       - local scoring: hooks, questions, emotion words, energy, pace
       - or an LLM picks (set ANTHROPIC_API_KEY or OPENAI_API_KEY + --picker llm)
    4. Render each clip as 1080x1920: face-tracked crop + word-by-word captions
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import wave
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

OUT_W, OUT_H = 1080, 1920


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
            sys.exit(f"'{tool}' not found. Install ffmpeg and make sure it's on your PATH "
                     "(see README).")


def slugify(text: str, limit: int = 60) -> str:
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    return (re.sub(r"[\s_-]+", "-", text)[:limit].strip("-")) or "video"


def probe(path: Path) -> dict:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate:format=duration",
        "-of", "json", str(path)])
    info = json.loads(out)
    st = info["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    if not (1 <= fps <= 120):
        fps = 30.0
    return {"width": int(st["width"]), "height": int(st["height"]),
            "fps": fps, "duration": float(info["format"]["duration"])}


# --------------------------------------------------------------------------- #
# 1. Download
# --------------------------------------------------------------------------- #
def download(source: str, workdir: Path, cookies_browser: str | None) -> tuple[Path, str]:
    if Path(source).is_file():
        p = Path(source)
        return p, p.stem
    try:
        import yt_dlp
    except ImportError:
        sys.exit("yt-dlp is not installed: pip install -r requirements.txt")

    log("Downloading video...")
    opts = {
        "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(workdir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    if cookies_browser:
        opts["cookiesfrombrowser"] = (cookies_browser,)
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(source, download=True)
        title = info.get("title") or "video"
    files = sorted(workdir.glob("source.*"), key=lambda p: p.stat().st_size, reverse=True)
    if not files:
        sys.exit("Download failed - no file was produced.")
    return files[0], title


# --------------------------------------------------------------------------- #
# 2. Transcribe
# --------------------------------------------------------------------------- #
def extract_audio(video: Path, wav: Path) -> None:
    run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1",
         "-ar", "16000", "-c:a", "pcm_s16le", str(wav)])


def transcribe(wav: Path, model_size: str, language: str | None, cache: Path) -> list[Word]:
    if cache.exists():
        log("Using cached transcript.")
        return [Word(**w) for w in json.loads(cache.read_text(encoding="utf-8"))]
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("faster-whisper is not installed: pip install -r requirements.txt")

    log(f"Transcribing with Whisper '{model_size}' (first run downloads the model)...")
    model = WhisperModel(model_size, device="auto", compute_type="auto")
    segments, info = model.transcribe(str(wav), language=language, word_timestamps=True,
                                      vad_filter=True)
    words: list[Word] = []
    for seg in segments:
        for w in seg.words or []:
            t = w.word.strip()
            if t:
                words.append(Word(round(w.start, 3), round(w.end, 3), t))
        print(f"\r  transcribed {seg.end:7.1f}s", end="", flush=True)
    print()
    cache.write_text(json.dumps([asdict(w) for w in words]), encoding="utf-8")
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
    with wave.open(str(wav), "rb") as wf:
        sr = wf.getframerate()
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)
    hop = int(sr * step)
    n = max(1, len(data) // hop)
    rms = np.sqrt(np.mean(data[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-9)
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
# 4a. Face tracking -> crop path
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
            gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            return list(cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6,
                                                 minSize=(int(w * 0.05), int(w * 0.05))))
        return detect_haar

    if not _warned_no_detector:
        log("No face detector available - using center crop.")
        _warned_no_detector = True
    return None


def face_track(video: Path, start: float, dur: float, src_w: int, src_h: int,
               crop_w: int, sample_fps: float = 3.0) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (times, crop_left_x) for the clip, or None if no faces were found."""
    import cv2

    aw = 640
    ah = int(round(src_h * aw / src_w / 2) * 2)
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(video),
           "-vf", f"fps={sample_fps},scale={aw}:{ah}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    detect = make_face_detector(aw, ah)
    if detect is None:
        return None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    frame_bytes = aw * ah * 3
    centers: list[float | None] = []
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(ah, aw, 3)
        faces = detect(frame)
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            centers.append((x + w / 2) * src_w / aw)
        else:
            centers.append(None)
    proc.stdout.close()
    proc.wait()

    found = [c for c in centers if c is not None]
    if len(found) < max(2, 0.15 * len(centers)):
        return None

    # fill gaps by carrying nearest detection
    filled = np.array(centers, dtype=object)
    last = found[0]
    for i, c in enumerate(filled):
        last = c if c is not None else last
        filled[i] = last
    arr = filled.astype(float)

    # median filter kills one-frame jumps
    k = 5
    padded = np.pad(arr, (k // 2, k // 2), mode="edge")
    arr = np.array([np.median(padded[i:i + k]) for i in range(len(arr))])

    # "lazy camera": stay still until the face drifts out of a dead zone
    dead = crop_w * 0.12
    cam = arr[0]
    path = []
    for t in arr:
        if abs(t - cam) > dead:
            cam += (t - cam) * 0.3
        path.append(cam)
    left = np.clip(np.array(path) - crop_w / 2, 0, src_w - crop_w)
    times = np.arange(len(left)) / sample_fps
    return times, left


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


def write_captions(words: list[Word], clip: Clip, path: Path, font: str,
                   max_words: int = 3, uppercase: bool = True) -> None:
    ws = [Word(w.start - clip.start, w.end - clip.start, w.text) for w in words
          if w.end > clip.start and w.start < clip.end]
    # chunk into short lines that break on punctuation
    chunks: list[list[Word]] = []
    cur: list[Word] = []
    for w in ws:
        cur.append(w)
        if len(cur) >= max_words or w.text[-1:] in ".,?!;:":
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},92,&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,1,0,1,7,3,2,80,80,560,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    dur = clip.duration
    for ci, chunk in enumerate(chunks):
        chunk_end = chunks[ci + 1][0].start if ci + 1 < len(chunks) else chunk[-1].end + 0.3
        chunk_end = min(chunk_end, chunk[-1].end + 0.6, dur)
        for wi, w in enumerate(chunk):
            st = max(0.0, w.start)
            en = chunk[wi + 1].start if wi + 1 < len(chunk) else chunk_end
            if en <= st:
                continue
            parts = []
            for k, x in enumerate(chunk):
                t = ass_escape(x.text.upper() if uppercase else x.text)
                if k == wi:  # current word: yellow + slight pop
                    parts.append(r"{\c&H00E5FF&\fscx112\fscy112}" + t + r"{\c&HFFFFFF&\fscx100\fscy100}")
                else:
                    parts.append(t)
            lines.append(f"Dialogue: 0,{ass_time(st)},{ass_time(en)},Cap,,0,0,0,,{' '.join(parts)}")
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# 4c. Render
# --------------------------------------------------------------------------- #
def encode_args(out: Path, crf: int, preset: str) -> list[str]:
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-shortest", str(out)]


def render_clip(video: Path, info: dict, clip: Clip, ass: Path, out: Path,
                layout: str, crf: int, preset: str) -> None:
    src_w, src_h, fps = info["width"], info["height"], info["fps"]
    start, dur = clip.start, clip.duration
    subs = f"subtitles={ass.name}"  # run with cwd=ass.parent -> no path-escaping issues
    seek = ["-ss", f"{start:.3f}", "-t", f"{dur:.3f}"]
    cwd = str(ass.parent)

    already_vertical = src_w / src_h <= 9 / 16 + 0.01
    if already_vertical or layout == "blur":
        if already_vertical:
            vf = (f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
                  f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2,setsar=1,{subs}")
        else:  # fit full frame, blurred copy behind it
            vf = (f"split[a][b];[a]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
                  f"crop={OUT_W}:{OUT_H},boxblur=30:3,eq=brightness=-0.08[bg];"
                  f"[b]scale={OUT_W}:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,{subs}")
        run(["ffmpeg", "-y", "-v", "error", *seek, "-i", str(video.resolve()),
             "-filter_complex" if layout == "blur" and not already_vertical else "-vf", vf,
             *encode_args(out.resolve(), crf, preset)], cwd=cwd)
        return

    crop_w = int(src_h * 9 / 16) // 2 * 2
    track = face_track(video, start, dur, src_w, src_h, crop_w) if layout == "face" else None
    if track is None:
        if layout == "face" and not _warned_no_detector:
            log("  no face found - using center crop")
        vf = f"crop={crop_w}:{src_h}:(iw-{crop_w})/2:0,scale={OUT_W}:{OUT_H},setsar=1,{subs}"
        run(["ffmpeg", "-y", "-v", "error", *seek, "-i", str(video.resolve()), "-vf", vf,
             *encode_args(out.resolve(), crf, preset)], cwd=cwd)
        return

    # Face-tracked: decode frames -> crop in Python -> pipe into encoder (single encode)
    times, lefts = track
    dec = subprocess.Popen(["ffmpeg", "-v", "error", *seek, "-i", str(video),
                            "-f", "rawvideo", "-pix_fmt", "bgr24", "-r", f"{fps:.5f}", "-"],
                           stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error",
                            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{crop_w}x{src_h}",
                            "-r", f"{fps:.5f}", "-i", "-",
                            *seek, "-i", str(video.resolve()),
                            "-map", "0:v", "-map", "1:a?",
                            "-vf", f"scale={OUT_W}:{OUT_H},setsar=1,{subs}",
                            *encode_args(out.resolve(), crf, preset)],
                           stdin=subprocess.PIPE, cwd=cwd)
    fb = src_w * src_h * 3
    i = 0
    try:
        while True:
            buf = dec.stdout.read(fb)
            if len(buf) < fb:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(src_h, src_w, 3)
            x = int(np.interp(i / fps, times, lefts)) // 2 * 2
            enc.stdin.write(np.ascontiguousarray(frame[:, x:x + crop_w]).tobytes())
            i += 1
    finally:
        enc.stdin.close()
        dec.wait()
        if enc.wait() != 0:
            raise RuntimeError("ffmpeg encoding failed")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def fmt(t: float) -> str:
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


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
    ap.add_argument("--crf", type=int, default=20, help="quality (lower = better, default 20)")
    ap.add_argument("--preset", default="veryfast", help="x264 preset (default veryfast)")
    ap.add_argument("--cookies-from-browser", default=None,
                    help="pass browser cookies to yt-dlp (chrome, firefox, edge...)")
    args = ap.parse_args()

    require_ffmpeg()
    out_root = Path(args.out)
    work = out_root / ".work"
    work.mkdir(parents=True, exist_ok=True)

    video, title = download(args.source, work, args.cookies_from_browser)
    info = probe(video)
    log(f"Video: '{title}' {info['width']}x{info['height']} {fmt(info['duration'])}")
    dest = out_root / slugify(title)
    dest.mkdir(parents=True, exist_ok=True)

    wav = work / f"{slugify(title)}.wav"
    extract_audio(video, wav)
    words = transcribe(wav, args.model, args.language, work / f"{slugify(title)}.words.json")
    if not words:
        sys.exit("No speech found in this video - nothing to clip.")
    sentences = build_sentences(words)
    energy = loudness_profile(wav)

    clips: list[Clip] = []
    use_llm = args.picker == "llm" or (
        args.picker == "auto" and (os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")))
    if use_llm:
        try:
            clips = pick_llm(sentences, args.clips, args.min_len, args.max_len)
        except Exception as e:  # never fail the whole run because of the LLM
            log(f"LLM picker failed ({e}); falling back to local scoring.")
    if len(clips) < args.clips:
        if not use_llm:
            log("Scoring moments locally...")
        local = pick_local(sentences, energy, args.clips * 5, args.min_len, args.max_len)
        clips = select_non_overlapping(local, args.clips, taken=clips)
    if not clips:
        sys.exit("Couldn't find any segments long enough. Try a lower --min.")
    if len(clips) < args.clips:
        log(f"Only found {len(clips)} good non-overlapping moments (video may be short).")

    meta = []
    for k, clip in enumerate(clips, 1):
        name = f"clip_{k}"
        log(f"Rendering {name}: {fmt(clip.start)}-{fmt(clip.end)} ({clip.duration:.0f}s) "
            f"- {clip.title}")
        ass = work / f"{name}.ass"
        if args.no_captions:
            ass.write_text("[Script Info]\nScriptType: v4.00+\n\n[Events]\n", encoding="utf-8")
        else:
            write_captions(words, clip, ass, args.font, uppercase=not args.no_upper)
        out = dest / f"{name}.mp4"
        render_clip(video, info, clip, ass, out, args.layout, args.crf, args.preset)
        text = " ".join(w.text for w in words if clip.start <= w.start < clip.end)
        meta.append({"file": out.name, "start": round(clip.start, 2), "end": round(clip.end, 2),
                     "duration": round(clip.duration, 1), "title": clip.title,
                     "why": clip.reason, "transcript": text})

    (dest / "clips.json").write_text(json.dumps({"source": args.source, "title": title,
                                                 "clips": meta}, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    log(f"Done! {len(meta)} clip(s) saved to {dest.resolve()}")


if __name__ == "__main__":
    main()
