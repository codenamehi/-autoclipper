# AutoClipper ✂️

Paste a video link → get **4 vertical, captioned short clips** ready for TikTok, YouTube Shorts and Reels. A free, open-source take on tools like Opus Clip.

```bash
python clipper.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

```
clips/
└── my-video-title/
    ├── clip_1.mp4      ← best moment
    ├── clip_2.mp4
    ├── clip_3.mp4
    ├── clip_4.mp4
    └── clips.json      ← timestamps, titles, why each was picked, transcript
```

## What it does

1. **Downloads** the video with [yt-dlp](https://github.com/yt-dlp/yt-dlp) (YouTube, Vimeo, X, TikTok and [1000+ sites](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)) — or use a local file.
2. **Transcribes** it locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), with timings for every word. No API key needed.
3. **Finds the best moments.** Every 20–60 second window that starts and ends on a full sentence is scored on:
   - a strong hook in the first line ("Here's the secret…", "Why…?", "Most people…")
   - emotional / high-interest words, questions and exclamations
   - vocal energy (loudness) and speaking pace
   - penalties for filler words ("um", "uh") and dead air

   The top 4 non-overlapping windows win.
   **Smarter option:** set a Claude or OpenAI API key and an LLM reads the transcript and picks the moments like a human editor would.
4. **Renders** each clip at 1080×1920:
   - crop that **follows the speaker's face** (with a smooth "lazy camera" so it doesn't jitter)
   - bold, **word-by-word captions** with the current word highlighted

## Install

Needs **Python 3.9+** and **ffmpeg**.

```bash
git clone https://github.com/YOUR_USERNAME/autoclipper.git
cd autoclipper
pip install -r requirements.txt
```

Install ffmpeg:
- **Windows:** `winget install ffmpeg` (then reopen your terminal)
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

## Usage

```bash
# 4 clips (default)
python clipper.py "https://youtu.be/VIDEO_ID"

# 6 clips, 30–45 seconds each
python clipper.py "https://youtu.be/VIDEO_ID" -n 6 --min 30 --max 45

# local file, screen recording/gameplay (whole frame on a blurred background)
python clipper.py my_stream.mp4 --layout blur

# non-English video, more accurate model
python clipper.py "https://youtu.be/VIDEO_ID" --language tl --model medium
```

| Option | Default | What it does |
|---|---|---|
| `-n`, `--clips` | `4` | Number of clips |
| `--min` / `--max` | `20` / `60` | Clip length range in seconds |
| `--layout` | `face` | `face` follows the speaker · `center` plain center crop · `blur` full frame on blurred background |
| `--picker` | `auto` | `auto` uses an LLM if a key is set, else local scoring · `local` · `llm` |
| `--model` | `small` | Whisper size: `tiny` (fastest) → `large-v3` (most accurate) |
| `--language` | auto | Force a language, e.g. `en`, `tl`, `es` |
| `--font` | `Arial` | Caption font (any font installed on your system) |
| `--no-upper` | | Keep caption casing instead of ALL CAPS |
| `--no-captions` | | Skip captions |
| `-o`, `--out` | `clips` | Output folder |
| `--cookies-from-browser` | | e.g. `chrome` — helps when YouTube asks you to sign in |

### Smarter clip picking with an LLM (optional)

```bash
# macOS / Linux
export ANTHROPIC_API_KEY=sk-ant-...      # or: export OPENAI_API_KEY=sk-...
# Windows PowerShell
$env:ANTHROPIC_API_KEY="sk-ant-..."

python clipper.py "https://youtu.be/VIDEO_ID"
```

Override the model with `ANTHROPIC_MODEL` or `OPENAI_MODEL`. If the API call fails for any reason, it falls back to local scoring automatically. Cost is usually a few cents per video.

## Run it on GitHub (no computer needed)

This repo includes a GitHub Action, so you can make clips from your phone:

1. Push this repo to GitHub.
2. Go to **Actions → Make clips → Run workflow**.
3. Paste the link, hit **Run**.
4. When it finishes (~5–20 min), download **clips** from the run page.

To use LLM picking there, add `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` under **Settings → Secrets and variables → Actions**.

> ⚠️ YouTube often blocks downloads from GitHub's servers ("Sign in to confirm you're not a bot"). If that happens, run it on your own computer instead — that almost always works.

## Speed

On a normal laptop CPU with the `small` model, transcription runs roughly 3–5× faster than real time (a 30-min video ≈ 6–10 min). With an NVIDIA GPU it's much faster. Use `--model tiny` or `base` if it's too slow. Transcripts are cached in `clips/.work`, so re-running the same video skips that step.

## Troubleshooting

- **`HTTP Error 403` / download fails** → `pip install -U yt-dlp` (sites change often), or add `--cookies-from-browser chrome`.
- **Captions in the wrong font** → pass a font you have installed, e.g. `--font "Montserrat"`.
- **Crop doesn't follow the person** → happens with side profiles or tiny faces; try `--layout center` or `--layout blur`.
- **No clips found** → the video may be too short or have little speech; lower `--min`.

## Notes

Only clip videos you own or have permission to use. Respect each platform's terms of service.

## License

MIT
