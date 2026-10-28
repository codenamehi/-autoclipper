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
   - crop that **follows the speaker's face** with a smooth camera (no jitter, clean cuts when the speaker changes)
   - steady 30 fps output that plays smoothly on phones
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
| `--fps` | `30` | Max output frame rate |
| `--at` | | Clip from this time instead of auto-picking, e.g. `--at 1:23` (repeat for more) |
| `--length` | `10` | Clip length in seconds when using `--at` |
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

## Run it as a website

AutoClipper includes a website with accounts and two plans:

| | Free | Pro |
|---|---|---|
| Choosing moments | You pick them on a video player | AutoClipper finds the best ones |
| Clip length | 10 seconds | 20–60 seconds |
| Limit | 4 clips per 24 hours | No daily limit |
| Links | Upload files only | Upload files or paste links |
| Captions and framing | Yes | Yes |
| Caption style | Standard | Choose font, colours, background box, size and position (with live preview) |
| Watermark | Small corner mark + 2-second "Made with AutoClipper" end card | None |

Everyone starts on Free. If a video fails, its free clips aren't used up.

```bash
pip install -r requirements.txt
ADMIN_PASSWORD=choose-a-password python server.py
```

Open http://localhost:8000. Each person only sees their own clips, and everything is deleted after 24 hours.

### Upgrading someone to Pro

- Open **/admin** (for example `http://localhost:8000/admin`), log in with any username and your `ADMIN_PASSWORD`, and click **Pro** next to their email. The admin page is switched off until you set `ADMIN_PASSWORD`.
- Or from the command line: `python server.py set-plan someone@example.com pro`

To take payments, set `UPGRADE_URL` to a payment page (a Stripe Payment Link works). An **Upgrade to Pro** button then appears for free users. When someone pays, upgrade them on the admin page. Upgrading automatically after payment needs a Stripe webhook, which can be added later.

### Put it online

GitHub stores the code but **can't run the website**. You need a server that runs Python and ffmpeg. The included `Dockerfile` works on most hosts:

- **Railway or Render:** connect your GitHub repo and they build the Dockerfile for you. Pick a plan with at least **4 GB RAM**, add a persistent disk mounted at `/data`, and set `ADMIN_PASSWORD`.
- **Your own cloud server (VPS)** from Hetzner, DigitalOcean, etc. Use at least 4 CPU cores and 8 GB RAM, then:
  ```bash
  docker build -t autoclipper .
  docker run -d -p 80:8000 -v autoclipper-data:/data -e ADMIN_PASSWORD=choose-a-password \
    --restart unless-stopped autoclipper
  ```

Once your site has https, set `SECURE_COOKIES=true`.

Free clips are quick: only the 10 seconds you picked get processed. Pro videos take longer. On a normal 4-core server, expect roughly 1 minute of work per 3–4 minutes of video (a 2-hour stream takes about 30–40 minutes). Videos are processed one at a time. To serve many people, use a server with an NVIDIA GPU (much faster) or raise `WORKERS` on a bigger machine.

### Settings (environment variables)

| Variable | Default | What it does |
|---|---|---|
| `ADMIN_PASSWORD` | | Turns on the /admin page for changing plans |
| `UPGRADE_URL` | | Payment page for the Upgrade to Pro button |
| `WATERMARK_TEXT` | `AutoClipper` | Name shown in the watermark and end card on free clips (empty = no watermark) |
| `WATERMARK_URL` | your site's address | Web address shown on the end card |
| `FREE_CLIPS_PER_DAY` | `4` | Free plan clips per 24 hours |
| `FREE_CLIP_SECONDS` | `10` | Free plan clip length |
| `MAX_CLIPS` | `8` | Most clips per video on Pro |
| `MAX_ACTIVE_JOBS` | `2` | Videos one person can have in progress |
| `ALLOW_URLS` | `true` | Let Pro users paste links (YouTube often blocks servers) |
| `MAX_UPLOAD_MB` | `4096` | Largest upload allowed |
| `MAX_MINUTES` | `180` | Longest video allowed |
| `KEEP_HOURS` | `24` | Results are deleted after this |
| `WORKERS` | `1` | Videos processed at the same time |
| `WHISPER_MODEL` | `small` | Speech model: `base` is faster, `medium` more accurate |
| `SECRET_KEY` | auto | Signs login cookies (made automatically if not set) |
| `SECURE_COOKIES` | `false` | Set to `true` once your site uses https |
| `PORT` / `DATA_DIR` | `8000` / `data` | Port, and where accounts, uploads and clips are stored |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | | Optional smarter clip picking for Pro |

### Before charging people

You'll still want terms of service, a privacy policy, a policy on copyrighted uploads, and automatic upgrades through Stripe webhooks.

### Caption fonts

Pro users can choose Classic, Montserrat, Anton, Bangers, Poppins, Bebas Neue or Permanent Marker. The `.ttf` font files sit next to `clipper.py`. Their licences (SIL Open Font License, and Apache 2.0 for Permanent Marker) are in `Caption-fonts-licenses.txt`.

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
