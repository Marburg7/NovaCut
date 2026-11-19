# NovaCut Studio

**Autonomous AI Clip Maker for Twitch, Kick & YouTube Live Streams**  
Created by **Syed Shahabuddin** • 100% Free & Open Source

Turn long videos and live streams into captioned vertical clips ready for YouTube Shorts, Instagram Reels, and TikTok. **Works 100% free out of the box with zero external API keys** using our built-in Autonomous Intelligence Engine, or bring your own keys for Gemini, OpenAI, Claude, Grok, and more.

Paste a YouTube, Twitch, or Kick link (or upload a file) and NovaCut Studio automatically detects viral moments, calculates dynamic retention scores, cuts them to 9:16, burns word-by-word highlighted captions with punchy hooks, and prepares them for posting. It can also monitor creators in real time, recording and clipping highlights while the stream is still live.

## Key Highlights

- **100% Free Built-in Autonomous Engine.** No external AI keys, subscriptions, or token fees needed! NovaCut includes an autonomous heuristic and semantic NLP engine that analyzes spoken transcripts, loudness dynamics, and conversational pacing to find viral moments, write engaging hooks, generate titles, and highlight key words automatically.
- **Clips from any video or stream.** Transcribes audio locally on your machine, extracts high-retention moments, and renders vertical 9:16 clips with animated captions in three custom styles (Bounce, Hype, Clean).
- **Real-Time Live Clipping.** Records live streams into a rolling buffer and generates ready-to-post clips while the broadcast is still going.
- **Workspaces per Creator.** Manage multiple creators with independent branding, caption styles, clipping thresholds, and posting rules.
- **Direct Social Posting.** Connect YouTube Shorts, Instagram Reels, and TikTok to queue and auto-publish clips with customizable schedules and daily limits.
- **Optional External AI Providers.** Want to use cloud LLMs? Easily plug in Google Gemini, OpenAI (GPT-4o), xAI (Grok), Groq, Anthropic (Claude), OpenRouter, or local models via Ollama.
- **Cyber Studio Dark UI.** Modern dark-mode dashboard with instant feedback, video scrubber, caption designer, and zero build steps.

## How to Run This Project

### 1. Prerequisites

You need **Python 3.10 or newer** and **`ffmpeg`**:

- **macOS:** `brew install ffmpeg`
- **Windows:** `winget install Gyan.FFmpeg`
- **Linux (Ubuntu / Debian):** `sudo apt install ffmpeg`

NovaCut Studio locates `ffmpeg` automatically and alerts you in the dashboard if it is missing.

### 2. Clone & Install Dependencies

```bash
git clone https://github.com/Marburg7/NovaCut.git
cd NovaCut
```

Create and activate a virtual environment, then install dependencies:

- **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- **Windows (PowerShell / Command Prompt):**
  ```bat
  python -m venv .venv
  .venv\Scripts\activate
  pip install -r requirements.txt
  ```

### 3. (Optional) Pre-download the Local Speech Model

On first startup, NovaCut automatically downloads the local speech recognition model (about 480 MB, once). To download it ahead of time:

```bash
python -m studio.fetch_model
```

### 4. Start NovaCut Studio

Make sure your virtual environment is active, then start the server:

- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  python -m studio.server
  ```
  *(Or run directly without activating first: `.venv/bin/python -m studio.server`)*
- **Windows:**
  ```bat
  .venv\Scripts\activate
  python -m studio.server
  ```
  *(Or run directly: `.venv\Scripts\python -m studio.server`)*

Open **http://127.0.0.1:8765** in your browser once you see `NovaCut Studio is running: http://127.0.0.1:8765`. There is no sign-in screen and **no external API key required** — the built-in **Antigravity AI** Autonomous Engine runs locally out of the box.

To use a custom port or data directory (for videos, clips, and the database):

```bash
NOVACUT_PORT=8766 NOVACUT_DATA=./my_data .venv/bin/python -m studio.server
```

*(Both `NOVACUT_*` and legacy `PEAKCUT_*` environment variable prefixes are supported.)*

### 5. Create Clips (Live Streams & Recorded Videos)

- **Live Stream Clipping (Twitch, Kick, YouTube Live):**
  - Open **Live Monitor** from the sidebar, paste a live stream URL (for example `https://twitch.tv/<channel>`, `https://kick.com/<channel>`, or `https://youtube.com/@<channel>/live`), and click **Connect & Monitor** — *or* paste a live channel URL directly on **Home** and click **Monitor Live Stream**.
  - NovaCut records the live stream into a rolling buffer, tracks live chat velocity, emote bursts, audio loudness, and speech locally to auto-generate captioned **9:16** shorts, and lets you manually cut moments anytime with **Clip last 30s** or **Last 60s**.
- **Video Clipping (VOD Links & Local Files):**
  - On **Home**, paste a YouTube, Twitch, or Kick video URL — or click **Upload video** / drag and drop a video file (`.mp4`, `.mov`, `.mkv`, `.webm`, etc.).
  - Open **Options** to pick clip length, layout (**Blurred fill**, **Zoomed in**, or **Fill screen**), and caption style (**Bounce**, **Hype**, or **Clean**), then click **Make clips**.

## Choose your AI

NovaCut Studio comes with the **Antigravity AI (`antigravity`)** built-in engine active by default — **100% free with zero API keys**.

If you want to connect an external cloud model or a local LLM instead, open **Settings → AI** in the dashboard, choose a provider, paste your key, press **Find models**, and click **Save and test**. Keys are stored locally in your data folder (`ai.json`).

You can also configure AI providers in `.env` (using either `NOVACUT_*` or `PEAKCUT_*` variable names; settings saved in the dashboard take priority):

```ini
NOVACUT_AI_PROVIDER=antigravity
# Or use an external provider, e.g.:
# NOVACUT_AI_PROVIDER=gemini
# NOVACUT_AI_MODELS=gemini-2.5-flash
# NOVACUT_AI_KEY=your-key-here
```

| `NOVACUT_AI_PROVIDER` / `PEAKCUT_AI_PROVIDER` | Service | Key / Notes |
|---|---|---|
| `antigravity` *(default)* | Built-in Autonomous Engine | **Free, no key needed** (runs 100% locally) |
| `gemini` | Google Gemini | https://aistudio.google.com/apikey |
| `openai` | OpenAI | https://platform.openai.com/api-keys |
| `xai` | Grok (xAI) | https://console.x.ai |
| `groq` | Groq | https://console.groq.com/keys |
| `openrouter` | OpenRouter (many models, one key) | https://openrouter.ai/keys |
| `anthropic` | Claude | https://console.anthropic.com |
| `ollama` | Local model via Ollama | https://ollama.com (no key needed) |
| `custom` | Any OpenAI-compatible endpoint | Also set `NOVACUT_AI_BASE_URL` / `PEAKCUT_AI_BASE_URL` |

- `NOVACUT_AI_MODELS` (or `PEAKCUT_AI_MODELS`) accepts one or more comma-separated model IDs, tried in order as fallbacks.
- When using external providers, a fast chat model that returns JSON works best, and the **AI usage** page tracks calls and token counts.
- Speech-to-text always runs locally on your own machine with `sherpa-onnx`, so audio never leaves your computer.

## Settings

Everything is set through environment variables, normally in a `.env` file next to this README.
[`.env.example`](.env.example) lists every setting with a short explanation. The ones that matter first:

| Setting | What it is for |
|---|---|
| `PEAKCUT_AI_PROVIDER`, `PEAKCUT_AI_MODELS`, `PEAKCUT_AI_KEY` | The AI that picks clips and writes titles (or set it in the dashboard) |
| `PEAKCUT_ACCOUNTS` | `1` makes everyone sign in. Use it on any server other people can reach |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | Optional: Sign in with Google, and connecting YouTube |
| `PEAKCUT_SMTP_*`, `PEAKCUT_MAIL_FROM` | Optional: confirmation and password emails |
| `PEAKCUT_DATA` | Where videos, clips and the database are kept (default `./data`) |

`.env`, the `secrets/` folder and all user data are ignored by git. Keep keys there, never in the code.

## Admins

This only matters with accounts switched on (`PEAKCUT_ACCOUNTS=1`); without them you are the owner and can do
everything. An account is an admin only when the database says so. No email address is special in the code and no web request
can change a role. Do it on the machine that runs NovaCut Studio:

```bash
python -m studio.admin grant you@example.com
```

`revoke` and `list` work the same way. The admin dashboard is at `/admin` and answers "not found" to everyone else.

## How it is built

| Part | What it is |
|---|---|
| `studio/` | The server: FastAPI, the clip pipeline, live recording, accounts |
| `studio/ai.py` | The only file that talks to an AI service. Add a provider here |
| `web/` | The pages: landing, sign-in, the app, the admin dashboard (plain HTML, CSS and JavaScript, no build step) |
| `tests/` | Tests that run against a throwaway copy of the app |
| `deploy/`, `Dockerfile`, `docker-compose.yml` | One-server deployment with PostgreSQL and automatic HTTPS |

Video work is done with `ffmpeg` and `yt-dlp`. Speech is transcribed with `sherpa-onnx` and the Parakeet model.
Data is kept per user: accounts in the database (SQLite on your own computer, PostgreSQL on a server), videos and
clips in each user's own folder. [AGENTS.md](AGENTS.md) is a short map of the code for contributors and AI coding
assistants.

## Where your videos and clips are kept

Everything NovaCut Studio keeps is in one folder: videos, clips, recordings, creators, settings and the speech model. By
default it is the `data` folder inside NovaCut Studio. **Settings, Storage** shows the folder and how much room its drive
has, opens it in one click, and lets you move it to another drive: choose the new folder, and NovaCut Studio moves
everything there the next time it starts. If the drive is nearly full, NovaCut Studio says so and stops taking new videos
until there is room. (`PEAKCUT_DATA` in `.env` still works and takes priority.)

## Updating

NovaCut Studio checks GitHub once a minute while it runs. When a newer version is out, the dashboard shows a notice
within a minute or two (no restart needed), with what changed
and how to update: ask your AI coding assistant to do it (the notice has a sentence to paste), or run
`git pull && pip install -r requirements.txt` in the NovaCut Studio folder and restart it (with Docker:
`git pull && docker compose up -d --build`). Your `.env`, keys and clips are not touched. Only the two commit ids
are sent to GitHub; set `PEAKCUT_UPDATE_CHECK=0` to turn the check off.

## Deploy

[DEPLOY.md](DEPLOY.md) walks through a single server with Docker: the app, PostgreSQL, and Caddy for HTTPS.

```bash
docker compose up -d --build
```

## How it behaves under load

- The page asks the server one small question every few seconds. When nothing has changed, the answer comes
  straight from memory.
- Each person's videos wait in their own line and the lines take turns, so one person cannot hold everyone up.
- The server measures its own processor, memory and disk every second. Extra videos and recordings start only
  while there is room, wait briefly when it is nearly full, and continue the moment it frees up.
- There are no limits on upload size, video length, recordings, creators or storage unless you set them in
  `.env`. Five videos can be sent at once. Request sizes are always capped, so junk can't fill the memory.

## Tests

Run the full integration test suite:

```bash
python tests/run.py
```

Run the autonomous live stream & viral detection test suite:

```bash
python tests/test_autonomous_live.py
```

`tests/run.py` starts a temporary copy of the app on a spare port with an empty data folder, runs permission, abuse, storage, AI, and queue tests against it, then cleans up. Both suites run completely locally with zero API keys and never touch real user data.

## Security

- With sign-in off, NovaCut Studio listens on this computer only (`127.0.0.1`), refuses changes sent by other websites,
  and only answers to its own address. Do not expose it to the internet that way: turn accounts on.
- Passwords are stored only as salted scrypt hashes. Sessions are random tokens stored hashed; cookies are HTTP-only.
- Every route works out who is asking from the session, never from the request, so users only reach their own data.
- Only known video sites are fetched, and text from videos is treated as data when it is shown to the AI.
- Found a problem? Please report it privately through the repository's security advisories instead of opening a
  public issue.

## Using it responsibly

Clip videos you made, or have permission to use. You are responsible for the rights to what you download, clip and
post, and for following the terms of the sites you fetch from and post to.

## Licence

[MIT](LICENSE). Caption fonts in `studio/fonts/` (Anton, Archivo Black, Poppins) are under the SIL Open Font
License; the licence texts are in the same folder.
