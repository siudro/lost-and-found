# Running and deploying

## Local, in three commands

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://127.0.0.1:8000
```

No keys, no GPU, no model downloads. Demo footage seeds itself on first run.

---

# Deploying to Railway

## Why the Railway deploy kept failing

The `Dockerfile` had two stages, `base` and then `test`. A build with no `--target`, which is what Railway runs, builds the **last** stage, and that was `test`. Two failures followed from it:

1. **The build itself failed.** The `test` stage ran `pip install -r requirements-dev.txt`, but that file was never `COPY`'d into the image, so pip stopped with "Could not open requirements file".
2. **Even with that file copied, the deploy would fail.** The `test` stage's `CMD` runs pytest and exits. The web server never starts, so `/api/health` never answers and Railway marks the deploy failed.

**Fix:** the runtime stage is now the last stage in the file, and the `test` stage copies `requirements-dev.txt` before installing it. Keep `runtime` last. `railway.json` also no longer sets `"startCommand": null`; the Dockerfile `CMD` is the only start command. `pytest` and `httpx` moved out of `requirements.txt` into `requirements-dev.txt`, so they are no longer installed in production.

`CMD` stays in shell form so Railway's `$PORT` expands:

```
$ PORT=9099 sh -c 'echo "uvicorn ... --port ${PORT:-8000}"'
uvicorn ... --port 9099
```

## Deploy in six steps

1. **Push this repo.** Already public — good, that removes a whole class of failure.
2. In Railway: **New Project → Deploy from GitHub repo → `siudro/lost-and-found`**.
3. Railway detects `railway.json` and builds with the `Dockerfile`. Nothing else to configure.
4. **Add a volume.** Right-click the service → **Volumes → New Volume** → mount path `/data`.
5. **Set the variables** (Service → Variables):

   | Variable | Value |
   |---|---|
   | `DATABASE_PATH` | `/data/foundit.db` |
   | `VENUE_TIMEZONE` | `Asia/Riyadh` |

   That is all that is required. Everything else in `.env.example` is optional.
6. **Generate a domain.** Service → **Settings → Networking → Generate Domain**.

First build takes 3–5 minutes (it compiles nothing, but apt-installs ffmpeg). Redeploys after that are fast because Docker layer caching keeps the ffmpeg layer.

## Why a volume matters

Without `/data` mounted, every redeploy gives you a fresh container with an empty filesystem. The app still starts and re-seeds demo fixtures, so it *looks* fine — but anything you ingested is gone. That is a silent failure, which is why it is step 4 and not a footnote.

## Verifying the deploy worked

Hit `/api/health` on your domain. You should see:

```json
{"ok": true, "checks": [{"name": "python", "ok": true, ...}, ...], "warnings": 1}
```

Then open `/` — the readiness panel should show `storage — sqlite writable (mounted volume)`. If it says `local ./data/foundit.db`, the volume is not mounted and you are writing inside the container.

### If it still fails

| Symptom | Cause | Fix |
|---|---|---|
| Build fails on `apt-get update` | Railway builder is not using the Dockerfile | Check `railway.json` says `"builder": "DOCKERFILE"` |
| Container starts then exits | Wrong stage built, or `$PORT` not expanding | `runtime` must be the last stage in the `Dockerfile`; `CMD` must be shell form |
| Build fails: `Could not open requirements file` | Old `Dockerfile` with `test` as the last stage | Pull the fixed `Dockerfile` |
| Healthcheck times out | App slower than 120s to boot | Already raised from 100s; check build logs |
| 502 on the domain | Domain generated before deploy finished | Wait, or check deploy logs |
| `/api/ingest` returns 422 | ffmpeg missing | The Dockerfile installs it; check the build log for the `apt-get` step |

Read the **Deploy Logs** tab, not just the build log. The app prints its readiness report to stdout on every boot, so the container log is the fastest place to see what is actually wrong.

## What runs where

Everything runs in one container: web server, chat, SQLite, video decoding. No separate worker, no external database, no GPU. A 60-second 1080p clip takes a few seconds to process at 1 fps sampling on a single CPU.

---

# Local Docker (optional)

```bash
docker build -t foundit .
docker run -p 8000:8000 -v foundit-data:/data foundit
```

The image is `python:3.12-slim` plus ffmpeg. Run the tests inside it too:

```bash
docker build --target test -t foundit:test .
docker run --rm foundit:test
```

---

# AI providers and how to get keys

**You do not need any of these to deploy or demo.** The app starts with every key unset and says so. Each row below is a quality improvement, not a dependency.

Check what is live at any time: `GET /api/providers` or the readiness panel in the UI.

| # | Provider | Key | Where to get it | Cost | What improves | Without it |
|---|---|---|---|---|---|---|
| 1 | `open_vocab_detector` | `DETECTOR_MODEL` | No key. Set to a model name, e.g. `yolov8x-worldv2`. Weights auto-download on first use. | Free | Finds items **by name** ("jacket") instead of by blob. Big recall gain in clutter. | Motion-based detection only |
| 2 | `sam_segmenter` | `SAM_CHECKPOINT` + `HF_TOKEN` | [huggingface.co/facebook/sam-vit-base](https://huggingface.co/facebook/sam-vit-base) — sign in, accept the licence, then **Settings → Access Tokens → New token (read)**. Set `SAM_CHECKPOINT` to `facebook/sam-vit-base`. | Free (gated) | Clean item masks → better appearance crops | Bounding boxes from the detector |
| 3 | `captioner` | `CAPTION_API_KEY` | Pick one provider: [OpenAI](https://platform.openai.com/api-keys) · [Anthropic](https://console.anthropic.com/) · [Google AI Studio](https://aistudio.google.com/apikey) (free tier). | ~$0.01–0.03 per 1,000 items | Real descriptions: "gray hooded jacket, zip front" | Colour + class label only |
| 4 | `llm_answerer` | `LLM_API_KEY` + `LLM_MODEL` | Same providers as above. Model name, e.g. `gpt-4o-mini` or `gemini-2.0-flash`. | < $0.01 per conversation | Friendlier phrasing | Template text, identical facts |
| 5 | `text_embedder` | `EMBED_MODEL` | No key for local models (e.g. `all-MiniLM-L6-v2`, auto-downloads). Key only if using a hosted one. | Free locally | Catches paraphrases ("overcoat" → "jacket") | Keyword retrieval only |

### Getting a Hugging Face token for SAM (the only fiddly one)

1. Create a free account at [huggingface.co](https://huggingface.co/join).
2. Open the model page and click **Agree and access repository** — SAM is licence-gated, so this step is required even though it is free.
3. **Settings → Access Tokens → New token**, type **Read**.
4. Copy it into `HF_TOKEN`. The `hf_...` value is real; `public` is not a token and will fail.

Do **not** commit this token. `.env` is gitignored, and on Railway set it under Variables, never in code.

### Important: the LLM cannot change what is true

The answer is composed from evidence in `app/core/answer.py` **first**, then optionally rephrased. The model never sees a version of the answer that contains a handover claim, because one is never constructed. Adding `LLM_API_KEY` changes the wording, not the facts — and there is a test that proves it.

---

# Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q                   # 77 tests
python tests/validate_fixtures.py            # 140 schema + rule checks
python tests/test_rules_catch_violations.py  # 15/15 forbidden states rejected
```

`pytest -q -k honesty` runs only the rules that decide whether the project is worth submitting.

# Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Chat interface |
| POST | `/api/chat` | `{"message": "..."}` |
| POST | `/api/ingest` | upload clip + `recorded_at` |
| GET | `/api/health` | readiness (also the Docker healthcheck) |
| GET | `/api/providers` | which AI is configured, and what happens without it |
| GET | `/api/stats` | row counts and sources |

# The interface

The home page is a chat. Footage (upload, samples, recordings on file) and How it works (limits, system status) are separate tabs. Greetings, thanks and "what can you do" get a short conversational reply (`app/core/conversation.py`); everything else goes to the footage search. A link such as `/?q=Did%20anyone%20leave%20a%20bag%20today%3F` opens the chat and asks that question, which is handy for demos.

# The LLM

Set `LLM_API_KEY` (an OpenAI key) in Railway > Variables. With it, the chat (`app/ai/llm.py`):

- rewrites each search answer into a short reply. The composed `body` stays the record; only `lead` changes, and the page labels the reply "Written with AI";
- answers chat that is not a search ("my name is Sara") and steers back to describing the item.

Guards: every clock time and number in a reply must appear in the search result, and replies claiming a handover or a sighting are discarded. In all those cases the composed text is used. Greetings never call the API, identical questions are cached, there is a daily call cap (`LLM_DAILY_CALL_LIMIT`, default 150) and a reply cap (`LLM_MAX_OUTPUT_TOKENS`, default 160). `GET /api/llm/status` shows today's usage and never shows the key. Without a key, everything still works with composed text.

# Testing with footage

On the home page:

1. **Add footage.** Upload a video, or click one of the real CAVIAR sample clips (the server downloads it). Processing runs in the background and the page shows progress. Uploads are limited to `MAX_UPLOAD_MB` (default 500). A playback copy (H.264, no audio) is kept on the volume under `media/`.
2. **Ask.** Example questions update to the day of the newest footage. Each match has a video player that jumps to the moment.

Good test footage: a phone on a shelf, not moving, filming a room. Someone puts a bag or jacket down, stays a few seconds, walks out of view, and the camera keeps recording for at least 30 more seconds.

Unattended thresholds for ingested footage (environment variables): `UNATTENDED_STATIONARY_S` (default 10), `UNATTENDED_DESERTED_S` (default 15; alone this long = possibly unattended, twice this long = likely unattended).

# How the detector works

`app/core/detector.py` keeps two background models of the scene. The long one remembers the empty room; the short one catches up within about a second. Pixels that differ from the long model but already match the short one belong to something that arrived and stopped moving. Those are grouped into items and tracked, and for every moment the distance to the nearest moving person is recorded. `pipeline.py` then judges each track with the rules in `unattended.py`, keeping first-seen, onset and confirmation times separate.

# Known limits

- The detector does not know what an item is. The item type is reported as unknown, and answers say so and point to the clip.
- Anything already in view when the recording starts becomes part of the background and is not detected.
- A person who stands completely still can look like an item; shapes whose pixels keep moving like a person's are ignored, but not every case is caught.
- Areas that look onto somewhere else (a glass railing over another floor, a window onto a street) need an ignore zone: `DetectorConfig(ignore_polygons=...)`. The CAVIAR samples use one for the glass railing in the lower left. Uploads have no ignore zone yet; there is no UI for drawing one.
- The camera must not move. A large sudden change (camera bumped, lights switched) rebuilds the background, and any items being tracked at that moment are dropped.
- No SAM or captioner enabled by default, so descriptions come from colour and position in the frame.
- Demo fixtures are hand-written to exercise the contract, not captured from real footage.
- No live camera ingest — recorded upload and replay only.
