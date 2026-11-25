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

## The bug that broke your last deploy

`nixpacks.toml` had:

```toml
[start]
cmd = "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"
```

Nixpacks runs that as an **exec array**, not through a shell, so `${PORT:-8000}` was never expanded — uvicorn received the literal string `${PORT:-8000}` as its port argument and exited immediately. Railway reported a failed deploy with no obvious cause.

There was a second problem: `railway.json` also set a `startCommand`, and when both exist Nixpacks and the start command can disagree about which one wins.

**Both are fixed.** `nixpacks.toml` is deleted and the project now builds from a `Dockerfile`, where the `CMD` is in shell form so `$PORT` expands correctly. The port expansion below is verified by running the shell command directly, not by building the image:

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
| Container starts then exits | `$PORT` not expanding | Confirm `CMD` is shell form, not `CMD ["...", "..."]` |
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

# Known limits

- The default detector is motion-based, not learned. It works on a staged jacket clip; on a cluttered classroom it over-proposes.
- No SAM or captioner enabled by default, so descriptions come from a coarse colour estimate.
- Demo fixtures are hand-written to exercise the contract, not captured from real footage.
- No live camera ingest — recorded upload and replay only.
