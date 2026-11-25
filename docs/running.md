# Running and deploying

## Local, in three commands

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://127.0.0.1:8000
```

The app starts with **no keys, no GPU, no model downloads**. It seeds demo fixtures on first run and prints a readiness report to the log.

## What the startup log tells you

Every start prints a report like this:

```
==============================================================
 FoundIt — startup readiness
==============================================================
  [PASS] python: 3.13
  [PASS] ffmpeg: ffmpeg and ffprobe found
  [PASS] media_decode: sample clip decodes
  [PASS] schemas: 7 contracts load
  [PASS] storage: sqlite writable at ./data/foundit.db
  [PASS] fixtures: 7 fixture files load
  [PASS] templates: jinja2 available
  [PASS] ai:rule_based_detector: Proposes candidate item regions…
  [WARN] ai:sam_segmenter: SAM_CHECKPOINT not set — crops come from detector boxes
--------------------------------------------------------------
 Ready. 1 optional feature(s) unavailable; answers stay honest but thinner.
==============================================================
```

`FAIL` means a required component is missing and the app cannot serve. `WARN` means an optional improvement is absent and the app still works. The distinction is deliberate: a missing captioner must not read like a broken app.

The same report is served at `/api/readiness`.

## Which AI is actually needed

**None.** That is the honest answer and it is designed in.

| Component | Key needed | What it does | If absent |
|---|---|---|---|
| `rule_based_detector` | no | Finds regions that stay still while people move | — always runs |
| `open_vocab_detector` | `DETECTOR_MODEL` | Finds "jacket" by name, better in clutter | motion detector only |
| `sam_segmenter` | `SAM_CHECKPOINT` + `HF_TOKEN` | Clean mask for the appearance crop | detector bounding box |
| `captioner` | `CAPTION_API_KEY` | Writes item type, colours, features | colour + class label |
| `llm_answerer` | `LLM_API_KEY` | Rephrases the composed answer | template text, same facts |
| `text_embedder` | `EMBED_MODEL` | Semantic retrieval | keyword only |

Check live status any time at `/api/providers`.

The `llm_answerer` is worth understanding: it **only rephrases**. The answer is composed from evidence first, in `app/core/answer.py`, and then optionally rewritten. That ordering is why a bad model cannot make the app claim a handover — it never sees a version of the truth that contains one.

## Railway

`railway.json` and `nixpacks.toml` are committed. Deploy:

1. Push this repo to GitHub.
2. In Railway: **New Project → Deploy from GitHub repo**.
3. Railway reads `railway.json`, installs `requirements.txt`, and runs
   `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
4. **Add a volume** and set `DATABASE_PATH=/data/foundit.db`. Without a volume,
   every redeploy wipes the reports. The app still starts; it just forgets.
5. Health check is `/api/health` — already configured.

`nixpacks.toml` installs **ffmpeg**, which the ingest path needs. Without it the
app runs fine and `/api/ingest` returns 422 with a clear message.

### Cost

Railway's trial/hobby tiers are enough. Nothing here needs a GPU: the baseline
pipeline is CPU-only and processes a 60-second 1080p clip in a few seconds at
1 fps sampling. A GPU only becomes necessary if you enable the SAM segmenter.

## Environment variables

Copy `.env.example`. Nothing is required. The one that matters in production is
`DATABASE_PATH` — point it at the mounted volume.

## Tests

```bash
python -m pytest tests/ -q          # 77 tests
python tests/validate_fixtures.py   # 140 schema + rule checks
python tests/test_rules_catch_violations.py  # 15/15 forbidden states rejected
```

Add `-v` to see each test name. `pytest -q -k honesty` runs only the
honesty rules, which are the ones that matter for the submission.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Chat UI |
| POST | `/api/chat` | `{"message": "..."}` → answer |
| POST | `/api/ingest` | upload a clip + `recorded_at` → genuine processing |
| GET | `/api/health` | full readiness report |
| GET | `/api/readiness` | same, smaller payload |
| GET | `/api/providers` | which AI is configured, and what happens without it |
| GET | `/api/stats` | row counts and sources |

## Known limits

- The baseline detector is motion-based, not learned. On a staged jacket clip it works; on a cluttered real classroom it will over-propose.
- No SAM or captioner is enabled by default, so item descriptions come from a coarse colour estimate.
- Demo fixtures are synthetic, written by hand to exercise the contract.
- No live camera ingest yet — recorded upload and replay only.
