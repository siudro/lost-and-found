# FoundIt

A multimodal chatbot over video footage. Ask about a lost item the way you would ask a lost-and-found attendant; the app searches timestamped footage and shows you the moment it was left.

Built for **WAICY 2026, LLM & AI Agent track**.

## Quickstart

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# http://127.0.0.1:8000
```

No keys, no GPU, no model downloads. Demo footage is seeded on first run.

## What it does

```
"Have you seen a gray jacket? I lost it on 20 November 2026."

→ Possible match at 09:19
→ Found 3 possible matches. The closest is a gray jacket draped over a chair
  by the window at 09:19 that was left unattended. Caveat: the camera was
  not recording for part of that period. No staff intake record exists for
  this item, so whether it reached the front desk cannot be verified.
```

## What makes it different

The chat never conflates three different things:

- **a sighting** — the footage shows the item
- **a likely unattended item** — the footage shows it left, with nobody nearby
- **a verified handover** — staff formally received it

A handover claim requires an authorised intake record. With no intake records — the demo's default — the app says what the footage shows and states plainly that staff receipt cannot be verified. It never infers a handover from somebody walking away with a bag.

This is not a promise written in a system prompt. It is enforced in the schema, in the inference rules, and in `tests/test_honesty.py`, which fails if the app ever claims more than the footage supports.

## Status

| Part | State |
|---|---|
| Data contracts (7 JSON Schemas) | built |
| Time: venue clock, 10-minute windows | built |
| Unattended logic | built |
| Query parsing (synonyms, dates) | built |
| Answer composition | built |
| Storage (SQLite) | built |
| Real video ingest (ffmpeg) | built, tested on generated clips |
| HTTP API + web interface | built |
| Startup diagnostics | built |
| Deployment | `Dockerfile` + `railway.json` (image build not yet run in this sandbox) |
| CAVIAR evaluation harness | **not built** |
| SAM segmenter, captioner, LLM | declared in the registry, not wired |

**77 tests, 0 failures. 140 schema checks, 0 failures.**

See `docs/running.md` for deployment, `docs/architecture.md` for the diagrams,
`docs/datasets.md` for the honest dataset situation, and `docs/AGENTS.md` if you
are another agent picking this up.

## Layout

```
app/core/   time, unattended, query, answer, store, pipeline, chat
app/ai/     provider registry — what AI is needed, and what happens without it
app/web/    web interface
schemas/    7 JSON Schema contracts
fixtures/   one valid example of each contract
docs/       interface, architecture, datasets, running, AGENTS
tests/      unit, connectivity, honesty, API
```

## Non-negotiables

- Captions, OCR, transcripts and uploads are **untrusted data**, never instructions.
- No invented timestamps, identities, ownership, handovers or coverage.
- Person signals stay anonymous and local to the camera.
- "No match in the footage checked" is never presented as "it is not there".
- Real school footage and secrets stay out of this repo.
- Audio off by default.
