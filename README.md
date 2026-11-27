# FoundIt

A lost-and-found assistant that searches camera footage. You ask it the way you would ask the person at the lost-and-found desk ("I forgot my gray jacket in the classroom today"). It searches timestamped video memory, explains each possible match, and plays the clip from the moment it found.

Built for **WAICY 2026, LLM & AI Agent track**.

**Live:** https://lost-and-found-production-d3e4966a.up.railway.app

## Quickstart

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# http://127.0.0.1:8000
```

No keys, GPU or model downloads are needed. Set `LLM_API_KEY` (an OpenAI key) to turn on the AI features; see `docs/running.md`.

## What it does

1. **Finds left items in footage.** A left-object detector (OpenCV, two background models) runs on CPU at about 12× real time. For each item it keeps three separate times: when it was put down, when it became unattended, and when the system was sure.
2. **Describes them.** With a key, a vision model gives each item a type, colours and features ("gray jacket with a collar"), within a fixed schema. Without a key, colour and position are used.
3. **Answers in chat.** The chat:
   - understands dates in the venue's time zone ("yesterday", "this morning") and synonyms (grey/gray, coat/hoodie);
   - ranks possible matches and shows a crop picture of each;
   - plays each match from the right second.
4. **Searches by photo.** Press the camera button and upload a picture of the item. The photo is never stored.
5. **Takes live cameras.** `tools/live_ingest.py` cuts a camera stream into segments and sends them through the same path as an upload.

## What makes it different

The chat never mixes up three things:

- **A sighting:** the footage shows the item.
- **A likely unattended item:** the footage shows it left with nobody nearby.
- **A verified handover:** staff formally received it.

A handover claim needs a staff intake record. Without one, the app says what the footage shows and that staff receipt cannot be verified. It also always says what footage it did not check, so "no match" never becomes "it isn't there".

The language model can only reword an answer that was already composed from the evidence. A reply that adds a time or number that isn't in the facts, or claims a handover or a sighting, is thrown away. These rules are enforced in the schemas, the inference code and the tests, not just in a prompt.

## Results

These are summarised from `docs/evaluation.md`. Every number comes from a run.

| Test | Result |
|---|---|
| 7 simulated clips with known ground truth, including a 10-minute classroom | precision 1.0, recall 1.0, 0 false alarms |
| Onset error / detection delay | +3 to +13 s / 30 s, reported separately |
| 4 real CAVIAR clips (CC BY-SA), checked by eye | 4 of 4 correct, one item each (small sample) |
| Retrieval, right item in top 3 | text 5/5, photo by colour only 5/5 |
| Speed | 11–13× real time on CPU, about 300 MB of memory |

**Not measured yet:** real jacket footage, which needs the staged shoot. No public dataset has labelled indoor jacket abandonment.

## Status

| Part | State |
|---|---|
| Data contracts (7 JSON Schemas) and validator | built |
| Left-object detection and unattended logic | built |
| AI item descriptions (OpenAI vision, schema-checked, capped) | built |
| Chat: dates, synonyms, ranking, evidence clips, crop pictures | built |
| AI wording with fact guards, daily cap and cache | built |
| Photo search | built |
| Live camera and replay ingest | built; track state is not carried across segments yet |
| Versioned prompts (`prompts/`) | built |
| Evaluation harness (`tools/evaluate.py`) | built for simulated clips; CAVIAR XML scoring not yet |
| CI (GitHub Actions) and Railway auto-deploy | built |
| SAM segmentation | not enabled |
| Staged real jacket footage | still to be filmed |

Check yourself: `python tests/validate_fixtures.py`, `python tests/test_rules_catch_violations.py`, `python -m pytest`.

## Docs

- `docs/running.md`: deploy, keys, live cameras.
- `docs/evaluation.md`: what was measured and what wasn't.
- `docs/adr.md`: architecture decisions.
- `docs/market_and_costs.md`: competitors, costs, accounts.
- `docs/interface.md`: why each schema field exists.
- `docs/datasets.md`: the dataset situation.
- `docs/AGENTS.md`: for another agent picking this up.

## Layout

```
app/core/   detector, pipeline, unattended logic, time, query, answer, store, chat
app/ai/     OpenAI client with guards, captioner, photo search, prompts loader, provider registry
app/web/    chat-first web interface
prompts/    versioned prompt files and changelog
schemas/    7 JSON Schema contracts; fixtures/ holds one valid example of each
tools/      evaluate, live_ingest, make_classroom_clip
tests/      unit, detector, API, honesty, AI (with a fake model), live ingest
docs/       running, evaluation, adr, market_and_costs, interface, datasets, AGENTS
```

## Non-negotiables

- Captions, OCR, transcripts and uploads are **untrusted data**, never instructions.
- No invented timestamps, identities, ownership, handovers or coverage.
- No face recognition. Person signals stay anonymous and local to the camera.
- "No match in the footage checked" is never presented as "it is not there".
- Visitor photos are not stored. Real school footage and secrets stay out of this repo.
- Audio off by default.
