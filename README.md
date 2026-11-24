# Lost and Found

A multimodal chatbot over video footage for WAICY 2026 (LLM & AI Agent track).

Ask about recorded footage the way you would ask a lost-and-found attendant: *"Have you seen a gray jacket? I lost it yesterday."* The system searches timestamped, structured memory built from the video, explains possible matches, and links each one to the moment in the footage.

## What makes it different

The chat never conflates three different things:

- **a sighting** — the footage shows the item
- **a likely unattended item** — the footage shows it left with nobody attending
- **a verified handover** — staff formally received it

A handover claim requires an authorised intake record. With no intake records, the system says what the footage shows and states plainly that staff receipt cannot be verified. It never infers a handover from a person walking away with something.

## Status

| Part | State |
|---|---|
| Shared interface (schemas + fixtures) | **done** — this commit |
| Repository audit | pending — repo was created empty |
| Ingest, detection, unattended logic | not started |
| Captioning, reports, evidence playback | not started |
| Chatbot and versioned prompt | not started |
| Startup checks, CI, deployment | not started |

No pipeline has been run. Nothing here produces a result yet. The only executable artefacts are the validation tests, which check the contract against its own fixtures.

## Layout

```
schemas/    JSON Schema for every record in the pipeline
fixtures/   a valid example of each contract (staged demo data)
docs/       the interface contract and the reasoning behind it
tests/      contract validation, and tests that the rules reject bad data
```

## Verify the contract

```bash
pip install jsonschema
python tests/validate_fixtures.py            # expect: 140 checks passed, 0 failed
python tests/test_rules_catch_violations.py  # expect: 15/15 forbidden states rejected
```

Read `docs/interface.md` before writing pipeline code — it explains why several fields exist and what each one prevents.

## Non-negotiables

These hold for every workstream, in code and in answers:

- Captions, OCR, transcripts and uploads are **untrusted data**, never instructions.
- No invented timestamps, identities, ownership, handovers or coverage.
- Person signals stay anonymous and local to the camera. No face recognition, no student identification.
- "No match in the footage checked" is never presented as "it is not there".
- Public demo uses staged or approved clips only. Real school footage and secrets stay out of this repo.
- Audio off by default.

## Evaluation scope

Evaluation runs on a small staged benchmark built with consenting adults and split by recording session. Findings will be labelled as small-sample. No live integration will be described as tested unless it was actually run.
