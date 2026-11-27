# For other agents

If you are an AI picking this project up, read this before touching anything. It exists because five agents once worked on the same repo and two of them edited the same file from opposite assumptions.

## Read in this order

1. `docs/interface.md` — why each schema field exists. Do not add fields without reading this.
2. `docs/architecture.md` — what calls what, and which modules are pure.
3. This file — ownership and conventions.
4. `docs/datasets.md` — what data exists and what is licensed.

## Ownership

Every file has one owner. If you need a change in a file you do not own, open a PR and say why.

| Area | Files | Owner |
|---|---|---|
| Data contract | `schemas/*`, `fixtures/*` | **integration owner** (currently the coordinating agent) |
| Time and windows | `app/core/timeutil.py` | pipeline owner |
| Unattended logic | `app/core/unattended.py` | pipeline owner |
| Ingest | `app/core/pipeline.py` | pipeline owner |
| Query parsing | `app/core/query.py` | chat owner |
| Answer composition | `app/core/answer.py` | chat owner |
| Chat service | `app/core/chat.py` | chat owner |
| Storage | `app/core/store.py` | storage owner |
| HTTP | `app/main.py`, `app/web/*` | deployment owner |
| AI providers | `app/ai/providers.py` | integration owner |
| Tests | `tests/*` | whoever changed the behaviour |

## The rules you must not break

These are enforced by `tests/test_honesty.py`. If your change breaks them, the test will tell you, and the right response is to change your code — not the test.

1. **A handover cannot be an event type.** `item_event.schema.json` has no handover value, and `answer.py` reads handovers only from the `handovers` table.
2. **Three timestamps stay separate.** `first_seen_at`, `unattended_onset_estimate`, `confirmed_at`.
3. **Worn is not unattended.** Validated in schema, in `unattended.infer`, and in retrieval.
4. **"No match" must state what was not checked.** `compose_no_match` requires limitations.
5. **Never invent a timestamp.** `offset_to_absolute` returns `None` without a clock, and callers must handle `None`.

## Adding an AI provider

1. Add an entry to `PROVIDERS` in `app/ai/providers.py`.
2. Fill in `purpose` and `without_it`. A provider with no honest fallback statement is not ready.
3. Add its env var to `.env.example`.
4. Do **not** make it required. The app must start with every key unset.

The provider registry is the single source of truth for "what AI does this need" — the startup diagnostics and `/api/providers` both read from it.

## Adding an endpoint

Endpoints live in `app/main.py`. Keep request handling thin: parse, call a service, return. Business rules belong in `app/core/`.

Add a test in `tests/test_api.py` that hits the real app object.

## Test policy

- Every behaviour change gets a test. No exceptions for "small" changes — the honesty rules are small changes.
- If a test fails, check whether the test or the code is wrong. Do not delete a failing test to make CI green.
- `tests/validate_fixtures.py` must stay at 0 failures. If you change a schema, update the fixtures in the same PR.

## Where the seams are

| Seam | Risk | Guarded by |
|---|---|---|
| ingest → store | column names drift | `test_search_reads_what_ingest_wrote` |
| store → answer | attributes not joined | `test_chat_service_and_store_agree_on_item_names` |
| query → store | synonyms not applied | `test_grey_and_gray_find_the_same_record` |
| pipeline → ffmpeg | decode fails silently | `test_pipeline_processes_a_real_generated_clip` |

## What is not built yet

Be suspicious of any claim that it is. As of this commit:

- No speaker/audio handling (audio off by design).
- Live camera and replay go through `tools/live_ingest.py` (ffmpeg segments posted to `/api/ingest` with `kind=live_feed` or `replay`). Track state is not carried across segments yet.
- No SAM segmenter enabled.
- Evaluation harness covers simulated clips with known truth (`tools/evaluate.py`, results in `docs/evaluation.md`). CAVIAR is scored by eye only; scoring against its XML ground truth is not written yet.

When you finish a piece, update this section. A stale "not built yet" list is how agents duplicate work.
