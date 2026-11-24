# Lost and Found — shared interface

This directory defines the data contract every workstream codes against. It was written before the pipeline exists, on purpose: five workstreams can build in parallel only if they agree on what a "sighting" is before any of them produces one.

| File | Contract |
|---|---|
| `schemas/video_source.schema.json` | One footage source: recorded upload, live feed, or replay |
| `schemas/observation.schema.json` | One tracked sighting of one item over time |
| `schemas/item_event.schema.json` | A dated change: placed, unattended, moved, picked up, worn |
| `schemas/report_window.schema.json` | One 10-minute window: scene summary, events, coverage |
| `schemas/evidence.schema.json` | A seekable claim to footage supporting an event |
| `schemas/search_result.schema.json` | What retrieval hands the chat, including what it could not search |
| `schemas/handover_record.schema.json` | An authorised staff intake record — the only handover authority |

`fixtures/` holds a valid example of every contract, using a single staged demo source. Workstreams can point their code at these files today.

## Run the checks

```bash
pip install jsonschema
python tests/validate_fixtures.py            # schema + cross-record invariants
python tests/test_rules_catch_violations.py  # proves the rules actually reject bad data
```

`validate_fixtures.py` checks more than shape. It verifies that events point at real windows, evidence seeks to its own event moment, windows are exactly ten minutes, coverage claims match coverage gaps, and — most importantly — that the project's hard rules hold across records.

`test_rules_catch_violations.py` mutates the good data into each forbidden state and asserts the validator rejects it. Without that second file, a green validator would prove only that it never fails.

## The four decisions that matter

These are the places where a casual design would let the system say something the footage does not support.

**1. A handover is not an event type.**

`ItemEvent` has no handover variant, and `item_event.schema.json` enumerates its `event_type` values so a handover cannot be added without deliberately editing the contract. A verified handover lives only in `handover_record`, which carries `intake_reference`, `recorded_by` and a `verification` block. The pipeline cannot write to that collection. So when the demo has no intake records — which it does not, initially — the answer is not "recovered" or "handed in", it is what the footage shows plus a statement that staff receipt cannot be verified.

**2. Three timestamps, never one.**

`Observation` separates `first_seen_at`, `unattended_onset_estimate` and `confirmed_at`. `confirmed_at` is when the threshold was crossed; the onset is when the item actually stopped being attended. Detection delay is `confirmed_at - onset`, and the two must not be reported as if they were the same instant. The validator rejects a confirmation that precedes its own onset estimate.

**3. Worn is not unattended, and offset zero is not the beginning of time.**

`is_worn` is a separate boolean, not a state. The validator rejects any worn observation in an unattended state and any retrieval result that returns a worn garment as a lost-item match. Separately, an item visible at media offset 0 has unknown history: its `onset_certainty` must be `uncertain` and `present_at_clip_start` must appear in `uncertainty`.

**4. "No match" and "no footage" are different answers.**

`SearchResult.coverage_limitations` is a required array, and `answer_mode` constrains what the chat may claim. `no_match_in_footage_checked` demands at least one stated limitation, so the chat cannot say "nothing matched" without also saying what it was unable to check. `footage_not_available` demands a footage-level reason, so missing or unprocessed footage is never rendered as "the item was not found". The fixtures include a query for footage that was never processed, and one where the search was broadened and must say so.

## Conventions

- Time is UTC in every stored field (`*_at`, ISO 8601 with offset). Venue-local rendering happens at the edges: `VideoSource.timezone` and `SearchResult.parsed.date_range.timezone_used`. Media-relative offsets (`*_offset_s`) are always kept alongside, because playback survives restarts while wall clocks do not.
- Identifiers are content-derived where the record must be stable across reprocessing: `event_id`, `window_id`, `evidence_id`. Reprocessing a window with the same `processing.content_hash` therefore overwrites rather than duplicates.
- Unknown is `null`, never a guess. Empty arrays mean "none observed", not "none exist".
- `caption_text` and `scene_summary.text` are untrusted data. They may be searched and quoted as evidence; they are never instructions.
- `access.visibility` is enforced when serving clips and when returning results, not when storing them.

## Changing the contract

This is a shared file. Change it in a pull request that names the workstream affected, and update the fixtures in the same change — the validation tests are the guard against one workstream silently redefining a field another depends on.
