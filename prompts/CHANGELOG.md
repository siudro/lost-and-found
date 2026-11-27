# Prompt changelog

The files in this folder are the exact system prompts the running app sends to the model. `app/ai/prompts.py` loads them at startup and the readiness check fails if one is missing or malformed. Change a prompt only together with its version number and an entry here.

| Date | Prompt | Version | Change |
|---|---|---|---|
| 2026-11-27 | chat_answer | 1.0.0 | First version: rephrase a composed search answer using only the facts given; code rejects replies that add times, numbers, handovers or sightings. |
| 2026-11-27 | chat_smalltalk | 1.0.0 | First version: reply to non-search chat and steer back to describing the lost item; may not claim to have seen anything. |
| 2026-11-27 | captioner | 1.0.0 | First version: structured description of a detected item from a crop and a context frame; unknown stays null; flags non-items and worn items. |
| 2026-11-27 | photo_query | 1.0.0 | First version: structured description of a visitor's photo for comparison with stored items. |
