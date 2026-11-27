# Architecture decisions

Short records of the choices that shape FoundIt, why they were made, and what would make us revisit them. Newest last.

## ADR-1: Evidence first, language second

**Decision.** Every answer is composed in code from stored records (`app/core/answer.py`). The language model may only rephrase that composed answer or make small talk. A rephrasing that adds a clock time or number not in the facts, or claims a handover or a sighting, is thrown away and the composed text is used.

**Why.** The failure that matters most is telling someone their jacket was handed in when it was not. A model asked to "answer from the footage" can say that. A model asked to reword a fact sheet that never contains a handover cannot get one past the guard.

**Revisit if** users find the composed text too stiff even with the key set.

## ADR-2: Classical left-object detection before a learned detector

**Decision.** The baseline detector (`app/core/detector.py`) is a dual-background left-object detector in OpenCV: a long background (median of the first 8 s, plus an untouched copy of the empty room) and a short one. A region that differs from the long background but matches the short one for 3 s is a still new thing. Ghost, lighting and edge-sliver checks remove the common false alarms. No YOLO or SAM is required to run.

**Why.** "Something new that stayed still" is the definition of a left item, and it works for any object, including ones no detector class covers. It runs on Railway's CPU at about 11–12× real time on 384×288 footage, needs no weights or keys, and its behaviour is explainable to judges frame by frame.

**Cost.** It knows "a gray thing", not "a jacket". Item type comes from the AI captioner (ADR-4) when a key is set. A learned detector (YOLO-World, or a detector plus MobileSAM on crops) is the planned upgrade for clutter; see `docs/running.md`.

**Revisit if** real staged footage shows recall below the synthetic results, or busy scenes swamp the background model.

## ADR-3: Three timestamps, never one

**Decision.** Each observation keeps `first_seen_at`, `unattended_onset_estimate` and `confirmed_at`. Events use the onset as their time. Detection delay (confirmed minus onset) is reported separately and never as the event time.

**Why.** "When did I lose it?" is answered by the onset; confirmation is just when the system became sure. Collapsing them misleads by 30 seconds or more (see `docs/evaluation.md`).

## ADR-4: Vision captions are bounded and schema-checked

**Decision.** With `LLM_API_KEY`, up to `CAPTIONS_PER_VIDEO` items per video get one vision call each: a crop and a context frame at detail "low", strict JSON schema, enums enforced again in code. Low confidence, worn, or "not an item" keeps the record marked uncertain.

**Why.** It adds what the classical detector cannot know (item type, features) at a predictable, small cost, and it cannot add free text that the chat might repeat as fact.

**Revisit if** a local vision model becomes fast enough on CPU to drop the API.

## ADR-5: Photos are processed in memory and never stored

**Decision.** `POST /api/chat/photo` decodes, describes and discards the photo. Without a key, only the main colour is used and the answer says so. Photo results are always "possible matches".

**Why.** A visitor's photo can contain faces and personal details. Not storing it removes a whole class of privacy risk at no cost to the feature.

## ADR-6: SQLite on a volume, one process

**Decision.** One FastAPI process with SQLite at `/data/foundit.db` on a Railway volume; media and crops under `/data/media`.

**Why.** A lost-and-found desk has one venue's footage and a handful of users. SQLite removes a moving part, survives redeploys on the volume, and is easy to inspect.

**Revisit if** several venues or many concurrent processing jobs are needed: move to Postgres and a job queue.

## ADR-7: Simulated footage is labelled, never passed off as real

**Decision.** The classroom clip is drawn by `tools/make_classroom_clip.py`, stamped "SIMULATED" in every frame, and its results are reported separately from real footage.

**Why.** No public dataset has labelled indoor jacket abandonment. The simulation proves the logic end to end with known ground truth; only staged real footage (still to be filmed) can give real jacket recall.
