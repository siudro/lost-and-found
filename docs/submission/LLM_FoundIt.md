# FoundIt — A Lost and Found Desk for Camera Footage

**WAICY 2026 · LLM & AI Agent Track**

Draft: 28 November 2026. Every number in this document comes from `docs/evaluation.md`, produced by `tools/evaluate.py` on 27 November 2026. Where something was not measured, it says so.

---

## 1. What specific service the AI provides

**FoundIt turns a stack of camera footage into something you can ask a question about.**

It is a lost and found desk — for one specific place, a school — where the person at the desk has watched every minute of every camera.

A student types:

> *"Have you seen a gray jacket? I lost it yesterday."*

and gets back:

> *Possible match at 09:19. The closest is a gray jacket draped over a chair by the window at 09:19 that was left unattended. Caveat: the camera was not recording for part of that period. No staff intake record exists for this item, so whether it reached the front desk cannot be verified from the footage.*

### Why this is specific and small

The track asks for a narrow domain, not a generalist. FoundIt is narrower than "video search" in a way that matters:

| Too broad | FoundIt |
|---|---|
| "Search all your video with AI" | Lost property in one school's fixed cameras |
| "Find a person in footage" | Finds a *left item*, and only reports it if it was left alone |
| "Describe what happened" | Answers who lost what, when it was abandoned, and whether it reached staff |

The scope is a school lost and found, because that is a concrete daily problem with a real owner (the front desk), a real user (a student who lost a jacket), and a real failure mode that hurts: telling someone their coat was handed in when it was not.

### The three things it must never confuse

This is the core of the service, and the reason it is an AI system rather than a search box:

1. **A sighting** — the footage shows the item.
2. **A likely unattended item** — the footage shows it left, with nobody nearby, for a sustained period.
3. **A verified handover** — staff formally received it.

A handover claim requires an authorised intake record. With no intake records — the demo's default — FoundIt says what the footage shows and states plainly that staff receipt cannot be verified. **It never infers a handover from somebody walking away with a bag.**

### What the user gets

- The moment, not a search result: *"09:19, chair by the window"*.
- A one-minute clip centred on that moment, with a seek offset.
- An honest statement of what footage was **not** checked.
- An explicit "cannot verify" for handovers.

---

## 2. Motivation

Every school has a lost and found. It is a box of things nobody can find, and a person at a desk who says "nothing has been handed in." The information a student needs — *where did I leave my jacket* — already exists, recorded, on the school's cameras. Nobody can search it.

The gap is not that cameras are missing. The gap is that footage is only usable by a person willing to scrub through an hour of video, and nobody does that for a lost jacket.

Existing tools do not close this gap:

- **Camera vendor search** (Verkada, Hikvision AcuSeek, Rhombus) returns *matching snapshots*. It does not know whether the item was abandoned, when, or whether its owner is still standing beside it.
- **TwelveLabs, Azure Video Indexer, Spot AI, Svid** are general video-AI platforms. They find "a gray jacket on a chair"; none has a notion of unattended onset or staff handover. Azure's face recognition is the opposite of what a school should run.
- **Sighti** (Riyadh, verified) turns CCTV into operations analytics for restaurants. Its focus is compliance, not lost property — but it is evidence that local venues already buy camera AI.

What is missing is not another general video search. It is the **specific judgement**: this item appeared, it stopped being attended, here is the second it happened, and here is what I cannot tell you.

### What inspired the honesty rule

Early in development the system was asked *"did anyone leave a red umbrella?"* on a day with no processed footage. A naive system answers "no." That answer is wrong and harmful: the footage did not exist. The distinction between **"no match in the footage checked"** and **"the footage does not exist"** became a first-class output field and a test.

The same logic applies to handovers. The failure mode that hurts most is telling a student their jacket is safe at the front desk when the footage only shows someone walking off with it.

---

## 3. Links

> **Chatbot (live):** _[insert deployed Railway URL]_
>
> **Chat history (≥ 10 messages, final edition):** _[insert link]_

Both are pass/fail items. The chat history must be recorded against the **final version** of the assistant, not an earlier draft.

---

## 4. Iterative AI design process

The track asks for **Idea → Advisor Design → Testing → Error Analysis → Improvement**. This section follows that shape with real evidence at each stage.

### 4.1 Idea

Started broader — "search all school footage" — and narrowed after testing whether the scope could support a single expert judgement. "Search video" cannot. "Was my item left behind, and did it reach staff" can, because it has one right answer and a measurable failure.

### 4.2 Advisor design

**Evidence first, language second** (ADR-1). Every answer is composed in code from stored records before any model sees it. The language model may rephrase; it may not decide. A rephrasing that adds a clock time not in the facts, or claims a handover or a sighting, is discarded and the composed text is used.

This ordering is the central design decision. A model asked to "answer from the footage" can invent a handover. A model asked to reword a fact sheet that never contains a handover cannot get one past the guard.

**Three timestamps, never one** (ADR-3): first seen, unattended onset, confirmation. Collapsing them misleads by 30 seconds or more.

**Classical detection before a learned one** (ADR-2): a dual-background left-object detector in OpenCV. "Something new that stayed still" is the definition of a left item and works for objects no detector class covers. It runs on a CPU at 11–12× real time, needs no weights or keys, and is explainable frame by frame.

**Bounded captioning** (ADR-4): up to 12 items per video get one vision call each — crop plus context frame, strict JSON schema, enums enforced again in code.

### 4.3 Testing

The prompts are versioned files in `prompts/`, loaded at startup, with the readiness check failing if one is missing. Current versions:

| Prompt | Version | Role |
|---|---|---|
| `chat_answer.md` | 1.0.0 | Rephrase a composed answer using only the given facts |
| `chat_smalltalk.md` | 1.0.0 | Reply to non-search chat, steer back to the item |
| `captioner.md` | 1.0.0 | Structured item description from a crop + context frame |
| `photo_query.md` | 1.0.0 | Structured description of a visitor's photo |

135 automated tests. The ones that matter most are the honesty tests, which fail the build if the app claims a handover, invents a timestamp, or says "no match" without stating coverage gaps.

**Adversarial testing was deliberate.** The system was pushed to fail on:

- an item present at clip start (onset must be marked *uncertain*, not guessed)
- a garment being worn (must never be reported as lost)
- a day with no processed footage (must not answer "not found")
- a query with no date (must ask, or say the date was assumed)

### 4.4 Error analysis — what actually went wrong

These are real defects found by running the system, not hypotheticals.

**Error 1 — the date parser silently fell back to today.**

Asked *"gray jacket, 20 November 2026"*, the parser did not recognise the format and defaulted to today. The app then answered *"no footage available"* — technically true and completely useless.

*Fix:* explicit date parsing for five formats (`20 November 2026`, `20 Nov 2026`, `November 20 2026`, `2026-11-20`, `20/11/2026`), including ordinal suffixes. A regression test pins all of them.

**Error 2 — the pickup detector used the wrong threshold.**

Item-movement detection reused the *person proximity* threshold (15% of frame width). A jacket being carried away moved less than that, so pickups were missed.

*Fix:* a separate `item_move_ratio` (3% of frame width). An item that shifts a few percent has been moved; a person standing a metre away is a different question entirely.

**Error 3 — retrieval joined the wrong table, producing "a item at 09:19".**

The chat read attributes from the event record, but item type and colour live on the observation. Every match came back with no colour and no type, and the answer degraded to *"a item at 09:19"*.

*Fix:* an explicit observation lookup in the store. Found by hitting the live HTTP endpoint — the unit tests passed because they checked each side separately, not the join.

**Error 4 — five false alarms on real CAVIAR footage.**

The detector fired on the empty-room background, on healed ghosts, on a lighting change, and on a glass railing's edge sliver. Each was fixed by a targeted check (see the PR history: `fix/ghost-recheck`, `fix/lighting-and-delete`, `fix/ignore-zones`, `fix/edge-slivers`).

**Error 5 — the deploy failed, and not because of Railway.**

The build config passed `${PORT:-8000}` as an exec argument rather than through a shell, so the variable was never expanded and the server exited with a literal string as its port. Replaced with a Dockerfile whose `CMD` is shell form.

### 4.5 Improvement

Each error produced a permanent guard rather than a patch:

| Error | Permanent guard |
|---|---|
| Silent date fallback | `tests/test_query_dates.py` — 6 tests across all formats |
| Wrong pickup threshold | `item_move_ratio` separated from `person_proximity_ratio` |
| Broken attribute join | `test_search_reads_what_ingest_wrote` |
| Detector false alarms | `tests/test_detector.py` + ignore zones |
| Money-losing handover claim | `tests/test_honesty.py` — 17 tests |

### 4.6 Measured results

From `tools/evaluate.py`, 27 November 2026, 2-core Xeon container, no GPU.

**Detection on simulated clips with known ground truth** (ground truth comes from the drawing script, so nothing is hand-labelled):

| Clip | Items found | TP | FP | FN | Onset error | Detection delay |
|---|---|---|---|---|---|---|
| left_behind (60 s) | 1 / 1 | 1 | 0 | 0 | +3.0 s | 30.0 s |
| attended (60 s) | 0 / 0 | – | 0 | – | – | – |
| picked_up (55 s) | 1 / 1 | 1 | 0 | 0 | +3.0 s | not confirmed before pickup |
| passers_by (25 s) | 0 / 0 | – | 0 | – | – | – |
| busy_start (50 s) | 1 / 1 | 1 | 0 | 0 | +4.0 s | 30.0 s |
| sunlight (45 s) | 1 / 1 | 1 | 0 | 0 | +4.0 s | not yet confirmed at clip end |
| classroom_jacket (10 min) | 1 / 1 | 1 | 0 | 0 | +13.4 s | 30.0 s |

**Totals: TP 5, FP 0, FN 0. Precision 1.0, recall 1.0.**

**What these numbers do not mean.** They are on *simulated* footage. They show the logic is correct end to end. They do not predict real-camera performance, and the classroom clip is stamped "SIMULATED" in every frame for exactly that reason.

**Retrieval:**

| Measure | Text query | Photo (colour only, no key) |
|---|---|---|
| Right item ranked first | 2 / 5 | 2 / 5 |
| Right item in top 3 | 5 / 5 | 5 / 5 |
| Median search time | 0.6 ms | 0.6 ms |

Top-1 is low because without an AI description the detector knows "a gray thing", not "a gray jacket". Three of five items are gray and tie. The system shows all of them with clips rather than guessing — which is the intended behaviour, not a defect.

**Speed:** 11.2–12.9× real time. A 10-minute classroom clip processes in about 54 seconds. Peak memory 304 MB. No GPU.

---

## 5. Limitations

### Practical

- **Jacket recall on real footage is unknown.** No public dataset has labelled indoor jacket abandonment. CAVIAR (CC BY-SA, 2003, 384×288) has bags, not garments. The staged jacket shoot has not been filmed. Any claim about real jacket performance would be unsupported.
- **The detector knows shape, not identity.** It reports "a gray thing", not "a gray jacket", unless a captioning key is set.
- **Four real clips is not a benchmark.** CAVIAR results were checked by eye, not scored against its XML ground truth — that harness is not written.
- **Simulated footage tests logic, not reality.** Lighting, occlusion and clutter in a real classroom are harder than the synthetic clips.
- **Live ingest does not carry track state across segments yet.**
- **Single venue.** SQLite, one process, one school's footage.

### Ethical

- **False reassurance is the serious harm.** Telling a student their jacket is at the front desk when it is not sends them away from where it actually is. This is why handovers require an intake record and why "no match" is never rendered as "not there".
- **Footage is sensitive even without faces.** A camera in a corridor records where specific children were at specific times. FoundIt stores no faces, identifies no one, and keeps person signals anonymous and local to the camera — but the footage itself remains personal data and needs a retention policy.
- **Unintended consequence: the system could become surveillance.** The same unattended-item logic could be pointed at people rather than objects. The design deliberately avoids person identification, and audio is off by default, but a deployment could still be misused. Worth stating in any school agreement.
- **Bias in detection.** A dark item on a dark floor, or a small item at 384×288, is harder to see. A system that misses some items more than others is unfair in a way users cannot detect.
- **Model output is untrusted data.** Captions and rephrasings are treated as content to check, never instructions to follow.

---

## 6. Future development

Each item answers a limitation above.

**Closed the jacket-recall gap** — film the staged benchmark (one consenting adult, gray hoodie on a chair, walked away from, 3–4 minutes; plus a similar attended jacket as a decoy, an item present at clip start, and a return-and-collect). Split evaluation by recording session. This is the single highest-value next step, because it is the only way to report a real number.

**Score CAVIAR properly** — the frame-level XML ground truth already exists. A harness that scores precision and recall against it would turn four eyeballed clips into a measured baseline.

**Upgrade the detector** — YOLO-World for open-vocabulary detection, with MobileSAM on crops. The interface is already designed for the swap: only `analyse_window` changes.

**Track state across live segments** — the known gap in live ingest.

**Local vision model** — dropping the captioning API would remove the last per-item cost and the last external dependency.

**Multi-venue** — Postgres and a job queue, if a district rather than a school wanted this.

**Intake desk integration** — the one change that would make verified handovers real rather than "cannot verify". The schema already separates handovers from events; the missing piece is a staff-facing log.

---

## 7. Conclusion

FoundIt turns a school's camera footage into something a student can ask a question about, and answers in the only way a lost and found desk should: *here is what the footage shows, here is when it was left, here is the clip — and here is what I could not check.*

The engineering is deliberately unglamorous. A classical detector instead of a learned one, because "something new that stayed still" is what a left item actually is. Answers composed in code before any model sees them, because a model that can rephrase a fact sheet cannot invent a handover. Three timestamps instead of one, because "when did I lose it" and "when did the system become sure" are different questions.

The measured results — precision 1.0, recall 1.0 on seven simulated clips, 11× real time on a CPU, no GPU and no mandatory API key — show the logic is correct end to end. They do not show real jacket performance, and this document says so rather than implying otherwise.

What makes the project worth submitting is not the detection accuracy. It is that the system is built to **refuse**: to refuse to claim a handover without a record, to refuse to say "not there" when the footage is missing, to refuse a confident onset when the item was already on the chair. That refusal is enforced in the schema, in the inference rules, and in seventeen tests that fail the build if it ever stops.

A lost and found desk that says "I don't know, and here is why" is more useful than one that guesses.

---

## 8. References

Berkeley, B., et al. (2024). *Segment Anything Model 2.* Meta AI. https://ai.meta.com/sam2/

CAVIAR Project. (2004). *Context Aware Vision using Image-based Active Recognition: test case scenarios.* EC Funded CAVIAR project/IST 2001 37540, University of Edinburgh. https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/

Cheng, T.-Y., et al. (2024). *YOLO-World: Real-Time Open-Vocabulary Object Detection.* CVPR 2024.

Drew, E., & Moore, T. (2026). *Railway pricing and plans.* Railway. https://docs.railway.com/pricing/plans

Krusch, P., Bochinski, E., Eiselein, V., & Sikora, T. (2017). A consistent two-level metric for evaluation of automated abandoned object detection methods. *IEEE ICIP 2017.* https://doi.org/10.5281/zenodo.1078507

Microsoft. (2026). *Azure AI Video Indexer pricing.* https://azure.microsoft.com/pricing/details/video-indexer/

OpenAI. (2026). *gpt-4o-mini model documentation and pricing.* https://developers.openai.com/api/docs/models/gpt-4o-mini

Sighti. (2026). *Case study: turning existing CCTV into business intelligence.* Gulf Startups. https://gulfstartups.co/case-study/sighti/

Spot AI. (2026). *Video AI platform.* https://www.spot.ai/

Sultani, W., Chen, C., & Shah, M. (2018). Real-world anomaly detection in surveillance videos. *CVPR 2018.* https://doi.org/10.1109/CVPR.2018.00678

Svid AI. (2026). *Natural-language search for CCTV.* https://apps.apple.com/app/id6791727276

TwelveLabs. (2026). *Video understanding platform and pricing.* https://www.twelvelabs.io/pricing

WAICY. (2026). *Evaluation rubrics.* https://www.waicy.org/RUBRICS/

Zhang, Z., et al. (2020). *DeepFashion2: A Versatile Benchmark for Detection, Pose Estimation, Segmentation and Re-Identification of Clothing Images.* CVPR 2019. https://github.com/switchablenorms/DeepFashion2

### Project artefacts

- Source: https://github.com/siudro/lost-and-found
- Evaluation: `docs/evaluation.md`, produced by `tools/evaluate.py`
- Architecture decisions: `docs/adr.md`
- Versioned prompts: `prompts/` (changlog in `prompts/CHANGELOG.md`)
- Competitor and cost analysis: `docs/market_and_costs.md`
- Dataset research: `docs/datasets.md`
