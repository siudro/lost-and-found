# Evaluation

What FoundIt has been measured on, the numbers, and what they do and do not show. Re-run with:

```
python tools/evaluate.py --json eval.json
```

Measured 27 Nov 2026 on a 2-core Intel Xeon 2.1 GHz container with no GPU and no API key.

## The honest headline

- On **simulated** clips with known ground truth, the pipeline found every left item and raised no false alarms: precision 1.0, recall 1.0 (5 true alerts, 2 clips with nothing to find).
- On **real** CAVIAR footage (4 clips), each clip gave exactly one correct item, checked by eye. That is a very small sample.
- **Jacket recall on real footage is unknown.** No public dataset has labelled indoor jacket abandonment. It needs the staged shoot (WAI-267).

The simulated numbers show the logic is right end to end. They do not predict real-camera performance.

## 1. Unattended detection on simulated clips

The ground truth comes from the scripts that draw the clips (`tests/scenes.py`, `tools/make_classroom_clip.py`), so nothing is hand-labelled. "Owner leaves" is when the owner starts walking away.

| Clip | What happens | Items found | TP | FP | FN | First seen error | Onset error | Detection delay | Pickup error | Colour right |
|---|---|---|---|---|---|---|---|---|---|---|
| left_behind (60 s) | bag put down at 6 s, owner leaves at 9 s | 1 / 1 | 1 | 0 | 0 | 0.0 s | +3.0 s | 30.0 s | – | yes |
| attended (60 s) | owner stays beside the bag | 0 / 0 | – | 0 | – | – | – | – | – | – |
| picked_up (55 s) | left at 6 s, collected at 40 s | 1 / 1 | 1 | 0 | 0 | 0.0 s | +3.0 s | not confirmed before pickup | +1.0 s | yes |
| passers_by (25 s) | three people walk through | 0 / 0 | – | 0 | – | – | – | – | – | – |
| busy_start (50 s) | someone in shot at the start; small dark bag left at 12 s | 1 / 1 | 1 | 0 | 0 | 0.0 s | +4.0 s | 30.0 s | – | yes |
| sunlight (45 s) | sun patch and shadow appear; dark bag left at 10 s | 1 / 1 | 1 | 0 | 0 | −1.0 s | +4.0 s | not yet confirmed at clip end | – | yes |
| classroom_jacket (10 min) | class, owner leaves the gray jacket on a chair, bell, room empties | 1 / 1 | 1 | 0 | 0 | −3.5 s | +13.4 s | 30.0 s | – | yes |

**Totals:** TP 5, FP 0, FN 0. Precision 1.0, recall 1.0.

How to read the columns:

- **Onset error** is the estimated unattended onset minus the moment the owner started leaving. It is positive because the estimate is taken when the owner is clearly gone, not when they take the first step. In the classroom, students mill around the jacket after the owner leaves, so the estimate lands later (+13.4 s).
- **Detection delay** is confirmation minus onset, the waiting period before "likely unattended". It is reported separately and is never used as the event time (ADR-3).
- **Not confirmed**: in `picked_up` the bag was collected, and in `sunlight` the clip ended, before the 30 s confirmation. Both still raised a "possibly unattended" alert with the correct onset.

## 2. Real footage: CAVIAR (small sample, checked by eye)

CAVIAR (CC BY-SA, 384×288, 2003) is the only freely licensed indoor left-bag footage we could reach. These four clips are the built-in samples on the website.

| Clip | Result | Checked against the video |
|---|---|---|
| LeftBag_AtChair | 1 item, left at 18.2 s, picked up at 37 s | correct |
| LeftBag_PickedUp | 1 item, left at 18.2 s, picked up at 41.6 s | correct |
| LeftBag | 1 item, left at 36.8 s, picked up at 54.8 s | correct |
| LeftBox | 1 item, left at 21.4 s, still there at the end | correct |

Getting here took five detector fixes, each driven by a false alarm on this footage: the empty-room background, ghost healing, the lighting test, periodic re-checks and the glass-railing ignore zone (see the PR history). Four clips is not a benchmark. The frame-level XML ground truth in CAVIAR would allow a proper score; that harness is not written yet.

## 3. Retrieval: text and photo

All items from section 1 were put in one database (all dated 20 Nov 2026), then each was searched for.

- **Text:** "Did anyone leave a {colour} {bag or jacket} today?"
- **Photo:** the item cut tightly from the footage stands in for a visitor's photo, searched without an API key, so by colour only.

| Measure | Text | Photo (colour only, no key) |
|---|---|---|
| Right item ranked first | 2 / 5 (0.4) | 2 / 5 (0.4) |
| Right item in the top 3 | 5 / 5 (1.0) | 5 / 5 (1.0) |
| Median search time | 0.6 ms | 0.6 ms (plus photo decoding) |

Why top-1 is low: without an AI description, the detector knows "a gray thing", not "a gray jacket" or "a gray bag". Three of the five items are gray, so they tie and the visitor sees all of them, with clips, in the first three results. That is the intended behaviour: show every possible match rather than guess. With `LLM_API_KEY` set, each item gets an item type and features (ADR-4), which separates jacket from bag. That path has unit tests with a fake model but has not been scored on real API output, to save the user's tokens.

## 4. Speed and resources

| Measure | Result |
|---|---|
| Processing speed | 11.2–12.9× real time (384×288 work size, 5 fps sampling) |
| 10-minute classroom clip | about 54 s to process |
| Peak memory (whole evaluation run) | 304 MB |
| Search latency | under 1 ms median, in-process |
| GPU | none needed |
| AI cost per processed video, with key | up to `CAPTIONS_PER_VIDEO` (default 12) vision calls at detail "low" |
| AI cost per chat answer, with key | one short call, at most 160 output tokens, cached, capped at 150 calls a day |

## 5. What is not measured yet

- Real jacket footage (needs the staged shoot).
- CAVIAR against its own XML ground truth.
- AI caption accuracy on real API output.
- Busy real scenes (crowded corridors, cafeterias).
- Night, infrared, or moving cameras.
