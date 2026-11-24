# Dataset research

Every entry below was checked by fetching it on **2026-11-24**. Status is what the server returned, not what a paper or a blog claims.

The project needs fixed-camera indoor video showing an item (especially a **jacket**) being placed and left, with labels for placement, departure, unattended onset and removal. That is a narrow thing, and most "abandoned object" datasets are not it. This document separates what we can actually use from what merely sounds right.

## Verdict up front

**No public dataset in this search contains labelled jacket-left-behind footage in a school setting.** The closest public data shows cups, bags and luggage abandoned outdoors or in stations. That is enough to build and test the *unattended logic*, which is the hard part, but not enough to claim jacket performance.

So the plan is two-track:

1. **Public data** proves the pipeline works on real abandonment footage — bags, cups, luggage.
2. **A staged jacket shoot** (WAI-267) supplies the garment-specific evaluation and the demo clips. This is not optional; without it we cannot honestly report jacket recall.

## Tier 1 — usable now, verified download

### Abandoned Object Detection Video Dataset (UniData)

- **URL:** https://huggingface.co/datasets/UniDataPro/abandoned-object-detection-video
- **Kaggle listing:** https://www.kaggle.com/datasets/unidpro/abandoned-object-detection-video-dataset
- **Claimed:** 1,062 videos, 1080p, 30 fps, 60–70 public locations, 50 object types
- **Actually downloadable:** **5 videos only.** Verified: each returned HTTP 200 and downloaded without a token.
- **Format:** MP4, H.264, 1920×1080, 30 fps, 31–53 seconds each, with an AAC audio track
- **Setting:** outdoor public locations — steps, plazas, walkways. Not school interiors.
- **Labels:** the description says "detailed annotations and metadata", but **no annotation files appear in the download**. Treat as unlabelled. Timestamps will have to be derived by us.
- **License:** CC BY-NC-ND 4.0
- **What it can test:** real placement and abandonment in a fixed camera; the appearance of a stationary object against moving people; whether our onset detection fires at a sensible time.

**Two problems with this dataset, both important:**

1. **It is a paid preview.** The Kaggle page says plainly: *"Buy the Dataset: This is a limited preview of the data. To access the full dataset, please contact us... to discuss your requirements and pricing options."* The 1,062-video headline is marketing; 5 clips are free.
2. **The license blocks use in the public demo.** CC BY-NC-ND means **NoDerivatives** — we may not distribute modified material. Our pipeline crops, annotates and re-renders footage, which is a derivative. The verification pass on this document confirmed the ND clause directly on the Creative Commons deed. Using these clips to *develop and evaluate* is fine; shipping them in the judge-facing demo is not. **Do not put these clips in the Space.**

**What we verified by watching it:** I downloaded `left-behined_item_0001.mp4` (78 MB), extracted frames, and confirmed by eye that a person sets down an item which then remains. The abandonment is real and legible. The item in the first clip is a **drink cup on a ledge**, not a jacket — which is the recurring pattern across this family of datasets.

### Recommendation

Use these 5 clips as the **pipeline smoke test**: does the system find a stationary object that people leave behind? Do not build the evaluation on them, and do not ship them.

## Tier 2 — blocked, dead, or unverified

### ABODA (Abandoned Object Detection Database)

- **URL cited in the brief:** http://www.mikelrodriguez.com/abandoned-object-database/
- **Status:** returns HTTP 200, **but the page is not the dataset.** It redirects to `/lander` — a domain-squatter parking page. The original host has lapsed.
- **Conclusion:** **treated as dead.** The dataset was published by Rodríguez et al. around the CAVIAR/PETS era and may exist in mirrors, but I could not verify a working download and will not cite one I have not checked. If we want it, the route is a mirror hunt, which is not worth the time compared to Tier 1 plus our own shoot.

### PETS 2006

- **URL:** https://www.cvg.reading.ac.uk/PETS2006/
- **Status:** **connection failed** (no response, timeout) from this environment. Not confirmed dead, but not reachable.
- **Likely content:** crowd-counting and LOITERING benchmark from Reading. Bags feature, but the benchmark is about crowds, not unattended items. Historical value only.
- **Note:** some PETS sets appear mirrored on Hugging Face; an `AVSS` name search returned only unrelated gesture data, so the mirrors are inconsistent. Nothing usable verified.

### AVSS 2007 Abandoned Baggage

- **URL tried:** https://www.eecs.qmul.ac.uk/~andrea/avss2007_d.html
- **Status:** **connection failed.**
- **Likely content:** the classic abandoned-baggage set — a bag left in a station, which is exactly the *behaviour* we need.
- **Verdict:** unverified. Worth one more access attempt from a normal network, because if it resolves it is the single most relevant public dataset for our unattended logic. Do not plan around it.

### UCF-Crime

- **URL:** https://www.crcv.ucf.edu/projects/real-world/ (live, 200) / https://huggingface.co/datasets/jinmang2/ucf_crime (mirror, CC0)
- **Content:** 1,900 real surveillance videos, 128 hours, 13 anomaly classes — fighting, robbery, arson and so on.
- **Relevance:** **low.** It contains "Stealing" and "Shoplifting", not abandonment. It is a useful *negative* set — normal surveillance activity our detector should ignore — but it is not abandonment footage.
- **License:** CC0 on the mirror; the original has academic terms. Fine for negatives.
- **Cheap use:** if we need "does the pipeline stay quiet on ordinary busy footage", sample normal-class clips from here.

### UA-DETRAC

- **URL:** https://huggingface.co/datasets/abhineet123/ua_detrac
- **Content:** vehicle detection/tracking on traffic video.
- **Relevance:** none for items. Listed only to close it out — the brief's pipeline needs person and object detection indoors, and this is cars outdoors.
- **License:** CC BY 4.0.

## Tier 3 — appearance, not footage

These are clothing datasets. They tell a model what a jacket looks like in a catalogue photo. They do not show a jacket on a chair, seen from a security camera, at an angle, partly occluded. **Do not treat them as footage** — the brief warns about exactly this.

### DeepFashion2

- **URL:** https://huggingface.co/datasets/sahirp/deepfashion2 (unverified content; `zyuzuguldu/deepfashion2-upper-body-masks` verified as a mask derivative, Apache-2.0)
- **Content:** garment photos on models, with segmentation for jackets, coats, hoodies.
- **Useful for:** prompts and class vocabulary — confirming "jacket", "hoodie", "coat" as detector terms, and possibly fine-tuning the segmenter on garment silhouettes.
- **Not useful for:** anything about placement, ownership or abandonment.

### Colorful Fashion Dataset (Kaggle) — from the brief

- Not verified in this pass. Same category as DeepFashion2: appearance only.

## What is missing and cannot be sourced

Nothing public gives us:

- a **jacket** specifically, left behind
- an **indoor school** setting
- labels for **placement / departure / onset / removal**
- footage that is **redistributable** for the public demo

That gap is the whole reason for WAI-267. Our staged shoot is not a convenience — it is the only way to produce a demo clip we are licensed to ship and the only way to report jacket recall.

## Shoot spec (feeds WAI-267)

Keep it small. One session, one phone or camera, fixed position:

1. Plain chair or table against a clear background, phone on a fixed mount, no zoom.
2. Adult sets down a **gray hooded jacket** and walks out of frame; leave 3–4 minutes untouched.
3. Second take: a **similar gray jacket** with someone sitting beside it the whole time — the attended decoy.
4. Third take: an item already on the chair when recording starts — tests the "present at clip start" uncertainty path.
5. Fourth take: a person returns and moves an item — tests the "moved" state.
6. One take outdoors or in a corridor if easy, for background variety.

Roughly 20 minutes total, one consenting adult, no faces close to camera. This gives every branch in `item_event.schema.json` at least one real clip.

## Honest summary for the written submission

- Two public sources verified downloadable: UniData (5 clips) and UCF-Crime (negatives).
- One named benchmark (ABODA) is dead at its published URL.
- Two (PETS 2006, AVSS 2007) unreachable from this environment and cannot be claimed.
- No public dataset supplies labelled jacket abandonment in a school — this must be stated, not hidden.
- Evaluation will be small-sample on staged footage, and will say so.
