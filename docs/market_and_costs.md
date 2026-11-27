# Market, costs and accounts

Checked 27 Nov 2026. Prices change; links go to the source of each figure.

## 1. Who else does this

Natural-language search over camera footage is now a common feature. What none of the products below are built around is the lost-and-found question itself: "was my jacket left here, when did it become unattended, and was it handed in?"

| Product | What it is | How it compares to FoundIt |
|---|---|---|
| [TwelveLabs](https://www.twelvelabs.io/pricing) | Video foundation models behind an API: semantic search returns time-stamped moments; an "Analyze" model answers questions about a video. | A general building block, priced per indexed hour (see section 2). It finds "a gray jacket on a chair" but has no notion of unattended onset, owner departure or handover. FoundIt could use it as a retrieval back end. |
| [Azure AI Video Indexer](https://azure.microsoft.com/pricing/details/video-indexer/) | Microsoft's cloud and edge video analytics: object detection, labels, OCR, scenes; higher presets add face and celebrity recognition. Priced per input minute. | Broad tagging, not lost-item reasoning. Its face recognition is the opposite of FoundIt's no-face design. |
| [Spot AI](https://www.spot.ai/) | Video AI platform on a business's existing cameras, with pre-built "AI coworker" agents for security, safety and operations, and a conversational interface (Iris). | Enterprise security and operations suite, sold through demos. Lost property is not a named use case. |
| [Svid AI](https://apps.apple.com/app/id6791727276) | iPhone/iPad app that makes CCTV, uploads, URLs or the phone camera searchable in plain English, with alerts. States it does no facial recognition. Paid by credits. | Closest in spirit (plain-English search, no faces). General search, no unattended-item logic or staff handover record. |
| Camera-vendor search (Verkada, Hikvision AcuSeek, March Networks, OpenEye, Rhombus) | Type-to-search features built into each vendor's cameras or recorders, often CLIP-style text-image matching. | Tied to that vendor's hardware. They return matching snapshots; they do not judge whether an item was abandoned. |
| [Sighti](https://gulfstartups.co/case-study/sighti/) (Riyadh) | Pre-seed Saudi startup founded 2025 that turns existing CCTV into business intelligence, focused on restaurants and food-and-beverage compliance ([PitchBook](https://pitchbook.com/profiles/company/819315-46), [Antler](https://careers.antler.co/companies/sighti-2)). | Verified as real and Riyadh-based. Its focus is operations analytics, not lost property. Useful as proof that local venues buy CCTV AI. |

**Where FoundIt is different**

- It answers the lost-property question, with three separate timestamps (first seen, unattended onset, confirmation).
- It refuses to claim a handover without a staff intake record.
- It always says what footage it did not check.
- No faces and no identification.
- It runs on one small CPU server, with or without an API key.

## 2. Costs

Rates used:

- **OpenAI gpt-4o-mini:** $0.15 per million input tokens and $0.60 per million output tokens ([OpenAI](https://developers.openai.com/api/docs/models/gpt-4o-mini)). A low-detail image costs gpt-4o-mini a fixed 2,833 input tokens ([measured write-up](https://strathweb.com/2024/10/how-gpt-4o-mini-can-be-simultaneously-20x-cheaper-and-2x-more-expensive-than-gpt-4o/)), about $0.00042 per image.
- **Railway** ([pricing](https://docs.railway.com/pricing/plans)):
  - $20 per vCPU-month, $10 per GB of RAM a month, $0.15 per GB of volume a month.
  - The Hobby plan is $5 a month and includes $5 of usage, with volumes up to 5 GB.
- **Measured speed** (`docs/evaluation.md`): 11–13× real time on one container, peak memory about 300 MB.

### One hour of recorded footage

| Item | Estimate |
|---|---|
| Processing (about 5 minutes of 1 vCPU and 0.3 GB) | about $0.003 |
| AI descriptions: up to 12 items × 2 images plus about 400 text tokens, about 100 tokens out | about $0.011 |
| Playback copy (960 px, H.264 CRF 26): roughly 0.25–0.7 GB for real camera footage | $0.04–0.11 a month to keep |
| **Total** | **about 1.5 cents once, plus a few cents a month of storage** |

For comparison, TwelveLabs indexing is listed at $0.042 a minute, which is $2.52 an hour, plus $0.0015 a minute ($0.09 an hour) of monthly infrastructure ([TwelveLabs pricing](https://www.twelvelabs.io/pricing)).

### One day of one live camera

| Item | Estimate |
|---|---|
| Processing, keeping up with real time (about 1/11 of a vCPU all day) | about $0.06 a day |
| AI descriptions, assuming about 50 found items | about $0.05 a day |
| Storage: 24 hours is roughly 6–17 GB at the same settings | $0.90–2.50 a month per day kept |

**Total:** about 11 cents a day of compute and AI, with storage as the real cost. One day of footage can exceed the Hobby plan's 5 GB volume, so live use needs Railway Pro or a retention limit (for example, keep 7 days at a lower resolution).

### One chat question

| Item | Estimate |
|---|---|
| Search (in-process, SQLite) | under 1 ms, no cost |
| Rephrasing, with key: about 600 tokens in, at most 160 out | about $0.0002 |
| Photo question, with key: add one image | about $0.0006 more |
| Repeated identical question | $0 (cached) |
| Daily cap | 150 AI calls a day by default (`LLM_DAILY_CALL_LIMIT`) |

At the default cap the worst case is about 150 × $0.0008 = $0.12 a day.

## 3. Accounts and keys

Only the first row is needed for the AI features. Everything else is optional or for later.

| What | Needed for | Where | Cost | Status |
|---|---|---|---|---|
| OpenAI API key, set as Railway variable `LLM_API_KEY` | AI wording, item descriptions, photo descriptions | [platform.openai.com/api-keys](https://platform.openai.com/api-keys), then Railway → service → Variables | Pay as you go; see section 2 | **You have the key. Add it in Railway yourself; never paste it into code.** |
| Railway account | Hosting the website | [railway.com](https://railway.com/pricing) | Hobby $5 a month (includes $5 usage); Pro $20 for live use | Done |
| GitHub | Code and CI | github.com/siudro/lost-and-found | Free | Done; CI runs on every PR |
| Hugging Face account and read token (`HF_TOKEN`) | Downloading gated model weights such as SAM, later | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Free account | Not needed now |
| Gated checkpoints (for example facebook/sam-vit-base) | Optional SAM segmentation | Model page → "Agree and access repository" | Free, but licence must be accepted | Not needed now; MobileSAM weights are on GitHub without gating |
| Hugging Face Spaces | Alternative hosting | [huggingface.co/pricing](https://huggingface.co/pricing) | Listed as free CPU Basic (2 vCPU, 16 GB); GPU from $0.40 an hour | **Not recommended for the deadline:** since mid-2026, users report that Docker and CPU-Basic Spaces need a PRO plan ($9 a month) on new accounts ([HF forum](https://discuss.huggingface.co/t/new-free-accounts-cannot-create-cpu-basic-gradio-spaces-only-zerogpu-available/177629)). Railway already works. |
| GPU | Faster learned detectors | Railway has none; HF Spaces T4 at $0.40 an hour | — | Not needed: the detector runs on CPU at 11× real time |
| Storage | Footage | Railway volume at `/data` ($0.15 per GB a month; 5 GB cap on Hobby) | — | Fine for the demo; plan Pro or retention for live |
