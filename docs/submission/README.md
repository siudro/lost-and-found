# Submission document

`LLM_FoundIt.md` is the source. `LLM_FoundIt.pdf` is generated from it — never edit the PDF.
`LLM_FoundIt.pdf` is the file that goes in the Drive folder.

Regenerate after any edit:

```bash
pip install markdown weasyprint
python tools/build_submission_pdf.py
```

## Before submitting

Three placeholders are highlighted yellow in the PDF. Search the source for `[insert`:

1. **Chatbot link** — the live Railway URL, verified from a fresh browser session.
2. **Chat history link** — at least 10 exchanged messages against the **final** version of the assistant. Not an older draft.

Both are pass/fail with the judges. A broken link or a 9-message history fails regardless of how good the document is.

## Naming

The Drive folder must be `LLM_FoundIt` and the file must be `LLM_FoundIt.pdf`. The track has no Elementary/Middle/High category, so the folder goes directly inside the `LLM` folder.

## What this document claims, and where it comes from

| Section | Source |
|---|---|
| Measured results (precision, recall, speed, memory) | `docs/evaluation.md`, produced by `tools/evaluate.py` on 27 Nov 2026 |
| Prompt versions | `prompts/CHANGELOG.md` |
| Design decisions | `docs/adr.md` |
| Competitors and costs | `docs/market_and_costs.md` |
| Dataset limitations | `docs/datasets.md` |

Every number is traceable to a file. If a number changes, regenerate from the source rather than editing the PDF — an outdated figure in a submission is worse than no figure.
