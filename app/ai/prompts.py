"""Load the versioned system prompts in /prompts.

The prompt files are the single source of truth for what the running app
sends to a model. Each has a small header (name, version) and the prompt
text. Nothing is hard-coded elsewhere, so the committed file is provably the
prompt in use; ``prompt_versions()`` is shown in the readiness report.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
REQUIRED = ("chat_answer", "chat_smalltalk", "captioner", "photo_query")


class PromptError(RuntimeError):
    pass


@lru_cache(maxsize=None)
def load(name: str) -> tuple[str, str]:
    """Return (version, text) for a prompt."""
    path = PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise PromptError(f"prompt file missing: prompts/{name}.md")
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---\n"):
        raise PromptError(f"prompts/{name}.md has no header")
    header, _, body = raw[4:].partition("\n---\n")
    meta = dict(line.split(":", 1) for line in header.splitlines() if ":" in line)
    meta = {k.strip(): v.strip() for k, v in meta.items()}
    if meta.get("name") != name or not meta.get("version"):
        raise PromptError(f"prompts/{name}.md header must name it '{name}' and give a version")
    text = body.strip()
    if len(text) < 40:
        raise PromptError(f"prompts/{name}.md is empty")
    return meta["version"], text


def text(name: str) -> str:
    return load(name)[1]


def prompt_versions() -> dict[str, str]:
    return {name: load(name)[0] for name in REQUIRED}
