"""OpenAI connection for the chat, built to be cheap and unable to invent facts.

What the model is allowed to do:

* rephrase an answer the footage search has already composed, so the reply
  reads like a person wrote it;
* reply to chat that is not a search ("how are you", "my name is…") and
  steer back to describing the lost item.

What it is not allowed to do, enforced in code rather than only asked for:

* state a time, number or sighting that is not in the search result. Every
  clock time and number in a reply is checked against the facts it was
  given; a reply that adds one is thrown away and the composed text is used;
* claim an item was handed in or collected by staff (there are no intake
  records), checked the same way.

Cost controls (the key belongs to the user and they asked for minimal use):

* no call at all for greetings and thanks (handled in ``conversation.py``);
* at most one call per chat message, with a short prompt and a hard cap on
  reply length (``LLM_MAX_OUTPUT_TOKENS``, default 160);
* identical questions with identical results are answered from a cache;
* a daily ceiling on calls (``LLM_DAILY_CALL_LIMIT``, default 150); past it,
  replies fall back to the composed text until midnight UTC;
* a 12-second timeout; any failure falls back silently to the composed text.

The key is read from the environment (``LLM_API_KEY`` or ``OPENAI_API_KEY``)
and is never logged, stored, returned by any endpoint or put in an error.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone

API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"

_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_FORBIDDEN = re.compile(
    r"\b(handed (it )?in|turned in|returned to (the )?(front desk|reception|owner)|"
    r"(staff|security|reception|front desk) (has|have|collected|received|is holding|are holding)|"
    r"is (at|with) the front desk|definitely yours|is yours)\b",
    re.IGNORECASE,
)

# The exact prompts live in /prompts as versioned files (see prompts/CHANGELOG.md).
from .prompts import text as prompt_text  # noqa: E402


def SEARCH_SYSTEM() -> str:
    return prompt_text("chat_answer")


def CHAT_SYSTEM() -> str:
    return prompt_text("chat_smalltalk")


@dataclass
class LLMUsage:
    day: str = ""
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    rejected: int = 0
    failures: int = 0
    cache_hits: int = 0

    def to_dict(self) -> dict:
        return {
            "day": self.day, "calls": self.calls, "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens, "rejected_replies": self.rejected,
            "failures": self.failures, "cache_hits": self.cache_hits,
        }


@dataclass
class LLMClient:
    api_key: str | None = None
    model: str = DEFAULT_MODEL
    max_output_tokens: int = 160
    daily_call_limit: int = 150
    timeout_s: float = 12.0
    transport: object = None
    """Callable(url, headers, body_bytes, timeout) -> dict. Tests replace it;
    by default a real HTTPS request is made."""
    usage: LLMUsage = field(default_factory=LLMUsage)
    _cache: "OrderedDict[str, str]" = field(default_factory=OrderedDict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def from_env(cls) -> "LLMClient":
        def number(name: str, default: int) -> int:
            try:
                return max(0, int(os.environ.get(name, default)))
            except ValueError:
                return default

        return cls(
            api_key=(os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip() or None,
            model=(os.environ.get("LLM_MODEL") or DEFAULT_MODEL).strip(),
            max_output_tokens=number("LLM_MAX_OUTPUT_TOKENS", 160),
            daily_call_limit=number("LLM_DAILY_CALL_LIMIT", 150),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def status(self) -> dict:
        """Safe to show publicly: never includes the key."""
        with self._lock:
            self._roll_day()
            return {
                "enabled": self.enabled,
                "provider": "openai",
                "model": self.model,
                "daily_call_limit": self.daily_call_limit,
                "max_output_tokens": self.max_output_tokens,
                "usage_today": self.usage.to_dict(),
            }

    # ------------------------------------------------------------------ public

    def phrase_answer(self, message: str, answer: dict) -> str | None:
        """A friendlier reply for a search answer, or None to keep the composed text."""
        if not self.enabled:
            return None
        facts = _facts(answer)
        user = f"Visitor: {message[:300]}\n\nFACTS:\n{facts}"
        reply = self._complete(SEARCH_SYSTEM(), user, cache_key=f"a:{message.strip().lower()}:{facts}")
        if reply is None:
            return None
        if not _supported_by(reply, facts):
            with self._lock:
                self.usage.rejected += 1
            return None
        return reply

    def chat_reply(self, message: str) -> str | None:
        """Reply to a message that is not a search, or None."""
        if not self.enabled:
            return None
        reply = self._complete(CHAT_SYSTEM(), message[:300], cache_key=f"c:{message.strip().lower()}")
        if reply is None:
            return None
        # A chat reply has no facts to rely on: it may not contain any time,
        # number or claim of a sighting at all.
        if _TIME.search(reply) or _FORBIDDEN.search(reply) or re.search(
            r"\b(I (found|saw|see|spotted|located)|in the footage|on camera)\b", reply, re.IGNORECASE
        ):
            with self._lock:
                self.usage.rejected += 1
            return None
        return reply

    def vision_json(self, system: str, user_text: str, images: list[bytes], schema: dict,
                    cache_key: str) -> dict | None:
        """One vision call with structured (JSON schema) output, or None.

        Images are JPEG bytes sent at detail "low" (a fixed, small token
        cost each). Shares the daily call limit and cache with the chat.
        """
        if not self.enabled:
            return None
        import base64

        digest = hashlib.sha256(b"".join(images) + cache_key.encode()).hexdigest()
        content = [{"type": "text", "text": user_text}] + [
            {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(img).decode(), "detail": "low"}}
            for img in images
        ]
        raw = self._complete(
            system, content, cache_key=f"v:{digest}",
            response_format={"type": "json_schema", "json_schema": {"name": "description", "schema": schema, "strict": True}},
            max_tokens=220,
        )
        if raw is None:
            return None
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            with self._lock:
                self.usage.rejected += 1
            return None
        return data if isinstance(data, dict) else None

    # ----------------------------------------------------------------- private

    def _roll_day(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.usage.day != today:
            self.usage = LLMUsage(day=today)

    def _complete(self, system: str, user, cache_key: str, response_format: dict | None = None,
                  max_tokens: int | None = None) -> str | None:
        key = hashlib.sha256(cache_key.encode()).hexdigest()
        with self._lock:
            self._roll_day()
            if key in self._cache:
                self._cache.move_to_end(key)
                self.usage.cache_hits += 1
                return self._cache[key]
            if self.usage.calls >= self.daily_call_limit:
                return None
            self.usage.calls += 1  # counted before the call so concurrent requests cannot overshoot

        request = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            # max_tokens is deprecated and rejected by reasoning models.
            "max_completion_tokens": max_tokens or self.max_output_tokens,
        }
        if response_format:
            request["response_format"] = response_format
        if not re.match(r"^(o\d|gpt-5|gpt-6)", self.model):
            request["temperature"] = 0.3  # reasoning models only accept the default
        body = json.dumps(request).encode()
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            data = (self.transport or _https_post)(API_URL, headers, body, self.timeout_s)
            text = data["choices"][0]["message"]["content"].strip()
            usage = data.get("usage") or {}
        except Exception:  # never surface details: they could echo request headers
            with self._lock:
                self.usage.failures += 1
            return None
        with self._lock:
            self.usage.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.usage.completion_tokens += int(usage.get("completion_tokens") or 0)
        if not text:
            return None
        with self._lock:
            self._cache[key] = text
            while len(self._cache) > 300:
                self._cache.popitem(last=False)
        return text


def _https_post(url: str, headers: dict, body: bytes, timeout: float) -> dict:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _facts(answer: dict) -> str:
    """The search result as compact text: everything the model may say."""
    lines = [f"Result: {answer.get('headline', '')}", f"Summary: {answer.get('lead') or answer.get('body', '')}"]
    for index, m in enumerate((answer.get("matches") or [])[:4], start=1):
        lines.append(
            f"Match {index}: time {m.get('local_time')}; colour {m.get('colour') or 'unclear'}; "
            f"type {m.get('item_type') or 'unknown'}; status {m.get('difference_note') or m.get('state')}; "
            f"place {m.get('location_label') or 'not recorded'}"
        )
    extra = len(answer.get("matches") or []) - 4
    if extra > 0:
        lines.append(f"Further matches: {extra}")
    for limit in (answer.get("coverage_limitations") or [])[:3]:
        lines.append(f"Could not check: {limit.get('description')}")
    for day in (answer.get("available_dates") or [])[:6]:
        lines.append(f"Footage exists for: {day}")
    if answer.get("scope_statement"):
        lines.append(f"Scope: {answer['scope_statement']}")
    return "\n".join(lines)


def _supported_by(reply: str, facts: str) -> bool:
    """Every clock time and number in the reply must appear in the facts."""
    if _FORBIDDEN.search(reply):
        return False
    times_in_facts = set(_TIME.findall(facts))
    for found in _TIME.findall(reply):
        if found not in times_in_facts and found.zfill(5) not in times_in_facts:
            return False
    numbers_in_facts = set(_NUMBER.findall(_TIME.sub(" ", facts)))
    for found in _NUMBER.findall(_TIME.sub(" ", reply)):
        if found not in numbers_in_facts:
            return False
    return True
