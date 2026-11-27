"""Small talk for the chat window.

A chat interface gets messages that are not searches at all: "hi",
"thanks", "what can you do?". Sending those to the footage search produced
answers like "No footage available for that period", which is true but
absurd. This module recognises the obvious cases and answers them directly.

It is deliberately narrow: it never says anything about what is in the
footage, and anything it does not recognise goes to the search as before.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_CLEAN = re.compile(r"[^a-z' ]+")

GREETINGS = {
    "hi", "hii", "hey", "hello", "hello there", "hi there", "hey there", "yo", "hiya",
    "good morning", "good afternoon", "good evening", "salam", "salaam",
    "assalamu alaikum", "as salamu alaykum", "marhaba", "ahlan",
}
THANKS = {
    "thanks", "thank you", "thank you so much", "thanks a lot", "thx", "ty",
    "cheers", "great thanks", "ok thanks", "okay thanks", "perfect thanks",
    "shukran", "jazakallah", "jazakallah khair",
}
BYES = {"bye", "goodbye", "see you", "see ya", "bye bye", "that's all", "thats all"}
HELP = {
    "help", "what can you do", "what do you do", "how does this work", "how do i use this",
    "who are you", "what are you", "how can you help", "how can you help me",
    "can you help me", "can you help", "hello can you help me", "hi can you help me",
    "what is this", "what is foundit", "i need help", "help me",
}


@dataclass
class Reply:
    kind: str
    text: str
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "text": self.text, "suggestions": self.suggestions}


def _normalise(message: str) -> str:
    text = _CLEAN.sub(" ", message.lower().replace("’", "'"))
    return " ".join(text.split())


def small_talk(message: str, suggestions: list[str]) -> Reply | None:
    """Return a reply for greetings, thanks, help and goodbyes, else None."""
    text = _normalise(message)
    if not text or len(text) > 60:
        return None
    if text in GREETINGS:
        return Reply(
            "greeting",
            "Hi! I'm FoundIt, the lost-and-found assistant. Tell me what you lost and "
            "roughly when, for example \"a gray bag this morning\", and I'll search the "
            "camera footage and show you the moment I find.",
            suggestions,
        )
    if text in HELP:
        return Reply(
            "help",
            "I search recorded camera footage for items that were put down and left. "
            "Describe the item (colour helps most) and when you lost it. I'll show "
            "possible matches with the video cued to the moment, how long nobody was "
            "near the item, and what I could not check. I can tell colour and position "
            "but not what kind of item something is, so always watch the clip.",
            suggestions,
        )
    if text in THANKS:
        return Reply(
            "thanks",
            "You're welcome. If you find it, collect it from the front desk, since "
            "I can't confirm whether anyone has handed it in.",
            [],
        )
    if text in BYES:
        return Reply("bye", "Good luck. I hope you find it.", [])
    return None
