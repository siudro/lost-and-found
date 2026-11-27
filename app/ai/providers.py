"""Provider registry.

Every external AI dependency is declared here with an honest statement of
what it needs and what happens without it. This module is what the startup
diagnostics check against, so "which AI parts are actually needed to run"
has one authoritative answer rather than scattered assumptions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class Requirement(str, Enum):
    NONE = "none"
    OPTIONAL_KEY = "optional_key"
    REQUIRED_KEY = "required_key"
    GATED_CHECKPOINT = "gated_checkpoint"


class Capability(str, Enum):
    DETECT = "detect"
    SEGMENT = "segment"
    CAPTION = "caption"
    ANSWER = "answer"
    EMBED = "embed"


@dataclass(frozen=True)
class Provider:
    name: str
    capability: Capability
    requirement: Requirement
    env_var: str | None
    purpose: str
    without_it: str
    enabled_in_demo: bool = False

    @property
    def configured(self) -> bool:
        if self.requirement is Requirement.NONE:
            return True
        return bool(self.env_var and os.environ.get(self.env_var))

    def status(self) -> str:
        if self.requirement is Requirement.NONE:
            return "built_in"
        if self.configured:
            return "configured"
        if self.requirement is Requirement.OPTIONAL_KEY:
            return "absent_optional"
        return "absent_required"


PROVIDERS: tuple[Provider, ...] = (
    Provider(
        name="rule_based_detector",
        capability=Capability.DETECT,
        requirement=Requirement.NONE,
        env_var=None,
        purpose=(
            "Proposes candidate item regions from frame differences and stable "
            "colour blobs. No model weights, no API, no GPU."
        ),
        without_it="Nothing — this is the built-in baseline and always runs.",
    ),
    Provider(
        name="open_vocab_detector",
        capability=Capability.DETECT,
        requirement=Requirement.OPTIONAL_KEY,
        env_var="DETECTOR_MODEL",
        purpose=(
            "Open-vocabulary detector (e.g. YOLO-World) that finds 'jacket' by "
            "name rather than by blob shape. Improves recall on cluttered scenes."
        ),
        without_it=(
            "Detection falls back to the rule-based baseline, which finds "
            "stationary blobs but cannot name them."
        ),
        enabled_in_demo=False,
    ),
    Provider(
        name="sam_segmenter",
        capability=Capability.SEGMENT,
        requirement=Requirement.GATED_CHECKPOINT,
        env_var="SAM_CHECKPOINT",
        purpose=(
            "Segment Anything, box-prompted from the detector, to get a clean "
            "item mask for the appearance crop."
        ),
        without_it=(
            "Crops come from the detector's bounding box, which is looser but "
            "still usable."
        ),
        enabled_in_demo=False,
    ),
    Provider(
        name="captioner",
        capability=Capability.CAPTION,
        requirement=Requirement.OPTIONAL_KEY,
        env_var="LLM_API_KEY",
        purpose=(
            "OpenAI vision model (same key as the chat) that writes the structured "
            "description of each found item and of visitor photos: item type, "
            "colours, visible features, location. Capped by CAPTIONS_PER_VIDEO."
        ),
        without_it=(
            "Captions come from the detector's dominant colour and position, and "
            "photo search matches by colour only, so descriptions are thinner but "
            "never invented."
        ),
        enabled_in_demo=False,
    ),
    Provider(
        name="llm_answerer",
        capability=Capability.ANSWER,
        requirement=Requirement.OPTIONAL_KEY,
        env_var="LLM_API_KEY",
        purpose=(
            "OpenAI model (LLM_MODEL, default gpt-4o-mini) that rewrites the composed "
            "answer into friendlier prose and handles chat that is not a search. It never "
            "decides what is true: every time and number in its reply is checked against "
            "the search result, and replies that add any are discarded."
        ),
        without_it=(
            "Answers are returned as template-composed text. Every fact is "
            "identical; only the phrasing is plainer."
        ),
        enabled_in_demo=False,
    ),
    Provider(
        name="text_embedder",
        capability=Capability.EMBED,
        requirement=Requirement.OPTIONAL_KEY,
        env_var="EMBED_MODEL",
        purpose="Semantic retrieval over captions, to catch paraphrases.",
        without_it="Retrieval stays keyword-only. Measured as an ablation.",
        enabled_in_demo=False,
    ),
)

BY_NAME = {p.name: p for p in PROVIDERS}
