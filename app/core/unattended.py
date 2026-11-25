"""Unattended-item inference.

The heart of the project. Given a track of an item over time, decide what
the footage shows about it — and be honest about what it cannot show.

Deliberately rule-based rather than learned. A threshold table is testable,
inspectable, and explainable to a judge; a black box that outputs 0.83
is none of those. Every threshold here is configurable and every decision
records which rule fired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class State(str, Enum):
    SEEN = "seen"
    STATIONARY = "stationary"
    POSSIBLY_UNATTENDED = "possibly_unattended"
    LIKELY_UNATTENDED = "likely_unattended"
    MOVED = "moved"
    UNCERTAIN = "uncertain"


class OnsetCertainty(str, Enum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    UNCERTAIN = "uncertain"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class Thresholds:
    """Tunable policy for the unattended decision."""

    stationary_seconds: float = 120.0
    """An item must be still this long before it counts as left alone."""

    deserted_seconds: float = 300.0
    """How long nobody may be near the item before 'likely unattended'."""

    person_proximity_ratio: float = 0.15
    """A person box counts as 'nearby' if within this fraction of frame width."""

    pickup_motion_seconds: float = 5.0
    """Movement of the item longer than this after settling is a pickup."""

    item_move_ratio: float = 0.03
    """An item counts as having moved if its position shifts by more than this
    fraction of frame width. Deliberately much tighter than person proximity:
    an item that shifts a few percent of the frame has been picked up or
    nudged, while a person standing a metre away is a different question."""


@dataclass
class TrackSample:
    """One observation of the item at one moment."""

    offset_s: float
    position: tuple[float, float]
    nearest_person_distance_ratio: float | None
    visible: bool = True


@dataclass
class Inference:
    """The verdict, plus the reasoning that produced it."""

    state: State
    onset_certainty: OnsetCertainty
    stationary_since_s: float | None
    deserted_since_s: float | None
    seconds_stationary: float
    seconds_deserted: float
    uncertainty: list[str] = field(default_factory=list)
    rules_fired: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "onset_certainty": self.onset_certainty.value,
            "stationary_since_s": self.stationary_since_s,
            "deserted_since_s": self.deserted_since_s,
            "seconds_stationary": round(self.seconds_stationary, 1),
            "seconds_deserted": round(self.seconds_deserted, 1),
            "uncertainty": list(self.uncertainty),
            "rules_fired": list(self.rules_fired),
            "confidence": round(self.confidence, 2),
        }


def _moved(
    samples: list[TrackSample],
    threshold: float,
) -> bool:
    """Did the item's position jump after a settled period?"""
    if len(samples) < 3:
        return False
    for previous, current in zip(samples, samples[1:]):
        dx = current.position[0] - previous.position[0]
        dy = current.position[1] - previous.position[1]
        distance = (dx * dx + dy * dy) ** 0.5
        gap = current.offset_s - previous.offset_s
        if distance > threshold and gap > 0:
            return True
    return False


def infer(
    samples: list[TrackSample],
    *,
    is_worn: bool,
    present_at_clip_start: bool,
    thresholds: Thresholds | None = None,
    frame_width: float = 1920.0,
    start_likely_attended: bool = True,
) -> Inference:
    """Decide the state of one track.

    Args:
        samples: observations in time order.
        is_worn: the garment is on a person's body.
        present_at_clip_start: the item was already visible at offset 0.
        start_likely_attended: a person was near it when recording began,
            e.g. because the person who placed it was still standing there.
    """
    policy = thresholds or Thresholds()
    rules: list[str] = []
    uncertainty: list[str] = []

    if not samples:
        return Inference(
            state=State.UNCERTAIN,
            onset_certainty=OnsetCertainty.NOT_APPLICABLE,
            stationary_since_s=None,
            deserted_since_s=None,
            seconds_stationary=0.0,
            seconds_deserted=0.0,
            uncertainty=["track_fragmented"],
            rules_fired=["no_samples"],
            confidence=0.0,
        )

    # Rule 1 — a worn garment can never be an unattended item.
    if is_worn:
        return Inference(
            state=State.SEEN,
            onset_certainty=OnsetCertainty.NOT_APPLICABLE,
            stationary_since_s=None,
            deserted_since_s=None,
            seconds_stationary=0.0,
            seconds_deserted=0.0,
            uncertainty=[],
            rules_fired=["worn_garment_excluded"],
            confidence=0.9,
        )

    first = samples[0]
    last = samples[-1]

    visible = [s for s in samples if s.visible]
    if present_at_clip_start:
        uncertainty.append("present_at_clip_start")
        rules.append("present_at_start")

    # Rule 2 — still, relative to the first sample.
    stationary_since = first.offset_s
    still_threshold = policy.stationary_seconds
    settled_at: float | None = None
    for sample in visible:
        dx = sample.position[0] - first.position[0]
        dy = sample.position[1] - first.position[1]
        if (dx * dx + dy * dy) ** 0.5 > policy.item_move_ratio * frame_width:
            stationary_since = sample.offset_s
            settled_at = None
        elif settled_at is None and sample.offset_s - stationary_since >= still_threshold:
            settled_at = stationary_since
            rules.append("stationary_threshold_met")

    seconds_stationary = max(0.0, last.offset_s - stationary_since)

    # Rule 3 — nobody near, for how long.
    deserted_since: float | None = None
    for sample in visible:
        near = sample.nearest_person_distance_ratio
        alone = near is None or near > policy.person_proximity_ratio
        if alone and deserted_since is None:
            deserted_since = sample.offset_s
        elif not alone:
            deserted_since = None
    seconds_deserted = (
        max(0.0, last.offset_s - deserted_since) if deserted_since is not None else 0.0
    )

    # Rule 4 — it moved again: someone picked it up.
    if _moved(visible, policy.item_move_ratio * frame_width):
        rules.append("pickup_motion_detected")
        return Inference(
            state=State.MOVED,
            onset_certainty=OnsetCertainty.OBSERVED,
            stationary_since_s=settled_at,
            deserted_since_s=deserted_since,
            seconds_stationary=seconds_stationary,
            seconds_deserted=seconds_deserted,
            uncertainty=uncertainty,
            rules_fired=rules,
            confidence=0.8,
        )

    # Rule 5 — classify by how long it has been both still and alone.
    if settled_at is None:
        return Inference(
            state=State.SEEN,
            onset_certainty=OnsetCertainty.NOT_APPLICABLE,
            stationary_since_s=None,
            deserted_since_s=deserted_since,
            seconds_stationary=seconds_stationary,
            seconds_deserted=seconds_deserted,
            uncertainty=uncertainty + (["track_fragmented"] if len(visible) < 3 else []),
            rules_fired=rules + ["not_settled"],
            confidence=0.5,
        )

    stationary_state = State.STATIONARY
    if seconds_deserted < policy.deserted_seconds:
        rules.append("person_nearby")
        state = stationary_state
        certainty = OnsetCertainty.NOT_APPLICABLE
        confidence = 0.6
    elif seconds_deserted < 2 * policy.deserted_seconds:
        rules.append("deserted_short")
        state = State.POSSIBLY_UNATTENDED
        certainty = OnsetCertainty.INFERRED
        confidence = 0.65
    else:
        rules.append("deserted_sustained")
        state = State.LIKELY_UNATTENDED
        certainty = OnsetCertainty.INFERRED
        confidence = 0.8

    # Rule 6 — the presence-at-start caveat outranks any confident onset.
    if present_at_clip_start and state in (
        State.POSSIBLY_UNATTENDED,
        State.LIKELY_UNATTENDED,
    ):
        certainty = OnsetCertainty.UNCERTAIN
        rules.append("onset_certainty_downgraded")
        confidence = min(confidence, 0.6)

    # Rule 7 — if we only ever saw it alone and still, onset is an inference,
    # not an observation. We never saw the person leave.
    if start_likely_attended is False and certainty is OnsetCertainty.INFERRED:
        rules.append("onset_inferred_no_departure_observed")

    return Inference(
        state=state,
        onset_certainty=certainty,
        stationary_since_s=settled_at,
        deserted_since_s=deserted_since,
        seconds_stationary=seconds_stationary,
        seconds_deserted=seconds_deserted,
        uncertainty=uncertainty,
        rules_fired=rules,
        confidence=confidence,
    )
