"""Unit tests for unattended-item inference.

The rules under test are the project's non-negotiables, expressed as
behaviour rather than prose. If one of these fails, the app is lying to a
user, so they are worth more than any coverage percentage.
"""

from __future__ import annotations

import pytest

from app.core.unattended import (
    Inference,
    OnsetCertainty,
    State,
    Thresholds,
    TrackSample,
    infer,
)

FAST = Thresholds(stationary_seconds=10, deserted_seconds=20, person_proximity_ratio=0.1)


def track(offsets, positions=None, persons=None, visible=None):
    positions = positions or [(100.0, 0.0)] * len(offsets)
    persons = persons if persons is not None else [None] * len(offsets)
    visible = visible or [True] * len(offsets)
    return [
        TrackSample(
            offset_s=o,
            position=p,
            nearest_person_distance_ratio=person,
            visible=v,
        )
        for o, p, person, v in zip(offsets, positions, persons, visible)
    ]


def test_worn_garment_is_never_unattended():
    result = infer(
        track([0, 10, 20, 30, 60, 120]),
        is_worn=True,
        present_at_clip_start=False,
        thresholds=FAST,
    )
    assert result.state is State.SEEN
    assert result.state not in (State.POSSIBLY_UNATTENDED, State.LIKELY_UNATTENDED)
    assert "worn_garment_excluded" in result.rules_fired


def test_worn_garment_is_excluded_even_if_it_sits_still_for_hours():
    result = infer(
        track([0, 600, 3600, 7200]),
        is_worn=True,
        present_at_clip_start=False,
        thresholds=FAST,
    )
    assert result.state is State.SEEN


def test_item_left_alone_becomes_likely_unattended():
    # Still for 120s, nobody near the whole time.
    result = infer(
        track([0, 20, 40, 60, 80, 100, 120]),
        is_worn=False,
        present_at_clip_start=False,
        thresholds=FAST,
    )
    assert result.state is State.LIKELY_UNATTENDED
    assert result.onset_certainty in (
        OnsetCertainty.INFERRED,
        OnsetCertainty.OBSERVED,
    )


def test_item_with_a_person_beside_it_is_only_stationary():
    # A person stays within the proximity ratio for the whole track.
    result = infer(
        track(
            [0, 20, 40, 60, 80, 100, 120],
            persons=[0.02, 0.02, 0.03, 0.02, 0.04, 0.02, 0.03],
        ),
        is_worn=False,
        present_at_clip_start=False,
        thresholds=FAST,
    )
    assert result.state is State.STATIONARY
    assert "person_nearby" in result.rules_fired


def test_item_present_at_clip_start_loses_onset_certainty():
    result = infer(
        track([0, 20, 40, 60, 80, 100, 120]),
        is_worn=False,
        present_at_clip_start=True,
        thresholds=FAST,
    )
    assert result.state is State.LIKELY_UNATTENDED
    assert result.onset_certainty is OnsetCertainty.UNCERTAIN
    assert "present_at_clip_start" in result.uncertainty
    assert "onset_certainty_downgraded" in result.rules_fired


def test_pickup_is_detected_as_moved():
    # Pixel-scale coordinates: the item is put down around x=800 and then
    # carried out of frame, so each step is far larger than item_move_ratio.
    positions = [(800.0, 0.0)] * 5 + [(1000.0, 0.0), (1200.0, 0.0)]
    result = infer(
        track([0, 20, 40, 60, 80, 100, 120], positions=positions),
        is_worn=False,
        present_at_clip_start=False,
        thresholds=FAST,
    )
    assert result.state is State.MOVED
    assert "pickup_motion_detected" in result.rules_fired


def test_movement_before_settling_is_not_a_pickup():
    # Dropped into place and then nudged a few centimetres. The item shifts
    # less than item_move_ratio per step, so this is placement, not pickup.
    positions = [(800.0, 0.0), (820.0, 0.0), (824.0, 0.0), (826.0, 0.0), (826.0, 0.0)]
    result = infer(
        track([0, 5, 30, 60, 90], positions=positions),
        is_worn=False,
        present_at_clip_start=False,
        thresholds=FAST,
    )
    assert result.state is not State.MOVED


def test_empty_track_is_uncertain_not_invented():
    result = infer([], is_worn=False, present_at_clip_start=False, thresholds=FAST)
    assert result.state is State.UNCERTAIN
    assert result.confidence == 0.0
    assert "track_fragmented" in result.uncertainty


def test_occluded_frames_do_not_become_a_ghost_item():
    result = infer(
        track([0, 20, 40, 60], visible=[True, False, False, False]),
        is_worn=False,
        present_at_clip_start=False,
        thresholds=FAST,
    )
    assert result.state is not State.LIKELY_UNATTENDED


def test_thresholds_are_configurable():
    slow = Thresholds(stationary_seconds=600, deserted_seconds=600)
    samples = track([0, 100, 200, 300])
    assert infer(samples, is_worn=False, present_at_clip_start=False, thresholds=slow).state is not (
        State.LIKELY_UNATTENDED
    )


def test_inference_serialises_without_losing_the_rules():
    result = infer(
        track([0, 20, 40, 60, 80, 100, 120]),
        is_worn=False,
        present_at_clip_start=True,
        thresholds=FAST,
    )
    payload = result.to_dict()
    assert payload["state"] == "likely_unattended"
    assert payload["onset_certainty"] == "uncertain"
    assert isinstance(payload["rules_fired"], list)
    assert payload["rules_fired"]


def test_onset_is_never_earlier_than_first_sighting():
    result = infer(
        track([100, 120, 140, 160, 180, 200, 220]),
        is_worn=False,
        present_at_clip_start=False,
        thresholds=FAST,
    )
    if result.stationary_since_s is not None:
        assert result.stationary_since_s >= 100


def test_deserted_duration_is_computed_from_the_track():
    result = infer(
        track([0, 20, 40, 60, 80, 100, 120]),
        is_worn=False,
        present_at_clip_start=False,
        thresholds=FAST,
    )
    assert result.seconds_deserted > 0
    assert result.seconds_deserted <= 120
