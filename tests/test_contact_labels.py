"""Tests for hoi_imitation.data.contact_labels."""

import numpy as np

from hoi_imitation.data.contact_labels import contact_gaps, repair

FPS = 30.0


def _moving(frames: int, speed: float = 0.2) -> np.ndarray:
    return np.stack([np.arange(frames) * speed / FPS, np.zeros(frames), np.zeros(frames)], axis=1)


def test_contact_gaps_returns_only_interior_runs():
    active = np.array([0, 1, 1, 0, 0, 1, 0, 1, 0], dtype=bool)
    assert contact_gaps(active) == [(3, 5), (6, 7)]


def test_short_gap_during_transport_is_filled():
    contact = np.zeros((2, 10), dtype=bool)
    contact[0, :3] = contact[0, 6:] = True
    repaired, filled = repair(contact, _moving(10), FPS, max_gap_s=1.0, min_speed=0.05)
    assert filled == 3 and repaired[0].all() and not repaired[1].any()


def test_gap_is_kept_when_the_object_rests():
    contact = np.zeros((2, 10), dtype=bool)
    contact[0, :3] = contact[0, 6:] = True
    repaired, filled = repair(contact, _moving(10, speed=0.0), FPS, max_gap_s=1.0, min_speed=0.05)
    assert filled == 0 and np.array_equal(repaired, contact)


def test_gap_is_kept_when_the_other_hand_holds_the_object():
    contact = np.zeros((2, 10), dtype=bool)
    contact[0, :3] = contact[0, 6:] = True
    contact[1, 3:6] = True
    repaired, filled = repair(contact, _moving(10), FPS, max_gap_s=1.0, min_speed=0.05)
    assert filled == 0 and not repaired[0, 3:6].any()


def test_long_gap_is_kept():
    contact = np.zeros((2, 200), dtype=bool)
    contact[0, :10] = contact[0, 190:] = True
    _, filled = repair(contact, _moving(200), FPS, max_gap_s=1.0, min_speed=0.05)
    assert filled == 0
