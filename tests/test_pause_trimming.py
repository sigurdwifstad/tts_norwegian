"""Unit tests for part_3 hesitation/silence parsing and audio trimming."""
import os
import tempfile

import torch

from dataset import (
    parse_freeform_xml,
    trim_pauses,
    _merge_pause_intervals,
    _estimate_trimmed_duration,
)


def _write_xml(body):
    fd, path = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd, "w") as f:
        f.write(f"<root>{body}</root>")
    return path


# ---------------------------------------------------------------------------
# _merge_pause_intervals
# ---------------------------------------------------------------------------

def test_merge_contiguous_intervals():
    intervals = [(1.0, 1.2), (1.2, 1.6), (1.6, 2.0)]
    assert _merge_pause_intervals(intervals) == [(1.0, 2.0)]


def test_merge_keeps_separated_intervals_distinct():
    intervals = [(1.0, 1.3), (2.0, 2.5)]
    assert _merge_pause_intervals(intervals) == [(1.0, 1.3), (2.0, 2.5)]


def test_merge_handles_small_float_gap_within_eps():
    intervals = [(1.0, 1.2), (1.2005, 1.5)]
    assert _merge_pause_intervals(intervals) == [(1.0, 1.5)]


# ---------------------------------------------------------------------------
# parse_freeform_xml — pause interval collection
# ---------------------------------------------------------------------------

def test_parse_freeform_xml_merges_contiguous_pause_tags():
    xml_path = _write_xml("""
        <annotation end="10.0" id="part_3/group_01/x-free" speaker="p1_g01_f1_1_t">
            <word text="Hei"/>
            <exhale start="1.0" end="1.2"/>
            <fp start="1.2" end="1.6"/>
            <sil start="1.6" end="2.0"/>
            <word text="der"/>
            <sentence_boundary start="2.5" end="2.5" text="."/>
        </annotation>
    """)
    try:
        records = parse_freeform_xml(xml_path)
        assert len(records) == 1
        assert records[0]["pauses"] == [(1.0, 2.0)]
        assert records[0]["text"] == "Hei <exhale> <fp> <sil> der."
    finally:
        os.remove(xml_path)


def test_parse_freeform_xml_keeps_separated_pauses_distinct():
    xml_path = _write_xml("""
        <annotation end="10.0" id="part_3/group_01/x-free" speaker="p1_g01_f1_1_t">
            <word text="A"/>
            <fp start="1.0" end="1.3"/>
            <word text="B"/>
            <fp start="2.0" end="2.5"/>
            <word text="C"/>
            <sentence_boundary start="3.0" end="3.0" text="."/>
        </annotation>
    """)
    try:
        records = parse_freeform_xml(xml_path)
        assert len(records) == 1
        assert records[0]["pauses"] == [(1.0, 1.3), (2.0, 2.5)]
    finally:
        os.remove(xml_path)


def test_parse_freeform_xml_ignores_leading_gap_before_first_word():
    # The leading silence before the first word of a sentence shifts
    # sentence_start; it must NOT be recorded as a mid-sentence pause.
    xml_path = _write_xml("""
        <annotation end="10.0" id="part_3/group_01/x-free" speaker="p1_g01_f1_1_t">
            <sil start="0.0" end="0.8"/>
            <word text="Hei"/>
            <sentence_boundary start="1.5" end="1.5" text="."/>
        </annotation>
    """)
    try:
        records = parse_freeform_xml(xml_path)
        assert len(records) == 1
        assert records[0]["pauses"] == []
        assert records[0]["start"] == 0.8
    finally:
        os.remove(xml_path)


# ---------------------------------------------------------------------------
# _estimate_trimmed_duration
# ---------------------------------------------------------------------------

def test_estimate_trimmed_duration_subtracts_excess_over_cap():
    # sentence spans 0..5s, one pause block of 1.0s where cap is 0.2s
    duration = _estimate_trimmed_duration(0.0, 5.0, [(2.0, 3.0)], max_pause_sec=0.2)
    assert duration == 5.0 - (1.0 - 0.2)


def test_estimate_trimmed_duration_ignores_short_pauses():
    duration = _estimate_trimmed_duration(0.0, 5.0, [(2.0, 2.1)], max_pause_sec=0.2)
    assert duration == 5.0


# ---------------------------------------------------------------------------
# trim_pauses
# ---------------------------------------------------------------------------

SR = 1000  # 1 sample == 1ms, makes the math easy to reason about


def _sample_waveform(n_samples):
    return torch.arange(n_samples, dtype=torch.float32).unsqueeze(0)


def test_trim_pauses_no_pauses_returns_original():
    waveform = _sample_waveform(1000)
    out = trim_pauses(waveform, SR, 0.0, [], max_pause_sec=0.2, fade_ms=10, zero_cross_ms=0)
    assert torch.equal(out, waveform)


def test_trim_pauses_short_pause_left_untouched():
    waveform = _sample_waveform(1000)
    # pause 0.3s -> 0.35s = 50ms, shorter than the 200ms cap
    out = trim_pauses(waveform, SR, 0.0, [(0.3, 0.35)], max_pause_sec=0.2, fade_ms=10, zero_cross_ms=0)
    assert torch.equal(out, waveform)


def test_trim_pauses_caps_long_pause_and_preserves_speech():
    waveform = _sample_waveform(1000)  # 1 second @ 1000Hz
    # pause from 0.3s to 0.9s (600ms), cap at 200ms -> keep samples [300, 500),
    # drop [500, 900), keep [900, 1000)
    out = trim_pauses(waveform, SR, 0.0, [(0.3, 0.9)], max_pause_sec=0.2, fade_ms=10, zero_cross_ms=0)

    expected_len = 1000 - (900 - 500)
    assert out.shape[-1] == expected_len

    # Samples well before/after the fade zones must be numerically unchanged.
    fade_samples = 10
    assert torch.equal(out[..., : 500 - fade_samples], waveform[..., : 500 - fade_samples])
    tail_start_in_out = expected_len - (1000 - 900 - fade_samples)
    assert torch.equal(
        out[..., tail_start_in_out:], waveform[..., 900 + fade_samples:]
    )


def test_trim_pauses_multiple_blocks_each_capped_independently():
    waveform = _sample_waveform(2000)
    pauses = [(0.2, 0.6), (1.0, 1.5)]  # 400ms and 500ms, both > 200ms cap
    out = trim_pauses(waveform, SR, 0.0, pauses, max_pause_sec=0.2, fade_ms=10, zero_cross_ms=0)

    removed = (600 - 400) + (1500 - 1200)
    assert out.shape[-1] == 2000 - removed


def test_trim_pauses_respects_nonzero_sentence_start():
    # sentence_start offset should shift absolute pause timestamps correctly
    waveform = _sample_waveform(1000)
    out = trim_pauses(waveform, SR, 5.0, [(5.3, 5.9)], max_pause_sec=0.2, fade_ms=10, zero_cross_ms=0)
    expected_len = 1000 - (900 - 500)
    assert out.shape[-1] == expected_len


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
