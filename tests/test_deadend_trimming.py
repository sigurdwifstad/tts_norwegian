"""Unit tests for part_3 deadend (false-start/repeated-word) parsing and trimming."""
import os
import tempfile
import xml.etree.ElementTree as ET

import torch

from dataset import (
    parse_freeform_xml,
    trim_pauses,
    _extract_deadend_cuts,
    _estimate_trimmed_duration,
)


def _write_xml(body):
    fd, path = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd, "w") as f:
        f.write(f"<root>{body}</root>")
    return path


def _elements(fragment):
    """Build a list of Elements from an XML fragment (for direct unit tests
    of _extract_deadend_cuts without going through full sentence parsing)."""
    root = ET.fromstring(f"<sentence>{fragment}</sentence>")
    return list(root)


# ---------------------------------------------------------------------------
# _extract_deadend_cuts — direct unit tests
# ---------------------------------------------------------------------------

def test_deadend_own_timestamps_bounded():
    elements = _elements("""
        <word text="Hallo"/>
        <word deadend="1" start="1.0" end="1.4" text="saa"/>
        <word text="Saa"/>
    """)
    cuts, unbounded = _extract_deadend_cuts(elements)
    assert unbounded is False
    assert cuts == [(1.0, 1.4)]


def test_deadend_neighbor_anchored_bounded():
    elements = _elements("""
        <word text="Jeg"/>
        <fp start="1.0" end="1.2"/>
        <word deadend="1" text="saa"/>
        <fp start="1.6" end="1.8"/>
        <word text="Saa"/>
    """)
    cuts, unbounded = _extract_deadend_cuts(elements)
    assert unbounded is False
    assert cuts == [(1.2, 1.6)]


def test_deadend_unbounded_when_untimed_words_on_both_sides():
    elements = _elements("""
        <word text="Vi"/>
        <word text="hadde"/>
        <word deadend="1" text="et"/>
        <word text="en"/>
        <word text="bil"/>
    """)
    cuts, unbounded = _extract_deadend_cuts(elements)
    assert unbounded is True
    assert cuts == []


def test_deadend_anchor_search_blocked_by_intervening_untimed_word():
    # There IS a timed element further back (the leading sil), but an
    # untimed real word ("Vi") sits between it and the deadend run, so the
    # anchor search must NOT reach past "Vi" and should report unbounded.
    elements = _elements("""
        <sil start="0.0" end="0.5"/>
        <word text="Vi"/>
        <word deadend="1" text="hadde"/>
        <word text="en"/>
    """)
    cuts, unbounded = _extract_deadend_cuts(elements)
    assert unbounded is True
    assert cuts == []


def test_deadend_multi_word_run_own_timestamps():
    elements = _elements("""
        <word deadend="1" start="0.5" end="0.8" text="saa"/>
        <word deadend="1" start="0.8" end="1.1" text="vi"/>
        <word text="Saa"/>
    """)
    cuts, unbounded = _extract_deadend_cuts(elements)
    assert unbounded is False
    assert cuts == [(0.5, 1.1)]


def test_no_deadend_returns_empty():
    elements = _elements("""
        <word text="Hallo"/>
        <word text="der"/>
    """)
    cuts, unbounded = _extract_deadend_cuts(elements)
    assert cuts == []
    assert unbounded is False


# ---------------------------------------------------------------------------
# parse_freeform_xml — deadend integration
# ---------------------------------------------------------------------------

def test_parse_strips_deadend_multiword_from_text():
    xml_path = _write_xml("""
        <annotation end="10.0" id="part_3/group_01/x-free" speaker="p1_g01_f1_1_t">
            <word text="Vi"/>
            <multiword deadend="1" start="1.0" end="1.5" text="to tusen"/>
            <word text="dro"/>
            <sentence_boundary start="2.0" end="2.0" text="."/>
        </annotation>
    """)
    try:
        records = parse_freeform_xml(xml_path)
        assert len(records) == 1
        assert records[0]["text"] == "Vi dro."
        assert records[0]["deadend_cuts"] == [(1.0, 1.5)]
    finally:
        os.remove(xml_path)


def test_parse_keeps_sentence_with_bounded_deadend():
    xml_path = _write_xml("""
        <annotation end="10.0" id="part_3/group_01/x-free" speaker="p1_g01_f1_1_t">
            <word text="Hallo"/>
            <word deadend="1" start="1.0" end="1.4" text="saa"/>
            <word text="Saa"/>
            <sentence_boundary start="2.0" end="2.0" text="."/>
        </annotation>
    """)
    try:
        records = parse_freeform_xml(xml_path)
        assert len(records) == 1
        assert records[0]["text"] == "Hallo Saa."
        assert records[0]["deadend_cuts"] == [(1.0, 1.4)]
    finally:
        os.remove(xml_path)


def test_parse_drops_sentence_with_unbounded_deadend():
    xml_path = _write_xml("""
        <annotation end="10.0" id="part_3/group_01/x-free" speaker="p1_g01_f1_1_t">
            <word text="Vi"/>
            <word text="hadde"/>
            <word deadend="1" text="et"/>
            <word text="en"/>
            <word text="bil"/>
            <sentence_boundary start="3.0" end="3.0" text="."/>
        </annotation>
    """)
    try:
        records = parse_freeform_xml(xml_path)
        assert len(records) == 0
    finally:
        os.remove(xml_path)


def test_parse_only_drops_the_affected_sentence():
    # First sentence has an unbounded deadend and should be dropped; the
    # second sentence is clean and should still be returned.
    xml_path = _write_xml("""
        <annotation end="10.0" id="part_3/group_01/x-free" speaker="p1_g01_f1_1_t">
            <word text="Vi"/>
            <word text="hadde"/>
            <word deadend="1" text="et"/>
            <word text="en"/>
            <word text="bil"/>
            <sentence_boundary start="3.0" end="3.0" text="."/>
            <word text="Hei"/>
            <word text="der"/>
            <sentence_boundary start="4.0" end="4.0" text="."/>
        </annotation>
    """)
    try:
        records = parse_freeform_xml(xml_path)
        assert len(records) == 1
        assert records[0]["text"] == "Hei der."
    finally:
        os.remove(xml_path)


# ---------------------------------------------------------------------------
# _estimate_trimmed_duration with deadend_cuts
# ---------------------------------------------------------------------------

def test_estimate_trimmed_duration_subtracts_full_deadend_duration():
    # deadend cut of 1.0s should be subtracted in FULL, unlike pauses which
    # are only reduced down to max_pause_sec.
    duration = _estimate_trimmed_duration(
        0.0, 5.0, pauses=[], deadend_cuts=[(2.0, 3.0)], max_pause_sec=0.2
    )
    assert duration == 4.0


def test_estimate_trimmed_duration_combines_pauses_and_deadends():
    duration = _estimate_trimmed_duration(
        0.0, 10.0, pauses=[(1.0, 2.0)], deadend_cuts=[(5.0, 5.5)], max_pause_sec=0.2
    )
    # pause: 1.0s duration - 0.2s cap = 0.8s excess removed
    # deadend: 0.5s removed in full
    assert duration == 10.0 - 0.8 - 0.5


# ---------------------------------------------------------------------------
# trim_pauses with hard_cuts
# ---------------------------------------------------------------------------

SR = 1000  # 1 sample == 1ms


def _sample_waveform(n_samples):
    return torch.arange(n_samples, dtype=torch.float32).unsqueeze(0)


def test_hard_cut_removed_in_full_regardless_of_max_pause_sec():
    waveform = _sample_waveform(1000)
    # hard cut from 0.3s to 0.5s (200ms) -- shorter than would normally be
    # capped by a pause, but must be removed IN FULL since it's a hard cut.
    out = trim_pauses(
        waveform, SR, 0.0, pauses=[], hard_cuts=[(0.3, 0.5)],
        max_pause_sec=0.2, fade_ms=10, zero_cross_ms=0,
    )
    assert out.shape[-1] == 1000 - 200


def test_combined_pause_and_hard_cut_in_one_call():
    waveform = _sample_waveform(2000)
    pauses = [(0.2, 0.6)]       # 400ms, capped to 200ms -> 200ms removed
    hard_cuts = [(1.0, 1.3)]    # 300ms, removed in full
    out = trim_pauses(
        waveform, SR, 0.0, pauses=pauses, hard_cuts=hard_cuts,
        max_pause_sec=0.2, fade_ms=10, zero_cross_ms=0,
    )
    removed = (600 - 400) + 300
    assert out.shape[-1] == 2000 - removed


def test_overlapping_pause_and_hard_cut_ranges_merge_without_double_counting():
    waveform = _sample_waveform(1000)
    # Pause block (0.2s-0.6s) capped to 200ms -> drop range [400ms, 600ms).
    # Hard cut (0.5s-0.7s) -> drop range [500ms, 700ms), overlapping the
    # pause's drop range. Total dropped should be the UNION: [400ms, 700ms).
    out = trim_pauses(
        waveform, SR, 0.0, pauses=[(0.2, 0.6)], hard_cuts=[(0.5, 0.7)],
        max_pause_sec=0.2, fade_ms=10, zero_cross_ms=0,
    )
    assert out.shape[-1] == 1000 - (700 - 400)


def test_no_pauses_no_hard_cuts_returns_original():
    waveform = _sample_waveform(500)
    out = trim_pauses(waveform, SR, 0.0, pauses=[], hard_cuts=[], zero_cross_ms=0)
    assert torch.equal(out, waveform)


def test_hard_cut_at_very_start_of_sentence_is_still_trimmed():
    # Regression test: a cut spanning from sample 0 leaves only ONE keep
    # segment. Earlier buggy code treated "only one keep segment" as "nothing
    # to trim" and incorrectly returned the untrimmed waveform.
    waveform = _sample_waveform(1000)
    out = trim_pauses(
        waveform, SR, 0.0, pauses=[], hard_cuts=[(0.0, 0.4)],
        max_pause_sec=0.2, fade_ms=10, zero_cross_ms=0,
    )
    assert out.shape[-1] == 1000 - 400
    # The kept tail (well past the fade zone) must be untouched.
    assert torch.equal(out[..., 10:], waveform[..., 410:])


def test_hard_cut_at_very_end_of_sentence_is_still_trimmed():
    waveform = _sample_waveform(1000)
    out = trim_pauses(
        waveform, SR, 0.0, pauses=[], hard_cuts=[(0.6, 1.0)],
        max_pause_sec=0.2, fade_ms=10, zero_cross_ms=0,
    )
    assert out.shape[-1] == 600
    assert torch.equal(out[..., : 600 - 10], waveform[..., : 600 - 10])


def test_entire_waveform_cut_returns_empty():
    waveform = _sample_waveform(1000)
    out = trim_pauses(
        waveform, SR, 0.0, pauses=[], hard_cuts=[(0.0, 1.0)],
        max_pause_sec=0.2, fade_ms=10, zero_cross_ms=0,
    )
    assert out.shape[-1] == 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
