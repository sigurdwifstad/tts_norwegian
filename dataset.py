import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import torchaudio
import torchaudio.functional as F
import re
import xml.etree.ElementTree as ET


def canonical_speaker_id(raw_speaker_id: str) -> str:
    """Map p1_g01_f1_1_t -> g01_f1_1 for speaker embedding lookup."""
    speaker_id = str(raw_speaker_id)
    parts = speaker_id.split("_")

    if len(parts) >= 5 and parts[1].startswith("g") and parts[2].startswith("f"):
        return "_".join(parts[1:4])

    if len(parts) >= 4 and parts[0].startswith("g") and parts[1].startswith("f"):
        return "_".join(parts[:3])

    match = re.search(r"(g\d+_f\d+_\d+)", speaker_id)
    if match:
        return match.group(1)

    return speaker_id


# ---------------------------------------------------------------------------
# Pause tokens – inserted into text so the model learns to produce pauses.
# Maps XML tag names from part_3 annotations to special token strings.
# ---------------------------------------------------------------------------
PAUSE_TOKENS = {
    "sil":    "<sil>",
    "inhale": "<inhale>",
    "exhale": "<exhale>",
    "fp":     "<fp>",
}
# Convenience set of all token strings for use elsewhere (tokenizer, regex).
PAUSE_TOKEN_SET = set(PAUSE_TOKENS.values())

# Tags whose audio should be capped/trimmed when they occur mid-sentence
# (silence, breathing, filled pauses, and hesitation vowels).
PAUSE_TRIM_TAGS = {"sil", "inhale", "exhale", "fp", "vowel"}

# Default cap for any single (merged) mid-sentence pause block, in seconds.
MAX_PAUSE_SEC = 0.2
# Default linear fade applied at each audio splice edge, in milliseconds.
PAUSE_FADE_MS = 10
# Tolerance for treating two consecutive pause-tag intervals as contiguous
# (i.e. part of the same hesitation block) despite tiny float gaps.
PAUSE_MERGE_EPS_SEC = 0.02


def _merge_pause_intervals(intervals, eps=PAUSE_MERGE_EPS_SEC):
    """Merge overlapping/adjacent (start, end) intervals into contiguous blocks.

    Consecutive pause-type tags (e.g. exhale -> fp -> sil -> fp) are usually
    back-to-back in time; we want to treat such a run as a single hesitation
    block for trimming purposes rather than trimming each tag separately.
    """
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start - merged[-1][1] <= eps:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _estimate_trimmed_duration(start, end, pauses, max_pause_sec=MAX_PAUSE_SEC):
    """Estimate a sentence's audio duration after capping long pause blocks."""
    duration = end - start
    for p_start, p_end in pauses:
        excess = (p_end - p_start) - max_pause_sec
        if excess > 0:
            duration -= excess
    return duration


def parse_sentence_level_xml(xml_path):
    """Parse part_1/part_2 style XML where each <annotation> is already a sentence.
    These annotations have a 'text' attribute and word-level phoneme segments.
    Returns a list of dicts with keys: id, speaker, text, start, end.
    """
    df = pd.read_xml(xml_path)
    records = []
    for _, row in df.iterrows():
        records.append({
            "id": row["id"],
            "speaker": row["speaker"],
            "text": row.get("text", ""),
            "start": 0.0,
            "end": float(row["end"]),
        })
    return records


def parse_freeform_xml(xml_path):
    """Parse part_3 style XML where each <annotation> is a long recording.
    Splits into sentence-level chunks using <sentence_boundary> timestamps.
    Handles word, multiword, complex_word, slang, and comma elements.

    Timestamp logic:
      - sentence END   = sentence_boundary's 'end' attribute
      - sentence START  = end timestamp of the last non-word element (sil, exhale,
        inhale, fp …) that appears *before* the first word after the previous
        sentence_boundary.  Falls back to the previous boundary's 'end' if no
        such element exists.  The very first sentence always starts at 0.

    Each record also carries a "pauses" list of (start, end) tuples (absolute
    time within the source .wav, merged where contiguous) for mid-sentence
    silence/breath/filler/hesitation-vowel tags (see PAUSE_TRIM_TAGS). These
    are used later to trim/cap hesitation audio without touching the speech
    in between, since individual words carry no timestamps of their own.

    Returns a list of dicts with keys: id, speaker, text, start, end, pauses.
    """
    WORD_TAGS = {"word", "multiword", "complex_word", "slang", "comma"}
    # Small buffer added after sentence_boundary timestamps so the tail of
    # the last phoneme is not clipped.  80 ms covers most stop releases.
    END_BUFFER_SEC = 0.08

    tree = ET.parse(xml_path)
    root = tree.getroot()
    records = []

    for annotation in root.findall("annotation"):
        ann_id = annotation.get("id")
        speaker = annotation.get("speaker")
        ann_end_str = annotation.get("end")
        if ann_end_str is None:
            continue  # skip annotations without end timestamp
        ann_end = float(ann_end_str)

        # Collect sentences by splitting on sentence_boundary elements
        current_words = []
        current_pauses = []           # mid-sentence (start, end) pause-tag intervals
        sentence_start = 0.0          # start of current sentence (updated by gap elements)
        awaiting_first_word = True     # True while we haven't seen a word since last boundary
        last_timed_end = 0.0           # tracks the latest timed element we've seen

        for elem in annotation:
            tag = elem.tag
            text_attr = elem.get("text", "")

            if tag in WORD_TAGS:
                # ---- word-like element ----
                awaiting_first_word = False
                if tag == "word":
                    if elem.get("deadend") != "1":
                        current_words.append(text_attr)
                elif tag == "multiword":
                    current_words.append(text_attr)
                elif tag == "complex_word":
                    if elem.get("deadend") != "1":
                        current_words.append(text_attr)
                elif tag == "slang":
                    if elem.get("deadend") != "1":
                        current_words.append(text_attr)
                elif tag == "comma":
                    current_words.append(text_attr)
                    # commas sometimes carry timestamps
                    end_str = elem.get("end")
                    if end_str is not None:
                        try:
                            last_timed_end = max(last_timed_end, float(end_str))
                        except ValueError:
                            pass

            elif tag == "sentence_boundary":
                # ---- end of current sentence ----
                boundary_end_str = elem.get("end")
                if boundary_end_str is None:
                    # No timestamp — merge with next sentence instead of splitting
                    continue
                boundary_end = float(boundary_end_str)
                last_timed_end = max(last_timed_end, boundary_end)

                sentence_text = " ".join(current_words).strip()
                if sentence_text and len(sentence_text) > 2:
                    records.append({
                        "id": ann_id,
                        "speaker": speaker,
                        "text": sentence_text + text_attr,  # append . or ?
                        "start": sentence_start,
                        "end": min(boundary_end + END_BUFFER_SEC, ann_end),
                        "pauses": _merge_pause_intervals(current_pauses),
                    })

                # Reset for the next sentence
                sentence_start = boundary_end   # default; may be pushed forward by gap elements
                awaiting_first_word = True
                current_words = []
                current_pauses = []

            else:
                # ---- non-word timed element (sil, fp, inhale, exhale …) ----
                # Always track latest timestamp for trailing-segment fallback.
                end_str = elem.get("end")
                if end_str is not None:
                    try:
                        last_timed_end = max(last_timed_end, float(end_str))
                    except ValueError:
                        pass

                if awaiting_first_word:
                    # Before the first word of a new sentence: advance
                    # sentence_start past these inter-sentence gaps.
                    if end_str is not None:
                        try:
                            sentence_start = float(end_str)
                        except ValueError:
                            pass
                else:
                    # Mid-sentence pause: inject a pause token into the text
                    # so the model learns to produce the corresponding pause.
                    pause_token = PAUSE_TOKENS.get(tag)
                    if pause_token is not None:
                        current_words.append(pause_token)

                    # Track exact (start, end) so the audio can be trimmed later.
                    if tag in PAUSE_TRIM_TAGS:
                        start_str = elem.get("start")
                        if start_str is not None and end_str is not None:
                            try:
                                current_pauses.append((float(start_str), float(end_str)))
                            except ValueError:
                                pass

        # Handle any remaining words after last sentence boundary.
        # Use last_timed_end (+ buffer) instead of ann_end to avoid
        # many seconds of trailing silence.
        remaining_text = " ".join(current_words).strip()
        if remaining_text and len(remaining_text) > 2:
            trailing_end = (last_timed_end + END_BUFFER_SEC
                            if last_timed_end > sentence_start else ann_end)
            trailing_end = min(trailing_end, ann_end)
            records.append({
                "id": ann_id,
                "speaker": speaker,
                "text": remaining_text,
                "start": sentence_start,
                "end": trailing_end,
                "pauses": _merge_pause_intervals(current_pauses),
            })

    return records


def trim_pauses(waveform, sr, sentence_start, pauses,
                 max_pause_sec=MAX_PAUSE_SEC, fade_ms=PAUSE_FADE_MS):
    """Shorten long mid-sentence pause/hesitation blocks in a sentence clip.

    Only the timing of pause-type tags (sil, inhale, exhale, fp, vowel) is
    known precisely — individual words carry no timestamps in part_3. So we
    can only cut/cap the known pause intervals and must leave everything else
    (implicitly speech) untouched.

    Args:
        waveform: tensor of shape (channels, samples), already sliced to the
            sentence's [start, end] time range.
        sr: sample rate of `waveform`.
        sentence_start: the sentence's absolute start time in seconds, used to
            convert `pauses` (absolute-time tuples) into sample offsets
            relative to `waveform`.
        pauses: list of (start, end) absolute-time tuples, already merged
            into contiguous blocks (see `_merge_pause_intervals`).
        max_pause_sec: cap for any single pause block; pauses longer than
            this are shortened, keeping their first `max_pause_sec` and
            dropping the remainder.
        fade_ms: short linear fade applied at each cut edge to avoid audible
            clicks/pops at the splice points.

    Returns:
        A new waveform tensor with long pauses capped. If no pause exceeds
        the cap, the original waveform is returned unchanged.
    """
    if not pauses:
        return waveform

    num_samples = waveform.shape[-1]
    fade_samples = max(0, int(sr * fade_ms / 1000))
    max_pause_samples = max(0, int(sr * max_pause_sec))

    # Sample ranges to DROP (the excess tail of any over-long pause block).
    cuts = []
    for p_start, p_end in pauses:
        rel_start = int(round((p_start - sentence_start) * sr))
        rel_end = int(round((p_end - sentence_start) * sr))
        rel_start = max(0, min(rel_start, num_samples))
        rel_end = max(0, min(rel_end, num_samples))
        if rel_end <= rel_start:
            continue
        if (rel_end - rel_start) <= max_pause_samples:
            continue  # short enough already, nothing to trim
        keep_until = rel_start + max_pause_samples
        cuts.append((keep_until, rel_end))

    if not cuts:
        return waveform

    # Sample ranges to KEEP = complement of `cuts`.
    cuts.sort()
    keep_segments = []
    cursor = 0
    for cut_start, cut_end in cuts:
        if cut_start > cursor:
            keep_segments.append((cursor, cut_start))
        cursor = max(cursor, cut_end)
    if cursor < num_samples:
        keep_segments.append((cursor, num_samples))

    if len(keep_segments) <= 1:
        return waveform

    n_segments = len(keep_segments)
    chunks = []
    for i, (seg_start, seg_end) in enumerate(keep_segments):
        chunk = waveform[..., seg_start:seg_end].clone()
        seg_len = chunk.shape[-1]
        this_fade = min(fade_samples, seg_len // 2) if seg_len > 0 else 0
        if this_fade > 0:
            fade_in = torch.linspace(0, 1, this_fade, device=chunk.device, dtype=chunk.dtype)
            fade_out = torch.linspace(1, 0, this_fade, device=chunk.device, dtype=chunk.dtype)
            if i > 0:
                # Fade in at the start of every segment except the very first
                # (a cut precedes this segment, so its start is a splice point).
                chunk[..., :this_fade] *= fade_in
            if i < n_segments - 1:
                # Fade out at the end of every segment except the very last
                # (a cut follows this segment, so its end is a splice point).
                chunk[..., -this_fade:] *= fade_out
        chunks.append(chunk)

    return torch.cat(chunks, dim=-1)


def detect_xml_format(xml_path):
    """Detect whether an XML file is sentence-level (part_1/2) or freeform (part_3).
    Sentence-level annotations have a 'text' attribute on <annotation>.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    first_ann = root.find("annotation")
    if first_ann is not None and first_ann.get("text") is not None:
        return "sentence_level"
    return "freeform"


class NBTaleDataset(Dataset):
    def __init__(self, data_path, processor, speaker_to_embedding, datasets=[1], max_audio_length=None,
                 use_pause_tokens=False, max_pause_sec=MAX_PAUSE_SEC, pause_fade_ms=PAUSE_FADE_MS):
        self.data_path = data_path
        self.processor = processor
        self.speaker_to_embedding = speaker_to_embedding
        self.max_audio_length = max_audio_length
        self.use_pause_tokens = use_pause_tokens
        self.max_pause_sec = max_pause_sec
        self.pause_fade_ms = pause_fade_ms

        os.makedirs("debug_audio", exist_ok=True)

        self.samples = []
        for i in datasets:
            xml_path = os.path.join(data_path, 'Annotation', f'part_{i}.xml')
            fmt = detect_xml_format(xml_path)
            if fmt == "sentence_level":
                records = parse_sentence_level_xml(xml_path)
            else:
                records = parse_freeform_xml(xml_path)
            self.samples.extend(records)

        print(f"Loaded {len(self.samples)} sentence-level samples from {len(datasets)} dataset(s)")

        if self.max_audio_length is not None:
            self._filter_by_audio_length()

    def __len__(self):
        return len(self.samples)

    def _filter_by_audio_length(self):
        """Filter out samples whose estimated spectrogram length exceeds max_audio_length,
        and samples that are too short (which may become empty after VAD trimming).
        """
        # SpeechT5 feature extractor: hop_length in config is in milliseconds (16ms)
        # Effective hop in samples = sr * hop_length_ms / 1000 = 16000 * 0.016 = 256
        hop_length_samples = 256
        sample_rate = 16000
        max_duration_sec = (self.max_audio_length * hop_length_samples) / sample_rate
        min_duration_sec = 0.5  # minimum to avoid empty waveforms after VAD trimming

        original_len = len(self.samples)
        self.samples = [
            s for s in self.samples
            if min_duration_sec <= _estimate_trimmed_duration(
                s["start"], s["end"], s.get("pauses", []), self.max_pause_sec
            ) <= max_duration_sec
        ]
        print(f"Filtered dataset: {original_len} -> {len(self.samples)} samples "
              f"(removed {original_len - len(self.samples)} samples outside "
              f"{min_duration_sec}s-{max_duration_sec:.1f}s / {self.max_audio_length} frames)")

    def __getitem__(self, idx):
        sample = self.samples[idx]

        wav_file = os.path.join(self.data_path, sample["id"] + ".wav")
        start_sec = sample["start"]
        end_sec = sample["end"]

        waveform, sr = torchaudio.load(wav_file)

        # Slice audio to the sentence time range
        start_frame = int(start_sec * sr)
        end_frame = int(end_sec * sr)
        waveform = waveform[:, start_frame:end_frame]

        if sr != 16000:
            waveform = F.resample(waveform, sr, 16000)
            sr = 16000

        if "part_3" in sample["id"]:
            waveform = trim_pauses(
                waveform, sr, start_sec, sample.get("pauses", []),
                max_pause_sec=self.max_pause_sec, fade_ms=self.pause_fade_ms,
            )
        else:
            waveform = self.edge_trim_vad(waveform)

        waveform = waveform.squeeze()

        normalized_text = text_normalizer(sample["text"], self.use_pause_tokens)
        speaker = sample["speaker"]
        speaker_key = canonical_speaker_id(speaker)
        if speaker_key not in self.speaker_to_embedding:
            raise KeyError(
                f"Missing speaker embedding for '{speaker}' (canonical '{speaker_key}')"
            )

        #torchaudio.save(
        #    f"debug_audio/sample_{idx}.wav",
        #    waveform.cpu(),
        #    sample_rate=16000
        #)

        processed_data = self.processor(
            text=normalized_text,
            audio_target=waveform,
            sampling_rate=16000,
            return_attention_mask=False,
            padding="longest",
        )

        labels = processed_data["labels"][0]
        input_ids = processed_data["input_ids"]

        return {
            "input_ids": input_ids,
            "labels": labels,  # [T, 80] per sample
            "speaker_embeddings": self.speaker_to_embedding[speaker_key],
            "normalized_text": normalized_text,
        }


    def edge_trim_vad(self, waveform):

        # Trim silence
        vad = torchaudio.transforms.Vad(sample_rate=16000)

        # 1. Trim the front
        trimmed_front = vad(waveform)

        # 2. Reverse the audio (flip along the last dimension)
        reversed_audio = torch.flip(trimmed_front, dims=[-1])

        # 3. Trim the 'new' front (which is the original back)
        trimmed_back_reversed = vad(reversed_audio)

        # 4. Reverse back to original orientation
        final_waveform = torch.flip(trimmed_back_reversed, dims=[-1])

        return final_waveform


# ---------------------------------------------------------------------------
# Norwegian number-to-words conversion (already in normalised spelling)
# ---------------------------------------------------------------------------
_ONES = {
    0: "null", 1: "en", 2: "to", 3: "tre", 4: "fire",
    5: "fem", 6: "seks", 7: "sju", 8: "aatte", 9: "ni",
}

_TEENS = {
    10: "ti", 11: "elleve", 12: "tolv", 13: "tretten", 14: "fjorten",
    15: "femten", 16: "seksten", 17: "sytten", 18: "atten", 19: "nitten",
}

_TENS = {
    20: "tjue", 30: "tretti", 40: "foerti", 50: "femti",
    60: "seksti", 70: "sytti", 80: "aatti", 90: "nitti",
}


def _number_to_norwegian(n: int) -> str:
    """Convert an integer 0–999 to Norwegian words (normalised spelling)."""
    if n < 0 or n > 999:
        return " ".join(_ONES[int(d)] for d in str(abs(n)))
    if n <= 9:
        return _ONES[n]
    if n <= 19:
        return _TEENS[n]
    if n <= 99:
        tens, ones = divmod(n, 10)
        word = _TENS[tens * 10]
        if ones:
            word += _ONES[ones]
        return word
    # 100–999
    hundreds, remainder = divmod(n, 100)
    word = "hundre" if hundreds == 1 else _ONES[hundreds] + " hundre"
    if remainder:
        word += " og " + _number_to_norwegian(remainder)
    return word


def _year_to_norwegian(n: int) -> str:
    """Convert a 4-digit year to Norwegian words.

    2000        → "to tusen"
    2001–2009   → "to tusen og en" … "to tusen og ni"
    all others  → century|remainder split, e.g. 1996 → "nitten nittiseks"
    """
    if n == 2000:
        return "to tusen"
    if 2001 <= n <= 2009:
        return "to tusen og " + _ONES[n - 2000]
    # Split into century prefix + two-digit remainder
    century, remainder = divmod(n, 100)
    century_word = _number_to_norwegian(century)
    if remainder == 0:
        return century_word + " hundre"
    return century_word + " " + _number_to_norwegian(remainder)


def _digits_to_words(match: re.Match) -> str:
    """``re.sub`` callback – convert a digit sequence to Norwegian words."""
    s = match.group()
    n = int(s)
    if len(s) == 4:
        return _year_to_norwegian(n)
    if n <= 999:
        return _number_to_norwegian(n)
    # 5+ digits: spell each digit individually
    return " ".join(_ONES[int(d)] for d in s)


def text_normalizer(text, use_pause_tokens):

    if not isinstance(text, str):
        return ""

    text = text.replace('ø', 'oe')
    text = text.replace('Ø', 'Oe')
    text = text.replace('Å', 'Aa')
    text = text.replace('å', 'aa')
    text = text.replace('Æ', 'æ')  # lowercase æ exists in tokenizer vocab
    text = text.replace('è', 'e')
    text = text.replace('ë', 'e')
    text = text.replace('ò', 'o')
    text = text.replace('ô', 'o')
    text = text.replace('ö', 'oe')
    text = text.replace('Ö', 'Oe')
    text = text.replace('ü', 'u')
    text = text.replace('â', 'a')
    text = text.replace('ä', 'æ')
    text = text.replace('É', 'E')
    text = text.replace('é', 'e')
    text = text.replace('ó', 'o')
    text = text.replace('_', ' ')
    text = text.replace("%", "prosent")

    # Strip commas and ellipses for simplicity
    text = text.replace(' ,', '')
    text = text.replace(',', '')
    text = text.replace('...', '')
    text = text.replace(' ...', '')


    if use_pause_tokens:
        # Remove unwanted angle-bracket tags but keep pause tokens.
        # 1) Temporarily replace pause tokens with safe placeholders.
        _placeholders = {}
        for i, tok in enumerate(sorted(PAUSE_TOKEN_SET)):
            placeholder = f"__PAUSE{i}__"
            _placeholders[placeholder] = tok
            text = text.replace(tok, placeholder)

    # 2) Strip all remaining angle-bracket tags.
    text = re.sub(r'<[^>]+> ', '', text)
    text = text.replace('\n', ' ')
    text = text.replace('™', '')
    text = text.replace('«', '')
    text = text.replace('»', '')
    text = text.replace('<', '')
    text = text.replace('|', '')

    if use_pause_tokens:
        # 3) Restore pause tokens.
        for placeholder, tok in _placeholders.items():
            text = text.replace(placeholder, tok)

    # Normalize digit sequences to Norwegian words (handles 0–9999+)
    text = re.sub(r'\d+', _digits_to_words, text)

    return text