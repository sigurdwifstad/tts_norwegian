import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import torchaudio
import torchaudio.functional as F
import re
import xml.etree.ElementTree as ET

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
    Returns a list of dicts with keys: id, speaker, text, start, end.
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
                    })

                # Reset for the next sentence
                sentence_start = boundary_end   # default; may be pushed forward by gap elements
                awaiting_first_word = True
                current_words = []

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
            })

    return records


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
    def __init__(self, data_path, processor, speaker_to_embedding, datasets=[1], max_audio_length=None):
        self.data_path = data_path
        self.processor = processor
        self.speaker_to_embedding = speaker_to_embedding
        self.max_audio_length = max_audio_length

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
            if min_duration_sec <= (s["end"] - s["start"]) <= max_duration_sec
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

        if "part_3" not in sample["id"]:
            waveform = self.edge_trim_vad(waveform)

        waveform = waveform.squeeze()

        normalized_text = text_normalizer(sample["text"])
        speaker = sample["speaker"]

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
            "speaker_embeddings": self.speaker_to_embedding[speaker],
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


def text_normalizer(text):

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

    # Remove unwanted angle-bracket tags but keep pause tokens.
    # 1) Temporarily replace pause tokens with safe placeholders.
    _placeholders = {}
    for i, tok in enumerate(sorted(PAUSE_TOKEN_SET)):
        placeholder = f"__PAUSE{i}__"
        _placeholders[placeholder] = tok
        text = text.replace(tok, placeholder)

    # 2) Strip all remaining angle-bracket tags.
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('\n', ' ')
    text = text.replace('™', '')
    text = text.replace('«', '')
    text = text.replace('»', '')
    text = text.replace('<', '')
    text = text.replace('|', '')

    # 3) Restore pause tokens.
    for placeholder, tok in _placeholders.items():
        text = text.replace(placeholder, tok)

    # Normalize digit sequences to Norwegian words (handles 0–9999+)
    text = re.sub(r'\d+', _digits_to_words, text)

    return text