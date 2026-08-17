from speechbrain.pretrained import EncoderClassifier
import torchaudio
import torch
import pandas as pd
import os
import re
import sounddevice as sd
import soundfile as sf


def _canonical_speaker_id(raw_speaker_id: str) -> str:
    """Map IDs like p1_g01_f1_1_t to g01_f1_1 (ignore part + train/test)."""
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


def _find_wav_path(data_path: str, utt_id: str) -> str:
    candidates = [
        os.path.join(data_path, f"{utt_id}.wav"),
        os.path.join(data_path, "part_1", f"{utt_id}.wav"),
        os.path.join(data_path, "part_2", f"{utt_id}.wav"),
        os.path.join(data_path, "part_3", f"{utt_id}.wav"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find wav for utterance id '{utt_id}'")


def _l2_normalize(vec: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(vec, p=2, dim=0)


def create_speaker_embeddings(data_path):

    df = pd.DataFrame()
    for i in [1, 2, 3]:
        df = pd.concat((df, pd.read_xml(os.path.join(data_path, "Annotation", f"part_{i}.xml"))))

    # Aggregate across parts (p1/p2/p3) and split suffix (_t/_x).
    df["canonical_speaker"] = df["speaker"].apply(_canonical_speaker_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    spk_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        run_opts={"device": device}
    )

    speaker_to_embedding = {}

    for speaker in sorted(df["canonical_speaker"].unique()):
        rows = df[df["canonical_speaker"] == speaker]
        utterance_embeddings = []

        for _, row in rows.iterrows():
            wav_path = _find_wav_path(data_path, row["id"])
            waveform, sr = torchaudio.load(wav_path)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)

            waveform = waveform.to(device)
            with torch.no_grad():
                emb = spk_model.encode_batch(waveform)
                emb = emb.squeeze(0).squeeze(0).cpu()
                utterance_embeddings.append(_l2_normalize(emb))

        if not utterance_embeddings:
            continue

        mean_embedding = torch.stack(utterance_embeddings, dim=0).mean(dim=0)
        speaker_to_embedding[speaker] = _l2_normalize(mean_embedding)

    return speaker_to_embedding


def record_voice(output_path, duration=10, sample_rate=16000):
    print("Recording...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32")
    sd.wait()
    sf.write(output_path, audio, sample_rate)
    print("Saved to", output_path)


def create_custom_speaker_embedding(wav_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    spk_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        run_opts={"device": device}
    )

    waveform, sr = torchaudio.load(wav_path)

    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    waveform = waveform.to(device)

    with torch.no_grad():
        emb = spk_model.encode_batch(waveform)
        emb = emb.squeeze(0).squeeze(0).cpu()

    return emb

if __name__ == "__main__":

    #record_voice("my_voice.wav", duration=60, sample_rate=16000)
    #embedding = create_custom_speaker_embedding("my_voice.wav")

    data_path = "data/shure"
    embeddings = create_speaker_embeddings(data_path)

    print("Number of speakers:", len(embeddings))
    if embeddings:
        first_speaker = next(iter(embeddings))
        print("Example speaker id:", first_speaker)
        print("Embedding shape:", embeddings[first_speaker].shape)
