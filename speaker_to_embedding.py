from speechbrain.pretrained import EncoderClassifier
import torchaudio
import torch
import pandas as pd
import os
import sounddevice as sd
import soundfile as sf

def create_speaker_embeddings(data_path):
    df = pd.read_xml(os.path.join(data_path, "Annotation", "part_1.xml"))

    spk_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    speaker_to_embedding = {}

    for speaker in df["speaker"].unique():
        row = df[df["speaker"] == speaker].iloc[0]
        wav_path = os.path.join(data_path, row["id"] + ".wav")

        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        with torch.no_grad():
            emb = spk_model.encode_batch(waveform)
            emb = emb.squeeze(0).squeeze(0).cpu()

        speaker_to_embedding[speaker] = emb

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

    record_voice("my_voice.wav", duration=60, sample_rate=16000)
    embedding = create_custom_speaker_embedding("my_voice.wav")
    print("Speaker embedding shape:", embedding.shape)
    print("Speaker embedding:", embedding)
