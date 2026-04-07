import argparse
import os
import random

import numpy as np
import torch
import torchaudio
from transformers import SpeechT5Processor

from dataset import NBTaleDataset, text_normalizer
from speaker_to_embedding import create_speaker_embeddings


def load_speaker_embeddings(embeddings_path, data_path):
    if os.path.exists(embeddings_path):
        return torch.load(embeddings_path, map_location="cpu")

    print(f"Embeddings file not found at {embeddings_path}. Creating from dataset...")
    speaker_to_embedding = create_speaker_embeddings(data_path)
    torch.save(speaker_to_embedding, embeddings_path)
    print(f"Saved embeddings to {embeddings_path}")
    return speaker_to_embedding


def load_audio_clip(data_path, record):
    wav_path = os.path.join(data_path, record["id"] + ".wav")
    waveform, sr = torchaudio.load(wav_path)

    start_frame = int(record["start"] * sr)
    end_frame = int(record["end"] * sr)
    clip = waveform[:, start_frame:end_frame]

    if sr != 16000:
        clip = torchaudio.functional.resample(clip, sr, 16000)
        sr = 16000

    return clip.squeeze(0), sr


def maybe_play_audio(waveform, sr, enabled):
    if not enabled:
        return

    try:
        import sounddevice as sd

        sd.play(waveform.numpy(), samplerate=sr)
        sd.wait()
    except Exception as exc:
        print(f"Audio playback skipped: {exc}")


def print_sample(dataset, idx, play_audio=False):
    raw = dataset.samples[idx]
    clip, sr = load_audio_clip(dataset.data_path, raw)

    print("\n" + "=" * 70)
    print(f"Index      : {idx}")
    print(f"Speaker    : {raw['speaker']}")
    print(f"Text       : {raw['text']}")
    print(f"Normalized : {text_normalizer(raw['text'])}")
    print(
        f"Time range : {raw['start']:.2f}s -> {raw['end']:.2f}s "
        f"({raw['end'] - raw['start']:.2f}s)"
    )
    print(f"Audio      : {clip.shape[0]} samples @ {sr} Hz")

    # Pull processed tensors to verify what training sees.
    processed = dataset[idx]
    print(f"input_ids shape   : {np.array(processed['input_ids']).shape}")
    print(f"labels shape      : {np.array(processed['labels']).shape}")
    print(f"speaker emb shape : {np.array(processed['speaker_embeddings']).shape}")

    maybe_play_audio(clip, sr, play_audio)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect NBTale part_3 dataloader samples as text/audio pairs from a regular Python script."
        )
    )
    parser.add_argument("--data-path", default="data/shure", help="Path to dataset root")
    parser.add_argument("--part", type=int, default=3, help="Dataset part index (default: 3)")
    parser.add_argument("--checkpoint", default="microsoft/speecht5_tts", help="SpeechT5 checkpoint")
    parser.add_argument("--embeddings-path", default="speaker_embeddings.pt", help="Speaker embeddings .pt file")
    parser.add_argument("--max-audio-length", type=int, default=1876, help="Max spectrogram frames (set -1 for no cap)")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of random samples to print")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--play-audio", action="store_true", help="Play each sample audio")
    parser.add_argument("--interactive", action="store_true", help="Interactive sample browser")
    args = parser.parse_args()

    random.seed(args.seed)

    max_audio_length = None if args.max_audio_length < 0 else args.max_audio_length

    print("Loading processor...")
    processor = SpeechT5Processor.from_pretrained(args.checkpoint)

    print("Loading speaker embeddings...")
    speaker_to_embedding = load_speaker_embeddings(args.embeddings_path, args.data_path)

    print("Building dataset...")
    dataset = NBTaleDataset(
        data_path=args.data_path,
        processor=processor,
        speaker_to_embedding=speaker_to_embedding,
        datasets=[args.part],
        max_audio_length=max_audio_length,
    )

    if len(dataset) == 0:
        raise ValueError("Dataset is empty after filtering. Increase max_audio_length or inspect XML parsing.")

    print(f"Dataset size: {len(dataset)}")

    n = min(args.num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), n)
    for idx in indices:
        print_sample(dataset, idx, play_audio=args.play_audio)

    if not args.interactive:
        return

    idx = 0
    print("\nInteractive mode: [n]ext, [p]rev, [j <idx>] jump, [q]uit")
    print_sample(dataset, idx, play_audio=args.play_audio)

    while True:
        cmd = input("cmd> ").strip().lower()
        if cmd == "q":
            break
        if cmd == "n":
            idx = (idx + 1) % len(dataset)
        elif cmd == "p":
            idx = (idx - 1) % len(dataset)
        elif cmd.startswith("j "):
            try:
                new_idx = int(cmd.split(maxsplit=1)[1])
                if 0 <= new_idx < len(dataset):
                    idx = new_idx
                else:
                    print(f"Index must be in [0, {len(dataset)-1}]")
                    continue
            except ValueError:
                print("Usage: j <index>")
                continue
        else:
            print("Unknown command. Use n, p, j <idx>, q")
            continue

        print_sample(dataset, idx, play_audio=args.play_audio)


if __name__ == "__main__":
    main()
