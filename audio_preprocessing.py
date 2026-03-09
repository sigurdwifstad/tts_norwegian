"""
Audio preprocessing utilities for TTS training data normalization.
Handles speaker normalization, loudness standardization, and spectral centering.
"""

import torch
import torchaudio
import torchaudio.functional as F
import numpy as np
from pathlib import Path


class AudioPreprocessor:
    """Preprocesses audio for consistent quality and loudness standards."""

    def __init__(self, target_loudness_lufs: float = -14.0, sample_rate: int = 16000):
        """
        Args:
            target_loudness_lufs: Target loudness in LUFS (Loudness Units relative to Full Scale).
                                  Typical values: -14 LUFS (streaming), -23 LUFS (broadcast).
            sample_rate: Target sample rate in Hz.
        """
        self.target_loudness_lufs = target_loudness_lufs
        self.sample_rate = sample_rate

    def normalize_loudness(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Normalize audio loudness to target LUFS using simplified loudness calculation.
        Falls back to RMS normalization if pyloudnorm unavailable.

        Args:
            waveform: Audio tensor of shape [T] or [1, T]

        Returns:
            Loudness-normalized waveform
        """
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        try:
            import pyloudnorm
            meter = pyloudnorm.Meter(self.sample_rate)
            loudness = meter.integrated_loudness(waveform.numpy())

            if np.isfinite(loudness):
                loudness_normalized = meter.normalize(waveform.numpy(), self.target_loudness_lufs)
                return torch.from_numpy(loudness_normalized).float()
        except (ImportError, ValueError, RuntimeError, AttributeError):
            # Fallback: RMS-based normalization
            pass

        # Fallback: Simple RMS normalization
        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms > 1e-8:
            target_rms = 10 ** (self.target_loudness_lufs / 20.0)
            waveform = waveform * (target_rms / rms)
            waveform = torch.clamp(waveform, -1.0, 1.0)  # Prevent clipping

        return waveform

    def remove_silence(self, waveform: torch.Tensor, threshold_db: float = -40.0) -> torch.Tensor:
        """
        Remove silence from audio using energy-based detection.

        Args:
            waveform: Audio tensor of shape [T] or [1, T]
            threshold_db: Silence threshold in dB relative to max amplitude

        Returns:
            Trimmed waveform
        """
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        # Use VAD for robust silence removal
        vad = torchaudio.transforms.Vad(sample_rate=self.sample_rate)

        # Trim from front
        trimmed_front = vad(waveform.unsqueeze(0)).squeeze(0)

        # Check if VAD removed everything (edge case with very noisy audio)
        if len(trimmed_front) == 0:
            return waveform

        # Trim from back by reversing
        reversed_audio = torch.flip(trimmed_front, dims=[0])
        trimmed_back_reversed = vad(reversed_audio.unsqueeze(0)).squeeze(0)

        # Check again
        if len(trimmed_back_reversed) == 0:
            return trimmed_front

        final_waveform = torch.flip(trimmed_back_reversed, dims=[0])

        return final_waveform

    def normalize_sample_rate(self, waveform: torch.Tensor, original_sr: int) -> torch.Tensor:
        """Resample audio to target sample rate if needed."""
        if original_sr != self.sample_rate:
            waveform = F.resample(waveform, original_sr, self.sample_rate)
        return waveform

    def process(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Apply full preprocessing pipeline to audio.

        Args:
            waveform: Raw audio tensor
            sr: Original sample rate

        Returns:
            Preprocessed audio at target sample rate with normalized loudness
        """
        # Resample if needed
        waveform = self.normalize_sample_rate(waveform, sr)

        # Remove silence
        waveform = self.remove_silence(waveform)

        # Normalize loudness
        waveform = self.normalize_loudness(waveform)

        return waveform

    @staticmethod
    def process_dataset_directory(input_dir: str, output_dir: str,
                                   target_loudness_lufs: float = -14.0) -> None:
        """
        Batch process all WAV files in a directory, writing normalized versions.

        Args:
            input_dir: Directory containing source WAV files
            output_dir: Directory for normalized WAV files
            target_loudness_lufs: Target loudness in LUFS
        """
        preprocessor = AudioPreprocessor(target_loudness_lufs=target_loudness_lufs)
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        wav_files = list(input_path.glob("**/*.wav"))
        print(f"Processing {len(wav_files)} audio files...")

        for wav_file in wav_files:
            try:
                waveform, sr = torchaudio.load(str(wav_file))
                processed = preprocessor.process(waveform, sr)

                # Maintain relative directory structure
                relative_path = wav_file.relative_to(input_path)
                output_file = output_path / relative_path
                output_file.parent.mkdir(parents=True, exist_ok=True)

                torchaudio.save(str(output_file), processed.unsqueeze(0), preprocessor.sample_rate)
                print(f"✓ {relative_path}")
            except Exception as e:
                print(f"✗ {wav_file.name}: {e}")

        print(f"Preprocessing complete. Output saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    preprocessor = AudioPreprocessor(target_loudness_lufs=-14.0)

    # Process a single file
    waveform, sr = torchaudio.load("sample.wav")
    processed = preprocessor.process(waveform, sr)
    torchaudio.save("sample_normalized.wav", processed.unsqueeze(0), 16000)

    # Or batch process a directory
    # AudioPreprocessor.process_dataset_directory("./data/raw", "./data/normalized")


