"""
Audio quality assessment utilities for evaluating TTS output quality.
Measures SNR, spectral characteristics, perceptual metrics, and vocoder artifacts.
"""

import torch
import torchaudio
import torchaudio.functional as F
import numpy as np
from typing import Dict, Tuple


class AudioQualityAssessment:
    """Analyzes audio quality metrics to detect artifacts and noise."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def calculate_snr(self, waveform: torch.Tensor,
                      noise_profile_start: int = 0,
                      noise_profile_duration: int = 0) -> float:
        """
        Calculate Signal-to-Noise Ratio using spectral analysis.

        Args:
            waveform: Audio tensor [T]
            noise_profile_start: Start sample for noise profile
            noise_profile_duration: Duration in samples for noise estimation

        Returns:
            SNR in dB
        """
        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        # Calculate STFT
        n_fft = 2048
        hop_length = 512
        stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length,
                          return_complex=True)
        magnitude = torch.abs(stft)

        # Estimate noise power from beginning or specified region
        if noise_profile_duration > 0:
            start_frame = noise_profile_start // hop_length
            end_frame = start_frame + (noise_profile_duration // hop_length)
            noise_power = torch.mean(magnitude[:, start_frame:end_frame] ** 2, dim=1)
        else:
            # Use first 500ms as noise estimate
            noise_frames = min(50, magnitude.shape[1] // 4)
            noise_power = torch.mean(magnitude[:, :noise_frames] ** 2, dim=1)

        # Signal power (entire audio)
        signal_power = torch.mean(magnitude ** 2, dim=1)

        # SNR = 10 * log10(signal_power / noise_power)
        snr_per_bin = 10 * torch.log10(signal_power / (noise_power + 1e-10) + 1e-10)
        snr_db = float(torch.mean(snr_per_bin))

        return snr_db

    def calculate_spectral_centroid(self, waveform: torch.Tensor) -> float:
        """
        Calculate spectral centroid (center of mass of spectrum).
        Lower values indicate duller sound, higher indicate brighter/tinny sound.

        Args:
            waveform: Audio tensor [T]

        Returns:
            Spectral centroid in Hz
        """
        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        n_fft = 2048
        stft = torch.stft(waveform, n_fft=n_fft, return_complex=True)
        magnitude = torch.abs(stft)

        # Frequency bins
        freqs = torch.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)

        # Power spectrum
        power = magnitude ** 2

        # Spectral centroid: sum(freq * power) / sum(power)
        numerator = torch.sum(freqs.unsqueeze(1) * power, dim=0)
        denominator = torch.sum(power, dim=0)
        centroid_per_frame = numerator / (denominator + 1e-10)

        return float(torch.mean(centroid_per_frame))

    def calculate_loudness_variance(self, waveform: torch.Tensor,
                                   frame_length: int = 2048) -> Tuple[float, float]:
        """
        Calculate loudness (LUFS approximation) and its variance.
        High variance indicates unnatural dynamics.

        Args:
            waveform: Audio tensor [T]
            frame_length: Frame length for loudness calculation

        Returns:
            (mean_loudness_db, variance_db)
        """
        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        # Simple frame-based loudness calculation
        frames = []
        for i in range(0, len(waveform) - frame_length, frame_length // 2):
            frame = waveform[i:i + frame_length]
            loudness = 20 * torch.log10(torch.sqrt(torch.mean(frame ** 2)) + 1e-10)
            frames.append(loudness.item())

        if not frames:
            return -100.0, 0.0

        frames = np.array(frames)
        return float(np.mean(frames)), float(np.var(frames))

    def detect_high_frequency_noise(self, waveform: torch.Tensor,
                                   threshold_hz: int = 8000) -> float:
        """
        Detect high-frequency noise (buzzing, artifacts above threshold_hz).
        Returns proportion of energy above threshold.

        Args:
            waveform: Audio tensor [T]
            threshold_hz: Frequency threshold

        Returns:
            Proportion of energy above threshold (0-1)
        """
        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        n_fft = 2048
        stft = torch.stft(waveform, n_fft=n_fft, return_complex=True)
        magnitude = torch.abs(stft)
        power = magnitude ** 2

        # Frequency resolution
        freqs = torch.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)

        # Find bins above threshold
        high_freq_bins = freqs > threshold_hz

        high_freq_energy = torch.sum(power[high_freq_bins, :])
        total_energy = torch.sum(power)

        proportion = float(high_freq_energy / (total_energy + 1e-10))
        return proportion

    def calculate_mel_spectrogram_smoothness(self, waveform: torch.Tensor) -> float:
        """
        Calculate mel-spectrogram smoothness (spectral continuity).
        Lower values indicate more artifacts/buzzing.

        Args:
            waveform: Audio tensor [T]

        Returns:
            Smoothness score (higher is smoother, better quality)
        """
        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        # Create mel-spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=80
        )
        mel_spec = mel_transform(waveform)
        mel_spec_db = 20 * torch.log10(mel_spec + 1e-10)

        # Calculate smoothness as negative variance of spectral deltas
        # (sudden changes in spectrum indicate artifacts)
        delta = torch.diff(mel_spec_db, dim=1)
        smoothness = float(-torch.mean(torch.abs(delta)))  # Negative because lower variance is better

        return smoothness

    def assess_quality(self, waveform: torch.Tensor) -> Dict[str, float]:
        """
        Comprehensive quality assessment of audio.

        Args:
            waveform: Audio tensor [T]

        Returns:
            Dictionary with quality metrics
        """
        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        metrics = {
            "snr_db": self.calculate_snr(waveform),
            "spectral_centroid_hz": self.calculate_spectral_centroid(waveform),
            "high_freq_noise_proportion": self.detect_high_frequency_noise(waveform),
            "mel_smoothness": self.calculate_mel_spectrogram_smoothness(waveform),
        }

        loudness_mean, loudness_var = self.calculate_loudness_variance(waveform)
        metrics["loudness_mean_db"] = loudness_mean
        metrics["loudness_variance_db"] = loudness_var

        return metrics

    @staticmethod
    def quality_report(metrics: Dict[str, float]) -> str:
        """Generate human-readable quality report from metrics."""
        report = "Audio Quality Assessment Report\n"
        report += "=" * 50 + "\n"
        report += f"SNR: {metrics['snr_db']:.2f} dB {'(Good)' if metrics['snr_db'] > 20 else '(Poor)'}\n"
        report += f"Spectral Centroid: {metrics['spectral_centroid_hz']:.0f} Hz "

        if metrics['spectral_centroid_hz'] < 2000:
            report += "(Muffled)\n"
        elif metrics['spectral_centroid_hz'] > 5000:
            report += "(Tinny/Bright)\n"
        else:
            report += "(Balanced)\n"

        report += f"High Freq Noise: {metrics['high_freq_noise_proportion']*100:.1f}% "
        if metrics['high_freq_noise_proportion'] > 0.3:
            report += "(High buzzing)\n"
        elif metrics['high_freq_noise_proportion'] > 0.1:
            report += "(Some buzzing)\n"
        else:
            report += "(Clean)\n"

        report += f"Mel Smoothness: {metrics['mel_smoothness']:.3f} "
        if metrics['mel_smoothness'] > -0.5:
            report += "(Smooth)\n"
        else:
            report += "(Artifacts)\n"

        report += f"Loudness: {metrics['loudness_mean_db']:.2f} dB (±{metrics['loudness_variance_db']:.2f} dB)\n"
        report += "=" * 50

        return report


if __name__ == "__main__":
    # Example usage
    assessor = AudioQualityAssessment(sample_rate=16000)

    # Load and assess an audio file
    waveform, sr = torchaudio.load("output.wav")

    if sr != 16000:
        waveform = F.resample(waveform, sr, 16000)

    metrics = assessor.assess_quality(waveform.squeeze())
    print(assessor.quality_report(metrics))



