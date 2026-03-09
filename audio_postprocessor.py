"""
Audio post-processing filters for TTS output enhancement.
Provides EQ, denoising, spectral gating, and quality assessment.
"""

import torch
import torchaudio
import torchaudio.functional as F
import numpy as np


class AudioPostProcessor:
    """Post-processes TTS audio to improve quality and reduce artifacts."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def remove_low_frequency_rumble(self, waveform: torch.Tensor,
                                    cutoff_hz: int = 80) -> torch.Tensor:
        """
        Remove low-frequency rumble and hum below cutoff_hz.

        Args:
            waveform: Audio tensor [T] or [1, T]
            cutoff_hz: Highpass filter cutoff frequency

        Returns:
            Filtered waveform
        """
        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        # High-pass Butterworth-like filter using biquad
        # Convert cutoff to normalized frequency (0-1, where 1 is Nyquist)
        normalized_freq = cutoff_hz / (self.sample_rate / 2.0)
        normalized_freq = np.clip(normalized_freq, 0.001, 0.999)

        try:
            # Apply highpass filter
            filtered = F.highpass_biquad(
                waveform.unsqueeze(0),
                self.sample_rate,
                cutoff_freq=cutoff_hz
            ).squeeze(0)
            return filtered
        except RuntimeError:
            # Fallback: simple high-pass via differentiation
            return waveform - F.highpass_biquad(
                waveform.unsqueeze(0),
                self.sample_rate,
                cutoff_freq=cutoff_hz * 0.5
            ).squeeze(0)

    def remove_high_frequency_harshness(self, waveform: torch.Tensor,
                                       cutoff_hz: int = 8000) -> torch.Tensor:
        """
        Remove harsh high-frequency content above cutoff_hz (buzzing, artifacts).

        Args:
            waveform: Audio tensor [T] or [1, T]
            cutoff_hz: Lowpass filter cutoff frequency

        Returns:
            Filtered waveform
        """
        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        normalized_freq = cutoff_hz / (self.sample_rate / 2.0)
        normalized_freq = np.clip(normalized_freq, 0.001, 0.999)

        try:
            filtered = F.lowpass_biquad(
                waveform.unsqueeze(0),
                self.sample_rate,
                cutoff_freq=cutoff_hz
            ).squeeze(0)
            return filtered
        except RuntimeError:
            return waveform

    def apply_eq(self, waveform: torch.Tensor, boost_mids: bool = True) -> torch.Tensor:
        """
        Apply parametric EQ to counter tinny character.
        Reduces metallic 3-4kHz peak, boosts warm 200-400Hz fundamentals.

        Args:
            waveform: Audio tensor [T] or [1, T]
            boost_mids: Whether to boost mid-range (warmer sound)

        Returns:
            EQ'd waveform
        """
        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        waveform_unsqueezed = waveform.unsqueeze(0)

        # Boost 200-400Hz (warmth)
        if boost_mids:
            try:
                # Use peaking EQ centered at ~300Hz
                waveform_unsqueezed = F.equalizer_biquad(
                    waveform_unsqueezed,
                    self.sample_rate,
                    center_freq=300,
                    gain=3.0,  # +3dB boost
                    Q=1.0
                )
            except:
                pass

        # Reduce 3-4kHz (metallic peak)
        try:
            waveform_unsqueezed = F.equalizer_biquad(
                waveform_unsqueezed,
                self.sample_rate,
                center_freq=3500,
                gain=-3.0,  # -3dB cut
                Q=1.5  # Narrower Q for precise targeting
            )
        except:
            pass

        return waveform_unsqueezed.squeeze(0)

    def spectral_gate(self, waveform: torch.Tensor,
                     threshold_db: float = -60.0) -> torch.Tensor:
        """
        Apply spectral gating to suppress noise/artifacts below threshold.

        Args:
            waveform: Audio tensor [T] or [1, T]
            threshold_db: Gate threshold in dB relative to peak

        Returns:
            Gated waveform
        """
        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        n_fft = 2048
        hop_length = 512

        stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length,
                          return_complex=True)
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)

        # Convert to dB
        magnitude_db = 20 * torch.log10(magnitude + 1e-10)

        # Find max amplitude per frequency bin
        max_db = torch.max(magnitude_db, dim=1, keepdim=True)[0]

        # Gate: suppress magnitude below threshold
        gate_mask = magnitude_db > (max_db + threshold_db)
        magnitude_gated = magnitude * gate_mask.float()

        # Reconstruct
        stft_gated = magnitude_gated * torch.exp(1j * phase)
        waveform_gated = torch.istft(stft_gated, n_fft=n_fft,
                                     hop_length=hop_length, length=len(waveform))

        return waveform_gated

    def denoise_simple(self, waveform: torch.Tensor,
                      noise_reduction_amount: float = 0.5) -> torch.Tensor:
        """
        Simple spectral subtraction denoising.
        Estimates noise from quiet portions and subtracts from signal.

        Args:
            waveform: Audio tensor [T] or [1, T]
            noise_reduction_amount: Amount to reduce (0-1, where 1 is maximum)

        Returns:
            Denoised waveform
        """
        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        try:
            import librosa
            # Use librosa's simple noise reduction
            waveform_np = waveform.numpy()
            # Apply harmonic-percussive decomposition, keep harmonic component
            harmonic, _ = librosa.effects.hpss(waveform_np)

            # Blend original and harmonic based on noise_reduction_amount
            denoised = waveform_np * (1 - noise_reduction_amount) + harmonic * noise_reduction_amount
            return torch.from_numpy(denoised).float()
        except ImportError:
            # Fallback: spectral subtraction
            n_fft = 2048
            hop_length = 512

            stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length,
                              return_complex=True)
            magnitude = torch.abs(stft)
            phase = torch.angle(stft)

            # Estimate noise from first 500ms
            noise_frames = min(50, magnitude.shape[1] // 4)
            noise_magnitude = torch.mean(magnitude[:, :noise_frames], dim=1, keepdim=True)

            # Spectral subtraction
            magnitude_denoised = magnitude - noise_reduction_amount * noise_magnitude
            magnitude_denoised = torch.clamp(magnitude_denoised, min=0)

            stft_denoised = magnitude_denoised * torch.exp(1j * phase)
            waveform_denoised = torch.istft(stft_denoised, n_fft=n_fft,
                                           hop_length=hop_length, length=len(waveform))

            return waveform_denoised

    def normalize_loudness(self, waveform: torch.Tensor, target_loudness_db: float = -14.0) -> torch.Tensor:
        """
        Normalize audio loudness using RMS scaling.

        Args:
            waveform: Audio tensor [T] or [1, T]
            target_loudness_db: Target loudness in dB

        Returns:
            Loudness-normalized waveform
        """
        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms < 1e-8:
            return waveform

        target_rms = 10 ** (target_loudness_db / 20.0)
        normalized = waveform * (target_rms / rms)

        # Soft clipping to prevent distortion
        normalized = torch.tanh(normalized / 1.2) * 1.2
        return normalized

    def add_subtle_reverb(self, waveform: torch.Tensor,
                         room_scale: float = 0.2) -> torch.Tensor:
        """
        Add subtle reverb/room ambience to reduce digital harshness.

        Args:
            waveform: Audio tensor [T] or [1, T]
            room_scale: Reverb amount (0-1)

        Returns:
            Reverb-enhanced waveform
        """
        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        # Simple reverb using delay and decay
        delay_samples = int(0.05 * self.sample_rate)  # 50ms delay
        decay = 0.4  # 40% of original amplitude

        if len(waveform) <= delay_samples:
            return waveform

        # Create delayed/decayed copy
        delayed = torch.zeros_like(waveform)
        delayed[delay_samples:] = waveform[:-delay_samples] * decay

        # Blend
        reverb_amount = room_scale * 0.5  # Keep subtle
        output = waveform * (1 - reverb_amount) + delayed * reverb_amount

        return output

    def process(self, waveform: torch.Tensor,
               remove_rumble: bool = True,
               remove_harshness: bool = True,
               apply_eq: bool = True,
               apply_gating: bool = False,
               denoise: bool = False,
               normalize: bool = True,
               add_reverb: bool = False) -> torch.Tensor:
        """
        Apply full post-processing pipeline.

        Args:
            waveform: Audio tensor [T] or [1, T]
            remove_rumble: Remove sub-100Hz rumble
            remove_harshness: Remove >8kHz harshness
            apply_eq: Apply parametric EQ for warmth
            apply_gating: Apply spectral gating
            denoise: Apply denoising
            normalize: Normalize loudness
            add_reverb: Add subtle reverb

        Returns:
            Processed waveform
        """
        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        # Remove rumble
        if remove_rumble:
            waveform = self.remove_low_frequency_rumble(waveform)

        # Remove harshness
        if remove_harshness:
            waveform = self.remove_high_frequency_harshness(waveform)

        # Apply EQ
        if apply_eq:
            waveform = self.apply_eq(waveform)

        # Spectral gating
        if apply_gating:
            waveform = self.spectral_gate(waveform)

        # Denoise
        if denoise:
            waveform = self.denoise_simple(waveform)

        # Add reverb
        if add_reverb:
            waveform = self.add_subtle_reverb(waveform)

        # Normalize
        if normalize:
            waveform = self.normalize_loudness(waveform)

        return waveform


if __name__ == "__main__":
    # Example usage
    processor = AudioPostProcessor(sample_rate=16000)

    # Load generated audio
    waveform, sr = torchaudio.load("output.wav")
    if sr != 16000:
        waveform = F.resample(waveform, sr, 16000)

    # Apply post-processing
    processed = processor.process(
        waveform,
        remove_rumble=True,
        remove_harshness=True,
        apply_eq=True,
        apply_gating=False,
        denoise=False,
        normalize=True,
        add_reverb=False
    )

    # Save
    torchaudio.save("output_processed.wav", processed.unsqueeze(0), 16000)
    print("Saved processed audio to output_processed.wav")





