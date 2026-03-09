import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import torchaudio
import torchaudio.functional as F
import re
import random
from typing import Tuple

class DataAugmentationPipeline:
    """Applies data augmentation to audio waveforms during training."""

    def __init__(self, augmentation_probability: float = 0.5,
                 pitch_shift_cents: int = 50, time_stretch_range: Tuple = (0.95, 1.05),
                 noise_factor: float = 0.005):
        """
        Args:
            augmentation_probability: Probability (0-1) of applying each augmentation
            pitch_shift_cents: Max pitch shift in cents (100 cents = 1 semitone)
            time_stretch_range: (min, max) multipliers for time-stretching
            noise_factor: Amplitude of Gaussian noise relative to signal RMS
        """
        self.augmentation_probability = augmentation_probability
        self.pitch_shift_cents = pitch_shift_cents
        self.time_stretch_range = time_stretch_range
        self.noise_factor = noise_factor

    def pitch_shift(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply random pitch shift (±pitch_shift_cents)."""
        if random.random() > self.augmentation_probability:
            return waveform

        # Pitch shift in cents: random between -pitch_shift_cents and +pitch_shift_cents
        shift_cents = random.uniform(-self.pitch_shift_cents, self.pitch_shift_cents)
        shift_steps = shift_cents / 100.0  # Convert cents to semitones

        try:
            # Use librosa if available for better quality
            import librosa
            waveform_np = waveform.numpy()
            shifted = librosa.effects.pitch_shift(waveform_np, sr=sample_rate, n_steps=shift_steps)
            return torch.from_numpy(shifted).float()
        except ImportError:
            # Fallback: use torchaudio (less smooth but functional)
            # Create a small frequency warping using resample trick
            shift_factor = 2 ** (shift_steps / 12.0)
            new_sr = int(sample_rate / shift_factor)

            # Resample down then up to shift pitch
            resampled = F.resample(waveform, sample_rate, new_sr)
            pitch_shifted = F.resample(resampled, new_sr, sample_rate)
            return pitch_shifted

    def time_stretch(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random time-stretching (speed change without pitch change)."""
        if random.random() > self.augmentation_probability:
            return waveform

        stretch_factor = random.uniform(self.time_stretch_range[0], self.time_stretch_range[1])
        original_length = len(waveform)

        try:
            # Use librosa for high-quality time-stretching
            import librosa
            waveform_np = waveform.numpy()
            stretched = librosa.effects.time_stretch(waveform_np, rate=stretch_factor)
            stretched_tensor = torch.from_numpy(stretched).float()
        except ImportError:
            # Fallback: simple resampling (changes pitch as well - not ideal)
            new_length = int(len(waveform) / stretch_factor)
            stretched_tensor = F.resample(
                waveform.unsqueeze(0),
                orig_freq=len(waveform),
                new_freq=new_length
            ).squeeze(0)

        # Ensure output matches original length
        stretched_length = len(stretched_tensor)
        if stretched_length > original_length:
            # Truncate if too long
            stretched_tensor = stretched_tensor[:original_length]
        elif stretched_length < original_length:
            # Pad with zeros if too short
            padding = original_length - stretched_length
            stretched_tensor = torch.nn.functional.pad(stretched_tensor, (0, padding))

        return stretched_tensor

    def add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to audio."""
        if random.random() > self.augmentation_probability:
            return waveform

        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms < 1e-8:
            return waveform

        noise = torch.randn_like(waveform) * self.noise_factor * rms
        noisy = waveform + noise

        # Soft clipping to prevent distortion
        noisy = torch.tanh(noisy)
        return noisy

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply augmentation pipeline to waveform."""
        # Apply augmentations in sequence
        waveform = self.pitch_shift(waveform, sample_rate)
        waveform = self.time_stretch(waveform)
        waveform = self.add_noise(waveform)

        return waveform


class NBTaleDataset(Dataset):
    def __init__(self, data_path, processor, speaker_to_embedding,
                 enable_augmentation: bool = True, augmentation_probability: float = 0.5):
        self.data_path = data_path
        self.df = pd.read_xml(os.path.join(data_path, 'Annotation', 'part_1.xml'))
        self.processor = processor
        self.speaker_to_embedding = speaker_to_embedding

        # Initialize augmentation pipeline
        self.augmentation = DataAugmentationPipeline(
            augmentation_probability=augmentation_probability
        ) if enable_augmentation else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        wav_file = os.path.join(self.data_path, row["id"] + ".wav")
        waveform, sr = torchaudio.load(wav_file)

        waveform = self.audio_normalizer(waveform, sr)  # [T] (time domain waveform)

        # Apply augmentation during training
        if self.augmentation is not None:
            waveform = self.augmentation(waveform, 16000)

        normalized_text = text_normalizer(row["text"])
        speaker = row["speaker"]

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

    def audio_normalizer(self, waveform, sr):
        if sr != 16000:
            waveform = F.resample(waveform, sr, 16000)

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

        return final_waveform.squeeze(0)

def text_normalizer(text):

    # TODO: <sil> tokens must be included somehow.

    text = re.sub(r'<[^>]+>', '', text)
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
    text = text.replace('ü', 'u')

    # Remove unwanted characters
    text = text.replace('\n', ' ')
    text = text.replace('™', '')
    text = text.replace('«', '')
    text = text.replace('»', '')
    text = text.replace('<', '')
    text = text.replace('|', '')

    # Normalize digits to words # TODO: are multi-digit numbers common and needs handling?
    text = text.replace('0', 'null')
    text = text.replace('1', 'en')
    text = text.replace('2', 'to')
    text = text.replace('3', 'tre')
    text = text.replace('4', 'fire')
    text = text.replace('5', 'fem')
    text = text.replace('6', 'seks')
    text = text.replace('7', 'sju')
    text = text.replace('8', 'aatte')
    text = text.replace('9', 'ni')

    return text