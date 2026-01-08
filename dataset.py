import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import torchaudio
import torchaudio.functional as F
import re

class NBTaleDataset(Dataset):
    def __init__(self, data_path, processor, speaker_to_embedding):
        self.data_path = data_path
        self.df = pd.read_xml(os.path.join(data_path, 'Annotation', 'part_1.xml'))
        self.processor = processor
        self.speaker_to_embedding = speaker_to_embedding

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        wav_file = os.path.join(self.data_path, row["id"] + ".wav")
        waveform, sr = torchaudio.load(wav_file)

        waveform = self.audio_normalizer(waveform, sr)  # [T] (time domain waveform)

        normalized_text = self.text_normalizer(row["text"])
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

    def text_normalizer(self, text):

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