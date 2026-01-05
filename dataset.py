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

    def filter_text(self, text):
        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)
        text = text.replace('æ', 'ae').replace('ø', 'oe').replace('å', 'aa')
        return text

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        wav_file = os.path.join(self.data_path, row["id"] + ".wav")
        waveform, sr = torchaudio.load(wav_file)

        # resample to 16 kHz
        if sr != 16000:
            waveform = F.resample(waveform, sr, 16000)

        waveform = waveform.squeeze(0)  # [T]

        normalized_text = self.filter_text(row["text"])
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
        }