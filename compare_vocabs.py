import os
import torch
import numpy as np
from dataclasses import dataclass
from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    Trainer,
    TrainingArguments,
)
from dataset import NBTaleDataset
from collections import Counter

checkpoint = "microsoft/speecht5_tts"
processor = SpeechT5Processor.from_pretrained(checkpoint)

tokenizer = processor.tokenizer

print("Vocab size:", tokenizer.vocab_size)

# Get all tokens
vocab = tokenizer.get_vocab()
tokens = set(vocab.keys())
# Show some examples
print(list(tokens)[:50])
# Show all <...> tokens:
special_tokens = [t for t in tokens if t.startswith("<") and t.endswith(">")]
print("Special tokens:", special_tokens)

speaker_to_embedding = torch.load("speaker_embeddings.pt")
train_dataset = NBTaleDataset(
    data_path="./data",
    processor=processor,
    speaker_to_embedding=speaker_to_embedding,
)


dataset_chars = Counter()

for example in train_dataset:
    text = example["normalized_text"]
    for ch in text:
        dataset_chars[ch] += 1

dataset_char_set = set(dataset_chars.keys())

print("Unique characters in dataset:", len(dataset_char_set))
print(sorted(dataset_char_set))

missing = dataset_char_set - tokens
extra = tokens - dataset_char_set

print("Characters in dataset but NOT in tokenizer:")
print(sorted(missing))

print("\nTokenizer tokens never seen in dataset (normal):")
print(list(extra)[:50])