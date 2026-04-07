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
from dataset import NBTaleDataset, PAUSE_TOKEN_SET
from collections import Counter

checkpoint = "microsoft/speecht5_tts"
processor = SpeechT5Processor.from_pretrained(checkpoint)

# Register pause tokens so they are recognized as whole tokens
num_added = processor.tokenizer.add_tokens(sorted(PAUSE_TOKEN_SET))
if num_added > 0:
    print(f"Registered {num_added} pause tokens with tokenizer")

tokenizer = processor.tokenizer

print("Vocab size:", len(tokenizer))

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
    data_path="data/shure",
    processor=processor,
    speaker_to_embedding=speaker_to_embedding,
    datasets=[3],
)


dataset_chars = Counter()

for example in train_dataset:
    text = example["normalized_text"]
    for ch in text:
        dataset_chars[ch] += 1

dataset_char_set = set(dataset_chars.keys())

# Remove < and > from the character set since they are part of pause tokens,
# not standalone characters the tokenizer needs to handle individually.
dataset_char_set -= {'<', '>'}

print("Unique characters in dataset:", len(dataset_char_set))
print(sorted(dataset_char_set))

missing = dataset_char_set - tokens
extra = tokens - dataset_char_set

print("Characters in dataset but NOT in tokenizer:")
print(sorted(missing))

print("\nTokenizer tokens never seen in dataset (normal):")
print(list(extra)[:50])

# Also verify pause tokens are properly tokenized (not <unk>)
print("\nPause token check:")
unk_id = tokenizer.unk_token_id
for tok in sorted(PAUSE_TOKEN_SET):
    tid = tokenizer.convert_tokens_to_ids(tok)
    status = "OK" if tid != unk_id else "MISSING"
    print(f"  {tok} -> id={tid}  [{status}]")
