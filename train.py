import os
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    Trainer,
    TrainingArguments,
)
from dataset import NBTaleDataset, PAUSE_TOKEN_SET
from speaker_to_embedding import create_speaker_embeddings


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)

output_dir = "models/speecht5_NBTale_tts_shure_123"

checkpoint = "microsoft/speecht5_tts"
data_path = "data/shure"

processor = SpeechT5Processor.from_pretrained(checkpoint)
model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)

# Register pause tokens (<sil>, <inhale>, <exhale>, <fp>) so the model
# can learn to produce pauses at the positions annotated in part_3 XML.
num_added = processor.tokenizer.add_tokens(sorted(PAUSE_TOKEN_SET))
if num_added > 0:
    model.resize_token_embeddings(len(processor.tokenizer))
    print(f"Added {num_added} pause tokens to tokenizer "
          f"(new vocab size: {len(processor.tokenizer)})")

model.config.use_cache = False

# TODO: is this better?
# Freeze encoder for stability
#for param in model.encoder.parameters():
#    param.requires_grad = False



if not os.path.exists("speaker_embeddings.pt"):
    speaker_to_embedding = create_speaker_embeddings(data_path)
    torch.save(speaker_to_embedding, "speaker_embeddings.pt")
else:
    speaker_to_embedding = torch.load("speaker_embeddings.pt")

# sanity check
assert next(iter(speaker_to_embedding.values())).shape[-1] == 512

# ===============================
# Dataset
# ===============================
train_dataset = NBTaleDataset(
    data_path=data_path,
    processor=processor,
    speaker_to_embedding=speaker_to_embedding,
    datasets=[1,2,3],
    max_audio_length=1876,  # SpeechT5 positional encoding limit
)


@dataclass
class TTSDataCollator:
    processor: Any
    max_length: int = None

    def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        if self.max_length is not None:
            features = [
                f for f in features
                if len(f["labels"]) <= self.max_length
            ]
            if len(features) == 0:
                raise ValueError("All samples in batch exceeded max_length. "
                                 "Increase max_length or filter the dataset.")

        input_ids = [
            {"input_ids": feature["input_ids"]}
            for feature in features
        ]

        label_features = [
            {"input_values": feature["labels"]}
            for feature in features
        ]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # Pad input_ids and labels separately
        batch = processor.pad(
            input_ids=input_ids, labels=label_features, return_tensors="pt"
        )

        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )

        del batch["decoder_attention_mask"]

        if model.config.reduction_factor > 1:
            target_lengths = torch.tensor(
                [len(feature["input_values"]) for feature in label_features]
            )
            target_lengths = target_lengths.new(
                [
                    length - length % model.config.reduction_factor
                    for length in target_lengths
                ]
            )
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        batch["speaker_embeddings"] = torch.tensor(np.array(speaker_features))

        return batch

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    warmup_steps=500,
    max_steps=2000,
    fp16=torch.cuda.is_available(),
    logging_steps=25,
    save_steps=200,
    eval_steps=200,
    report_to=["tensorboard"],
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=TTSDataCollator(processor, max_length=1876),
)

if __name__ == "__main__":
    trainer.train()
    # save the model and processor
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)