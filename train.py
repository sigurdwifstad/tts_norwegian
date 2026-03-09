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
from dataset import NBTaleDataset
from speaker_to_embedding import create_speaker_embeddings

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)

output_dir = "models/speecht5_NBTale_tts_shure_1_vibes"

checkpoint = "microsoft/speecht5_tts"
data_path = "./data/shure_1"

processor = SpeechT5Processor.from_pretrained(checkpoint)
model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)
model.to(device)

model.config.use_cache = False
# Only enable gradient checkpointing for CUDA, not for MPS
if device.type == "cuda":
    model.gradient_checkpointing_enable()

if not os.path.exists("speaker_embeddings.pt"):
    speaker_to_embedding = create_speaker_embeddings(data_path)
    torch.save(speaker_to_embedding, "speaker_embeddings.pt")
else:
    speaker_to_embedding = torch.load("speaker_embeddings.pt")

assert next(iter(speaker_to_embedding.values())).shape[-1] == 512

# ===============================
# Dataset
# ===============================
train_dataset = NBTaleDataset(
    data_path=data_path,
    processor=processor,
    speaker_to_embedding=speaker_to_embedding,
    enable_augmentation=True,
    augmentation_probability=0.5,
)

# Split into train/eval (80/20 split)
train_size = int(0.8 * len(train_dataset))
eval_size = len(train_dataset) - train_size
train_dataset, eval_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, eval_size]
)

@dataclass
class TTSDataCollator:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        batch = processor.pad(
            input_ids=input_ids, labels=label_features, return_tensors="pt"
        )

        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )

        del batch["decoder_attention_mask"]

        if model.config.reduction_factor > 1:
            target_lengths = torch.tensor(
                [len(feature["labels"]) for feature in features]
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
    learning_rate=5e-5,
    warmup_steps=500,
    max_steps=2000,
    max_grad_norm=1.0,
    fp16=torch.cuda.is_available(),
    logging_steps=25,
    save_steps=200,
    eval_steps=200,
    report_to=["tensorboard"],
    remove_unused_columns=False,
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=TTSDataCollator(processor),
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)