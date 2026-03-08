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
from speaker_to_embedding import create_speaker_embeddings

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)

output_dir = "models/speecht5_NBTale_tts_shure_1"

checkpoint = "microsoft/speecht5_tts"
data_path = "./data/shure_1"

processor = SpeechT5Processor.from_pretrained(checkpoint)
model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)

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
)

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class TTSDataCollator:
        processor: Any

        def __call__(
                self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
        ) -> Dict[str, torch.Tensor]:
            input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
            label_features = [{"input_values": feature["labels"]} for feature in features]
            speaker_features = [feature["speaker_embeddings"] for feature in features]

            # collate the inputs and targets into a batch
            batch = processor.pad(
                input_ids=input_ids, labels=label_features, return_tensors="pt"
            )

            # replace padding with -100 to ignore loss correctly
            batch["labels"] = batch["labels"].masked_fill(
                batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
            )

            # not used during fine-tuning
            del batch["decoder_attention_mask"]

            # round down target lengths to multiple of reduction factor
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

            # also add in the speaker embeddings
            batch["speaker_embeddings"] = torch.tensor(np.array(speaker_features))

            return batch

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    warmup_steps=200,
    max_steps=1000,
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
    data_collator=TTSDataCollator(processor),
)

if __name__ == "__main__":
    trainer.train()
    # save the model and processor
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)