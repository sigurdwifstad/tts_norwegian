import torch
import torchaudio
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from dataset import text_normalizer, PAUSE_TOKEN_SET
from speaker_to_embedding import create_custom_speaker_embedding

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

base_path = "microsoft/speecht5_tts"
finetune_path = "models/speecht5_NBTale_tts_shure_1/checkpoint-1000"
#finetune_path = "models/speecht5_NBTale_tts_shure_123/checkpoint-1400"

# Load processor from the finetuned model dir so the extended tokenizer
# (with pause tokens) is used.  Fall back to base if not saved there yet.
import os
if os.path.exists(os.path.join(finetune_path, "tokenizer_config.json")):
    processor = SpeechT5Processor.from_pretrained(finetune_path)
else:
    processor = SpeechT5Processor.from_pretrained(base_path)
    # Ensure pause tokens are registered even when using base processor
    processor.tokenizer.add_tokens(sorted(PAUSE_TOKEN_SET))

# Load fine-tuned TTS model
model = SpeechT5ForTextToSpeech.from_pretrained(
    finetune_path,
).to(device)

# Resize embeddings to match the (possibly extended) tokenizer
model.resize_token_embeddings(len(processor.tokenizer))

model.eval()

# Use a sample speaker or your own speaker embedding
embeddings_dataset = torch.load("speaker_embeddings.pt")
print(embeddings_dataset.keys())

speaker_embeddings = torch.tensor(
    embeddings_dataset['p1_g04_m2_4_t']
).unsqueeze(0).to(device)

# custom speaker embedding from your own recording
speaker_embeddings = create_custom_speaker_embedding("voice_recordings/ingvild.wav").unsqueeze(0).to(device)

# Prepare input text
text = "Jeg er en syntetisk stemme fra Finnmark."
inputs = processor(text=text_normalizer(text), return_tensors="pt").to(device)

# Load vocoder for waveform generation
vocoder = SpeechT5HifiGan.from_pretrained(
    "microsoft/speecht5_hifigan").to(device)
vocoder.eval()

# Generate speech
with torch.no_grad():
    speech = model.generate_speech(
        inputs["input_ids"],
        speaker_embeddings=speaker_embeddings,
        vocoder=vocoder
    )

# TODO: How to reduce metallic feel of generated speech?
from noisereduce import reduce_noise
# Reduce noise
speech = reduce_noise(
    y=speech.cpu().numpy(),
    sr=16000,
    prop_decrease=1.0,
    stationary=False
)
speech = torch.tensor(speech).to(device)

# EQ
#torchaudio.functional.equalizer_biquad(
#    speech,
#    sample_rate=16000,
#    center_freq=3000,
#    gain=5.0,
#    Q=1.0
#)

# Save audio
torchaudio.save(
    "output.wav",
    speech.unsqueeze(0).cpu(),
    sample_rate=16000
)

print("Saved output.wav")