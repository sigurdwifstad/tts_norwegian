import torch
import torchaudio
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

device = "cuda" if torch.cuda.is_available() else "cpu"

base_path = "microsoft/speecht5_tts"
finetune_path = "./models/speecht5_NBTale_tts_3/checkpoint-500"

#  Load processor from base model
processor = SpeechT5Processor.from_pretrained(base_path)

# ⃣ Load fine-tuned TTS model
model = SpeechT5ForTextToSpeech.from_pretrained(
    finetune_path,
).to(device)

model.eval()

# Use a sample speaker or your own speaker embedding
embeddings_dataset = torch.load("speaker_embeddings.pt")

speaker_embeddings = torch.tensor(
    embeddings_dataset['p1_g02_f1_1_t']
).unsqueeze(0).to(device)

# Prepare input text
text = "Ibsens ripsbaerbusker og andre buskevekster."
inputs = processor(text=text, return_tensors="pt").to(device)

# Load vocoder for waveform generation
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Generate speech
with torch.no_grad():
    speech = model.generate_speech(
        inputs["input_ids"],
        speaker_embeddings=speaker_embeddings,
        vocoder=vocoder
    )


# Save audio
torchaudio.save(
    "output.wav",
    speech.unsqueeze(0).cpu(),
    sample_rate=16000
)

print("Saved output.wav")