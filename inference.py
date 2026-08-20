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
torch.backends.cudnn.deterministic = True

base_path = "microsoft/speecht5_tts"
finetune_path = "models/speecht5_NBTale_tts_shure_1/checkpoint-1000"
finetune_path = "models/speecht5_NBTale_tts_shure_1_august/checkpoint-5000"
#finetune_path = "models/speecht5_NBTale_tts_shure_123/checkpoint-4800"

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
    embeddings_dataset['g01_f1_1']
).unsqueeze(0).to(device)

# custom speaker embedding from your own recording
#speaker_embeddings = create_custom_speaker_embedding("voice_recordings/AnneKat.m4a").unsqueeze(0).to(device)

# Load vocoder for waveform generation
vocoder = SpeechT5HifiGan.from_pretrained(
    "microsoft/speecht5_hifigan").to(device)
vocoder.eval()

supress_noise = True

# Prepare input text
text = "NRK spurte Paulsen per SMS om hva som skjer med varslingssaken like etter klokken 07 torsdag morgen. Hun har ikke besvart henvendelsen. Partileder i moderpartiet Frp, Sylvi Listhaug, er glad for at Løvold setter partiet først. Jeg tar til orientering at Lars Løvold har trukket seg som leder av FpU og hans begrunnelse for det. Jeg er glad for at han setter partiet først, sier hun i en uttalelse til NRK"
text = text.replace("NRK", "N-R-K")
text = text.replace("SMS", "S-M-S")
text = text.replace("Frp", "F-R-P")
text = text.replace("FpU", "F-P-U")

for part in text.split('.'):
    inputs = processor(text=text_normalizer(part, use_pause_tokens=False), return_tensors="pt").to(device)

    # Generate speech
    with torch.no_grad():
        speech = model.generate_speech(
            inputs["input_ids"],
            speaker_embeddings=speaker_embeddings,
            vocoder=vocoder,
        )

    # TODO: How to reduce metallic feel of generated speech?
    if supress_noise:
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
    #torchaudio.save(
    #    "output.wav",
    #    speech.unsqueeze(0).cpu(),
    #    sample_rate=16000
    #)
    #print("Saved output.wav")

    # Play audio

    import sounddevice as sd
    print("Playing audio...")
    sd.play(speech.numpy(), samplerate=16000)
    sd.wait()