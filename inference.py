import torch
import torchaudio
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from dataset import text_normalizer
from audio_postprocessor import AudioPostProcessor
from audio_quality_assessment import AudioQualityAssessment

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

base_path = "microsoft/speecht5_tts"
finetune_path = "models/speecht5_NBTale_tts_shure_1/checkpoint-1000"
#finetune_path = "models/speecht5_NBTale_tts_senheiser_1/checkpoint-1000"

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
    embeddings_dataset['p1_g02_f2_1_t']
).unsqueeze(0).to(device)

# custom speaker embedding from your own recording
#from speaker_to_embedding import create_custom_speaker_embedding
#speaker_embeddings = create_custom_speaker_embedding("sigurd.wav").unsqueeze(0).to(device)

# Prepare input text
text = "Cathrine og Kjell bor i Elvegata i Trondheim."
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

# ============================================
# POST-PROCESSING AND QUALITY ASSESSMENT
# ============================================

# Initialize post-processor
post_processor = AudioPostProcessor(sample_rate=16000)

# Initialize quality assessor
quality_assessor = AudioQualityAssessment(sample_rate=16000)

# Apply post-processing to reduce tinny/buzzing artifacts
print("\nApplying post-processing filters...")
speech_processed = post_processor.process(
    speech,
    remove_rumble=False,        # Remove sub-100Hz rumble
    remove_harshness=False,      # Remove >8kHz harshness (buzzing)
    apply_eq=False,              # Apply EQ for warmth (boost 300Hz, cut 3.5kHz)
    apply_gating=False,         # Optional: spectral gating (can over-process)
    denoise=False,              # Optional: denoising (set to True if background noise)
    normalize=False,             # Normalize loudness
    add_reverb=False            # Optional: subtle reverb (set to True for less digital sound)
)

# Assess quality before and after
print("\nQuality Assessment:")
print("=" * 60)

metrics_before = quality_assessor.assess_quality(speech)
print("BEFORE post-processing:")
print(quality_assessor.quality_report(metrics_before))

print("\n")

metrics_after = quality_assessor.assess_quality(speech_processed)
print("AFTER post-processing:")
print(quality_assessor.quality_report(metrics_after))

print("=" * 60)

# Save audio
torchaudio.save(
    "output.wav",
    speech_processed.unsqueeze(0).cpu(),
    sample_rate=16000
)

print("\nSaved output.wav (post-processed)")
