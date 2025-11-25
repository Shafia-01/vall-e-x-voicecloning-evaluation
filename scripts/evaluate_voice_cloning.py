import time
import torch
import torchaudio
from scipy.io.wavfile import write as write_wav
from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import whisper
import jiwer

# ------------------------------------------------------
# Load models
# ------------------------------------------------------
print("Loading VALL-E-X Models...")
preload_models()

print("Loading Whisper ASR...")
whisper_model = whisper.load_model("base")

print("Loading Speaker Embedding Model...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
speaker_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# ------------------------------------------------------
# Speaker embedding function
# ------------------------------------------------------

def get_embedding(path):
    # Load using soundfile (never torchaudio)
    import soundfile as sf
    import torch

    wav, sr = sf.read(path)

    # Convert numpy to tensor
    wav = torch.tensor(wav, dtype=torch.float32)

    # If stereo -> convert to mono
    if wav.dim() == 2:
        wav = wav.mean(dim=1)

    # Ensure shape [1, samples]
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    # Resample manually to 16k
    wav = torchaudio.functional.resample(wav, sr, 16000)

    # Run through Wav2Vec2
    inputs = processor(wav.squeeze(), sampling_rate=16000, return_tensors="pt", padding=False)
    input_values = inputs["input_values"]
    if input_values.dim() == 3:
        input_values = input_values.squeeze(1)
    with torch.no_grad():
        outputs = speaker_model(input_values)

    # Return embedding
    return outputs.last_hidden_state.mean(dim=1).squeeze()

reference_path = "prompts/shafia_clone_fixed/audio.wav"

ref_emb = get_embedding(reference_path)

# ------------------------------------------------------
# Test sentences (cloned)
# ------------------------------------------------------
test_sentences = [
    "Hello, this is a voice cloning accuracy test.",
    "Artificial intelligence enables realistic voice synthesis.",
    "The quick brown fox jumps over the lazy dog.",
    "CyArt Tech is evaluating cloned speech quality.",
    "Longer sentences help us measure prosody and consistency over time.",
]

results = []

# ------------------------------------------------------
# Evaluation for each sentence
# ------------------------------------------------------
for i, text in enumerate(test_sentences):
    print(f"\n=== Generating Clone {i+1} ===")

    start = time.time()
    audio = generate_audio(text, prompt="shafia_clone_fixed")
    end = time.time()

    latency = end - start
    file_name = f"clone_eval_{i+1}.wav"
    write_wav(file_name, SAMPLE_RATE, audio)

    print(f"Saved: {file_name}")
    print(f"Latency: {latency:.2f} sec")

    # Compute speaker similarity
    clone_emb = get_embedding(file_name)
    similarity = torch.nn.functional.cosine_similarity(
        ref_emb.unsqueeze(0), clone_emb.unsqueeze(0)
    ).item()

    # Transcription & WER
    asr = whisper_model.transcribe(file_name)["text"]
    wer = jiwer.wer(text.lower(), asr.lower())

    results.append({
        "original": text,
        "transcribed": asr,
        "similarity": similarity,
        "latency": latency,
        "wer": wer,
        "file": file_name
    })

# ------------------------------------------------------
# Print final results
# ------------------------------------------------------
print("\n\n=== FINAL VOICE CLONING ACCURACY RESULTS ===\n")
for i, r in enumerate(results):
    print(f"Sentence {i+1}:")
    print(f"  Text         : {r['original']}")
    print(f"  Transcribed  : {r['transcribed']}")
    print(f"  Similarity   : {r['similarity']:.4f}")
    print(f"  WER          : {r['wer']:.3f}")
    print(f"  Latency      : {r['latency']:.2f} sec")
    print(f"  Audio File   : {r['file']}")
    print("")
