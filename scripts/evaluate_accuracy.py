import time
from utils.generation import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import whisper
import jiwer

# ---------------------------
# Load models
# ---------------------------
print("Loading VALL-E-X models...")
preload_models()

# Load Whisper for transcription
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")

# ---------------------------
# Test sentences
# ---------------------------
test_sentences = [
    "Hello, this is a test for accuracy evaluation of the VALL E X model.",
    "Today is a beautiful day and we are conducting speech synthesis experiments.",
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming text to speech technologies.",
]

# ---------------------------
# Function to evaluate each sentence
# ---------------------------
def evaluate_sentence(text, index):
    print(f"\nEvaluating sentence {index+1}...")
    
    # Latency measurement
    start = time.time()
    audio = generate_audio(text)
    end = time.time()
    
    generation_time = end - start
    print(f"Generation Time: {generation_time:.2f} seconds")

    # Save the audio file
    file_name = f"eval_{index+1}.wav"
    write_wav(file_name, SAMPLE_RATE, audio)
    print(f"Saved: {file_name}")

    # Transcribe using Whisper
    result = whisper_model.transcribe(file_name)
    predicted_text = result["text"].strip()

    # WER calculation
    wer = jiwer.wer(text.lower(), predicted_text.lower())

    return {
        "original": text,
        "transcribed": predicted_text,
        "wer": wer,
        "latency": generation_time,
        "audio_file": file_name
    }

# ---------------------------
# Run evaluation
# ---------------------------
results = []

for i, sentence in enumerate(test_sentences):
    result = evaluate_sentence(sentence, i)
    results.append(result)

# ---------------------------
# Print summary
# ---------------------------
print("\n\n=== ACCURACY SUMMARY ===\n")

for i, r in enumerate(results):
    print(f"Sentence {i+1}:")
    print(f"  Original    : {r['original']}")
    print(f"  Transcribed : {r['transcribed']}")
    print(f"  WER         : {r['wer']:.3f}")
    print(f"  Latency     : {r['latency']:.2f} sec")
    print(f"  Audio File  : {r['audio_file']}")
    print("")
