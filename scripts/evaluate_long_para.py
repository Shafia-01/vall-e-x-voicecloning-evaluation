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

print("Loading Whisper model...")
whisper_model = whisper.load_model("base")

# ---------------------------
# Long paragraph to test
# ---------------------------
long_para = """
Artificial intelligence has rapidly transformed the field of speech synthesis. 
Modern text-to-speech systems like VALL-E-X aim to produce natural, expressive, and 
human-like audio by leveraging neural codec language models. These systems capture 
the intricacies of human speech, such as rhythm, tone, and accent. As AI continues 
to advance, the future of speech generation looks promising, with applications in 
assistive technology, communication tools, voice cloning, and immersive user experiences.
"""

# ---------------------------
# Evaluation function
# ---------------------------
def evaluate_long_text(text):
    print("\nGenerating long paragraph audio...")

    start = time.time()
    audio = generate_audio(text)
    end = time.time()

    gen_time = end - start
    print(f"Generation Time: {gen_time:.2f} seconds")

    # Save file
    filename = "long_para.wav"
    write_wav(filename, SAMPLE_RATE, audio)
    print(f"Saved: {filename}")

    # Transcription
    result = whisper_model.transcribe(filename)
    predicted = result['text']

    # Accuracy (WER)
    wer = jiwer.wer(text.lower(), predicted.lower())

    return text, predicted, wer, gen_time

# ---------------------------
# Run evaluation
# ---------------------------
original, transcript, wer, latency = evaluate_long_text(long_para)

# ---------------------------
# Print results
# ---------------------------
print("\n=== LONG PARAGRAPH ACCURACY RESULTS ===\n")
print("Original Paragraph:\n", original)
print("\nTranscribed Paragraph:\n", transcript)
print("\nWER:", wer)
print("Latency:", latency, "seconds")
