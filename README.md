# VALL-E-X Voice Cloning Evaluation

This repository contains my evaluation work on the **VALL-E-X** text-to-speech and zero-shot voice cloning model.  
The goal was to test:

- ✔ General TTS accuracy  
- ✔ Long-paragraph stability  
- ✔ Zero-shot voice cloning accuracy  
- ✔ Latency & speed behaviour  
- ✔ Transcript correctness using Whisper  
- ✔ Speaker-similarity scoring using Wav2Vec2

The repo includes the **evaluation scripts**, **audio prompts**, and **results** generated during testing—without containing model weights or heavy files.

### 🔗 Base Model Used
This evaluation is based on the official VALL-E-X model:
https://github.com/Plachtaa/VALL-E-X


## 📌 Repository Structure
```
VALL-E-X-voicecloning-evaluation/
│
├── prompts/
│ ├── voice_reference.wav
│ └── shafia_clone_fixed/
│ └── audio.wav
│
├── results/
│ ├── eval_1.wav … eval_4.wav
│ ├── long_para.wav
│ ├── clone_eval_1.wav … clone_eval_5.wav
│ └── clone_test_fixed.wav
│
├── scripts/
│ ├── evaluate_accuracy.py
│ ├── evaluate_long_para.py
│ └── evaluate_voice_cloning.py
│
├── README.md
└── .gitignore
```

## 🚀 What Was Evaluated

### **1. TTS Accuracy**
- Whisper ASR used to transcribe generated audio  
- WER (Word Error Rate) measured  
- Latency measured per sample  
- Performance:  
  - **3/4 sentences had perfect transcription**
  - Avg WER: **0.053**
  - Avg Latency (CPU): **18–22 sec**

### **2. Long-Paragraph Stability**
- Generated a long paragraph to check consistency & prosody  
- VALL-E-X handled long sequences with stable tone  
- Slight drift observed after ~20 seconds (expected for CPU inference)

### **3. Voice Cloning Accuracy**
Used:
- **Facebook Wav2Vec2-base-960h** for speaker embeddings  
- Cosine similarity + WER + Whisper transcription

**Final Results:**

| Sentence | Similarity | WER | Notes |
|---------|------------|-----|-------|
| 1 | 0.936 | 0.875 | Text mismatch but voice preserved |
| 2 | 0.918 | 0.167 | Mostly accurate |
| 3 | 0.955 | 0.556 | Minor articulation drift |
| 4 | 0.941 | 0.429 | Good prosody |
| 5 | 0.965 | 0.200 | Best overall |

### **4. Overall Model Behavior**
- Clone sounds very close to the reference speaker  
- Accuracy depends heavily on:  
  - quality of prompt  
  - duration (≥ 5 seconds recommended)  
  - noise  
  - accent  
  - CPU vs GPU  


## 🧪 How to Run Scripts
### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Run TTS accuracy
```
python scripts/evaluate_accuracy.py
```

### 3. Run long-paragraph evaluation
```
python scripts/evaluate_long_para.py
```

### 4. Run voice-cloning accuracy
```
python scripts/evaluate_voice_cloning.py
```

## 📌 Notes on VALL-E-X Variants
Users have discussed “small / medium / large” VALL-E-X, but **the public GitHub release does not provide multiple model sizes**.  
Only one trained checkpoint is available.

“Variants” mentioned online refer to:
- Different *training configurations*
- Custom *fine-tuned models* from researchers
- Not official model sizes