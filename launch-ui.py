# import argparse
# import logging
# import os
# import pathlib
# import time
# import tempfile
# import platform
# import webbrowser
# import sys
# print(f"default encoding is {sys.getdefaultencoding()},file system encoding is {sys.getfilesystemencoding()}")
# if(sys.version_info[0]<3 or sys.version_info[1]<7):
#     print("The Python version is too low and may cause problems")

# if platform.system().lower() == 'windows':
#     temp = pathlib.PosixPath
#     pathlib.PosixPath = pathlib.WindowsPath
# else:
#     temp = pathlib.WindowsPath
#     pathlib.WindowsPath = pathlib.PosixPath
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# import langid
# # langid.set_languages(['en', 'zh', 'ja'])
# langid.set_languages(['en'])
# import nltk
# nltk.data.path = nltk.data.path + [os.path.join(os.getcwd(), "nltk_data")]

# import torch
# import torchaudio
# import random
# import soundfile as sf

# import numpy as np

# from data.tokenizer import (
#     AudioTokenizer,
#     tokenize_audio,
# )
# from data.collation import get_text_token_collater
# from models.vallex import VALLE
# from utils.g2p import PhonemeBpeTokenizer
# from descriptions import *
# from macros import *
# from examples import *

# import gradio as gr
# import whisper
# from vocos import Vocos
# import multiprocessing

# # ------------------------------
# # Simple KMeans Speaker Diarization
# # ------------------------------
# import librosa
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.metrics.pairwise import cosine_distances

# import time
# import os
# import numpy as np

# # Lazy imports for embedding & ASR evaluation
# try:
#     import soundfile as sf
# except Exception:
#     sf = None
# try:
#     import torchaudio
# except Exception:
#     torchaudio = None

# # transformers Wav2Vec2 for speaker embeddings (lazy loaded)
# _w2v_processor = None
# _w2v_model = None
# def _ensure_w2v_loaded():
#     global _w2v_processor, _w2v_model
#     if _w2v_processor is None or _w2v_model is None:
#         try:
#             from transformers import Wav2Vec2Processor, Wav2Vec2Model
#             _w2v_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
#             _w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
#             _w2v_model.eval()
#             # turn off gradients
#             for p in _w2v_model.parameters():
#                 p.requires_grad = False
#         except Exception as e:
#             _w2v_processor = None
#             _w2v_model = None
#             print(f"[Eval Helper] Failed to load Wav2Vec2 model: {e}")

# def get_embedding(wav_path):
#     """
#     Return a 1D torch tensor embedding for wav_path (mean pool of last_hidden_state).
#     Resamples to 16k if necessary.
#     """
#     import torch
#     global _w2v_processor, _w2v_model
#     if sf is None:
#         raise RuntimeError("soundfile (pysoundfile) is required for get_embedding")
#     _ensure_w2v_loaded()
#     if _w2v_processor is None or _w2v_model is None:
#         raise RuntimeError("Wav2Vec2 processor/model not loaded")
#     wav, sr = sf.read(wav_path)
#     # Convert to mono if stereo
#     if wav.ndim == 2:
#         wav = wav.mean(axis=1)
#     wav = wav.astype("float32")
#     # Resample to 16k if needed
#     if sr != 16000:
#         if torchaudio is None:
#             raise RuntimeError("torchaudio required for resampling to 16k")
#         wav_t = torch.from_numpy(wav).unsqueeze(0)  # (1, samples)
#         wav_t = torchaudio.functional.resample(wav_t, sr, 16000)
#         wav = wav_t.squeeze(0).numpy()
#         sr = 16000
#     inputs = _w2v_processor(wav, sampling_rate=sr, return_tensors="pt", padding=False)
#     input_values = inputs["input_values"]
#     if input_values.dim() == 3:
#         input_values = input_values.squeeze(1)
#     with torch.no_grad():
#         outputs = _w2v_model(input_values)
#     emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)  # (hidden,)
#     return emb

# # Evaluation helper: writes audio, runs Whisper ASR, WER, similarity, prints to terminal
# def evaluate_and_print_metrics(audio_numpy, sample_rate, original_text="", reference_prompt_wav_path=None, save_prefix="eval"):
#     """
#     audio_numpy: 1D numpy array (float32) of generated audio samples
#     sample_rate: int (e.g., 24000)
#     original_text: str - ground truth text that was synthesized (for WER)
#     reference_prompt_wav_path: path to reference enrollment wav (for similarity). If None, similarity skipped.
#     save_prefix: filename prefix for saved generated file
#     """
#     import torch
#     import torch.nn.functional as F
#     import jiwer
#     import whisper
#     from scipy.io.wavfile import write as write_wav
#     # Normalize audio to -1..1 float32
#     audio = np.asarray(audio_numpy)
#     if audio.dtype not in (np.float32, np.float64):
#         # Convert integer -> float
#         peak = np.max(np.abs(audio)) if audio.size else 1.0
#         if peak > 0:
#             audio = audio.astype("float32") / float(peak)
#         else:
#             audio = audio.astype("float32")
#     else:
#         audio = audio.astype("float32")
#     peak = np.max(np.abs(audio)) if audio.size else 1.0
#     if peak > 1.0:
#         audio = audio / peak

#     timestamp = int(time.time())
#     out_filename = f"{save_prefix}_{timestamp}.wav"
#     try:
#         write_wav(out_filename, sample_rate, audio)
#     except Exception:
#         # fallback to soundfile if scipy write fails
#         if sf is None:
#             print("[Eval] Could not save audio: scipy write failed and soundfile not present.")
#         else:
#             sf.write(out_filename, audio, sample_rate)
#     print(f"[Eval] Saved generated audio -> {out_filename}")

#     # Whisper transcription (use existing global whisper_model if present)
#     try:
#         try:
#             whisper_model  # check if exists in globals
#             model_for_asr = whisper_model
#         except NameError:
#             model_for_asr = whisper.load_model("base")
#         tstart = time.time()
#         res = model_for_asr.transcribe(out_filename)
#         transcribed = res.get("text", "").strip()
#         tasr = time.time() - tstart
#     except Exception as e:
#         transcribed = "[ASR_ERROR]"
#         tasr = 0.0
#         print(f"[Eval] Whisper transcription failed: {e}")

#     # Compute WER
#     wer_val = None
#     if original_text and original_text.strip() != "":
#         try:
#             wer_val = jiwer.wer(original_text.lower(), transcribed.lower())
#         except Exception as e:
#             print(f"[Eval] WER computation error: {e}")
#             wer_val = None

#     # Similarity (if requested and available)
#     similarity = None
#     if reference_prompt_wav_path:
#         try:
#             emb_gen = get_embedding(out_filename)  # embedding for generated audio
#             emb_ref = get_embedding(reference_prompt_wav_path)  # embedding for reference prompt
#             # Convert to numpy arrays
#             emb_gen_np = emb_gen.detach().cpu().numpy().reshape(1, -1)
#             emb_ref_np = emb_ref.detach().cpu().numpy().reshape(1, -1)
#             # Calculate cosine similarity
#             similarity = float(cosine_similarity(emb_gen_np, emb_ref_np)[0][0])
#         except Exception as e:
#             print(f"[Eval] Speaker similarity calculation failed: {e}")
#             similarity = None

#     # Printable summary
#     print("\n=== ACCURACY METRICS ===")
#     if original_text:
#         print(f"Original Text     : {original_text}")
#     print(f"Transcribed Text  : {transcribed}")
#     if wer_val is not None:
#         print(f"WER               : {wer_val:.4f}")
#     else:
#         print("WER               : N/A")
#     print(f"ASR Time (sec)    : {tasr:.3f}")
#     if similarity is not None:
#         print(f"Speaker Similarity: {similarity:.4f} (cosine)")
#         if similarity >= 0.80:
#             remark = "Excellent cloning"
#         elif similarity >= 0.60:
#             remark = "Good cloning"
#         elif similarity >= 0.40:
#             remark = "Average cloning"
#         else:
#             remark = "Poor cloning"
#         print(f"Cloning Quality   : {remark}")
#     else:
#         print("Speaker Similarity: N/A")
#     print(f"Saved audio file  : {out_filename}")
#     print("=========================\n")

# def detect_speakers(audio_path, threshold=0.25):
#     """
#     Detects number of speakers in audio using KMeans clustering.
    
#     Returns:
#         tuple: (num_speakers: int, message: str)
#     """
#     try:
#         # Load audio
#         y, sr = librosa.load(audio_path, sr=16000)
#         duration = len(y) / sr
        
#         # If audio is very short, assume single speaker
#         if duration < 1.0:
#             return 1, "✓ Single speaker (audio too short for analysis)"
        
#         # Voice Activity Detection using energy
#         energy = librosa.feature.rms(y=y)[0]
#         frames = librosa.util.frame(y, frame_length=2048, hop_length=512)
        
#         voiced_frames = []
#         for i in range(min(frames.shape[1], len(energy))):
#             if energy[i] > np.mean(energy) * 0.8:
#                 voiced_frames.append(frames[:, i])
        
#         if len(voiced_frames) < 5:
#             return 1, "✓ Single speaker (insufficient voice activity)"
        
#         # Extract MFCC embeddings
#         embeddings = []
#         for frame in voiced_frames:
#             mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=20)
#             emb = np.mean(mfcc, axis=1)
#             embeddings.append(emb)
        
#         if len(embeddings) < 10:
#             return 1, "✓ Single speaker (insufficient features)"
        
#         embeddings = np.array(embeddings)
        
#         # KMeans clustering with 2 clusters
#         kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(embeddings)
#         labels = kmeans.labels_
        
#         # Calculate cluster centroids
#         cluster_0 = embeddings[labels == 0]
#         cluster_1 = embeddings[labels == 1]
        
#         if len(cluster_0) == 0 or len(cluster_1) == 0:
#             return 1, "✓ Single speaker detected"
        
#         centroid_0 = cluster_0.mean(axis=0)
#         centroid_1 = cluster_1.mean(axis=0)
        
#         # Calculate distance between centroids
#         distance = cosine_distances([centroid_0], [centroid_1])[0][0]
        
#         # Calculate cluster sizes
#         cluster_0_pct = len(cluster_0) / len(embeddings) * 100
#         cluster_1_pct = len(cluster_1) / len(embeddings) * 100
        
#         print(f"Cluster analysis: {cluster_0_pct:.1f}% vs {cluster_1_pct:.1f}%, distance={distance:.3f}")
        
#         # Decision logic
#         if distance < threshold:
#             return 1, f"✓ Single speaker detected (distance={distance:.3f})"
#         else:
#             # Check if one cluster is too small (might be noise)
#             min_cluster_size = 15  # minimum 15% to be considered a separate speaker
#             if cluster_0_pct < min_cluster_size or cluster_1_pct < min_cluster_size:
#                 return 1, f"✓ Single speaker (minor variation detected, distance={distance:.3f})"
            
#             return 2, f"⚠ Multiple speakers detected ({cluster_0_pct:.0f}%/{cluster_1_pct:.0f}% split, distance={distance:.3f})"
    
#     except Exception as e:
#         print(f"Speaker detection error: {e}")
#         return 1, "✓ Single speaker (analysis failed, assuming single)"


# def check_audio_for_generation(audio_path):
#     """
#     Checks if audio is suitable for voice cloning.
#     Returns: (is_valid: bool, message: str)
#     """
#     num_speakers, detail_msg = detect_speakers(audio_path)
    
#     if num_speakers > 1:
#         error_msg = (
#             "❌ Multiple speakers detected in the audio!\n\n"
#             "This voice cloning system requires audio with only ONE speaker.\n"
#             f"Analysis: {detail_msg}\n\n"
#             "Please provide a different audio sample with only one person speaking."
#         )
#         return False, error_msg
#     else:
#         success_msg = f"✓ Audio validated for voice cloning\n{detail_msg}"
#         return True, success_msg

# # Try to import SpeechBrain for speaker diarization
# # try:
# #     # from speechbrain.pretrained import SpeakerRecognition
# #     SPEECHBRAIN_AVAILABLE = True
# #     print("SpeechBrain loaded successfully for speaker diarization")
# # except Exception as e:
# #     print(f"Warning: SpeechBrain not available ({e}). Speaker detection will be skipped.")
# #     SPEECHBRAIN_AVAILABLE = False

# thread_count = multiprocessing.cpu_count()

# print("Use",thread_count,"cpu cores for computing")

# torch.set_num_threads(thread_count)
# torch.set_num_interop_threads(thread_count)
# torch._C._jit_set_profiling_executor(False)
# torch._C._jit_set_profiling_mode(False)
# torch._C._set_graph_executor_optimize(False)

# text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_69.json")
# text_collater = get_text_token_collater()

# device = torch.device("cpu")
# if torch.cuda.is_available():
#     device = torch.device("cuda", 0)
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# # VALL-E-X model
# if not os.path.exists("./checkpoints/"): os.mkdir("./checkpoints/")
# CHECKPOINT_PATH = "checkpoints/vallex-checkpoint.pt"

# # Ensure folder exists
# os.makedirs("checkpoints", exist_ok=True)

# if not os.path.isfile(CHECKPOINT_PATH):
#     import wget
#     try:
#         print("Model checkpoint not found. Downloading it now...")
#         logging.info("Downloading VALLE-X model (first time only)...")
#         wget.download(
#             "https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt",
#             out=CHECKPOINT_PATH,
#             bar=wget.bar_adaptive
#         )
#         print("\nDownload complete!")
#     except Exception as e:
#         logging.info(e)
#         raise Exception(
#             "\nModel weights download failed.\n"
#             "Please manually download from https://huggingface.co/Plachta/VALL-E-X\n"
#             f"and put vallex-checkpoint.pt inside: {os.getcwd()}/checkpoints/"
#         )
# else:
#     print("✔ Using existing model checkpoint — skipping download.")


# model = VALLE(
#         N_DIM,
#         NUM_HEAD,
#         NUM_LAYERS,
#         norm_first=True,
#         add_prenet=False,
#         prefix_mode=PREFIX_MODE,
#         share_embedding=True,
#         nar_scale_factor=1.0,
#         prepend_bos=True,
#         num_quantizers=NUM_QUANTIZERS,
#     )
# checkpoint = torch.load("./checkpoints/vallex-checkpoint.pt", map_location='cpu', weights_only=False)
# missing_keys, unexpected_keys = model.load_state_dict(
#     checkpoint["model"], strict=True
# )
# assert not missing_keys
# model.eval()

# # Encodec model
# audio_tokenizer = AudioTokenizer(device)

# # Vocos decoder
# vocos = Vocos.from_pretrained('charactr/vocos-encodec-24khz').to(device)

# # ASR
# if not os.path.exists("./whisper/"): os.mkdir("./whisper/")
# try:
#     whisper_model = whisper.load_model("medium",download_root=os.path.join(os.getcwd(), "whisper")).cpu()
# except Exception as e:
#     logging.info(e)
#     raise Exception(
#         "\n Whisper download failed or damaged, please go to "
#         "'https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt'"
#         "\n manually download model and put it to {} .".format(os.getcwd() + "\whisper"))

# # Voice Presets
# preset_list = os.walk("./presets/").__next__()[2]
# preset_list = [preset[:-4] for preset in preset_list if preset.endswith(".npz")]

# # Speaker diarization model
# speaker_diarization_model = None

# def load_speaker_diarization():
#     # global speaker_diarization_model
#     # if not SPEECHBRAIN_AVAILABLE:
#     #     print("SpeechBrain not available, skipping speaker diarization")
#     #     return
#     # try:
#     #     print("Loading SpeechBrain speaker diarization model...")
#     #     speaker_diarization_model = SpeakerRecognition.from_hparams(
#     #         source="speechbrain/spkrec-ecapa-voxceleb",
#     #         savedir="pretrained_models/spkrec"
#     #     )
#     #     print("Speaker diarization model loaded successfully")
#     # except Exception as e:
#     #     print(f"Warning: Could not load speaker diarization model: {e}")
#         print("Speaker detection will be skipped")

# def check_single_speaker(audio_path):
#     """
#     Check if audio contains only one speaker using SpeechBrain.
#     Returns: (is_single_speaker: bool, num_speakers: int, error_message: str)
#     """
#     # if not SPEECHBRAIN_AVAILABLE or speaker_diarization_model is None:
#     #     return True, 1, ""  # Skip check if model not loaded
    
#     try:
#         # Simple check: if audio is very short, assume single speaker
#         import librosa
#         y, sr = librosa.load(audio_path, sr=16000)
#         duration = len(y) / sr
        
#         if duration < 1.0:
#             return True, 1, ""
        
#         # For simplicity, we assume single speaker if model is loaded
#         # You can implement more sophisticated multi-speaker detection here
#         # This is a placeholder - actual implementation would require
#         # speaker segmentation and clustering
        
#         return True, 1, ""
        
#     except Exception as e:
#         print(f"Warning: Speaker diarization check failed: {e}")
#         return True, 1, ""  # If check fails, allow it through

# def clear_prompts():
#     try:
#         path = tempfile.gettempdir()
#         for eachfile in os.listdir(path):
#             filename = os.path.join(path, eachfile)
#             if os.path.isfile(filename) and filename.endswith(".npz"):
#                 lastmodifytime = os.stat(filename).st_mtime
#                 endfiletime = time.time() - 60
#                 if endfiletime > lastmodifytime:
#                     os.remove(filename)
#     except:
#         return

# def transcribe_one(model, audio_path):
#     # load audio and pad/trim it to fit 30 seconds
#     audio = whisper.load_audio(audio_path)
#     audio = whisper.pad_or_trim(audio)

#     # make log-Mel spectrogram and move to the same device as the model
#     mel = whisper.log_mel_spectrogram(audio).to(model.device)

#     # detect the spoken language
#     _, probs = model.detect_language(mel)
#     print(f"Detected language: {max(probs, key=probs.get)}")
#     lang = max(probs, key=probs.get)
#     # decode the audio
#     options = whisper.DecodingOptions(temperature=1.0, best_of=5, fp16=False if device == torch.device("cpu") else True, sample_len=150)
#     result = whisper.decode(model, mel, options)

#     # print the recognized text
#     print(result.text)

#     text_pr = result.text
#     if text_pr.strip(" ")[-1] not in "?!.,。，？！。、":
#         text_pr += "."
#     return lang, text_pr

# def make_npz_prompt(name, uploaded_audio, recorded_audio, transcript_content):
#     global model, text_collater, text_tokenizer, audio_tokenizer
#     clear_prompts()
#     audio_prompt = uploaded_audio if uploaded_audio is not None else recorded_audio
    
#     if audio_prompt is None:
#         return "Error: No audio provided", None
    
#     # Handle different Gradio audio formats
#     if isinstance(audio_prompt, str):
#         # New format: file path string
#         wav_pr, sr = torchaudio.load(audio_prompt)
#     elif isinstance(audio_prompt, dict):
#         # Dictionary format with 'name' or 'path' key
#         audio_path = audio_prompt.get('name') or audio_prompt.get('path')
#         wav_pr, sr = torchaudio.load(audio_path)
#     elif isinstance(audio_prompt, tuple) and len(audio_prompt) == 2:
#         # Old format: (sample_rate, waveform) tuple
#         sr, wav_pr = audio_prompt
#         if not isinstance(wav_pr, torch.Tensor):
#             wav_pr = torch.FloatTensor(wav_pr)
#     else:
#         return "Error: Unsupported audio format", None
    
#     # Ensure wav_pr is a tensor
#     if not isinstance(wav_pr, torch.Tensor):
#         wav_pr = torch.FloatTensor(wav_pr)
    
#     # Normalize
#     if wav_pr.abs().max() > 1:
#         wav_pr /= wav_pr.abs().max()
    
#     # Handle stereo to mono
#     if wav_pr.ndim > 1 and wav_pr.size(0) == 2:
#         wav_pr = wav_pr.mean(dim=0, keepdim=True)
#     elif wav_pr.ndim > 1 and wav_pr.size(-1) == 2:
#         wav_pr = wav_pr.mean(dim=-1, keepdim=True).squeeze(-1)
    
#     # Ensure correct shape
#     if wav_pr.ndim == 1:
#         wav_pr = wav_pr.unsqueeze(0)
    
#     assert wav_pr.ndim == 2 and wav_pr.size(0) == 1, f"Expected shape (1, length), got {wav_pr.shape}"
    
#     if transcript_content == "":
#         text_pr, lang_pr = make_prompt(name, wav_pr, sr, save=False)
#     else:
#         lang_pr = langid.classify(str(transcript_content))[0]
#         if lang_pr != "en":
#             return f"Error: Only English text is supported. Detected language: {lang_pr}", None
#         lang_token = lang2token[lang_pr]
#         text_pr = f"{lang_token}{str(transcript_content)}{lang_token}"
    
#     # tokenize audio
#     encoded_frames = tokenize_audio(audio_tokenizer, (wav_pr, sr))
#     audio_tokens = encoded_frames[0][0].transpose(2, 1).cpu().numpy()

#     # tokenize text
#     phonemes, _ = text_tokenizer.tokenize(text=f"{text_pr}".strip())
#     text_tokens, enroll_x_lens = text_collater([phonemes])

#     message = f"Detected language: {lang_pr}\n Detected text {text_pr}\n"

#     # save as npz file
#     np.savez(os.path.join(tempfile.gettempdir(), f"{name}.npz"),
#              audio_tokens=audio_tokens, text_tokens=text_tokens, lang_code=lang2code[lang_pr])
#     return message, os.path.join(tempfile.gettempdir(), f"{name}.npz")


# # At the top (around line 32), add:
# # import soundfile as sf

# # Then update make_prompt function (around line 299-340):
# def make_prompt(name, wav, sr, save=True):
#     global whisper_model
#     whisper_model.to(device)
    
#     if not isinstance(wav, torch.FloatTensor):
#         wav = torch.tensor(wav)
    
#     # Normalize to [-1, 1]
#     if wav.abs().max() > 1:
#         wav = wav / wav.abs().max()
    
#     # Convert stereo to mono
#     if wav.size(-1) == 2:
#         wav = wav.mean(-1, keepdim=False)
    
#     # Ensure shape is (1, N)
#     if wav.ndim == 1:
#         wav = wav.unsqueeze(0)
    
#     assert wav.ndim == 2 and wav.size(0) == 1, f"Expected shape (1, N), got {wav.shape}"
    
#     # Convert to numpy and ensure it's in the correct range
#     data = wav.squeeze(0).cpu().numpy()
    
#     # Clip to ensure values are strictly in [-1.0, 1.0]
#     data = np.clip(data, -1.0, 1.0)
    
#     # Convert to float32 (soundfile expects this)
#     data = data.astype(np.float32)
    
#     # Save with soundfile - explicitly set subtype to avoid casting issues
#     # Convert float32 audio to int16 before saving as PCM_16
#     data = np.clip(data, -1.0, 1.0)  # Ensure values are in [-1, 1]
#     data_int16 = (data * 32767).astype(np.int16)
#     sf.write(f"./prompts/{name}.wav", data_int16, sr, subtype='PCM_16')
    
#     # Transcribe
#     lang, text = transcribe_one(whisper_model, f"./prompts/{name}.wav")
    
#     if lang != "en":
#         raise ValueError(f"Error: Only English audio is supported. Detected language: {lang}")
    
#     lang_token = lang2token[lang]
#     text = lang_token + text + lang_token
    
#     with open(f"./prompts/{name}.txt", 'w', encoding='utf-8') as f:
#         f.write(text)
    
#     if not save:
#         os.remove(f"./prompts/{name}.wav")
#         os.remove(f"./prompts/{name}.txt")

#     whisper_model.cpu()
#     torch.cuda.empty_cache()
    
#     return text, lang



# import torch
# @torch.no_grad()
# def infer_from_audio(text, language, accent, audio_prompt, record_audio_prompt, transcript_content):
#     """
#     Full replacement function for your original infer_from_audio.
#     Returns (message, (sample_rate, numpy_audio))
#     Prints evaluation results to terminal (WER, ASR text, similarity if enrollment prompt provided).
#     """
#     global model, text_collater, text_tokenizer, audio_tokenizer, vocos, device
#     try:
#         audio_prompt = audio_prompt if audio_prompt is not None else record_audio_prompt

#         if audio_prompt is None:
#             return "Error: No audio provided", None

#         # Track a path if we can for similarity
#         temp_wav_path = None

#         # Normalize and load provided audio prompt into wav_pr, sr
#         if isinstance(audio_prompt, str):
#             # filepath
#             wav_pr, sr = torchaudio.load(audio_prompt)
#             temp_wav_path = audio_prompt
#         elif isinstance(audio_prompt, dict):
#             # gradio dict format: may have 'name' or 'path'
#             audio_path = audio_prompt.get('name') or audio_prompt.get('path')
#             if audio_path and os.path.isfile(audio_path):
#                 wav_pr, sr = torchaudio.load(audio_path)
#                 temp_wav_path = audio_path
#             else:
#                 # maybe it's a base64 style or temporary data; try 'array' or fallback
#                 # If dict contains 'array' tuple like (sr, np_array)
#                 if 'array' in audio_prompt:
#                     arr = audio_prompt['array']
#                     if isinstance(arr, (list, tuple)) and len(arr) == 2:
#                         sr, data = arr
#                         wav_pr = torch.FloatTensor(data)
#                     else:
#                         raise ValueError("Unsupported audio_prompt dict array format")
#                 else:
#                     raise ValueError("Unsupported audio_prompt dict format; missing file path")
#         elif isinstance(audio_prompt, tuple) and len(audio_prompt) == 2:
#             # old format: (sample_rate, waveform)
#             sr, wav_pr = audio_prompt
#             if not isinstance(wav_pr, torch.Tensor):
#                 wav_pr = torch.FloatTensor(wav_pr)
#             # save to temp wav for diarization/similarity path
#             if sf is not None:
#                 tmp_path = os.path.join(os.getcwd(), f"temp_enroll_{int(time.time())}.wav")
#                 # wav_pr shape may be (samples,) or (1, samples)
#                 arr = wav_pr.squeeze(0).cpu().numpy() if isinstance(wav_pr, torch.Tensor) else wav_pr
#                 sf.write(tmp_path, arr, sr)
#                 temp_wav_path = tmp_path
#         else:
#             return "Error: Unsupported audio format", None

#         if temp_wav_path:
#             is_valid, check_message = check_audio_for_generation(temp_wav_path)
#             if not is_valid:
#                 return check_message, None
#             else:
#                 print(check_message)
#         # Ensure wav_pr is a tensor
#         if not isinstance(wav_pr, torch.Tensor):
#             wav_pr = torch.FloatTensor(wav_pr)

#         # Normalize amplitude
#         if wav_pr.abs().max() > 1:
#             wav_pr = wav_pr / wav_pr.abs().max()

#         # stereo -> mono handling
#         if wav_pr.ndim > 1 and wav_pr.size(0) == 2:
#             wav_pr = wav_pr.mean(dim=0, keepdim=True)
#         elif wav_pr.ndim > 1 and wav_pr.size(-1) == 2:
#             wav_pr = wav_pr.mean(dim=-1, keepdim=True).squeeze(-1)

#         if wav_pr.ndim == 1:
#             wav_pr = wav_pr.unsqueeze(0)

#         assert wav_pr.ndim == 2 and wav_pr.size(0) == 1, f"Expected shape (1, length), got {wav_pr.shape}"

#         # transcript content / prompt extraction
#         if transcript_content == "":
#             text_pr, lang_pr = make_prompt('dummy', wav_pr, sr, save=False)
#         else:
#             lang_pr = langid.classify(str(transcript_content))[0]
#             if lang_pr != "en":
#                 return f"Error: Only English text is supported. Detected language: {lang_pr}", None
#             lang_token = lang2token[lang_pr]
#             text_pr = f"{lang_token}{str(transcript_content)}{lang_token}"

#         if language == 'auto-detect':
#             lang_token = lang2token[langid.classify(text)[0]]
#         else:
#             lang_token = langdropdown2token[language]
#         lang = token2lang[lang_token]
#         text = lang_token + text + lang_token

#         # Move model to device
#         model.to(device)

#         # tokenize audio prompt (enrollment)
#         encoded_frames = tokenize_audio(audio_tokenizer, (wav_pr, sr))
#         audio_prompts = encoded_frames[0][0].transpose(2, 1).to(device)

#         # tokenize text
#         logging.info(f"synthesize text: {text}")
#         phone_tokens, langs = text_tokenizer.tokenize(text=f"_{text}".strip())
#         text_tokens, text_tokens_lens = text_collater([phone_tokens])

#         enroll_x_lens = None
#         if text_pr:
#             text_prompts, _ = text_tokenizer.tokenize(text=f"{text_pr}".strip())
#             text_prompts, enroll_x_lens = text_collater([text_prompts])

#         text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
#         text_tokens_lens += enroll_x_lens
#         lang = lang if accent == "no-accent" else token2lang[langdropdown2token[accent]]

#         # measure generation time
#         gen_start = time.time()
#         encoded_frames = model.inference(
#             text_tokens.to(device),
#             text_tokens_lens.to(device),
#             audio_prompts,
#             enroll_x_lens=enroll_x_lens,
#             top_k=10,
#             temperature=1,
#             prompt_language=lang_pr,
#             text_language=langs if accent == "no-accent" else lang,
#             best_of=1,
#         )
#         gen_end = time.time()

#         # Decode with Vocos
#         frames = encoded_frames.permute(2,0,1)
#         features = vocos.codes_to_features(frames)
#         samples = vocos.decode(features, bandwidth_id=torch.tensor([2], device=device))

#         # offload model
#         model.to('cpu')
#         torch.cuda.empty_cache()

#         audio_numpy = samples.squeeze(0).cpu().numpy()
#         # clip extremely long audio
#         if len(audio_numpy) > 24000 * 300:
#             logging.warning(f"Audio too long ({len(audio_numpy)/24000:.1f}s), clipping to 5 minutes")
#             audio_numpy = audio_numpy[:24000 * 300]

#         message = f"✓ Generated successfully!\nPrompt: {text_pr}\nSynthesized: {text}\nDuration: {len(audio_numpy)/24000:.1f}s"

#         # Evaluate and print metrics to terminal
#         try:
#             ref_path = temp_wav_path
#             if isinstance(audio_prompt, dict):
#                 ref_path = audio_prompt.get('name') or audio_prompt.get('path') or ref_path
#             evaluate_and_print_metrics(audio_numpy, 24000, original_text=text, reference_prompt_wav_path=ref_path, save_prefix="infer_from_audio")
#         except Exception as e:
#             print(f"[Eval] evaluation failed: {e}")

#         return message, (24000, audio_numpy)

#     except Exception as e:
#         logging.error(f"Error in infer_from_audio: {str(e)}")
#         model.to('cpu')
#         torch.cuda.empty_cache()
#         return f"Error: {str(e)}", None

# @torch.no_grad()
# def infer_from_prompt(text, language, accent, preset_prompt, prompt_file):
#     """
#     Replacement infer_from_prompt (text -> synth using a prompt file or preset).
#     Returns (message, (sample_rate, numpy_audio))
#     """
#     global model, text_collater, text_tokenizer, audio_tokenizer, vocos, device
#     try:
#         # check language token
#         if language == 'auto-detect':
#             lang_token = lang2token[langid.classify(text)[0]]
#         else:
#             lang_token = langdropdown2token[language]
#         lang = token2lang[lang_token]
#         text = lang_token + text + lang_token

#         # load prompt data (either from uploaded .npz or preset)
#         if prompt_file is not None:
#             # prompt_file is a file-like object (gradio) - it likely has .name path
#             try:
#                 prompt_data = np.load(prompt_file.name)
#             except Exception:
#                 # try reading via file object
#                 prompt_file.seek(0)
#                 prompt_data = np.load(prompt_file)
#         else:
#             prompt_data = np.load(os.path.join("./presets/", f"{preset_prompt}.npz"))

#         audio_prompts = prompt_data['audio_tokens']
#         text_prompts = prompt_data['text_tokens']
#         lang_pr = prompt_data['lang_code']
#         lang_pr = code2lang[int(lang_pr)]

#         # numpy -> tensor
#         audio_prompts = torch.tensor(audio_prompts).type(torch.int32).to(device)
#         text_prompts = torch.tensor(text_prompts).type(torch.int32)

#         enroll_x_lens = text_prompts.shape[-1]
#         logging.info(f"synthesize text: {text}")
#         phone_tokens, langs = text_tokenizer.tokenize(text=f"_{text}".strip())
#         text_tokens, text_tokens_lens = text_collater([phone_tokens])
#         text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
#         text_tokens_lens += enroll_x_lens

#         # accent control
#         lang = lang if accent == "no-accent" else token2lang[langdropdown2token[accent]]
#         encoded_frames = model.inference(
#             text_tokens.to(device),
#             text_tokens_lens.to(device),
#             audio_prompts,
#             enroll_x_lens=enroll_x_lens,
#             top_k=-100,
#             temperature=1,
#             prompt_language=lang_pr,
#             text_language=langs if accent == "no-accent" else lang,
#             best_of=5,
#         )

#         # Decode
#         frames = encoded_frames.permute(2,0,1)
#         features = vocos.codes_to_features(frames)
#         samples = vocos.decode(features, bandwidth_id=torch.tensor([2], device=device))

#         model.to('cpu')
#         torch.cuda.empty_cache()

#         audio_numpy = samples.squeeze(0).cpu().numpy()
#         if len(audio_numpy) > 24000 * 300:
#             logging.warning(f"Audio too long ({len(audio_numpy)/24000:.1f}s), clipping to 5 minutes")
#             audio_numpy = audio_numpy[:24000 * 300]

#         message = f"sythesized text: {text}"

#         # Determine reference path for similarity (if prompt_file had a path)
#         reference_path = None
#         try:
#             if prompt_file is not None and hasattr(prompt_file, 'name'):
#                 # If the user uploaded an npz produced from recorded audio, may not have wav
#                 reference_path = prompt_file.name
#         except Exception:
#             reference_path = None

#         # evaluate and print metrics
#         try:
#             evaluate_and_print_metrics(audio_numpy, 24000, original_text=text, reference_prompt_wav_path=reference_path, save_prefix="infer_from_prompt")
#         except Exception as e:
#             print(f"[Eval] evaluation failed: {e}")

#         return message, (24000, audio_numpy)

#     except Exception as e:
#         logging.error(f"Error in infer_from_prompt: {e}")
#         model.to('cpu')
#         torch.cuda.empty_cache()
#         return f"Error: {e}", None

# # ------------------------------
# # 3) infer_long_text (full ready-to-replace)
# # ------------------------------
# from utils.sentence_cutter import split_text_into_sentences
# @torch.no_grad()
# def infer_long_text(text, preset_prompt, prompt=None, language='auto', accent='no-accent'):
#     """
#     Replacement infer_long_text with evaluation printing.
#     Returns (message, (sample_rate, numpy_audio))
#     """
#     global model, audio_tokenizer, text_tokenizer, text_collater, vocos, device, NUM_QUANTIZERS
#     model.to(device)
#     mode = 'fixed-prompt'
#     if (prompt is None or prompt == "") and preset_prompt == "":
#         mode = 'sliding-window'
#     sentences = split_text_into_sentences(text)

#     # detect language
#     if language == "auto-detect" or language == 'auto':
#         language = langid.classify(text)[0]
#     else:
#         language = token2lang[langdropdown2token[language]]

#     # prepare prompts
#     if prompt is not None and prompt != "":
#         # load prompt from uploaded file object
#         if hasattr(prompt, "name"):
#             prompt_data = np.load(prompt.name)
#         else:
#             prompt_data = np.load(prompt)
#         audio_prompts = prompt_data['audio_tokens']
#         text_prompts = prompt_data['text_tokens']
#         lang_pr = prompt_data['lang_code']
#         lang_pr = code2lang[int(lang_pr)]
#         audio_prompts = torch.tensor(audio_prompts).type(torch.int32).to(device)
#         text_prompts = torch.tensor(text_prompts).type(torch.int32)
#     elif preset_prompt is not None and preset_prompt != "":
#         prompt_data = np.load(os.path.join("./presets/", f"{preset_prompt}.npz"))
#         audio_prompts = prompt_data['audio_tokens']
#         text_prompts = prompt_data['text_tokens']
#         lang_pr = prompt_data['lang_code']
#         lang_pr = code2lang[int(lang_pr)]
#         audio_prompts = torch.tensor(audio_prompts).type(torch.int32).to(device)
#         text_prompts = torch.tensor(text_prompts).type(torch.int32)
#     else:
#         audio_prompts = torch.zeros([1, 0, NUM_QUANTIZERS]).type(torch.int32).to(device)
#         text_prompts = torch.zeros([1, 0]).type(torch.int32)
#         lang_pr = language if language != 'mix' else 'en'

#     if mode == 'fixed-prompt':
#         complete_tokens = torch.zeros([1, NUM_QUANTIZERS, 0]).type(torch.LongTensor).to(device)
#         for text_sent in sentences:
#             text_sent = text_sent.replace("\n", "").strip(" ")
#             if text_sent == "":
#                 continue
#             lang_token = lang2token[language]
#             lang = token2lang[lang_token]
#             text_in = lang_token + text_sent + lang_token

#             enroll_x_lens = text_prompts.shape[-1]
#             logging.info(f"synthesize text: {text_in}")
#             phone_tokens, langs = text_tokenizer.tokenize(text=f"_{text_in}".strip())
#             text_tokens, text_tokens_lens = text_collater([phone_tokens])
#             text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
#             text_tokens_lens += enroll_x_lens
#             lang_use = lang if accent == "no-accent" else token2lang[langdropdown2token[accent]]
#             encoded_frames = model.inference(
#                 text_tokens.to(device),
#                 text_tokens_lens.to(device),
#                 audio_prompts,
#                 enroll_x_lens=enroll_x_lens,
#                 top_k=-100,
#                 temperature=1,
#                 prompt_language=lang_pr,
#                 text_language=langs if accent == "no-accent" else lang_use,
#                 best_of=5,
#             )
#             complete_tokens = torch.cat([complete_tokens, encoded_frames.transpose(2, 1)], dim=-1)
#         # Decode all frames
#         frames = complete_tokens.permute(1, 0, 2)
#         features = vocos.codes_to_features(frames)
#         samples = vocos.decode(features, bandwidth_id=torch.tensor([2], device=device))

#         model.to('cpu')
#         message = f"Cut into {len(sentences)} sentences"
#         audio_numpy = samples.squeeze(0).cpu().numpy()

#         if len(audio_numpy) > 24000 * 300:
#             logging.warning(f"Audio too long ({len(audio_numpy)/24000:.1f}s), clipping to 5 minutes")
#             audio_numpy = audio_numpy[:24000 * 300]

#         # evaluation: we won't have a simple enrollment file for long_text usually
#         try:
#             evaluate_and_print_metrics(audio_numpy, 24000, original_text=text, reference_prompt_wav_path=None, save_prefix="infer_long")
#         except Exception as e:
#             print(f"[Eval] evaluation failed: {e}")

#         return message, (24000, audio_numpy)

#     elif mode == "sliding-window":
#         complete_tokens = torch.zeros([1, NUM_QUANTIZERS, 0]).type(torch.LongTensor).to(device)
#         original_audio_prompts = audio_prompts
#         original_text_prompts = text_prompts
#         for text_sent in sentences:
#             text_sent = text_sent.replace("\n", "").strip(" ")
#             if text_sent == "":
#                 continue
#             lang_token = lang2token[language]
#             lang = token2lang[lang_token]
#             text_in = lang_token + text_sent + lang_token

#             enroll_x_lens = text_prompts.shape[-1]
#             logging.info(f"synthesize text: {text_in}")
#             phone_tokens, langs = text_tokenizer.tokenize(text=f"_{text_in}".strip())
#             text_tokens, text_tokens_lens = text_collater([phone_tokens])
#             text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
#             text_tokens_lens += enroll_x_lens
#             lang_use = lang if accent == "no-accent" else token2lang[langdropdown2token[accent]]
#             encoded_frames = model.inference(
#                 text_tokens.to(device),
#                 text_tokens_lens.to(device),
#                 audio_prompts,
#                 enroll_x_lens=enroll_x_lens,
#                 top_k=-100,
#                 temperature=1,
#                 prompt_language=lang_pr,
#                 text_language=langs if accent == "no-accent" else lang_use,
#                 best_of=5,
#             )
#             complete_tokens = torch.cat([complete_tokens, encoded_frames.transpose(2, 1)], dim=-1)
#             if torch.rand(1) < 1.0:
#                 audio_prompts = encoded_frames[:, :, -NUM_QUANTIZERS:]
#                 text_prompts = text_tokens[:, enroll_x_lens:]
#             else:
#                 audio_prompts = original_audio_prompts
#                 text_prompts = original_text_prompts
#         frames = complete_tokens.permute(1, 0, 2)
#         features = vocos.codes_to_features(frames)
#         samples = vocos.decode(features, bandwidth_id=torch.tensor([2], device=device))

#         model.to('cpu')
#         message = f"Cut into {len(sentences)} sentences"
#         audio_numpy = samples.squeeze(0).cpu().numpy()

#         if len(audio_numpy) > 24000 * 300:
#             logging.warning(f"Audio too long ({len(audio_numpy)/24000:.1f}s), clipping to 5 minutes")
#             audio_numpy = audio_numpy[:24000 * 300]

#         try:
#             evaluate_and_print_metrics(audio_numpy, 24000, original_text=text, reference_prompt_wav_path=None, save_prefix="infer_long")
#         except Exception as e:
#             print(f"[Eval] evaluation failed: {e}")

#         return message, (24000, audio_numpy)
#     else:
#         raise ValueError(f"No such mode {mode}")


# def main():
#     # Load speaker diarization model
#     try:
#         load_speaker_diarization()
#     except Exception as e:
#         print(f"Failed to load speaker diarization: {e}")
#         print("Continuing without speaker diarization...")
    
#     app = gr.Blocks(title="VALL-E X")
#     with app:
#         gr.Markdown(top_md)
#         with gr.Tab("Infer from audio"):
#             gr.Markdown(infer_from_audio_md)
#             with gr.Row():
#                 with gr.Column():

#                     textbox = gr.TextArea(label="Text",
#                                           placeholder="Type your sentence here",
#                                           value="Welcome back, Master. What can I do for you today?", elem_id=f"tts-input")
#                     # language_dropdown = gr.Dropdown(choices=['auto-detect', 'English', '中文', '日本語'], value='auto-detect', label='language')
#                     language_dropdown = gr.Dropdown(choices=['English'], value='English', label='language')
#                     accent_dropdown = gr.Dropdown(choices=['English'], value='English', label='accent')
#                     # accent_dropdown = gr.Dropdown(choices=['no-accent', 'English', '中文', '日本語'], value='no-accent', label='accent')
#                     textbox_transcript = gr.TextArea(label="Transcript",
#                                           placeholder="Write transcript here. (leave empty to use whisper)",
#                                           value="", elem_id=f"prompt-name")
#                     upload_audio_prompt = gr.Audio(label='uploaded audio prompt', interactive=True)
#                     record_audio_prompt = gr.Audio(label='recorded audio prompt', interactive=True)
#                 with gr.Column():
#                     text_output = gr.Textbox(label="Message")
#                     audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
#                     btn = gr.Button("Generate!")
#                     btn.click(infer_from_audio,
#                               inputs=[textbox, language_dropdown, accent_dropdown, upload_audio_prompt, record_audio_prompt, textbox_transcript],
#                               outputs=[text_output, audio_output])
#                     textbox_mp = gr.TextArea(label="Prompt name",
#                                           placeholder="Name your prompt here",
#                                           value="prompt_1", elem_id=f"prompt-name")
#                     btn_mp = gr.Button("Make prompt!")
#                     prompt_output = gr.File(interactive=False)
#                     btn_mp.click(make_npz_prompt,
#                                 inputs=[textbox_mp, upload_audio_prompt, record_audio_prompt, textbox_transcript],
#                                 outputs=[text_output, prompt_output])
#             gr.Examples(examples=infer_from_audio_examples,
#                         inputs=[textbox, language_dropdown, accent_dropdown, upload_audio_prompt, record_audio_prompt, textbox_transcript],
#                         outputs=[text_output, audio_output],
#                         fn=infer_from_audio,
#                         cache_examples=False,)
#         with gr.Tab("Make prompt"):
#             gr.Markdown(make_prompt_md)
#             with gr.Row():
#                 with gr.Column():
#                     textbox2 = gr.TextArea(label="Prompt name",
#                                           placeholder="Name your prompt here",
#                                           value="prompt_1", elem_id=f"prompt-name")
#                     # 添加选择语言和输入台本的地方
#                     textbox_transcript2 = gr.TextArea(label="Transcript",
#                                           placeholder="Write transcript here. (leave empty to use whisper)",
#                                           value="", elem_id=f"prompt-name")
#                     upload_audio_prompt_2 = gr.Audio(label='uploaded audio prompt', interactive=True)
#                     record_audio_prompt_2 = gr.Audio(label='recorded audio prompt', interactive=True)
#                 with gr.Column():
#                     text_output_2 = gr.Textbox(label="Message")
#                     prompt_output_2 = gr.File(interactive=False)
#                     btn_2 = gr.Button("Make!")
#                     btn_2.click(make_npz_prompt,
#                               inputs=[textbox2, upload_audio_prompt_2, record_audio_prompt_2, textbox_transcript2],
#                               outputs=[text_output_2, prompt_output_2])
#             gr.Examples(examples=make_npz_prompt_examples,
#                         inputs=[textbox2, upload_audio_prompt_2, record_audio_prompt_2, textbox_transcript2],
#                         outputs=[text_output_2, prompt_output_2],
#                         fn=make_npz_prompt,
#                         cache_examples=False,)
#         with gr.Tab("Infer from prompt"):
#             gr.Markdown(infer_from_prompt_md)
#             with gr.Row():
#                 with gr.Column():
#                     textbox_3 = gr.TextArea(label="Text",
#                                           placeholder="Type your sentence here",
#                                           value="Welcome back, Master. What can I do for you today?", elem_id=f"tts-input")
#                     language_dropdown_3 = gr.Dropdown(choices=['English'], value='English',
#                                                     label='language')
#                     accent_dropdown_3 = gr.Dropdown(choices=['English'], value='English',
#                                                   label='accent')
#                     preset_dropdown_3 = gr.Dropdown(choices=preset_list, value=None, label='Voice preset')
#                     prompt_file = gr.File(file_count='single', file_types=['.npz'], interactive=True)
#                 with gr.Column():
#                     text_output_3 = gr.Textbox(label="Message")
#                     audio_output_3 = gr.Audio(label="Output Audio", elem_id="tts-audio")
#                     btn_3 = gr.Button("Generate!")
#                     btn_3.click(infer_from_prompt,
#                               inputs=[textbox_3, language_dropdown_3, accent_dropdown_3, preset_dropdown_3, prompt_file],
#                               outputs=[text_output_3, audio_output_3])
#             gr.Examples(examples=infer_from_prompt_examples,
#                         inputs=[textbox_3, language_dropdown_3, accent_dropdown_3, preset_dropdown_3, prompt_file],
#                         outputs=[text_output_3, audio_output_3],
#                         fn=infer_from_prompt,
#                         cache_examples=False,)
#         with gr.Tab("Infer long text"):
#             gr.Markdown("This is a long text generation demo. You can use this to generate long audio. ")
#             with gr.Row():
#                 with gr.Column():
#                     textbox_4 = gr.TextArea(label="Text",
#                                           placeholder="Type your sentence here",
#                                           value=long_text_example, elem_id=f"tts-input")
#                     language_dropdown_4 = gr.Dropdown(choices=['English'], value='English',
#                                                     label='language')
#                     accent_dropdown_4 = gr.Dropdown(choices=['English'], value='English',
#                                                     label='accent')
#                     preset_dropdown_4 = gr.Dropdown(choices=preset_list, value=None, label='Voice preset')
#                     prompt_file_4 = gr.File(file_count='single', file_types=['.npz'], interactive=True)
#                 with gr.Column():
#                     text_output_4 = gr.TextArea(label="Message")
#                     audio_output_4 = gr.Audio(label="Output Audio", elem_id="tts-audio")
#                     btn_4 = gr.Button("Generate!")
#                     btn_4.click(infer_long_text,
#                               inputs=[textbox_4, preset_dropdown_4, prompt_file_4, language_dropdown_4, accent_dropdown_4],
#                               outputs=[text_output_4, audio_output_4])

#     try:
#         webbrowser.open("http://127.0.0.1:7860")
#         app.launch(inbrowser=True)
#     except Exception as e:
#         print(f"Error launching app: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     formatter = (
#         "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
#     )
#     logging.basicConfig(format=formatter, level=logging.INFO)
    
#     try:
#         main()
#     except Exception as e:
#         print(f"Fatal error: {e}")
#         import traceback
#         traceback.print_exc()







































# coding: utf-8
import argparse
import logging
import os
import pathlib
import time
import tempfile
import platform
import webbrowser
import sys
print(f"default encoding is {sys.getdefaultencoding()},file system encoding is {sys.getfilesystemencoding()}")
if(sys.version_info[0]<3 or sys.version_info[1]<7):
    print("The Python version is too low and may cause problems")

if platform.system().lower() == 'windows':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
else:
    temp = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import langid
# langid.set_languages(['en', 'zh', 'ja'])
langid.set_languages(['en'])
import nltk
nltk.data.path = nltk.data.path + [os.path.join(os.getcwd(), "nltk_data")]

import torch
import torchaudio
import random
import soundfile as sf

import numpy as np

from data.tokenizer import (
    AudioTokenizer,
    tokenize_audio,
)
from data.collation import get_text_token_collater
from models.vallex import VALLE
from utils.g2p import PhonemeBpeTokenizer
from descriptions import *
from macros import *
from examples import *

import gradio as gr
import whisper
from vocos import Vocos
import multiprocessing

# ------------------------------
# Simple KMeans Speaker Diarization
# ------------------------------
import librosa
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances

import time
import os
import numpy as np

# Lazy imports for embedding & ASR evaluation
try:
    import soundfile as sf
except Exception:
    sf = None
try:
    import torchaudio
except Exception:
    torchaudio = None

# transformers Wav2Vec2 for speaker embeddings (lazy loaded)
_w2v_processor = None
_w2v_model = None
def _ensure_w2v_loaded():
    global _w2v_processor, _w2v_model
    if _w2v_processor is None or _w2v_model is None:
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
            _w2v_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            _w2v_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            _w2v_model.eval()
            # turn off gradients
            for p in _w2v_model.parameters():
                p.requires_grad = False
        except Exception as e:
            _w2v_processor = None
            _w2v_model = None
            print(f"[Eval Helper] Failed to load Wav2Vec2 model: {e}")

def get_embedding(wav_path):
    """
    Return a 1D torch tensor embedding for wav_path (mean pool of last_hidden_state).
    Resamples to 16k if necessary.
    """
    import torch
    global _w2v_processor, _w2v_model
    if sf is None:
        raise RuntimeError("soundfile (pysoundfile) is required for get_embedding")
    _ensure_w2v_loaded()
    if _w2v_processor is None or _w2v_model is None:
        raise RuntimeError("Wav2Vec2 processor/model not loaded")
    wav, sr = sf.read(wav_path)
    # Convert to mono if stereo
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype("float32")
    # Resample to 16k if needed
    if sr != 16000:
        if torchaudio is None:
            raise RuntimeError("torchaudio required for resampling to 16k")
        wav_t = torch.from_numpy(wav).unsqueeze(0)  # (1, samples)
        wav_t = torchaudio.functional.resample(wav_t, sr, 16000)
        wav = wav_t.squeeze(0).numpy()
        sr = 16000
    inputs = _w2v_processor(wav, sampling_rate=sr, return_tensors="pt", padding=False)
    input_values = inputs["input_values"]
    if input_values.dim() == 3:
        input_values = input_values.squeeze(1)
    with torch.no_grad():
        outputs = _w2v_model(input_values)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)  # (hidden,)
    return emb

# Evaluation helper: writes audio, runs Whisper ASR, WER, similarity, prints to terminal
def evaluate_and_print_metrics(audio_numpy, sample_rate, original_text="", reference_prompt_wav_path=None, save_prefix="eval"):
    """
    audio_numpy: 1D numpy array (float32) of generated audio samples
    sample_rate: int (e.g., 24000)
    original_text: str - ground truth text that was synthesized (for WER)
    reference_prompt_wav_path: path to reference enrollment wav (for similarity). If None, similarity skipped.
    save_prefix: filename prefix for saved generated file
    """
    import torch
    import torch.nn.functional as F
    import jiwer
    import whisper
    from scipy.io.wavfile import write as write_wav
    # Normalize audio to -1..1 float32
    audio = np.asarray(audio_numpy)
    if audio.dtype not in (np.float32, np.float64):
        # Convert integer -> float
        peak = np.max(np.abs(audio)) if audio.size else 1.0
        if peak > 0:
            audio = audio.astype("float32") / float(peak)
        else:
            audio = audio.astype("float32")
    else:
        audio = audio.astype("float32")
    peak = np.max(np.abs(audio)) if audio.size else 1.0
    if peak > 1.0:
        audio = audio / peak

    timestamp = int(time.time())
    out_filename = f"{save_prefix}_{timestamp}.wav"
    try:
        write_wav(out_filename, sample_rate, audio)
    except Exception:
        # fallback to soundfile if scipy write fails
        if sf is None:
            print("[Eval] Could not save audio: scipy write failed and soundfile not present.")
        else:
            sf.write(out_filename, audio, sample_rate)
    print(f"[Eval] Saved generated audio -> {out_filename}")

    # Whisper transcription (use existing global whisper_model if present)
    try:
        try:
            whisper_model  # check if exists in globals
            model_for_asr = whisper_model
        except NameError:
            model_for_asr = whisper.load_model("base")
        tstart = time.time()
        res = model_for_asr.transcribe(out_filename)
        transcribed = res.get("text", "").strip()
        tasr = time.time() - tstart
    except Exception as e:
        transcribed = "[ASR_ERROR]"
        tasr = 0.0
        print(f"[Eval] Whisper transcription failed: {e}")
    from sklearn.metrics.pairwise import cosine_similarity
    # Compute WER
    wer_val = None
    if original_text and original_text.strip() != "":
        try:
            wer_val = jiwer.wer(original_text.lower(), transcribed.lower())
        except Exception as e:
            print(f"[Eval] WER computation error: {e}")
            wer_val = None

    # Similarity (if requested and available)
    similarity = None
    if reference_prompt_wav_path:
        try:
            emb_gen = get_embedding(out_filename)  # embedding for generated audio
            emb_ref = get_embedding(reference_prompt_wav_path)  # embedding for reference prompt
            # Convert to numpy arrays
            emb_gen_np = emb_gen.detach().cpu().numpy().reshape(1, -1)
            emb_ref_np = emb_ref.detach().cpu().numpy().reshape(1, -1)
            # Calculate cosine similarity
            similarity = float(cosine_similarity(emb_gen_np, emb_ref_np)[0][0])
        except Exception as e:
            print(f"[Eval] Speaker similarity calculation failed: {e}")
            similarity = None

    # Printable summary
    print("\n=== ACCURACY METRICS ===")
    if original_text:
        print(f"Original Text     : {original_text}")
    print(f"Transcribed Text  : {transcribed}")
    if wer_val is not None:
        print(f"WER               : {wer_val:.4f}")
    else:
        print("WER               : N/A")
    print(f"ASR Time (sec)    : {tasr:.3f}")
    if similarity is not None:
        print(f"Speaker Similarity: {similarity:.4f} (cosine)")
        if similarity >= 0.80:
            remark = "Excellent cloning"
        elif similarity >= 0.60:
            remark = "Good cloning"
        elif similarity >= 0.40:
            remark = "Average cloning"
        else:
            remark = "Poor cloning"
        print(f"Cloning Quality   : {remark}")
    else:
        print("Speaker Similarity: N/A")
    print(f"Saved audio file  : {out_filename}")
    print("=========================\n")

def detect_speakers(audio_path, threshold=0.15):
    """
    Detects number of speakers in audio using KMeans clustering.
    
    Returns:
        tuple: (num_speakers: int, message: str)
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        duration = len(y) / sr
        
        # If audio is very short, assume single speaker
        if duration < 1.0:
            return 1, "✓ Single speaker (audio too short for analysis)"
        
        # Voice Activity Detection using energy
        energy = librosa.feature.rms(y=y)[0]
        frames = librosa.util.frame(y, frame_length=2048, hop_length=512)
        
        voiced_frames = []
        for i in range(min(frames.shape[1], len(energy))):
            if energy[i] > np.mean(energy) * 0.8:
                voiced_frames.append(frames[:, i])
        
        if len(voiced_frames) < 5:
            return 1, "✓ Single speaker (insufficient voice activity)"
        
        # Extract MFCC embeddings
        embeddings = []
        for frame in voiced_frames:
            mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=20)
            emb = np.mean(mfcc, axis=1)
            embeddings.append(emb)
        
        if len(embeddings) < 10:
            return 1, "✓ Single speaker (insufficient features)"
        
        embeddings = np.array(embeddings)
        
        # KMeans clustering with 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(embeddings)
        labels = kmeans.labels_
        
        # Calculate cluster centroids
        cluster_0 = embeddings[labels == 0]
        cluster_1 = embeddings[labels == 1]
        
        if len(cluster_0) == 0 or len(cluster_1) == 0:
            return 1, "✓ Single speaker detected"
        
        centroid_0 = cluster_0.mean(axis=0)
        centroid_1 = cluster_1.mean(axis=0)
        
        # Calculate distance between centroids
        distance = cosine_distances([centroid_0], [centroid_1])[0][0]
        
        # Calculate cluster sizes
        cluster_0_pct = len(cluster_0) / len(embeddings) * 100
        cluster_1_pct = len(cluster_1) / len(embeddings) * 100
        
        print(f"Cluster analysis: {cluster_0_pct:.1f}% vs {cluster_1_pct:.1f}%, distance={distance:.3f}")
        
        # Decision logic
        if distance < threshold:
            return 1, f"✓ Single speaker detected (distance={distance:.3f})"
        else:
            # Check if one cluster is too small (might be noise)
            min_cluster_size = 15  # minimum 15% to be considered a separate speaker
            if cluster_0_pct < min_cluster_size or cluster_1_pct < min_cluster_size:
                return 1, f"✓ Single speaker (minor variation detected, distance={distance:.3f})"
            
            return 2, f"⚠ Multiple speakers detected ({cluster_0_pct:.0f}%/{cluster_1_pct:.0f}% split, distance={distance:.3f})"
    
    except Exception as e:
        print(f"Speaker detection error: {e}")
        return 1, "✓ Single speaker (analysis failed, assuming single)"


def check_audio_for_generation(audio_path):
    """
    Checks if audio is suitable for voice cloning.
    Returns: (is_valid: bool, message: str)
    """
    num_speakers, detail_msg = detect_speakers(audio_path)
    
    if num_speakers > 1:
        error_msg = (
            "❌ Multiple speakers detected in the audio!\n\n"
            "This voice cloning system requires audio with only ONE speaker.\n"
            f"Analysis: {detail_msg}\n\n"
            "Please provide a different audio sample with only one person speaking."
        )
        return False, error_msg
    else:
        success_msg = f"✓ Audio validated for voice cloning\n{detail_msg}"
        return True, success_msg

# Try to import SpeechBrain for speaker diarization
# try:
#     # from speechbrain.pretrained import SpeakerRecognition
#     SPEECHBRAIN_AVAILABLE = True
#     print("SpeechBrain loaded successfully for speaker diarization")
# except Exception as e:
#     print(f"Warning: SpeechBrain not available ({e}). Speaker detection will be skipped.")
#     SPEECHBRAIN_AVAILABLE = False

thread_count = multiprocessing.cpu_count()

print("Use",thread_count,"cpu cores for computing")

torch.set_num_threads(thread_count)
torch.set_num_interop_threads(thread_count)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)

text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_69.json")
text_collater = get_text_token_collater()

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda", 0)
if torch.backends.mps.is_available():
    device = torch.device("mps")
# VALL-E-X model
if not os.path.exists("./checkpoints/"): os.mkdir("./checkpoints/")
CHECKPOINT_PATH = "checkpoints/vallex-checkpoint.pt"

# Ensure folder exists
os.makedirs("checkpoints", exist_ok=True)

if not os.path.isfile(CHECKPOINT_PATH):
    import wget
    try:
        print("Model checkpoint not found. Downloading it now...")
        logging.info("Downloading VALLE-X model (first time only)...")
        wget.download(
            "https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt",
            out=CHECKPOINT_PATH,
            bar=wget.bar_adaptive
        )
        print("\nDownload complete!")
    except Exception as e:
        logging.info(e)
        raise Exception(
            "\nModel weights download failed.\n"
            "Please manually download from https://huggingface.co/Plachta/VALL-E-X\n"
            f"and put vallex-checkpoint.pt inside: {os.getcwd()}/checkpoints/"
        )
else:
    print("✔ Using existing model checkpoint — skipping download.")


model = VALLE(
        N_DIM,
        NUM_HEAD,
        NUM_LAYERS,
        norm_first=True,
        add_prenet=False,
        prefix_mode=PREFIX_MODE,
        share_embedding=True,
        nar_scale_factor=1.0,
        prepend_bos=True,
        num_quantizers=NUM_QUANTIZERS,
    )
checkpoint = torch.load("./checkpoints/vallex-checkpoint.pt", map_location='cpu', weights_only=False)
missing_keys, unexpected_keys = model.load_state_dict(
    checkpoint["model"], strict=True
)
assert not missing_keys
model.eval()

# Encodec model
audio_tokenizer = AudioTokenizer(device)

# Vocos decoder
vocos = Vocos.from_pretrained('charactr/vocos-encodec-24khz').to(device)

# ASR
if not os.path.exists("./whisper/"): os.mkdir("./whisper/")
try:
    whisper_model = whisper.load_model("medium",download_root=os.path.join(os.getcwd(), "whisper")).cpu()
except Exception as e:
    logging.info(e)
    raise Exception(
        "\n Whisper download failed or damaged, please go to "
        "'https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt'"
        "\n manually download model and put it to {} .".format(os.getcwd() + "\whisper"))

# Voice Presets
preset_list = os.walk("./presets/").__next__()[2]
preset_list = [preset[:-4] for preset in preset_list if preset.endswith(".npz")]

# Speaker diarization model
speaker_diarization_model = None

def load_speaker_diarization():
    # global speaker_diarization_model
    # if not SPEECHBRAIN_AVAILABLE:
    #     print("SpeechBrain not available, skipping speaker diarization")
    #     return
    # try:
    #     print("Loading SpeechBrain speaker diarization model...")
    #     speaker_diarization_model = SpeakerRecognition.from_hparams(
    #         source="speechbrain/spkrec-ecapa-voxceleb",
    #         savedir="pretrained_models/spkrec"
    #     )
    #     print("Speaker diarization model loaded successfully")
    # except Exception as e:
    #     print(f"Warning: Could not load speaker diarization model: {e}")
        print("Speaker detection will be skipped")

def check_single_speaker(audio_path):
    """
    Check if audio contains only one speaker using SpeechBrain.
    Returns: (is_single_speaker: bool, num_speakers: int, error_message: str)
    """
    # if not SPEECHBRAIN_AVAILABLE or speaker_diarization_model is None:
    #     return True, 1, ""  # Skip check if model not loaded
    
    try:
        # Simple check: if audio is very short, assume single speaker
        import librosa
        y, sr = librosa.load(audio_path, sr=16000)
        duration = len(y) / sr
        
        if duration < 1.0:
            return True, 1, ""
        
        # For simplicity, we assume single speaker if model is loaded
        # You can implement more sophisticated multi-speaker detection here
        # This is a placeholder - actual implementation would require
        # speaker segmentation and clustering
        
        return True, 1, ""
        
    except Exception as e:
        print(f"Warning: Speaker diarization check failed: {e}")
        return True, 1, ""  # If check fails, allow it through

def clear_prompts():
    try:
        path = tempfile.gettempdir()
        for eachfile in os.listdir(path):
            filename = os.path.join(path, eachfile)
            if os.path.isfile(filename) and filename.endswith(".npz"):
                lastmodifytime = os.stat(filename).st_mtime
                endfiletime = time.time() - 60
                if endfiletime > lastmodifytime:
                    os.remove(filename)
    except:
        return

def transcribe_one(model, audio_path):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    lang = max(probs, key=probs.get)
    # decode the audio
    options = whisper.DecodingOptions(temperature=1.0, best_of=5, fp16=False if device == torch.device("cpu") else True, sample_len=150)
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)

    text_pr = result.text
    if text_pr.strip(" ")[-1] not in "?!.,。，？！。、":
        text_pr += "."
    return lang, text_pr

def make_npz_prompt(name, uploaded_audio, recorded_audio, transcript_content):
    global model, text_collater, text_tokenizer, audio_tokenizer
    clear_prompts()
    audio_prompt = uploaded_audio if uploaded_audio is not None else recorded_audio
    
    if audio_prompt is None:
        return "Error: No audio provided", None
    
    # Handle different Gradio audio formats
    if isinstance(audio_prompt, str):
        # New format: file path string
        wav_pr, sr = torchaudio.load(audio_prompt)
    elif isinstance(audio_prompt, dict):
        # Dictionary format with 'name' or 'path' key
        audio_path = audio_prompt.get('name') or audio_prompt.get('path')
        wav_pr, sr = torchaudio.load(audio_path)
    elif isinstance(audio_prompt, tuple) and len(audio_prompt) == 2:
        # Old format: (sample_rate, waveform) tuple
        sr, wav_pr = audio_prompt
        if not isinstance(wav_pr, torch.Tensor):
            wav_pr = torch.FloatTensor(wav_pr)
    else:
        return "Error: Unsupported audio format", None
    
    # Ensure wav_pr is a tensor
    if not isinstance(wav_pr, torch.Tensor):
        wav_pr = torch.FloatTensor(wav_pr)
    
    # Normalize
    if wav_pr.abs().max() > 1:
        wav_pr /= wav_pr.abs().max()
    
    # Handle stereo to mono
    if wav_pr.ndim > 1 and wav_pr.size(0) == 2:
        wav_pr = wav_pr.mean(dim=0, keepdim=True)
    elif wav_pr.ndim > 1 and wav_pr.size(-1) == 2:
        wav_pr = wav_pr.mean(dim=-1, keepdim=True).squeeze(-1)
    
    # Ensure correct shape
    if wav_pr.ndim == 1:
        wav_pr = wav_pr.unsqueeze(0)
    
    assert wav_pr.ndim == 2 and wav_pr.size(0) == 1, f"Expected shape (1, length), got {wav_pr.shape}"
    
    if transcript_content == "":
        text_pr, lang_pr = make_prompt(name, wav_pr, sr, save=False)
    else:
        lang_pr = langid.classify(str(transcript_content))[0]
        if lang_pr != "en":
            return f"Error: Only English text is supported. Detected language: {lang_pr}", None
        lang_token = lang2token[lang_pr]
        text_pr = f"{lang_token}{str(transcript_content)}{lang_token}"
    
    # tokenize audio
    encoded_frames = tokenize_audio(audio_tokenizer, (wav_pr, sr))
    audio_tokens = encoded_frames[0][0].transpose(2, 1).cpu().numpy()

    # tokenize text
    phonemes, _ = text_tokenizer.tokenize(text=f"{text_pr}".strip())
    text_tokens, enroll_x_lens = text_collater([phonemes])

    message = f"Detected language: {lang_pr}\n Detected text {text_pr}\n"

    # save as npz file
    np.savez(os.path.join(tempfile.gettempdir(), f"{name}.npz"),
             audio_tokens=audio_tokens, text_tokens=text_tokens, lang_code=lang2code[lang_pr])
    return message, os.path.join(tempfile.gettempdir(), f"{name}.npz")


# At the top (around line 32), add:
# import soundfile as sf

# Then update make_prompt function (around line 299-340):
def make_prompt(name, wav, sr, save=True):
    global whisper_model
    whisper_model.to(device)
    
    if not isinstance(wav, torch.FloatTensor):
        wav = torch.tensor(wav)
    
    # Normalize to [-1, 1]
    if wav.abs().max() > 1:
        wav = wav / wav.abs().max()
    
    # Convert stereo to mono
    if wav.size(-1) == 2:
        wav = wav.mean(-1, keepdim=False)
    
    # Ensure shape is (1, N)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    
    assert wav.ndim == 2 and wav.size(0) == 1, f"Expected shape (1, N), got {wav.shape}"
    
    # Convert to numpy and ensure it's in the correct range
    data = wav.squeeze(0).cpu().numpy()
    
    # Clip to ensure values are strictly in [-1.0, 1.0]
    data = np.clip(data, -1.0, 1.0)
    
    # Convert to float32 (soundfile expects this)
    data = data.astype(np.float32)
    
    # Save with soundfile - explicitly set subtype to avoid casting issues
    # Convert float32 audio to int16 before saving as PCM_16
    data = np.clip(data, -1.0, 1.0)  # Ensure values are in [-1, 1]
    data_int16 = (data * 32767).astype(np.int16)
    sf.write(f"./prompts/{name}.wav", data_int16, sr, subtype='PCM_16')
    
    # Transcribe
    lang, text = transcribe_one(whisper_model, f"./prompts/{name}.wav")
    
    if lang != "en":
        raise ValueError(f"Error: Only English audio is supported. Detected language: {lang}")
    
    lang_token = lang2token[lang]
    text = lang_token + text + lang_token
    
    with open(f"./prompts/{name}.txt", 'w', encoding='utf-8') as f:
        f.write(text)
    
    if not save:
        os.remove(f"./prompts/{name}.wav")
        os.remove(f"./prompts/{name}.txt")

    whisper_model.cpu()
    torch.cuda.empty_cache()
    
    return text, lang



import torch
@torch.no_grad()
def infer_from_audio(text, language, accent, audio_prompt, record_audio_prompt, transcript_content):
    """
    Full replacement function for your original infer_from_audio.
    Returns (message, (sample_rate, numpy_audio))
    Prints evaluation results to terminal (WER, ASR text, similarity if enrollment prompt provided).
    """
    global model, text_collater, text_tokenizer, audio_tokenizer, vocos, device
    try:
        audio_prompt = audio_prompt if audio_prompt is not None else record_audio_prompt

        if audio_prompt is None:
            return "Error: No audio provided", None

        # Track a path if we can for similarity
        temp_wav_path = None

        # Normalize and load provided audio prompt into wav_pr, sr
        if isinstance(audio_prompt, str):
            # filepath
            wav_pr, sr = torchaudio.load(audio_prompt)
            temp_wav_path = audio_prompt
        elif isinstance(audio_prompt, dict):
            # gradio dict format: may have 'name' or 'path'
            audio_path = audio_prompt.get('name') or audio_prompt.get('path')
            if audio_path and os.path.isfile(audio_path):
                wav_pr, sr = torchaudio.load(audio_path)
                temp_wav_path = audio_path
            else:
                # maybe it's a base64 style or temporary data; try 'array' or fallback
                # If dict contains 'array' tuple like (sr, np_array)
                if 'array' in audio_prompt:
                    arr = audio_prompt['array']
                    if isinstance(arr, (list, tuple)) and len(arr) == 2:
                        sr, data = arr
                        wav_pr = torch.FloatTensor(data)
                    else:
                        raise ValueError("Unsupported audio_prompt dict array format")
                else:
                    raise ValueError("Unsupported audio_prompt dict format; missing file path")
        elif isinstance(audio_prompt, tuple) and len(audio_prompt) == 2:
            # old format: (sample_rate, waveform)
            sr, wav_pr = audio_prompt
            if not isinstance(wav_pr, torch.Tensor):
                wav_pr = torch.FloatTensor(wav_pr)
            # save to temp wav for diarization/similarity path
            if sf is not None:
                tmp_path = os.path.join(os.getcwd(), f"temp_enroll_{int(time.time())}.wav")
                # wav_pr shape may be (samples,) or (1, samples)
                arr = wav_pr.squeeze(0).cpu().numpy() if isinstance(wav_pr, torch.Tensor) else wav_pr
                sf.write(tmp_path, arr, sr)
                temp_wav_path = tmp_path
        else:
            return "Error: Unsupported audio format", None

        if temp_wav_path:
            is_valid, check_message = check_audio_for_generation(temp_wav_path)
            if not is_valid:
                return check_message, None
            else:
                print(check_message)
        # Ensure wav_pr is a tensor
        if not isinstance(wav_pr, torch.Tensor):
            wav_pr = torch.FloatTensor(wav_pr)

        # Normalize amplitude
        if wav_pr.abs().max() > 1:
            wav_pr = wav_pr / wav_pr.abs().max()

        # stereo -> mono handling
        if wav_pr.ndim > 1 and wav_pr.size(0) == 2:
            wav_pr = wav_pr.mean(dim=0, keepdim=True)
        elif wav_pr.ndim > 1 and wav_pr.size(-1) == 2:
            wav_pr = wav_pr.mean(dim=-1, keepdim=True).squeeze(-1)

        if wav_pr.ndim == 1:
            wav_pr = wav_pr.unsqueeze(0)

        assert wav_pr.ndim == 2 and wav_pr.size(0) == 1, f"Expected shape (1, length), got {wav_pr.shape}"

        # transcript content / prompt extraction
        if transcript_content == "":
            text_pr, lang_pr = make_prompt('dummy', wav_pr, sr, save=False)
        else:
            lang_pr = langid.classify(str(transcript_content))[0]
            if lang_pr != "en":
                return f"Error: Only English text is supported. Detected language: {lang_pr}", None
            lang_token = lang2token[lang_pr]
            text_pr = f"{lang_token}{str(transcript_content)}{lang_token}"

        if language == 'auto-detect':
            lang_token = lang2token[langid.classify(text)[0]]
        else:
            lang_token = langdropdown2token[language]
        lang = token2lang[lang_token]
        text = lang_token + text + lang_token

        # Move model to device
        model.to(device)

        # tokenize audio prompt (enrollment)
        encoded_frames = tokenize_audio(audio_tokenizer, (wav_pr, sr))
        audio_prompts = encoded_frames[0][0].transpose(2, 1).to(device)

        # tokenize text
        logging.info(f"synthesize text: {text}")
        phone_tokens, langs = text_tokenizer.tokenize(text=f"_{text}".strip())
        text_tokens, text_tokens_lens = text_collater([phone_tokens])

        enroll_x_lens = None
        if text_pr:
            text_prompts, _ = text_tokenizer.tokenize(text=f"{text_pr}".strip())
            text_prompts, enroll_x_lens = text_collater([text_prompts])

        text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
        text_tokens_lens += enroll_x_lens
        lang = lang if accent == "no-accent" else token2lang[langdropdown2token[accent]]

        # measure generation time
        gen_start = time.time()
        encoded_frames = model.inference(
            text_tokens.to(device),
            text_tokens_lens.to(device),
            audio_prompts,
            enroll_x_lens=enroll_x_lens,
            top_k=10,
            temperature=1,
            prompt_language=lang_pr,
            text_language=langs if accent == "no-accent" else lang,
            best_of=1,
        )
        gen_end = time.time()

        # Decode with Vocos
        frames = encoded_frames.permute(2,0,1)
        features = vocos.codes_to_features(frames)
        samples = vocos.decode(features, bandwidth_id=torch.tensor([2], device=device))

        # offload model
        model.to('cpu')
        torch.cuda.empty_cache()

        audio_numpy = samples.squeeze(0).cpu().numpy()
        # clip extremely long audio
        if len(audio_numpy) > 24000 * 300:
            logging.warning(f"Audio too long ({len(audio_numpy)/24000:.1f}s), clipping to 5 minutes")
            audio_numpy = audio_numpy[:24000 * 300]

        message = f"✓ Generated successfully!\nPrompt: {text_pr}\nSynthesized: {text}\nDuration: {len(audio_numpy)/24000:.1f}s"

        # Evaluate and print metrics to terminal
        try:
            ref_path = temp_wav_path
            if isinstance(audio_prompt, dict):
                ref_path = audio_prompt.get('name') or audio_prompt.get('path') or ref_path
            evaluate_and_print_metrics(audio_numpy, 24000, original_text=text, reference_prompt_wav_path=ref_path, save_prefix="infer_from_audio")
        except Exception as e:
            print(f"[Eval] evaluation failed: {e}")

        return message, (24000, audio_numpy)

    except Exception as e:
        logging.error(f"Error in infer_from_audio: {str(e)}")
        model.to('cpu')
        torch.cuda.empty_cache()
        return f"Error: {str(e)}", None

@torch.no_grad()
def infer_from_prompt(text, language, accent, preset_prompt, prompt_file):
    """
    Replacement infer_from_prompt (text -> synth using a prompt file or preset).
    Returns (message, (sample_rate, numpy_audio))
    """
    global model, text_collater, text_tokenizer, audio_tokenizer, vocos, device
    try:
        # check language token
        if language == 'auto-detect':
            lang_token = lang2token[langid.classify(text)[0]]
        else:
            lang_token = langdropdown2token[language]
        lang = token2lang[lang_token]
        text = lang_token + text + lang_token

        # load prompt data (either from uploaded .npz or preset)
        if prompt_file is not None:
            # prompt_file is a file-like object (gradio) - it likely has .name path
            try:
                prompt_data = np.load(prompt_file.name)
            except Exception:
                # try reading via file object
                prompt_file.seek(0)
                prompt_data = np.load(prompt_file)
        else:
            prompt_data = np.load(os.path.join("./presets/", f"{preset_prompt}.npz"))

        audio_prompts = prompt_data['audio_tokens']
        text_prompts = prompt_data['text_tokens']
        lang_pr = prompt_data['lang_code']
        lang_pr = code2lang[int(lang_pr)]

        # numpy -> tensor
        audio_prompts = torch.tensor(audio_prompts).type(torch.int32).to(device)
        text_prompts = torch.tensor(text_prompts).type(torch.int32)

        enroll_x_lens = text_prompts.shape[-1]
        logging.info(f"synthesize text: {text}")
        phone_tokens, langs = text_tokenizer.tokenize(text=f"_{text}".strip())
        text_tokens, text_tokens_lens = text_collater([phone_tokens])
        text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
        text_tokens_lens += enroll_x_lens

        # accent control
        lang = lang if accent == "no-accent" else token2lang[langdropdown2token[accent]]
        encoded_frames = model.inference(
            text_tokens.to(device),
            text_tokens_lens.to(device),
            audio_prompts,
            enroll_x_lens=enroll_x_lens,
            top_k=-100,
            temperature=1,
            prompt_language=lang_pr,
            text_language=langs if accent == "no-accent" else lang,
            best_of=5,
        )

        # Decode
        frames = encoded_frames.permute(2,0,1)
        features = vocos.codes_to_features(frames)
        samples = vocos.decode(features, bandwidth_id=torch.tensor([2], device=device))

        model.to('cpu')
        torch.cuda.empty_cache()

        audio_numpy = samples.squeeze(0).cpu().numpy()
        if len(audio_numpy) > 24000 * 300:
            logging.warning(f"Audio too long ({len(audio_numpy)/24000:.1f}s), clipping to 5 minutes")
            audio_numpy = audio_numpy[:24000 * 300]

        message = f"sythesized text: {text}"

        # Determine reference path for similarity (if prompt_file had a path)
        reference_path = None
        try:
            if prompt_file is not None and hasattr(prompt_file, 'name'):
                # If the user uploaded an npz produced from recorded audio, may not have wav
                reference_path = prompt_file.name
        except Exception:
            reference_path = None

        # evaluate and print metrics
        try:
            evaluate_and_print_metrics(audio_numpy, 24000, original_text=text, reference_prompt_wav_path=reference_path, save_prefix="infer_from_prompt")
        except Exception as e:
            print(f"[Eval] evaluation failed: {e}")

        return message, (24000, audio_numpy)

    except Exception as e:
        logging.error(f"Error in infer_from_prompt: {e}")
        model.to('cpu')
        torch.cuda.empty_cache()
        return f"Error: {e}", None

# ------------------------------
# 3) infer_long_text (full ready-to-replace)
# ------------------------------
from utils.sentence_cutter import split_text_into_sentences
@torch.no_grad()
def infer_long_text(text, preset_prompt, prompt=None, language='auto', accent='no-accent'):
    """
    Replacement infer_long_text with evaluation printing.
    Returns (message, (sample_rate, numpy_audio))
    """
    global model, audio_tokenizer, text_tokenizer, text_collater, vocos, device, NUM_QUANTIZERS
    model.to(device)
    mode = 'fixed-prompt'
    if (prompt is None or prompt == "") and preset_prompt == "":
        mode = 'sliding-window'
    sentences = split_text_into_sentences(text)

    # detect language
    if language == "auto-detect" or language == 'auto':
        language = langid.classify(text)[0]
    else:
        language = token2lang[langdropdown2token[language]]

    # prepare prompts
    if prompt is not None and prompt != "":
        # load prompt from uploaded file object
        if hasattr(prompt, "name"):
            prompt_data = np.load(prompt.name)
        else:
            prompt_data = np.load(prompt)
        audio_prompts = prompt_data['audio_tokens']
        text_prompts = prompt_data['text_tokens']
        lang_pr = prompt_data['lang_code']
        lang_pr = code2lang[int(lang_pr)]
        audio_prompts = torch.tensor(audio_prompts).type(torch.int32).to(device)
        text_prompts = torch.tensor(text_prompts).type(torch.int32)
    elif preset_prompt is not None and preset_prompt != "":
        prompt_data = np.load(os.path.join("./presets/", f"{preset_prompt}.npz"))
        audio_prompts = prompt_data['audio_tokens']
        text_prompts = prompt_data['text_tokens']
        lang_pr = prompt_data['lang_code']
        lang_pr = code2lang[int(lang_pr)]
        audio_prompts = torch.tensor(audio_prompts).type(torch.int32).to(device)
        text_prompts = torch.tensor(text_prompts).type(torch.int32)
    else:
        audio_prompts = torch.zeros([1, 0, NUM_QUANTIZERS]).type(torch.int32).to(device)
        text_prompts = torch.zeros([1, 0]).type(torch.int32)
        lang_pr = language if language != 'mix' else 'en'

    if mode == 'fixed-prompt':
        complete_tokens = torch.zeros([1, NUM_QUANTIZERS, 0]).type(torch.LongTensor).to(device)
        for text_sent in sentences:
            text_sent = text_sent.replace("\n", "").strip(" ")
            if text_sent == "":
                continue
            lang_token = lang2token[language]
            lang = token2lang[lang_token]
            text_in = lang_token + text_sent + lang_token

            enroll_x_lens = text_prompts.shape[-1]
            logging.info(f"synthesize text: {text_in}")
            phone_tokens, langs = text_tokenizer.tokenize(text=f"_{text_in}".strip())
            text_tokens, text_tokens_lens = text_collater([phone_tokens])
            text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
            text_tokens_lens += enroll_x_lens
            lang_use = lang if accent == "no-accent" else token2lang[langdropdown2token[accent]]
            encoded_frames = model.inference(
                text_tokens.to(device),
                text_tokens_lens.to(device),
                audio_prompts,
                enroll_x_lens=enroll_x_lens,
                top_k=-100,
                temperature=1,
                prompt_language=lang_pr,
                text_language=langs if accent == "no-accent" else lang_use,
                best_of=5,
            )
            complete_tokens = torch.cat([complete_tokens, encoded_frames.transpose(2, 1)], dim=-1)
        # Decode all frames
        frames = complete_tokens.permute(1, 0, 2)
        features = vocos.codes_to_features(frames)
        samples = vocos.decode(features, bandwidth_id=torch.tensor([2], device=device))

        model.to('cpu')
        message = f"Cut into {len(sentences)} sentences"
        audio_numpy = samples.squeeze(0).cpu().numpy()

        if len(audio_numpy) > 24000 * 300:
            logging.warning(f"Audio too long ({len(audio_numpy)/24000:.1f}s), clipping to 5 minutes")
            audio_numpy = audio_numpy[:24000 * 300]

        # evaluation: we won't have a simple enrollment file for long_text usually
        try:
            evaluate_and_print_metrics(audio_numpy, 24000, original_text=text, reference_prompt_wav_path=None, save_prefix="infer_long")
        except Exception as e:
            print(f"[Eval] evaluation failed: {e}")

        return message, (24000, audio_numpy)

    elif mode == "sliding-window":
        complete_tokens = torch.zeros([1, NUM_QUANTIZERS, 0]).type(torch.LongTensor).to(device)
        original_audio_prompts = audio_prompts
        original_text_prompts = text_prompts
        for text_sent in sentences:
            text_sent = text_sent.replace("\n", "").strip(" ")
            if text_sent == "":
                continue
            lang_token = lang2token[language]
            lang = token2lang[lang_token]
            text_in = lang_token + text_sent + lang_token

            enroll_x_lens = text_prompts.shape[-1]
            logging.info(f"synthesize text: {text_in}")
            phone_tokens, langs = text_tokenizer.tokenize(text=f"_{text_in}".strip())
            text_tokens, text_tokens_lens = text_collater([phone_tokens])
            text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
            text_tokens_lens += enroll_x_lens
            lang_use = lang if accent == "no-accent" else token2lang[langdropdown2token[accent]]
            encoded_frames = model.inference(
                text_tokens.to(device),
                text_tokens_lens.to(device),
                audio_prompts,
                enroll_x_lens=enroll_x_lens,
                top_k=-100,
                temperature=1,
                prompt_language=lang_pr,
                text_language=langs if accent == "no-accent" else lang_use,
                best_of=5,
            )
            complete_tokens = torch.cat([complete_tokens, encoded_frames.transpose(2, 1)], dim=-1)
            if torch.rand(1) < 1.0:
                audio_prompts = encoded_frames[:, :, -NUM_QUANTIZERS:]
                text_prompts = text_tokens[:, enroll_x_lens:]
            else:
                audio_prompts = original_audio_prompts
                text_prompts = original_text_prompts
        frames = complete_tokens.permute(1, 0, 2)
        features = vocos.codes_to_features(frames)
        samples = vocos.decode(features, bandwidth_id=torch.tensor([2], device=device))

        model.to('cpu')
        message = f"Cut into {len(sentences)} sentences"
        audio_numpy = samples.squeeze(0).cpu().numpy()

        if len(audio_numpy) > 24000 * 300:
            logging.warning(f"Audio too long ({len(audio_numpy)/24000:.1f}s), clipping to 5 minutes")
            audio_numpy = audio_numpy[:24000 * 300]

        try:
            evaluate_and_print_metrics(audio_numpy, 24000, original_text=text, reference_prompt_wav_path=None, save_prefix="infer_long")
        except Exception as e:
            print(f"[Eval] evaluation failed: {e}")

        return message, (24000, audio_numpy)
    else:
        raise ValueError(f"No such mode {mode}")


def main():
    # Load speaker diarization model
    try:
        load_speaker_diarization()
    except Exception as e:
        print(f"Failed to load speaker diarization: {e}")
        print("Continuing without speaker diarization...")
    
    app = gr.Blocks(title="VALL-E X")
    with app:
        gr.Markdown(top_md)
        with gr.Tab("Infer from audio"):
            gr.Markdown(infer_from_audio_md)
            with gr.Row():
                with gr.Column():
                    textbox = gr.TextArea(label="Text",
                                          placeholder="Type your sentence here",
                                          value="Welcome back, Master. What can I do for you today?", elem_id=f"tts-input")
                    language_dropdown = gr.Dropdown(choices=['English'], value='English', label='language')
                    accent_dropdown = gr.Dropdown(choices=['English'], value='English', label='accent')
                    textbox_transcript = gr.TextArea(label="Transcript",
                                          placeholder="Write transcript here. (leave empty to use whisper)",
                                          value="", elem_id=f"prompt-name")
                    upload_audio_prompt = gr.Audio(label='Upload audio prompt', interactive=True)
                    # Microphone button for recording
                    mic_audio_prompt = gr.Audio(source="microphone", label="Record with Microphone", interactive=True)
                with gr.Column():
                    text_output = gr.Textbox(label="Message")
                    audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                    btn = gr.Button("Generate!")
                    # Use either uploaded or mic audio
                    btn.click(
                        infer_from_audio,
                        inputs=[textbox, language_dropdown, accent_dropdown, upload_audio_prompt, mic_audio_prompt, textbox_transcript],
                        outputs=[text_output, audio_output]
                    )
                    textbox_mp = gr.TextArea(label="Prompt name",
                                          placeholder="Name your prompt here",
                                          value="prompt_1", elem_id=f"prompt-name")
                    btn_mp = gr.Button("Make prompt!")
                    prompt_output = gr.File(interactive=False)
                    btn_mp.click(
                        make_npz_prompt,
                        inputs=[textbox_mp, upload_audio_prompt, mic_audio_prompt, textbox_transcript],
                        outputs=[text_output, prompt_output]
                    )
            gr.Examples(examples=infer_from_audio_examples,
                        inputs=[textbox, language_dropdown, accent_dropdown, upload_audio_prompt, mic_audio_prompt, textbox_transcript],
                        outputs=[text_output, audio_output],
                        fn=infer_from_audio,
                        cache_examples=False,)
        with gr.Tab("Make prompt"):
            gr.Markdown(make_prompt_md)
            with gr.Row():
                with gr.Column():
                    textbox2 = gr.TextArea(label="Prompt name",
                                          placeholder="Name your prompt here",
                                          value="prompt_1", elem_id=f"prompt-name")
                    # 添加选择语言和输入台本的地方
                    textbox_transcript2 = gr.TextArea(label="Transcript",
                                          placeholder="Write transcript here. (leave empty to use whisper)",
                                          value="", elem_id=f"prompt-name")
                    upload_audio_prompt_2 = gr.Audio(label='uploaded audio prompt', interactive=True)
                    record_audio_prompt_2 = gr.Audio(label='recorded audio prompt', interactive=True)
                with gr.Column():
                    text_output_2 = gr.Textbox(label="Message")
                    prompt_output_2 = gr.File(interactive=False)
                    btn_2 = gr.Button("Make!")
                    btn_2.click(make_npz_prompt,
                              inputs=[textbox2, upload_audio_prompt_2, record_audio_prompt_2, textbox_transcript2],
                              outputs=[text_output_2, prompt_output_2])
            gr.Examples(examples=make_npz_prompt_examples,
                        inputs=[textbox2, upload_audio_prompt_2, record_audio_prompt_2, textbox_transcript2],
                        outputs=[text_output_2, prompt_output_2],
                        fn=make_npz_prompt,
                        cache_examples=False,)
        with gr.Tab("Infer from prompt"):
            gr.Markdown(infer_from_prompt_md)
            with gr.Row():
                with gr.Column():
                    textbox_3 = gr.TextArea(label="Text",
                                          placeholder="Type your sentence here",
                                          value="Welcome back, Master. What can I do for you today?", elem_id=f"tts-input")
                    language_dropdown_3 = gr.Dropdown(choices=['English'], value='English',
                                                    label='language')
                    accent_dropdown_3 = gr.Dropdown(choices=['English'], value='English',
                                                  label='accent')
                    preset_dropdown_3 = gr.Dropdown(choices=preset_list, value=None, label='Voice preset')
                    prompt_file = gr.File(file_count='single', file_types=['.npz'], interactive=True)
                with gr.Column():
                    text_output_3 = gr.Textbox(label="Message")
                    audio_output_3 = gr.Audio(label="Output Audio", elem_id="tts-audio")
                    btn_3 = gr.Button("Generate!")
                    btn_3.click(infer_from_prompt,
                              inputs=[textbox_3, language_dropdown_3, accent_dropdown_3, preset_dropdown_3, prompt_file],
                              outputs=[text_output_3, audio_output_3])
            gr.Examples(examples=infer_from_prompt_examples,
                        inputs=[textbox_3, language_dropdown_3, accent_dropdown_3, preset_dropdown_3, prompt_file],
                        outputs=[text_output_3, audio_output_3],
                        fn=infer_from_prompt,
                        cache_examples=False,)
        with gr.Tab("Infer long text"):
            gr.Markdown("This is a long text generation demo. You can use this to generate long audio. ")
            with gr.Row():
                with gr.Column():
                    textbox_4 = gr.TextArea(label="Text",
                                          placeholder="Type your sentence here",
                                          value=long_text_example, elem_id=f"tts-input")
                    language_dropdown_4 = gr.Dropdown(choices=['English'], value='English',
                                                    label='language')
                    accent_dropdown_4 = gr.Dropdown(choices=['English'], value='English',
                                                    label='accent')
                    preset_dropdown_4 = gr.Dropdown(choices=preset_list, value=None, label='Voice preset')
                    prompt_file_4 = gr.File(file_count='single', file_types=['.npz'], interactive=True)
                with gr.Column():
                    text_output_4 = gr.TextArea(label="Message")
                    audio_output_4 = gr.Audio(label="Output Audio", elem_id="tts-audio")
                    btn_4 = gr.Button("Generate!")
                    btn_4.click(infer_long_text,
                              inputs=[textbox_4, preset_dropdown_4, prompt_file_4, language_dropdown_4, accent_dropdown_4],
                              outputs=[text_output_4, audio_output_4])

    try:
        webbrowser.open("http://127.0.0.1:7860")
        app.launch(inbrowser=True)
    except Exception as e:
        print(f"Error launching app: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
