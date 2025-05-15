"""
Audio Feature Extraction Module for Alzheimer's Risk Prediction (Modular)
"""
import os
import numpy as np
import librosa
import parselmouth
from textblob import TextBlob
import webrtcvad
import soundfile as sf
import scipy.signal
# Add Google Cloud Speech imports
try:
    from google.cloud import speech
    import io
except ImportError:
    speech = None
    io = None
# client = speech.SpeechClient()

# Helper: Transcribe audio using Google Speech-to-Text API
# Requires GOOGLE_APPLICATION_CREDENTIALS environment variable set to your service account JSON

def transcribe_audio(file_path):
    if speech is None or io is None:
        raise ImportError("Google Cloud Speech libraries not installed. Please install google-cloud-speech and set up authentication.")
    try:

        # Assume file_path is the local path, but we want to use the GCS URI instead
        # Construct the GCS URI based on the known bucket and file structure
        # Example: file_path = 'D:\Docathon\dataset\alzehiemer\audio1.wav'
        # GCS URI: gs://transcript_speech/alzehiemer/audio1.wav
        # import ntpath
        # import re
        from google.oauth2 import service_account
        credentials_path = r"C:\Users\Durgesh Babu\Downloads\cloud-speech-459509-1648e5d1c5c1.json"
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        client = speech.SpeechClient(credentials=credentials)
        # # Extract class and filename from file_path
        # match = re.search(r"dataset[\\/](.*?)[\\/](.*\.wav)$", file_path, re.IGNORECASE)
        # if not match:
        #     print(f"Could not determine GCS URI for file: {file_path}")
        #     return "(GCS URI not found)"
        # cls, fname = match.group(1), match.group(2)
        # bucket_name = "transcript_speech"
        # gcs_uri = f"gs://{bucket_name}/dataset/{cls}/{fname}"
        # # Always use async recognition for GCS files
        # audio = speech.RecognitionAudio(uri=gcs_uri)

        # Use local file path directly
        with io.open(file_path, "rb") as audio_file:
            content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code="en-US",
            enable_automatic_punctuation=True
        )
        operation = client.long_running_recognize(config=config, audio=audio)
        response = operation.result(timeout=600)
        transcript = " ".join([result.alternatives[0].transcript for result in response.results])
        if not transcript.strip():
            return "(No speech detected)"
        return transcript
    except Exception as e:
        print(f"Google Speech-to-Text API error: {e}")
        return "(Transcription failed)"

def detect_pauses(y, sr, frame_duration=30):
    vad = webrtcvad.Vad(2)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y_pcm = np.clip(y, -1, 1)
    y_pcm = (y_pcm * 32767).astype(np.int16)
    frames = int(sr * frame_duration / 1000)
    num_frames = len(y_pcm) // frames
    pause_count = 0
    pause_durations = []
    current_pause = 0
    for i in range(num_frames):
        frame = y_pcm[i*frames:(i+1)*frames]
        if len(frame) < frames:
            continue
        try:
            pcm = frame.tobytes()
            if sr not in [8000, 16000, 32000, 48000]:
                continue
            if frames not in [int(sr*0.01), int(sr*0.02), int(sr*0.03)]:
                continue
            if not vad.is_speech(pcm, sr):
                current_pause += 1
            else:
                if current_pause > 0:
                    pause_durations.append(current_pause * frame_duration / 1000)
                    pause_count += 1
                    current_pause = 0
        except Exception as e:
            continue
    avg_pause_duration = np.mean(pause_durations) if pause_durations else 0
    max_pause_duration = np.max(pause_durations) if pause_durations else 0
    return {'pause_count': pause_count, 'avg_pause_duration': avg_pause_duration, 'max_pause_duration': max_pause_duration}

def detect_stammering(transcript):
    import re
    words = transcript.lower().split()
    stammer_count = 0
    stammer_sounds = ['uh', 'um', 'er', 'hmm', 'ah', 'eh', 'mm', 'uhh', 'umm']
    for i, w in enumerate(words):
        # Consecutive repeated short words or filler words
        if i > 0 and words[i] == words[i-1] and (len(w) <= 3 or w in stammer_sounds):
            stammer_count += 1
        # Partial word repetition (e.g., "I-I-I want")
        if re.match(r'^(\w-)+\w+$', w):
            stammer_count += 1
        # Common stammering sounds
        if w in stammer_sounds:
            stammer_count += 1
        # Repeated first letter (e.g., "b-b-but")
        if len(w) > 2 and w[1] == '-' and w[0] == w[2]:
            stammer_count += 1
        # Elongated sounds (e.g., "soooo")
        if re.match(r'^(\w)\1{2,}$', w):
            stammer_count += 1
    return stammer_count

def detect_repeated_words(transcript):
    import re
    words = re.findall(r'\b\w+\b', transcript.lower())
    repeated = set()
    for i in range(1, len(words)):
        if words[i] == words[i-1]:
            repeated.add(words[i])
    word_counts = {}
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1
    for w, c in word_counts.items():
        if c > 1:
            repeated.add(w)
    # Add context: count repeated phrases (bigrams)
    bigrams = zip(words, words[1:])
    phrase_counts = {}
    for bg in bigrams:
        phrase = ' '.join(bg)
        phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
    repeated_phrases = [p for p, c in phrase_counts.items() if c > 1]
    return len(repeated) + len(repeated_phrases)

def detect_forgetfulness(transcript):
    fillers = ['um', 'uh', 'er', 'hmm', 'like', 'you know', 'forgot', 'i don\'t remember', 'i can\'t recall', 'what was it', 'let me think']
    count = sum(transcript.lower().count(f) for f in fillers)
    # Add context: count explicit forgetfulness phrases
    import re
    forget_patterns = [r"i (don'?t|cannot|can't) remember", r"i (forgot|forget)", r"i (don'?t|cannot|can't) recall", r"what was it", r"let me think"]
    for pat in forget_patterns:
        count += len(re.findall(pat, transcript.lower()))
    return count

def sentence_complexity(transcript):
    blob = TextBlob(transcript)
    sentences = blob.sentences
    avg_len = np.mean([len(s.words) for s in sentences]) if sentences else 0
    clause_count = sum(s.raw.count(',') + s.raw.lower().count(' and ') + s.raw.lower().count(' but ') for s in sentences)
    # Add: count subordinate clauses ("because", "although", etc.)
    sub_clauses = ['because', 'although', 'since', 'unless', 'while', 'whereas', 'though']
    sub_count = sum(sum(s.raw.lower().count(sc) for sc in sub_clauses) for s in sentences)
    return avg_len + 0.5 * clause_count + 0.7 * sub_count

def extract_vocal_emotion(file_path):
    snd = parselmouth.Sound(file_path)
    pitch = snd.to_pitch()
    mean_pitch = pitch.selected_array['frequency'][pitch.selected_array['frequency'] > 0].mean() if np.any(pitch.selected_array['frequency'] > 0) else 0
    intensity = snd.to_intensity()
    mean_intensity = intensity.values.mean() if intensity.values.size > 0 else 0
    return mean_pitch, mean_intensity

def extract_audio_features(file_path):
    try:
        y, sr = sf.read(file_path)
        if y.ndim > 1:
            y = y.mean(axis=1)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio with soundfile: {e}. Please ensure the file is a valid WAV/FLAC/OGG or convert it to WAV.")
    transcript = transcribe_audio(file_path)
    features = {}
    pause_feats = detect_pauses(y, sr)
    features['pause_count'] = pause_feats['pause_count']
    features['avg_pause_duration'] = pause_feats['avg_pause_duration']
    features['max_pause_duration'] = pause_feats['max_pause_duration']
    features['stammer_count'] = detect_stammering(transcript)
    features['repeated_words'] = detect_repeated_words(transcript)
    features['forgetfulness'] = detect_forgetfulness(transcript)
    features['sentence_complexity'] = sentence_complexity(transcript)
    pitch, intensity = extract_vocal_emotion(file_path)
    features['mean_pitch'] = pitch
    features['mean_intensity'] = intensity
    num_words = len(transcript.split())
    duration_sec = len(y) / sr if sr else 1
    features['speech_rate'] = num_words / duration_sec if duration_sec > 0 else 0
    # Add: articulation rate (excluding pauses)
    total_pause = features['avg_pause_duration'] * features['pause_count']
    speech_time = duration_sec - total_pause
    features['articulation_rate'] = num_words / speech_time if speech_time > 0 else 0
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    features['mfccs_mean'] = mfccs_mean

    # Extract Delta MFCCs
    delta_mfccs = librosa.feature.delta(mfccs)
    delta_mfccs_mean = np.mean(delta_mfccs, axis=1)
    features['delta_mfccs_mean'] = delta_mfccs_mean

    # Extract Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    features['chroma_mean'] = chroma_mean

    # Extract Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    features['contrast_mean'] = contrast_mean

    # Extract Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    features['zcr_mean'] = zcr_mean

    return features

if __name__ == "__main__":
    data_dir = 'D:\Docathon\dataset'
    classes = ['alzehiemer', 'normal']
    results = []
    for label, cls in enumerate(classes):
        folder = os.path.join(data_dir, cls)
        for fname in os.listdir(folder):
            if fname.endswith('.wav'):
                fpath = os.path.join(folder, fname)
                feats = extract_audio_features(fpath)
                feats['label'] = cls
                feats['file'] = fname
                results.append(feats)
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('audio_features_extracted.csv', index=False)
    print('Feature extraction complete. Saved to audio_features_extracted.csv')