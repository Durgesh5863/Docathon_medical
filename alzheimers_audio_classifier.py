import os
import numpy as np
import pandas as pd
import librosa


data_dir = 'dataset'
classes = ['alzehiemer', 'normal']

def extract_features(file_path, n_mfcc=13):
    try:
        y, sr = librosa.load(file_path, sr=None)
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        # Delta MFCCs
        delta_mfccs = librosa.feature.delta(mfccs)
        delta_mfccs_mean = np.mean(delta_mfccs, axis=1)
        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        # Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        # Concatenate all features
        features = np.concatenate([
            mfccs_mean,
            delta_mfccs_mean,
            chroma_mean,
            contrast_mean,
            [zcr_mean]
        ])
        return features
    except Exception as e:
        print(f'Error processing {file_path}: {e}')
        return None

features = []
labels = []
file_names = []
for label, cls in enumerate(classes):
    folder = os.path.join(data_dir, cls)
    for fname in os.listdir(folder):
        if fname.endswith('.wav'):
            fpath = os.path.join(folder, fname)
            feats = extract_features(fpath)
            if feats is not None:
                features.append(feats)
                labels.append(label)
                file_names.append(fname)

# Define feature names
feature_names = [
    *(f"mfcc_{i+1}" for i in range(13)),
    *(f"delta_mfcc_{i+1}" for i in range(13)),
    *(f"chroma_{i+1}" for i in range(12)),
    *(f"contrast_{i+1}" for i in range(7)),
    "zcr"
]

# Convert features to DataFrame and save to CSV
features_df = pd.DataFrame(features, columns = feature_names)
features_df['file'] = file_names
features_df['label'] = labels
features_df['label'] = features_df['label'].map({0: 'alzehiemer', 1: 'normal'})
features_df.to_csv('audio_features.csv', index=False)
print("Saved features and file names to audio_features.csv")
