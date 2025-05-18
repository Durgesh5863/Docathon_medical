import pandas as pd

# Replace with your actual file paths
csv1 = 'audio_features.csv'  # low-level features (MFCC, chroma, etc.)
csv2 = 'audio_features_extracted.csv'  # high-level features (pause, stammer, etc.)

# Load both CSVs
features1 = pd.read_csv(csv1)
features2 = pd.read_csv(csv2)

# Merge on 'file' and 'label' columns
merged = pd.merge(features1, features2, on=['file', 'label'], how='inner')

# print(merged.columns.tolist())
# Save the merged DataFrame
merged.to_csv('merged_features.csv', index=False)

print('Merged CSV saved as Extracted_data/merged_features.csv')