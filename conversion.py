import os
import soundfile as sf

def convert_to_mono(input_path):
    data, samplerate = sf.read(input_path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)  # Convert to mono
        sf.write(input_path, data, samplerate)  # Overwrite original file
        print(f"Converted to mono: {input_path}")
    else:
        print(f"Already mono: {input_path}")

base_dir = 'dataset'
for subfolder in ['alzehiemer', 'normal']:
    folder_path = os.path.join(base_dir, subfolder)
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            convert_to_mono(os.path.join(folder_path, filename))
