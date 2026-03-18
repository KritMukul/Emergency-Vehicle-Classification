import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.fftpack import dct

#  PATHS 
BASE_DIR = "data"
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
TRAFFIC_DIR = os.path.join(BASE_DIR, "traffic")

os.makedirs("features_csv", exist_ok=True)
os.makedirs("saved_features", exist_ok=True)

#  SETTINGS 
SAMPLE_RATE = 22050
DURATION = 3  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

#  LOAD & TRIM & PAD 
def load_audio(path):
    """Load an audio file, trim silence, and pad/crop to 3 seconds."""
    try:
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        y, _ = librosa.effects.trim(y)
        
        if len(y) > SAMPLES_PER_TRACK:
            # Find the loudest 3-second window
            max_energy = 0
            best_start = 0
            for i in range(len(y) - SAMPLES_PER_TRACK + 1):
                window = y[i:i + SAMPLES_PER_TRACK]
                energy = np.sum(window ** 2)
                if energy > max_energy:
                    max_energy = energy
                    best_start = i
            y = y[best_start:best_start + SAMPLES_PER_TRACK]
        elif len(y) < SAMPLES_PER_TRACK:
            y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)))
        return y, sr
    except Exception as e:
        print(f"[ERROR] Could not load {path}: {e}")
        return None, None

#  FEATURE EXTRACTION 
def extract_mfcc(y, sr):
    return np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

def extract_lfcc(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    log_S = np.log(S + 1e-10)
    lfcc = dct(log_S, type=2, axis=0, norm='ortho')[:13]
    return np.mean(lfcc, axis=1)

def extract_cfcc(y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20)
    logmel = np.log(mel + 1e-10)
    cfcc = dct(logmel, type=2, axis=0, norm='ortho')[:13]
    return np.mean(cfcc, axis=1)

def extract_sfcc(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    logS = np.log(S + 1e-10)
    sfcc = dct(logS, type=3, axis=0, norm='ortho')[:13]
    return np.mean(sfcc, axis=1)

FEATURES = {
    "mfcc": extract_mfcc,
    "lfcc": extract_lfcc,
    "cfcc": extract_cfcc,
    "sfcc": extract_sfcc
}

#  LABEL RULE 
def get_label(path):
    path_lower = path.lower()
    base = os.path.basename(path)
    if "dataset" in path_lower:
        return 1
    elif "traffic" in path_lower:
        return 0
    elif "audio" in path_lower:
        parts = base.split("-")
        try:
            class_num = int(parts[1])
            return 1 if class_num == 8 else 0
        except:
            return 0
    return 0

#  MAIN PIPELINE 
def process_and_extract():
    all_files = []
    for folder in [AUDIO_DIR, DATASET_DIR, TRAFFIC_DIR]:
        if os.path.exists(folder):
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(".wav"):
                        all_files.append(os.path.join(root, file))

    all_files.sort()
    print(f" Found {len(all_files)} audio files.\n")

    mfcc_data, lfcc_data, cfcc_data, sfcc_data, labels, names = [], [], [], [], [], []

    for file_path in tqdm(all_files, desc="Extracting features"):
        y, sr = load_audio(file_path)
        if y is None:
            continue

        names.append(os.path.basename(file_path))
        labels.append(get_label(file_path))
        mfcc_data.append(extract_mfcc(y, sr))
        lfcc_data.append(extract_lfcc(y, sr))
        cfcc_data.append(extract_cfcc(y, sr))
        sfcc_data.append(extract_sfcc(y, sr))

    #  SAVE AS NUMPY FILES 
    np.save("saved_features/X_mfcc.npy", np.array(mfcc_data))
    np.save("saved_features/X_lfcc.npy", np.array(lfcc_data))
    np.save("saved_features/X_cfcc.npy", np.array(cfcc_data))
    np.save("saved_features/X_sfcc.npy", np.array(sfcc_data))
    np.save("saved_features/y_labels.npy", np.array(labels))
    print(" Saved .npy feature arrays in /saved_features")

    #  SAVE COMBINED CSV 
    df = pd.DataFrame({"filename": names, "label": labels})
    for name, data in {
        "mfcc": mfcc_data,
        "lfcc": lfcc_data,
        "cfcc": cfcc_data,
        "sfcc": sfcc_data
    }.items():
        temp = pd.DataFrame(data, columns=[f"{name}_{i+1}" for i in range(13)])
        df = pd.concat([df, temp], axis=1)

    df.to_csv("features_csv/all_features.csv", index=False)
    print(" Saved combined CSV: features_csv/all_features.csv")

if __name__ == "__main__":
    process_and_extract()
