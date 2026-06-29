import librosa
import numpy as np
from scipy import signal

def get_adaptive_spectrogram(file_path, target_shape=(128, 216)):
    """
    High-Pass Filter spectrogram extractor.
    Steps:
      1. Load and clip audio to 5 seconds
      2. Apply high-pass filter (cutoff 2000 Hz) to expose LSB noise
      3. Generate Mel spectrogram
      4. Pad or truncate to fixed width
    """
    try:
        # Load audio, preserving original sample rate (typically 44.1 kHz)
        y, sr = librosa.load(file_path, sr=None, duration=5.0)

        # If audio is shorter than 5s, tile it to reach the required length
        required_samples = int(5 * sr)
        if len(y) < required_samples:
            y = np.tile(y, int(np.ceil(required_samples / len(y))))
        y = y[:required_samples]

        # Apply high-pass filter to remove frequencies below 2000 Hz,
        # which helps reveal LSB steganography noise in the upper spectrum
        sos = signal.butter(10, 2000, 'hp', fs=sr, output='sos')
        y = signal.sosfilt(sos, y)

        # Compute Mel spectrogram (height = n_mels = 128)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_shape[0], fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)

        # Fix time-axis width by truncating or zero-padding
        fixed_length = target_shape[1]  # typically 216 frames for 5s audio
        if S_db.shape[1] > fixed_length:
            S_db = S_db[:, :fixed_length]
        else:
            pad_width = fixed_length - S_db.shape[1]
            S_db = np.pad(S_db, pad_width=((0, 0), (0, pad_width)), mode='constant')

        # Add channel dimension for CNN input: (H, W) -> (H, W, 1)
        S_db = S_db[..., np.newaxis]

        return S_db

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def get_statistical_features(file_path):
    """
    Extract hand-crafted statistical features for traditional ML classifiers.
    Concatenates MFCC, chroma, mel, spectral contrast, and tonnetz features.
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
        return np.concatenate([mfcc, chroma, mel, contrast, tonnetz])
    except Exception as e:
        return None