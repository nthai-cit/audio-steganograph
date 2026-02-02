import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split

def get_adaptive_spectrogram(file_path, target_sr=16000, duration=3.0):
    try:
        y, sr = librosa.load(file_path, sr=target_sr)
        target_len = int(target_sr * duration)
        if len(y) > target_len:
            y = y[:target_len]
        else:
            y = np.pad(y, (0, target_len - len(y)), 'constant')
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=2048, hop_length=512)
        S_dB = librosa.power_to_db(S, ref=np.max)
        S_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-6)
        return S_norm
    except Exception as e:
        print(f"[Warn] Error processing {os.path.basename(file_path)}: {e}")
        return None

def get_statistical_features(file_path):
    return None

class StegoDataset:
    def __init__(self, cover_dir, stego_dir, model_type='cnn', cache_path=None):
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        self.model_type = model_type
        self.cache_path = cache_path
        if self.cache_path and os.path.exists(self.cache_path):
            print(f"[Cache] Found cache file: {self.cache_path}")
        else:
            self.files = []
            self.labels = []
            self._scan_files()

    def _scan_files(self):
        directories = [(self.cover_dir, 0), (self.stego_dir, 1)]
        print(f"[Dataset] Scanning files...")
        total_found = 0
        for folder_path, label in directories:
            if not os.path.exists(folder_path):
                print(f"[Warning] Directory not found: {folder_path}")
                continue
            count = 0
            for root, _, filenames in os.walk(folder_path):
                for f in filenames:
                    if f.lower().endswith(('.wav', '.flac', '.mp3')):
                        self.files.append(os.path.join(root, f))
                        self.labels.append(label)
                        count += 1
            print(f" -> Found {count} files in {folder_path}")
            total_found += count
        if total_found == 0:
            raise ValueError("[ERROR] No audio files found in directories!")

    def load_data(self):
        X, y = None, None
        if self.cache_path and os.path.exists(self.cache_path):
            print(f"[Fast Load] Loading from cache...")
            try:
                data = np.load(self.cache_path)
                keys = list(data.keys())
                if 'X' in keys and 'y' in keys:
                    X, y = data['X'], data['y']
                else:
                    X = data[keys[0]]
                    y = data[keys[1]]
                print(f" -> Loaded: X={X.shape}, y={y.shape}")
            except Exception as e:
                print(f"[ERROR] Cache load failed: {e}. Rescanning.")

        if X is None:
            if self.cache_path: 
                X, y = self._process_in_chunks()
            else:
                X, y = self._process_all_memory()

        if self.model_type in ['cnn', 'bilstm']:
            if X.ndim == 3:
                print(f"[Auto-Fix] Reshaping 3D -> 4D channel: {X.shape} -> {(X.shape + (1,))}")
                X = X[..., np.newaxis]
        return X, y

    def _process_in_chunks(self, chunk_size=500):
        total_files = len(self.files)
        base_dir = os.path.dirname(self.cache_path)
        base_name = os.path.splitext(os.path.basename(self.cache_path))[0]
        temp_dir = os.path.join(base_dir, f"{base_name}_parts")
        os.makedirs(temp_dir, exist_ok=True)

        print(f"\n[Safe Mode] Processing in chunks...")
        num_chunks = (total_files + chunk_size - 1) // chunk_size
        X_parts, y_parts = [], []

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_files)
            part_file = os.path.join(temp_dir, f"part_{i}.npz")
            
            if os.path.exists(part_file):
                print(f"   [Skip] Part {i+1}/{num_chunks} exists.")
                try:
                    data = np.load(part_file)
                    X_parts.append(data['X'])
                    y_parts.append(data['y'])
                    continue
                except:
                    pass

            print(f"   [Work] Part {i+1}/{num_chunks}...")
            X_curr, y_curr = [], []
            for idx in range(start_idx, end_idx):
                f_path = self.files[idx]
                try:
                    if self.model_type in ['cnn', 'bilstm']:
                        feat = get_adaptive_spectrogram(f_path)
                    else:
                        feat = get_statistical_features(f_path)
                    if feat is not None:
                        X_curr.append(feat)
                        y_curr.append(self.labels[idx])
                except:
                    pass
            
            X_curr = np.array(X_curr)
            y_curr = np.array(y_curr)
            if len(X_curr) > 0:
                np.savez_compressed(part_file, X=X_curr, y=y_curr)
                X_parts.append(X_curr)
                y_parts.append(y_curr)

        if not X_parts:
             raise ValueError("[ERROR] No data extracted from files.")

        X_final = np.concatenate(X_parts, axis=0)
        y_final = np.concatenate(y_parts, axis=0)
        np.savez_compressed(self.cache_path, X=X_final, y=y_final)
        return X_final, y_final

    def _process_all_memory(self):
        X, y = [], []
        for i, f in enumerate(self.files):
            try:
                if self.model_type in ['cnn', 'bilstm']:
                    feat = get_adaptive_spectrogram(f)
                else:
                    feat = get_statistical_features(f)
                if feat is not None:
                    X.append(feat)
                    y.append(self.labels[i])
            except:
                pass
        return np.array(X), np.array(y)

    def get_train_val_split(self, test_size=0.2):
        X, y = self.load_data()
        if len(X) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)