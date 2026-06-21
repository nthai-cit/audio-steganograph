import os
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from .features import get_adaptive_spectrogram, get_statistical_features

class StegoDataset:
    def __init__(self, cover_dir, stego_dir, model_type='cnn', cache_path=None):
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        self.model_type = model_type
        self.cache_path = cache_path
        self.files = []
        self.labels = []
        if self.cache_path and os.path.exists(self.cache_path):
            print(f"[Cache] Found cache file: {self.cache_path}")
        else:
            self._scan_files()

    
    def _scan_files(self):
        directories = [(self.cover_dir, 0), (self.stego_dir, 1)]
        print(f"[Dataset] Scanning files...")
        total_found = 0
        self.files = []
        self.labels = []
        for folder_path, label in directories:
            if not os.path.exists(folder_path):
                print(f"[Warning] Directory not found: {folder_path}")
                continue
            count = 0
            for root, _, filenames in os.walk(folder_path):
                # QUAN TRỌNG: Phải sorted() để Cover và Stego khớp thứ tự 1-1
                for f in sorted(filenames):
                    if f.lower().endswith(('.wav', '.flac', '.mp3')):
                        self.files.append(os.path.join(root, f))
                        self.labels.append(label)
                        count += 1
            print(f" -> Found {count} files in {folder_path}")

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

    # def get_train_val_split(self, test_size=0.2):
    #     X, y = self.load_data()
    #     if len(X) == 0:
    #         return np.array([]), np.array([]), np.array([]), np.array([])
    #     return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)



    def get_train_val_split(self, test_size=0.15, val_size=0.15, random_state=42):
        X, y = self.load_data()
        if len(X) == 0:
            return [np.array([]) for _ in range(6)]
        
        # Tách index của cover (0) và stego (1)
        cover_idx = np.where(y == 0)[0]
        stego_idx = np.where(y == 1)[0]
        
        # Khắc phục lỗi Phase Coding (150 vs 139): Lấy số lượng nhỏ nhất
        n_pairs = min(len(cover_idx), len(stego_idx))
        
        if n_pairs == 0:
            raise ValueError("[ERROR] Dataset không có đủ cả cover lẫn stego samples!")
        
        # Ghép cặp an toàn, loại bỏ các file bị dư
        paired_cover = cover_idx[:n_pairs]
        paired_stego = stego_idx[:n_pairs]
        
        valid_indices = np.concatenate([paired_cover, paired_stego])
        X_valid = X[valid_indices]
        y_valid = y[valid_indices]
        
        # Tạo Group ID chuẩn xác
        groups = np.concatenate([np.arange(n_pairs), np.arange(n_pairs)])
        
        # 1. Tách tập TEST (Cố định, không đổi qua 10 runs)
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_val_idx, test_idx = next(gss_test.split(X_valid, y_valid, groups))
        
        X_train_val = X_valid[train_val_idx]
        y_train_val = y_valid[train_val_idx]
        groups_train_val = groups[train_val_idx]
        
        X_test = X_valid[test_idx]
        y_test = y_valid[test_idx]
        
        # 2. Tách tập TRAIN và VAL (Xáo trộn theo seed qua từng run)
        # Tính tỷ lệ val thực tế trên phần dữ liệu còn lại (0.15 / 0.85 ≈ 0.1764)
        relative_val_size = val_size / (1.0 - test_size)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
        train_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, groups_train_val))
        
        X_train = X_train_val[train_idx]
        y_train = y_train_val[train_idx]
        
        X_val = X_train_val[val_idx]
        y_val = y_train_val[val_idx]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
