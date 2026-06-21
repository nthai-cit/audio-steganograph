import librosa
import numpy as np
from scipy import signal

def get_adaptive_spectrogram(file_path, target_shape=(128, 216)):
    """
    Phiên bản "High-Pass Filter" (Dựa trên code thành công của bạn)
    1. Cắt 5s
    2. Lọc bỏ tần số < 2000Hz (Quan trọng nhất để phát hiện LSB)
    3. Tạo Mel Spectrogram
    4. Padding/Cutting thay vì Resize
    """
    try:
        # 1. Load Audio (Duration 5.0s như code cũ)
        # Lưu ý: sr=None để giữ nguyên sample rate gốc (thường là 44.1kHz)
        y, sr = librosa.load(file_path, sr=None, duration=5.0)
        
        # Nếu file ngắn hơn 5s, lặp lại cho đủ
        required_samples = int(5 * sr)
        if len(y) < required_samples:
            y = np.tile(y, int(np.ceil(required_samples / len(y))))
        y = y[:required_samples]

        # 2. ÁP DỤNG HIGH-PASS FILTER
        # Loại bỏ tần số dưới 2000Hz để làm lộ nhiễu LSB
        sos = signal.butter(10, 2000, 'hp', fs=sr, output='sos')
        y = signal.sosfilt(sos, y)

        # 3. Tạo Mel Spectrogram
        # n_mels=128 (Chiều cao ảnh)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_shape[0], fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)

        # 4. Cố định kích thước chiều rộng (Time steps)
        fixed_length = target_shape[1] # Thường là 216 cho 5s
        
        if S_db.shape[1] > fixed_length:
            S_db = S_db[:, :fixed_length]
        else:
            pad_width = fixed_length - S_db.shape[1]
            S_db = np.pad(S_db, pad_width=((0, 0), (0, pad_width)), mode='constant')

        # 5. Reshape cho CNN (H, W, 1)
        # Code cũ của bạn chuẩn hóa Z-score bên ngoài (theo batch), 
        # nhưng để an toàn khi chạy từng file lẻ, ta thêm 1 trục channel ở đây.
        S_db = S_db[..., np.newaxis]
        
        return S_db

    except Exception as e:
        print(f"Lỗi xử lý file {file_path}: {e}")
        return None

def get_statistical_features(file_path):
    """
    Giữ nguyên cho ML truyền thống
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