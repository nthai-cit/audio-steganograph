import os
import numpy as np
from scipy.io import wavfile
import hashlib
import io
from PIL import Image

# --- CAU HINH ---
SENTINEL = b"||END||"
ANCHOR_SIZE = 1024 
# Salt co dinh de dam bao doi ten file khong bi loi giai ma
FIXED_SALT = "AUDIO_STEGO_PROJECT_2026" 

# --- CLASS XU LY ANH ---
class ImageProcessor:
    @staticmethod
    def compress_image_to_fit(image_path, max_bytes):
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                sizes = [512, 256, 128, 64] 
                for size in sizes:
                    buffer = io.BytesIO()
                    img_resized = img.resize((size, size))
                    img_resized.save(buffer, format="JPEG", quality=75, subsampling=0)
                    data = buffer.getvalue()
                    if len(data) < max_bytes:
                        print(f"   [Improved LSB] Da nen anh xuong: {size}x{size} (4:2:0). Size: {len(data)} bytes")
                        return data
            return None
        except Exception as e:
            return None

# --- CAC HAM HO TRO ---

def _generate_seed(password):
    """Tao Seed chi dua tren Password va Salt co dinh."""
    # Su dung FIXED_SALT thay vi ten file de tranh loi khi doi ten file
    if password is None: 
        password = "default"
    combined = f"{password}__{FIXED_SALT}"
    return int(hashlib.sha256(combined.encode()).hexdigest(), 16) % (2**32)

def _get_data_bytes(secret_input, max_capacity_bytes):
    if os.path.isfile(secret_input):
        ext = os.path.splitext(secret_input)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            print(f"   [Improved LSB] Phat hien file anh. Dang toi uu dung luong...")
            compressed_data = ImageProcessor.compress_image_to_fit(secret_input, max_capacity_bytes)
            if compressed_data:
                return compressed_data
        
        with open(secret_input, 'rb') as f:
            return f.read()
    else:
        return secret_input.encode('utf-8')

def calculate_metrics(original, stego):
    min_len = min(len(original), len(stego))
    orig = original.flatten()[:min_len].astype(float)
    mod = stego.flatten()[:min_len].astype(float)
    mse = np.mean((orig - mod) ** 2)
    if mse == 0: return 0, float('inf'), float('inf')
    
    max_val = 32768.0
    psnr = 10 * np.log10((max_val**2) / mse)
    signal_power = np.mean(orig**2)
    snr = 10 * np.log10(signal_power / mse)
    return mse, snr, psnr

# --- ENCODE / DECODE ---

def encode(cover_path, secret_input, output_path, k=2, password=None):
    try:
        rate, audio_data = wavfile.read(cover_path)
        if audio_data.dtype != np.int16:
             audio_data = (audio_data * 32767).astype(np.int16) if audio_data.dtype == np.float32 else audio_data.astype(np.int16)
            
        original_shape = audio_data.shape
        audio_flat = audio_data.flatten().copy()
        
        num_slots = len(audio_flat) - ANCHOR_SIZE
        max_bytes = (num_slots * k) // 8
        
        secret_bytes = _get_data_bytes(secret_input, max_bytes)
        full_payload = secret_bytes + SENTINEL
        
        bits = np.unpackbits(np.frombuffer(full_payload, dtype=np.uint8))
        remainder = len(bits) % k
        if remainder != 0:
            bits = np.append(bits, [0] * (k - remainder))
        
        powers = 1 << np.arange(k)[::-1]
        secret_values = bits.reshape(-1, k).dot(powers).astype(np.int16)
        
        if len(secret_values) > num_slots:
            raise ValueError(f"Du lieu qua lon! Can {len(secret_values)} slots, co {num_slots} slots.")
            
        print(f"   [Improved LSB] K={k}. Password='{password if password else 'default'}'. Dang nhung {len(secret_bytes)} bytes...")

        # TAO SEED
        seed = _generate_seed(password)
        rng = np.random.default_rng(seed)
        
        valid_range = np.arange(ANCHOR_SIZE, len(audio_flat))
        shuffled_indices = rng.permutation(valid_range)
        target_indices = shuffled_indices[:len(secret_values)]
        
        # NHUNG
        mask = (1 << k) - 1
        audio_flat[target_indices] &= ~mask
        audio_flat[target_indices] |= secret_values
        
        stego_data = audio_flat.reshape(original_shape)
        wavfile.write(output_path, rate, stego_data)
        
        print(f"   [DANH GIA] Dang tinh toan chi so...")
        mse, snr, psnr = calculate_metrics(audio_data, stego_data)
        
        print("-" * 40)
        print(f"   BANG KET QUA (IMPROVED RANDOM LSB):")
        print(f"   [-] MSE  : {mse:.4f}")
        print(f"   [+] SNR  : {snr:.2f} dB")
        print(f"   [+] PSNR : {psnr:.2f} dB")
        print("-" * 40)
        return output_path

    except Exception as e:
        raise RuntimeError(f"Loi Encode Improved: {e}")

def decode(stego_path, k=2, password=None):
    """
    Trich xuat tin Improved LSB.
    Nang cap: Tu dong nhan dien nhieu loai file hon (WAV, MP3, ZIP, BMP...)
    """
    try:
        if not os.path.exists(stego_path):
            return {'type': 'error', 'message': "Khong tim thay file stego."}
            
        rate, stego_data = wavfile.read(stego_path)
        stego_flat = stego_data.flatten()
        
        # --- Logic tao Seed (giu nguyen) ---
        salt = os.path.basename(stego_path)
        FIXED_SALT = "AUDIO_STEGO_PROJECT_2026"
        if password is None: password = "default"
        combined = f"{password}__{FIXED_SALT}"
        seed = int(hashlib.sha256(combined.encode()).hexdigest(), 16) % (2**32)
        
        rng = np.random.default_rng(seed)
        ANCHOR_SIZE = 1024
        
        valid_range = np.arange(ANCHOR_SIZE, len(stego_flat))
        shuffled_indices = rng.permutation(valid_range)
        
        mask = (1 << k) - 1
        extracted_values = (stego_flat[shuffled_indices] & mask).astype(np.uint8)
        
        bits_matrix = np.unpackbits(extracted_values[:, np.newaxis], axis=1)
        relevant_bits = bits_matrix[:, -k:]
        all_bytes = np.packbits(relevant_bits.flatten()).tobytes()
        
        SENTINEL = b"||END||"
        pos = all_bytes.find(SENTINEL)
        
        if pos != -1:
            content = all_bytes[:pos]
            
            # --- BO NHAN DIEN FILE THONG MINH ---
            
            # 1. Hinh anh
            if content.startswith(b'\xff\xd8\xff'): return {'type': 'image', 'data': content, 'ext': '.jpg'}
            if content.startswith(b'\x89\x50\x4e\x47'): return {'type': 'image', 'data': content, 'ext': '.png'}
            if content.startswith(b'BM'): return {'type': 'image', 'data': content, 'ext': '.bmp'}
            if content.startswith(b'GIF8'): return {'type': 'image', 'data': content, 'ext': '.gif'}
            
            # 2. Am thanh
            if content.startswith(b'RIFF') and content[8:12] == b'WAVE': return {'type': 'audio', 'data': content, 'ext': '.wav'}
            if content.startswith(b'ID3') or content.startswith(b'\xff\xfb') or content.startswith(b'\xff\xf3'): 
                return {'type': 'audio', 'data': content, 'ext': '.mp3'}
            
            # 3. File nen / Tai lieu
            if content.startswith(b'PK\x03\x04'): return {'type': 'archive', 'data': content, 'ext': '.zip'}
            if content.startswith(b'%PDF'): return {'type': 'document', 'data': content, 'ext': '.pdf'}
            
            # 4. Van ban (UTF-8)
            try:
                text = content.decode('utf-8')
                # Neu decode thanh cong thi la text
                return {'type': 'text', 'data': content, 'ext': '.txt'}
            except:
                # 5. Khong nhan dien duoc -> Binary
                return {'type': 'binary', 'data': content, 'ext': '.bin'}
        else:
            return {'type': 'error', 'message': "Sai mat khau hoac khong co thong diep."}

    except Exception as e:
        return {'type': 'error', 'message': str(e)}
    
def process_batch(input_dir, secret_input, k=2, password=None):
    """
    Chay Improved LSB hang loat (Khong luu file).
    Ho tro K va Password.
    """
    results = []
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
    total_files = len(files)
    
    if total_files == 0: return []
    print(f"[Improved Batch] Tim thay {total_files} file. K={k}, Pass={'CO' if password else 'KHONG'}")

    # Luu y: Voi Improved LSB, ta phai doc file bi mat cho TUNG file audio
    # vi dung luong moi file audio khac nhau -> muc do nen anh khac nhau.

    for idx, filename in enumerate(files):
        filepath = os.path.join(input_dir, filename)
        try:
            # 1. Doc Audio
            rate, audio_data = wavfile.read(filepath)
            if audio_data.dtype != np.int16:
                 audio_data = (audio_data * 32767).astype(np.int16) if audio_data.dtype == np.float32 else audio_data.astype(np.int16)
            
            audio_flat = audio_data.flatten().copy()
            stego_flat = audio_flat.copy()
            
            # Tinh dung luong
            ANCHOR_SIZE = 1024
            SENTINEL = b"||END||"
            num_slots = len(audio_flat) - ANCHOR_SIZE
            max_bytes = (num_slots * k) // 8
            
            # 2. Chuan bi du lieu (Tu dong nen anh neu can thiet dua theo max_bytes)
            # Goi lai ham _get_data_bytes cho tung file
            secret_bytes = _get_data_bytes(secret_input, max_bytes)
            
            if secret_bytes is None:
                 print(f"  [Bo qua] {filename}: Anh qua lon, khong nen vua.")
                 continue

            full_payload = secret_bytes + SENTINEL
            bits = np.unpackbits(np.frombuffer(full_payload, dtype=np.uint8))
            
            # Padding
            remainder = len(bits) % k
            if remainder != 0: bits = np.append(bits, [0] * (k - remainder))
            
            powers = 1 << np.arange(k)[::-1]
            secret_values = bits.reshape(-1, k).dot(powers).astype(np.int16)
            
            if len(secret_values) > num_slots:
                print(f"  [Bo qua] {filename}: Qua tai (Can {len(secret_values)}, co {num_slots}).")
                continue
            
            # 3. Shuffle & Embed (Trong RAM)
            # Salt = Ten file hien tai
            salt = filename 
            seed = _generate_seed(password if password else "default", salt)
            rng = np.random.default_rng(seed)
            
            valid_range = np.arange(ANCHOR_SIZE, len(audio_flat))
            shuffled_indices = rng.permutation(valid_range)
            target_indices = shuffled_indices[:len(secret_values)]
            
            mask = (1 << k) - 1
            stego_flat[target_indices] &= ~mask
            stego_flat[target_indices] |= secret_values
            
            # 4. Danh gia
            stego_data = stego_flat.reshape(audio_data.shape)
            mse, snr, psnr = calculate_metrics(audio_data, stego_data)
            
            results.append({
                "Filename": filename,
                "MSE": mse,
                "PSNR": psnr,
                "SNR": snr,
                "Status": "Success"
            })
            print(f"  [{idx+1}/{total_files}] {filename} -> PSNR: {psnr:.2f} dB")

        except Exception as e:
            print(f"  [Loi] {filename}: {e}")
            results.append({"Filename": filename, "Status": "Error"})
            
    return results