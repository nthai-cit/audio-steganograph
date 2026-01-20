import os
import numpy as np
from scipy.io import wavfile
import hashlib
import io
from PIL import Image

class ImageProcessor:
    @staticmethod
    def compress_image_to_bytes(image_path, target_bytes):
        """Nen anh (Resize + JPEG Quality) de vua voi dung luong Audio."""
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB': img = img.convert('RGB')
            
            # Giam kich thuoc buoc 1
            img.thumbnail((1024, 1024)) 
            
            quality = 95
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=quality)
            data = output.getvalue()
            
            # Giam chat luong tu tu neu van qua lon
            while len(data) > target_bytes and quality > 10:
                quality -= 5
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=quality)
                data = output.getvalue()
                
            # Neu van qua lon, resize nho di
            if len(data) > target_bytes:
                w, h = img.size
                img = img.resize((int(w*0.7), int(h*0.7)))
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=quality)
                data = output.getvalue()
                
            print(f"   [Improved LSB] Da nen anh xuong: {img.size}. Size: {len(data)} bytes")
            return data
        except Exception as e:
            print(f"   [Warning] Khong the nen anh: {e}. Su dung du lieu goc.")
            with open(image_path, 'rb') as f: return f.read()


def _generate_seed(password, salt):
    """
    Tao seed ngau nhien dua tren mat khau va muoi (ten file).
    """
    key = f"{password}__{salt}"
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)

def _get_data_bytes(secret_input, max_bytes=None):
    # Neu la duong dan file
    if os.path.isfile(secret_input):
        ext = os.path.splitext(secret_input)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            print(f"   [Improved LSB] Phat hien file anh. Dang toi uu dung luong...")
            if max_bytes:
                return ImageProcessor.compress_image_to_bytes(secret_input, max_bytes)
            else:
                with open(secret_input, 'rb') as f: return f.read()
        else:
            print(f"   [Improved LSB] Dang doc file: {os.path.basename(secret_input)}")
            with open(secret_input, 'rb') as f: return f.read()
    # Neu la chuoi van ban (nhu truong hop cua ban '132134...')
    else:
        print(f"   [Improved LSB] Nhan dien tin nhan van ban.")
        return secret_input.encode('utf-8')

def calculate_metrics(original, stego):
    orig = original.astype(np.float64)
    mod = stego.astype(np.float64)
    diff = orig - mod
    mse = np.mean(diff ** 2)
    
    if mse == 0: return 0, float('inf'), float('inf')
    
    rmse = np.sqrt(mse)
    max_val = 32767.0 
    psnr = 20 * np.log10(max_val / rmse)
    signal_power = np.sum(orig ** 2)
    snr = 10 * np.log10(signal_power / mse)
    return mse, snr, psnr

def encode(cover_path, secret_input, output_path, k=2, password=None):
    try:
        rate, audio_data = wavfile.read(cover_path)
        if audio_data.dtype != np.int16:
             audio_data = (audio_data * 32767).astype(np.int16) if audio_data.dtype == np.float32 else audio_data.astype(np.int16)
        
        audio_flat = audio_data.flatten()
        stego_flat = audio_flat.copy()
        
        ANCHOR_SIZE = 1024 
        num_slots = len(audio_flat) - ANCHOR_SIZE
        max_bytes = (num_slots * k) // 8
        
        secret_bytes = _get_data_bytes(secret_input, max_bytes)
        
        print(f"   [Improved LSB] K={k}. Embedding {len(secret_bytes)} bytes...")
        
        full_payload = secret_bytes + b"||END||"
        bits = np.unpackbits(np.frombuffer(full_payload, dtype=np.uint8))
        
        remainder = len(bits) % k
        if remainder != 0: bits = np.append(bits, [0] * (k - remainder))
        
        powers = 1 << np.arange(k)[::-1]
        secret_values = bits.reshape(-1, k).dot(powers).astype(np.int16)
        
        if len(secret_values) > num_slots:
            raise ValueError(f"Khong du dung luong! Can {len(secret_values)} slots, co {num_slots}.")
            
        # Shuffle
        salt = os.path.basename(output_path)
        seed = _generate_seed(password if password else "default", salt)
        rng = np.random.default_rng(seed)
        
        valid_range = np.arange(ANCHOR_SIZE, len(audio_flat))
        shuffled_indices = rng.permutation(valid_range)
        target_indices = shuffled_indices[:len(secret_values)]
        
        # Embed
        mask = (1 << k) - 1
        stego_flat[target_indices] &= ~mask
        stego_flat[target_indices] |= secret_values
        
        stego_data = stego_flat.reshape(audio_data.shape)
        wavfile.write(output_path, rate, stego_data)
        
        print(f"   [DANH GIA] Dang tinh toan chi so...")
        mse, snr, psnr = calculate_metrics(audio_data, stego_data)
        
        print("-" * 40)
        print(f"   BANG KET QUA (IMPROVED):")
        print(f"   [-] MSE  : {mse:.4f}")
        print(f"   [+] SNR  : {snr:.2f} dB")
        print(f"   [+] PSNR : {psnr:.2f} dB")
        print("-" * 40)
        
        return {
            "status": "success",
            "output_path": output_path,
            "mse": mse,
            "psnr": psnr,
            "snr": snr,
            "k": k,
            "capacity": len(secret_bytes)
        }

    except Exception as e:
        raise RuntimeError(f"Loi Improved Encode: {e}")


def decode(stego_path, k=2, password=None):
    try:
        if not os.path.exists(stego_path): return {'type': 'error', 'message': "File khong ton tai"}
            
        rate, stego_data = wavfile.read(stego_path)
        if stego_data.dtype != np.int16:
             stego_data = (stego_data * 32767).astype(np.int16) if stego_data.dtype == np.float32 else stego_data.astype(np.int16)
        stego_flat = stego_data.flatten()
        
        salt = os.path.basename(stego_path)
        seed = _generate_seed(password if password else "default", salt)
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
            if content.startswith(b'\xff\xd8\xff'): return {'type': 'image', 'data': content, 'ext': '.jpg'}
            if content.startswith(b'\x89\x50\x4e\x47'): return {'type': 'image', 'data': content, 'ext': '.png'}
            try: return {'type': 'text', 'data': content, 'ext': '.txt'}
            except: return {'type': 'binary', 'data': content, 'ext': '.bin'}
        else:
            return {'type': 'error', 'message': "Sai mat khau hoac khong co thong diep."}
    except Exception as e:
        return {'type': 'error', 'message': str(e)}


def process_batch(input_dir, secret_input, k=2, password=None):
    results = []
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
    total_files = len(files)
    
    if total_files == 0: 
        print("[Batch] Khong tim thay file wav.")
        return []

    print(f"[Improved Batch] Tim thay {total_files} file. K={k}...")

    # Chuan bi du lieu (Neu la text thi encode 1 lan, neu la file anh thi phai xu ly tung file vi dung luong khac nhau)
    is_text = not os.path.isfile(secret_input)

    for idx, filename in enumerate(files):
        filepath = os.path.join(input_dir, filename)
        try:
            rate, audio_data = wavfile.read(filepath)
            if audio_data.dtype != np.int16:
                 audio_data = (audio_data * 32767).astype(np.int16) if audio_data.dtype == np.float32 else audio_data.astype(np.int16)
            
            audio_flat = audio_data.flatten()
            stego_flat = audio_flat.copy()
            
            ANCHOR_SIZE = 1024
            num_slots = len(audio_flat) - ANCHOR_SIZE
            max_bytes = (num_slots * k) // 8
            
            # Lay du lieu
            secret_bytes = _get_data_bytes(secret_input, max_bytes)
            
            full_payload = secret_bytes + b"||END||"
            bits = np.unpackbits(np.frombuffer(full_payload, dtype=np.uint8))
            
            remainder = len(bits) % k
            if remainder != 0: bits = np.append(bits, [0] * (k - remainder))
            
            powers = 1 << np.arange(k)[::-1]
            secret_values = bits.reshape(-1, k).dot(powers).astype(np.int16)
            
            if len(secret_values) > num_slots:
                print(f"  [Bo qua] {filename}: Qua tai.")
                continue
            
            # Shuffle & Embed
            salt = filename # Trong batch, dung ten file goc lam muoi
            seed = _generate_seed(password if password else "default", salt)
            rng = np.random.default_rng(seed)
            
            shuffled_indices = rng.permutation(np.arange(ANCHOR_SIZE, len(audio_flat)))
            target_indices = shuffled_indices[:len(secret_values)]
            
            mask = (1 << k) - 1
            stego_flat[target_indices] &= ~mask
            stego_flat[target_indices] |= secret_values
            
            stego_data = stego_flat.reshape(audio_data.shape)
            mse, snr, psnr = calculate_metrics(audio_data, stego_data)
            
            results.append({
                "Filename": filename,
                "MSE": mse, "PSNR": psnr, "SNR": snr,
                "Status": "Success"
            })
            print(f"  [{idx+1}/{total_files}] {filename} -> PSNR: {psnr:.2f} dB")

        except Exception as e:
            print(f"  [Loi] {filename}: {e}")
            results.append({"Filename": filename, "Status": "Error"})
            
    return results