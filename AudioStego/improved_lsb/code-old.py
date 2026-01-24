import os
import io
import hashlib
import numpy as np
from scipy.io import wavfile
from PIL import Image

# --- CORE 1: CHAOS ENGINE (FIXED - PREFIX CONSISTENT) ---
class QuantumChaosGenerator:
    @staticmethod
    def generate_indices(seed, total_slots, required_count):
        """
        Sinh vị trí ngẫu nhiên đảm bảo tính đồng bộ tuyệt đối giữa Encode và Decode.
        Sử dụng Hoán vị (Permutation) thay vì Chọn mẫu (Choice) để đảm bảo:
        Encode lấy 100 số đầu tiên khớp hoàn toàn với 100 số đầu tiên trong 1000 số của Decode.
        """
        rng = np.random.default_rng(seed)
        
        # Tạo hoán vị đầy đủ của các slot [0, 1, ..., total_slots-1]
        # Lưu ý: Với file WAV 50MB (~25 triệu mẫu), mảng này tốn khoảng 100MB RAM.
        # Đây là mức chấp nhận được và an toàn nhất cho tính đúng đắn của thuật toán.
        full_permutation = rng.permutation(total_slots)
        
        # Lấy đúng số lượng cần thiết từ đầu dãy
        indices = full_permutation[:required_count]
        
        return indices.astype(np.int32)

# --- CORE 2: PROCESSING ---
class DataProcessor:
    @staticmethod
    def compress_image(image_path, target_bytes):
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB': img = img.convert('RGB')
            img.thumbnail((1024, 1024))
            
            quality = 95
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=quality)
            data = output.getvalue()
            
            while len(data) > target_bytes and quality > 10:
                quality -= 5
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=quality)
                data = output.getvalue()
            
            while len(data) > target_bytes:
                w, h = img.size
                if w < 50 or h < 50: break
                img = img.resize((int(w*0.8), int(h*0.8)))
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=quality)
                data = output.getvalue()
            return data
        except:
            with open(image_path, 'rb') as f: return f.read()

class StegoMetrics:
    @staticmethod
    def calculate(original, stego):
        orig, mod = original.astype(np.float64), stego.astype(np.float64)
        mse = np.mean((orig - mod) ** 2)
        if mse == 0: return 0.0, float('inf'), float('inf')
        psnr = 20 * np.log10(32767.0 / np.sqrt(mse))
        snr = 10 * np.log10(np.sum(orig ** 2) / mse)
        return mse, psnr, snr

# --- CORE 3: ALGORITHM ---
class QuantumLSB:
    ANCHOR_SIZE = 4096 
    SENTINEL = b"||END||"

    @staticmethod
    def _generate_seed(password, salt):
        key = f"{password}__{salt}"
        return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)

    @staticmethod
    def encode(cover_path, secret_path, output_path, k=2, password="default"):
        try:
            rate, audio_data = wavfile.read(cover_path)
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16) if audio_data.dtype == np.float32 else audio_data.astype(np.int16)
                
            audio_flat = audio_data.flatten()
            stego_flat = audio_flat.copy()
            
            num_slots = len(audio_flat) - QuantumLSB.ANCHOR_SIZE
            max_bytes = (num_slots * k) // 8
            
            # Xử lý input
            if os.path.exists(secret_path):
                ext = os.path.splitext(secret_path)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png']:
                    secret_bytes = DataProcessor.compress_image(secret_path, max_bytes)
                else:
                    with open(secret_path, 'rb') as f: secret_bytes = f.read()
            else:
                 secret_bytes = secret_path.encode('utf-8')

            full_payload = secret_bytes + QuantumLSB.SENTINEL
            bits = np.unpackbits(np.frombuffer(full_payload, dtype=np.uint8))
            
            remainder = len(bits) % k
            if remainder != 0: bits = np.append(bits, [0] * (k - remainder))
            
            powers = 1 << np.arange(k)[::-1]
            secret_values = bits.reshape(-1, k).dot(powers).astype(np.int16)
            
            if len(secret_values) > num_slots:
                return {"status": "error", "mse": 0, "psnr": 0, "snr": 0, "message": "Oversize"}

            salt = os.path.basename(output_path)
            seed = QuantumLSB._generate_seed(password, salt)
            
            # Sinh vị trí (Permutation - Đồng bộ chính xác)
            target_indices = QuantumChaosGenerator.generate_indices(seed, num_slots, len(secret_values))
            target_indices += QuantumLSB.ANCHOR_SIZE
            
            mask = (1 << k) - 1
            stego_flat[target_indices] &= ~mask
            stego_flat[target_indices] |= secret_values
            
            stego_data = stego_flat.reshape(audio_data.shape)
            wavfile.write(output_path, rate, stego_data)
            
            mse, psnr, snr = StegoMetrics.calculate(audio_data, stego_data)
            return {"status": "success", "mse": mse, "psnr": psnr, "snr": snr}
            
        except Exception as e:
            return {"status": "error", "mse": 0, "psnr": 0, "snr": 0, "message": str(e)}

    @staticmethod
    def decode(stego_path, k=2, password="default"):
        try:
            if not os.path.exists(stego_path): 
                return {'status': 'error', 'type': 'error', 'message': "File khong ton tai"}
            
            # 1. Đọc Audio
            rate, stego_data = wavfile.read(stego_path)
            if stego_data.dtype != np.int16:
                 stego_data = (stego_data * 32767).astype(np.int16) if stego_data.dtype == np.float32 else stego_data.astype(np.int16)
            
            stego_flat = stego_data.flatten()
            num_slots = len(stego_flat) - QuantumLSB.ANCHOR_SIZE
            
            # 2. Tái tạo Seed
            salt = os.path.basename(stego_path)
            seed = QuantumLSB._generate_seed(password, salt)
            
            # 3. Đọc dữ liệu
            # Giới hạn đọc khoảng 5MB (40 triệu bits) để tối ưu tốc độ
            max_read_bits = 40_000_000 
            read_count = min(num_slots, max_read_bits // k)
            
            # Sinh lại vị trí (Dùng Permutation giống hệt Encode)
            target_indices = QuantumChaosGenerator.generate_indices(seed, num_slots, read_count)
            target_indices += QuantumLSB.ANCHOR_SIZE
            
            # 4. Trích xuất
            mask = (1 << k) - 1
            extracted_values = (stego_flat[target_indices] & mask).astype(np.uint8)
            
            bits_matrix = np.unpackbits(extracted_values[:, np.newaxis], axis=1)
            relevant_bits = bits_matrix[:, -k:]
            all_bytes = np.packbits(relevant_bits.flatten()).tobytes()
            
            # 5. Tìm Sentinel
            pos = all_bytes.find(QuantumLSB.SENTINEL)
            
            if pos != -1:
                content = all_bytes[:pos]
                # Nhận diện loại file
                file_type = 'binary'
                ext = '.bin'
                if content.startswith(b'\xff\xd8\xff'): 
                    file_type = 'image'; ext = '.jpg'
                elif content.startswith(b'\x89\x50\x4e\x47'): 
                    file_type = 'image'; ext = '.png'
                else:
                    try: 
                        content.decode('utf-8')
                        file_type = 'text'; ext = '.txt'
                    except: pass
                
                return {
                    'status': 'success',
                    'type': file_type,
                    'data': content,
                    'ext': ext
                }
            else:
                return {
                    'status': 'error', 
                    'type': 'error', 
                    'message': "Sai mat khau hoac khong tim thay du lieu."
                }
                
        except Exception as e:
            return {'status': 'error', 'type': 'error', 'message': str(e)}

# --- WRAPPERS ---
def encode(cover, secret, output, k=2, password="123"):
    return QuantumLSB.encode(cover, secret, output, k, password)

def decode(input_file, k=2, password="123"):
    return QuantumLSB.decode(input_file, k, password)