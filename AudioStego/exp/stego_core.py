import os
import numpy as np
from scipy.io import wavfile
import hashlib
import math
import io

# --- THÊM: Import PIL để xử lý ảnh ---
try:
    from PIL import Image
except ImportError:
    print("Warning: PIL not installed. Image resizing will be disabled.")
    Image = None
# -------------------------------------

class StegoUtils:
    @staticmethod
    def compress_image(file_path, target_size):
        """
        Resize ảnh về kích thước target_size (vuông) và nén JPEG tối đa.
        """
        if Image is None:
            with open(file_path, 'rb') as f: return f.read()
            
        try:
            img = Image.open(file_path)
            if img.mode != 'RGB': img = img.convert('RGB')
            
            # 1. Resize cứng về kích thước yêu cầu (VD: 128x128)
            img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            
            # 2. Nén JPEG với chất lượng thấp dần để giảm dung lượng file
            # TIMIT rất nhỏ, nên ta ưu tiên dung lượng nhỏ hơn chất lượng ảnh
            best_data = None
            
            # Thử quality từ 85 xuống 10
            for q in range(85, 9, -15):
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=q)
                data = output.getvalue()
                best_data = data
                # Nếu dung lượng < 10KB (ước lượng cho TIMIT) thì dừng luôn
                if len(data) < 10240: 
                    break
            
            return best_data
        except Exception as e:
            print(f"[Image Error] {e}. Using raw bytes.")
            with open(file_path, 'rb') as f: return f.read()

    @staticmethod
    def prepare_payload(file_path, target_size=256):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        
        # Nếu là ảnh -> Resize
        if ext in image_exts:
            return StegoUtils.compress_image(file_path, target_size)
        else:
            # Nếu là text/binary -> Đọc nguyên gốc
            with open(file_path, 'rb') as f:
                return f.read()

    @staticmethod
    def generate_seed(password, salt):
        key = f"{password}__{salt}"
        return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)

class StegoBasic:
    # CASE 1
    @staticmethod
    def encode(cover_path, payload_bytes, output_path):
        try:
            rate, audio_data = wavfile.read(cover_path)
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16) if audio_data.dtype == np.float32 else audio_data.astype(np.int16)

            audio_flat = audio_data.flatten()
            stego_flat = audio_flat.copy()
            
            full_payload = payload_bytes + b"||END||"
            bits = np.unpackbits(np.frombuffer(full_payload, dtype=np.uint8))
            
            if len(bits) > len(stego_flat):
                return {"status": "error", "message": "Oversize"}

            stego_flat[:len(bits)] &= ~1
            stego_flat[:len(bits)] |= bits.astype(np.int16)
            
            stego_data = stego_flat.reshape(audio_data.shape)
            wavfile.write(output_path, rate, stego_data)
            return {"status": "success"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

class StegoImproved:
    # CASE 2-5
    @staticmethod
    def encode(cover_path, payload_bytes, output_path, password, k_strategy='fixed', salt_source='default', fixed_k_val=1):
        try:
            rate, audio_data = wavfile.read(cover_path)
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16) if audio_data.dtype == np.float32 else audio_data.astype(np.int16)
            
            audio_flat = audio_data.flatten()
            stego_flat = audio_flat.copy()
            ANCHOR_SIZE = 1024
            num_slots = len(audio_flat) - ANCHOR_SIZE

            if k_strategy == 'adaptive':
                required_bits = (len(payload_bytes) + 10) * 8
                calc_k = math.ceil(required_bits / num_slots)
                # Giới hạn k max = 6 để tránh vỡ tiếng quá nhiều (TIMIT quá ngắn)
                k = max(1, min(calc_k, 6))
            else:
                k = fixed_k_val

            full_payload = payload_bytes + b"||END||"
            bits = np.unpackbits(np.frombuffer(full_payload, dtype=np.uint8))
            
            remainder = len(bits) % k
            if remainder != 0: bits = np.append(bits, [0] * (k - remainder))
            
            powers = 1 << np.arange(k)[::-1]
            secret_values = bits.reshape(-1, k).dot(powers).astype(np.int16)
            
            if len(secret_values) > num_slots:
                return {"status": "error", "message": f"Oversize (K={k}, Need {len(secret_values)} slots, Have {num_slots})"}

            if salt_source == 'content':
                salt = hashlib.md5(payload_bytes).hexdigest()
            else:
                salt = "STATIC_DEFAULT_SALT"

            seed = StegoUtils.generate_seed(password, salt)
            rng = np.random.default_rng(seed)
            
            valid_range = np.arange(ANCHOR_SIZE, len(audio_flat))
            shuffled_indices = rng.permutation(valid_range)
            target_indices = shuffled_indices[:len(secret_values)]
            
            mask = (1 << k) - 1
            stego_flat[target_indices] &= ~mask
            stego_flat[target_indices] |= secret_values

            stego_data = stego_flat.reshape(audio_data.shape)
            wavfile.write(output_path, rate, stego_data)
            
            return {"status": "success", "k_used": k, "salt_used": salt}
        except Exception as e:
            return {"status": "error", "message": str(e)}

class StegoPhase:
    # CASE 7: Phase Coding
    @staticmethod
    def _read_audio_float(filepath):
        rate, audio_data = wavfile.read(filepath)
        if np.issubdtype(audio_data.dtype, np.integer):
            max_val = np.iinfo(audio_data.dtype).max
            audio_data = audio_data.astype(np.float32) / max_val
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0] # Force Mono
        return rate, audio_data

    @staticmethod
    def _write_audio_float(filepath, rate, audio_data):
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = audio_data / max_val
        wavfile.write(filepath, rate, audio_data.astype(np.float32))

    @staticmethod
    def _calculate_segment_params(audio_len, sample_rate):
        seg_len = 2 ** int(np.log2(sample_rate * 1.5)) 
        seg_len = min(seg_len, 131072)
        seg_len = max(seg_len, 8192)
        seg_num = int(np.ceil(audio_len / seg_len))
        usable_bins = int((seg_len // 2) * 0.8) 
        return seg_len, seg_num, usable_bins

    @staticmethod
    def encode(cover_path, payload_bytes, output_path):
        try:
            rate, audio = StegoPhase._read_audio_float(cover_path)
            
            full_payload = payload_bytes + b"||DATA_END||" 
            msg_bits = np.unpackbits(np.frombuffer(full_payload, dtype=np.uint8))
            msg_len = len(msg_bits)
            
            seg_len, seg_num, bits_per_seg = StegoPhase._calculate_segment_params(len(audio), rate)
            capacity = bits_per_seg * seg_num
            
            if msg_len > capacity:
                return {"status": "error", "message": f"Oversize: Can {msg_len} bits, chua {capacity}"}
            
            target_len = seg_num * seg_len
            if len(audio) < target_len: 
                audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
            
            segs = audio.reshape((seg_num, seg_len))
            fft_segs = np.fft.fft(segs)
            
            M = np.abs(fft_segs)
            P = np.angle(fft_segs)
            
            PHASE_0, PHASE_1 = np.pi/4, -np.pi/4
            phase_values = np.where(msg_bits == 0, PHASE_0, PHASE_1)
            
            seg_mid = seg_len // 2
            start_idx = int(seg_mid * 0.1)
            curr = 0
            
            for i in range(seg_num):
                bits_here = min(bits_per_seg, msg_len - curr)
                if bits_here <= 0: break
                
                seg_phases = phase_values[curr : curr + bits_here]
                embed_start = start_idx
                
                P[i, embed_start:embed_start+bits_here] = seg_phases
                P[i, seg_len - (embed_start+bits_here) + 1 : seg_len - embed_start + 1] = -seg_phases[::-1]
                
                curr += bits_here
                
            mod_fft = M * np.exp(1j * P)
            stego_audio = np.fft.ifft(mod_fft).real.ravel()
            
            StegoPhase._write_audio_float(output_path, rate, stego_audio)
            
            return {"status": "success", "info": "Phase_FFT"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}