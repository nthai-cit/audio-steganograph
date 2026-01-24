import os
import numpy as np
from scipy.io import wavfile
import hashlib
import io
from PIL import Image

class ImageProcessor:
    @staticmethod
    def compress_image_to_bytes(image_path, target_bytes):
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB': img = img.convert('RGB')

            img.thumbnail((1024, 1024)) 
            
            quality = 95
            output = io.BytesIO()
            
            img.save(output, format='JPEG', quality=quality, subsampling=2)

            data = output.getvalue()

            while len(data) > target_bytes and quality > 10:
                quality -= 5
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=quality, subsampling=2)
                data = output.getvalue()
       
            while len(data) > target_bytes:
                w, h = img.size
                if w < 10 or h < 10: break
                img = img.resize((int(w*0.9), int(h*0.9)))
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=quality, subsampling=2)
                data = output.getvalue()
                
            return data
        except Exception as e:
            print(f"[Warn] Loi xu ly anh: {e}. Dung file goc.")
            with open(image_path, 'rb') as f: return f.read()

def _generate_seed(password, salt):
    """Tạo seed ngẫu nhiên thống nhất dựa trên mật khẩu và muối."""
    key = f"{password}__{salt}"
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)

def _get_data_bytes(secret_input, max_bytes=None):
    # Nếu là đường dẫn file
    if os.path.isfile(secret_input):
        ext = os.path.splitext(secret_input)[1].lower()
        # Neu la anh -> Nen
        if ext in ['.jpg', '.jpeg', '.png', '.bmp'] and max_bytes:
            return ImageProcessor.compress_image_to_bytes(secret_input, max_bytes)
        else:
            with open(secret_input, 'rb') as f: return f.read()
    # Nếu là chuỗi văn bản nhập tay
    else:
        return secret_input.encode('utf-8')

def calculate_metrics(original, stego):
    """Tính toán MSE, PSNR, SNR."""
    orig = original.astype(np.float64)
    mod = stego.astype(np.float64)
    diff = orig - mod
    mse = np.mean(diff ** 2)
    
    if mse == 0: return 0.0, float('inf'), float('inf')
    
    rmse = np.sqrt(mse)
    max_val = 32767.0 
    psnr = 20 * np.log10(max_val / rmse)
    
    signal_power = np.sum(orig ** 2)
    noise_power = np.sum(diff ** 2)
    if noise_power == 0:
        snr = float('inf')
    else:
        snr = 10 * np.log10(signal_power / noise_power)
        
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
        full_payload = secret_bytes + b"||END||"
        
        print(f"   [Improved] Dang nhung {len(secret_bytes)} bytes (k={k})...")

      
        bits = np.unpackbits(np.frombuffer(full_payload, dtype=np.uint8))
        
       
        remainder = len(bits) % k
        if remainder != 0: bits = np.append(bits, [0] * (k - remainder))
    
        powers = 1 << np.arange(k)[::-1]
        secret_values = bits.reshape(-1, k).dot(powers).astype(np.int16)
        
        if len(secret_values) > num_slots:
            return {"status": "error", "message": f"Oversize: Can {len(secret_values)} slots, Co {num_slots}"}
            
      
        salt = os.path.basename(output_path)
        seed = _generate_seed(password if password else "default", salt)
        rng = np.random.default_rng(seed)

        valid_range = np.arange(ANCHOR_SIZE, len(audio_flat))
        shuffled_indices = rng.permutation(valid_range)
        

        target_indices = shuffled_indices[:len(secret_values)]
        
  
        mask = (1 << k) - 1
        stego_flat[target_indices] &= ~mask          # Xóa k bit cuối
        stego_flat[target_indices] |= secret_values  # Ghi giá trị mới

        stego_data = stego_flat.reshape(audio_data.shape)
        wavfile.write(output_path, rate, stego_data)
        
    
        print(f"   [DANH GIA] Dang tinh toan chi so...")
        mse, snr, psnr = calculate_metrics(audio_data, stego_data)
        
        print("-" * 40)
        print(f"   BANG KET QUA (IMPROVED LSB):")
        print(f"   [-] MSE  : {mse:.6f}")
        print(f"   [+] SNR  : {snr:.2f} dB")
        print(f"   [+] PSNR : {psnr:.2f} dB")
        print("-" * 40)
        
        return {
            "status": "success",
            "output_path": output_path,
            "mse": mse, "psnr": psnr, "snr": snr,
            "k": k, "capacity": len(secret_bytes)
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


def decode(stego_path, k=2, password=None):
    try:
        stego_path = os.path.abspath(stego_path)
        if not os.path.exists(stego_path): 
            return {'status': 'error', 'message': "File khong ton tai"}
        
        print(f"   [Improved Decode] Dang xu ly: {os.path.basename(stego_path)}")

        # Đọc Audio
        rate, stego_data = wavfile.read(stego_path)
        if stego_data.dtype != np.int16:
             stego_data = (stego_data * 32767).astype(np.int16) if stego_data.dtype == np.float32 else stego_data.astype(np.int16)
        stego_flat = stego_data.flatten()
        
        #  Tái tạo Seed để lấy đúng vị trí đã xáo trộn
        salt = os.path.basename(stego_path)
        seed = _generate_seed(password if password else "default", salt)
        rng = np.random.default_rng(seed)
        
        ANCHOR_SIZE = 1024
        valid_range = np.arange(ANCHOR_SIZE, len(stego_flat))
        shuffled_indices = rng.permutation(valid_range)
        
        #  Trích xuất giá trị từ các vị trí ngẫu nhiên đó
        mask = (1 << k) - 1
        extracted_values = (stego_flat[shuffled_indices] & mask).astype(np.uint8)
        
        # Chuyển đổi ngược: Giá trị k-bit -> Bits -> Bytes
        bits_matrix = np.unpackbits(extracted_values[:, np.newaxis], axis=1)
        relevant_bits = bits_matrix[:, -k:] 
        all_bytes = np.packbits(relevant_bits.flatten()).tobytes()
        
        # Tìm dấu hiệu kết thúc
        SENTINEL = b"||END||"
        pos = all_bytes.find(SENTINEL)
        
        if pos != -1:
            content = all_bytes[:pos]
            result = {'status': 'success', 'data': content}
            
            # Nhận diện định dạng
            if content.startswith(b'\xff\xd8\xff'): result.update({'type': 'image', 'ext': '.jpg'})
            elif content.startswith(b'\x89\x50\x4e\x47'): result.update({'type': 'image', 'ext': '.png'})
            elif content.startswith(b'BM'): result.update({'type': 'image', 'ext': '.bmp'})
            elif content.startswith(b'RIFF'): result.update({'type': 'audio', 'ext': '.wav'})
            else:
                try: 
                    text = content.decode('utf-8')
                    if all(c.isprintable() or c.isspace() for c in text):
                        result.update({'type': 'text', 'ext': '.txt', 'content_text': text})
                    else:
                        result.update({'type': 'binary', 'ext': '.bin'})
                except: 
                    result.update({'type': 'binary', 'ext': '.bin'})
            
            return result
        else:
            return {'status': 'error', 'message': "Sai mat khau hoac khong tim thay dau hieu ket thuc."}
            
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def process_batch(input_dir, secret_input, output_dir=None, k=2, password=None):
    results = []
    
    if not os.path.exists(input_dir):
        print(f"[Batch] Khong tim thay thu muc input: {input_dir}")
        return []

    all_audio_paths = []
    print(f"[Batch] Dang quet tap tin trong: {input_dir}...")
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.wav') and not file.startswith('.'):
                full_path = os.path.join(root, file)
                all_audio_paths.append(full_path)

    total_files = len(all_audio_paths)
    if total_files == 0: 
        print("[Batch] Khong tim thay file wav nao.")
        return []

    print(f"[Improved Batch] Tim thay {total_files} file. K={k}. Password={'YES' if password else 'NO'}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    success_count = 0
    fail_count = 0
    skip_count = 0
    total_psnr = 0
    total_snr = 0

    for idx, filepath in enumerate(all_audio_paths):
        print(f"\r[Xu ly] {idx+1}/{total_files} file... ", end="")
        
        filename = os.path.basename(filepath)
        unique_filename = f"{idx:03d}_{filename}"
        out_path = os.path.join(output_dir, unique_filename) if output_dir else f"temp_{unique_filename}"
        
        res = encode(filepath, secret_input, out_path, k=k, password=password)
        
        if res['status'] == 'success':
            success_count += 1
            if res['psnr'] != float('inf'): total_psnr += res['psnr']
            if res['snr'] != float('inf'): total_snr += res['snr']
            
            results.append({
                "Filename": unique_filename,
                "MSE": f"{res['mse']:.6f}", "PSNR": f"{res['psnr']:.2f}", "SNR": f"{res['snr']:.2f}",
                "Status": "Success"
            })
        else:
            if "Oversize" in res['message']:
                skip_count += 1
                results.append({"Filename": unique_filename, "Status": "Skipped (Oversize)"})
            else:
                fail_count += 1
                results.append({"Filename": unique_filename, "Status": "Error", "Info": res['message']})

    print("\n" + "="*50)
    print("KET QUA TONG HOP (BATCH IMPROVED)")
    print(f"Thanh cong      : {success_count}")
    print(f"Bo qua (day)    : {skip_count}")
    print(f"Loi             : {fail_count}")
    
    if success_count > 0:
        print(f"PSNR Trung binh : {total_psnr/success_count:.2f} dB")
        print(f"SNR Trung binh  : {total_snr/success_count:.2f} dB")
    print("="*50)
            
    return results