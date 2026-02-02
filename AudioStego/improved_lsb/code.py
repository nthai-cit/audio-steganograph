import os
import numpy as np
from scipy.io import wavfile
import hashlib
import io
import math
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
        except Exception:
            with open(image_path, 'rb') as f: return f.read()

def _generate_seed(password, salt):
    key = f"{password}__{salt}"
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)

def _get_content_salt(audio_data, anchor_size=1024):
    anchor_data = audio_data[:anchor_size].tobytes()
    return hashlib.sha256(anchor_data).hexdigest()

def _get_data_bytes(secret_input, max_bytes=None):
    if os.path.isfile(secret_input):
        ext = os.path.splitext(secret_input)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp'] and max_bytes:
            return ImageProcessor.compress_image_to_bytes(secret_input, max_bytes)
        else:
            with open(secret_input, 'rb') as f: return f.read()
    else:
        return secret_input.encode('utf-8')

def calculate_metrics(original, stego):
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
    snr = 10 * np.log10(signal_power / noise_power) if noise_power != 0 else float('inf')
    return mse, snr, psnr

def encode(cover_path, secret_input, output_path, k=None, password=None):
    try:
        rate, audio_data = wavfile.read(cover_path)
        if audio_data.dtype != np.int16:
             audio_data = (audio_data * 32767).astype(np.int16) if audio_data.dtype == np.float32 else audio_data.astype(np.int16)
        
        audio_flat = audio_data.flatten()
        stego_flat = audio_flat.copy()
        
        ANCHOR_SIZE = 1024 
        num_slots = len(audio_flat) - ANCHOR_SIZE
        
        if k is not None and k > 0:
            print(f"   [Mode] Fixed k = {k}")
            max_bytes_allowed = (num_slots * k) // 8
            raw_secret_bytes = _get_data_bytes(secret_input, max_bytes=max_bytes_allowed)
            payload_bits_needed = (len(raw_secret_bytes) + 7) * 8
            if payload_bits_needed > num_slots * k:
                return {"status": "error", "message": f"Oversize! K={k} is too small."}
        else:
            raw_secret_bytes = _get_data_bytes(secret_input, max_bytes=None)
            payload_bits_needed = (len(raw_secret_bytes) + 7) * 8
            k_calculated = math.ceil(payload_bits_needed / num_slots)
            k = max(1, k_calculated)
            if k > 8:
                 return {"status": "error", "message": f"File too large! Auto-calculated k={k} (>8)."}
            print(f"   [Mode] Adaptive Auto-K: Calculated k = {k}")

        full_payload = raw_secret_bytes + b"||END||"
        bits = np.unpackbits(np.frombuffer(full_payload, dtype=np.uint8))
        
        remainder = len(bits) % k
        if remainder != 0: bits = np.append(bits, [0] * (k - remainder))
    
        powers = 1 << np.arange(k)[::-1]
        secret_values = bits.reshape(-1, k).dot(powers).astype(np.int16)
        
        salt = _get_content_salt(audio_flat, ANCHOR_SIZE) 
        seed = _generate_seed(password if password else "default", salt)
        rng = np.random.default_rng(seed)

        valid_range = np.arange(ANCHOR_SIZE, len(audio_flat))
        shuffled_indices = rng.permutation(valid_range)
        target_indices = shuffled_indices[:len(secret_values)]
        
        mask = (1 << k) - 1
        stego_flat[target_indices] &= ~mask          
        stego_flat[target_indices] |= secret_values  

        stego_data = stego_flat.reshape(audio_data.shape)
        wavfile.write(output_path, rate, stego_data)
        
        mse, snr, psnr = calculate_metrics(audio_data, stego_data)
        
        return {
            "status": "success",
            "output_path": output_path,
            "mse": mse, "psnr": psnr, "snr": snr,
            "k": k, 
            "capacity": len(raw_secret_bytes)
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

def decode(stego_path, k=None, password=None):
    try:
        stego_path = os.path.abspath(stego_path)
        if not os.path.exists(stego_path): 
            return {'status': 'error', 'message': "File khong ton tai"}
        
        rate, stego_data = wavfile.read(stego_path)
        if stego_data.dtype != np.int16:
             stego_data = (stego_data * 32767).astype(np.int16) if stego_data.dtype == np.float32 else stego_data.astype(np.int16)
        stego_flat = stego_data.flatten()
        
        ANCHOR_SIZE = 1024
        salt = _get_content_salt(stego_flat, ANCHOR_SIZE)
        seed = _generate_seed(password if password else "default", salt)
        rng = np.random.default_rng(seed)
        
        valid_range = np.arange(ANCHOR_SIZE, len(stego_flat))
        shuffled_indices = rng.permutation(valid_range)
        
        SENTINEL = b"||END||"
        
        candidates = [k] if k else []
        candidates += [x for x in range(1, 9) if x not in candidates]
        
        final_content = None
        detected_k = -1

        print(f"   [Decode] Scanning k in {candidates}...")

        for k_try in candidates:
            if k_try is None: continue
            
            mask = (1 << k_try) - 1
            check_len = min(len(shuffled_indices), len(stego_flat))
            
            extracted_values = (stego_flat[shuffled_indices[:check_len]] & mask).astype(np.uint8)
            bits_matrix = np.unpackbits(extracted_values[:, np.newaxis], axis=1)
            relevant_bits = bits_matrix[:, -k_try:] 
            all_bytes = np.packbits(relevant_bits.flatten()).tobytes()
            
            pos = all_bytes.find(SENTINEL)
            
            if pos != -1:
                final_content = all_bytes[:pos]
                detected_k = k_try
                print(f"   [Decode] Success! Found data at k={detected_k}")
                break
        
        if final_content is not None:
            content = final_content
            result = {'status': 'success', 'data': content, 'k_detected': detected_k}
            
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
            return {'status': 'error', 'message': "Sentinel not found."}
            
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

def process_batch(input_dir, secret_input, output_dir=None, k=2, password=None):
    results = []
    
    if not os.path.exists(input_dir):
        print(f"[Batch] Path not found: {input_dir}")
        return []

    all_audio_paths = []
    print(f"[Batch] Scanning: {input_dir}...")
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.wav') and not file.startswith('.'):
                full_path = os.path.join(root, file)
                all_audio_paths.append(full_path)

    total_files = len(all_audio_paths)
    if total_files == 0: 
        print("[Batch] No wav files found.")
        return []

    print(f"[Batch] Found {total_files} files.")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    success_count = 0
    fail_count = 0
    skip_count = 0
    total_psnr = 0
    total_snr = 0

    for idx, filepath in enumerate(all_audio_paths):
        print(f"\r[Processing] {idx+1}/{total_files}... ", end="")
        
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
    print("BATCH SUMMARY")
    print(f"Success         : {success_count}")
    print(f"Skipped         : {skip_count}")
    print(f"Failed          : {fail_count}")
    
    if success_count > 0:
        print(f"Avg PSNR        : {total_psnr/success_count:.2f} dB")
        print(f"Avg SNR         : {total_snr/success_count:.2f} dB")
    print("="*50)
            
    return results