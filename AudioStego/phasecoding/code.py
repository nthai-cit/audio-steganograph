import os
import numpy as np
import scipy.io.wavfile as wavfile
import math

# --- CAC HAM HO TRO ---

def _get_data_bytes(secret_input):
    if os.path.isfile(secret_input):
        print(f"   [Phase Info] Nhan dien du lieu la FILE: '{os.path.basename(secret_input)}'")
        with open(secret_input, 'rb') as f:
            return f.read()
    else:
        print(f"   [Phase Info] Nhan dien du lieu la VAN BAN.")
        return secret_input.encode('utf-8')

def _bytes_to_bits(data_bytes):
    delimiter = b"||DATA_END||"
    full_data = data_bytes + delimiter
    return np.unpackbits(np.frombuffer(full_data, dtype=np.uint8))

def _bits_to_bytes(bits):
    padding = (8 - (len(bits) % 8)) % 8
    if padding > 0:
        bits = np.concatenate((bits, np.zeros(padding, dtype=np.uint8)))
    return np.packbits(bits).tobytes()

def _read_audio_float(filepath):
    rate, audio_data = wavfile.read(filepath)
    if np.issubdtype(audio_data.dtype, np.integer):
        max_val = np.iinfo(audio_data.dtype).max
        audio_data = audio_data.astype(np.float32) / max_val
    if len(audio_data.shape) > 1:
        audio_data = audio_data[:, 0] # Mono
    return rate, audio_data

def _write_audio_float(filepath, rate, audio_data):
    max_val = np.max(np.abs(audio_data))
    if max_val > 1.0:
        audio_data = audio_data / max_val
    wavfile.write(filepath, rate, audio_data.astype(np.float32))

def _calculate_segment_params(audio_len, sample_rate):
    seg_len = 2 ** int(np.log2(sample_rate * 1.5))
    seg_len = min(seg_len, 131072)
    seg_len = max(seg_len, 8192)
    seg_num = int(np.ceil(audio_len / seg_len))
    usable_bins = int((seg_len // 2) * 0.8) 
    return seg_len, seg_num, usable_bins

# --- HAM DANH GIA CHAT LUONG (FLOAT) ---
def calculate_metrics_float(original, stego):
    """Tinh toan metrics cho du lieu float (-1.0 den 1.0)."""
    # Cat do dai cho bang nhau (do padding luc encode)
    min_len = min(len(original), len(stego))
    orig = original[:min_len].astype(np.float64)
    mod = stego[:min_len].astype(np.float64)
    
    diff = orig - mod
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    
    if mse == 0:
        psnr = float('inf')
        snr = float('inf')
    else:
        max_val = 1.0  # Voi float, bien do lon nhat la 1.0
        psnr = 20 * np.log10(max_val / rmse)
        
        signal_power = np.sum(orig ** 2)
        noise_power = np.sum(diff ** 2)
        if noise_power == 0:
            snr = float('inf')
        else:
            snr = 10 * np.log10(signal_power / noise_power)
            
    return mse, rmse, psnr, snr


def encode(cover_path, secret_input, output_path):
    rate, audio = _read_audio_float(cover_path)
    original_copy = audio.copy()
    
    data_bytes = _get_data_bytes(secret_input)
    msg_bits = _bytes_to_bits(data_bytes)
    msg_len = len(msg_bits)
    
    seg_len, seg_num, bits_per_seg = _calculate_segment_params(len(audio), rate)
    capacity = bits_per_seg * seg_num
    
    if msg_len > capacity:
        raise ValueError(f"Phase Capacity Error: Can {msg_len}, co {capacity}.")
    
    print(f"   [Phase] Dang nhung {msg_len} bits...")
    
    # Xu ly Phase (Embed)
    target_len = seg_num * seg_len
    if len(audio) < target_len: audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
    segs = audio.reshape((seg_num, seg_len))
    fft_segs = np.fft.fft(segs)
    M = np.abs(fft_segs)
    P = np.angle(fft_segs)
    M += 1e-12 
    
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
    
    _write_audio_float(output_path, rate, stego_audio)
    
    mse, rmse, psnr, snr = calculate_metrics_float(original_copy, stego_audio)
    
    # TRA VE DICT CHO MAIN.PY
    return {
        "status": "success",
        "output_path": output_path,
        "mse": mse,
        "psnr": psnr,
        "snr": snr,
        "capacity": capacity
    }

def decode(stego_path):
    try:
        rate, stego_audio = _read_audio_float(stego_path)
        seg_len, seg_num, bits_per_seg = _calculate_segment_params(len(stego_audio), rate)
        
        print(f"   [Phase Process] Dang quet {seg_num} phan doan pha...")
        
        extracted_bits_list = []
        seg_mid = seg_len // 2
        start_idx = int(seg_mid * 0.1)
        
        for i in range(seg_num):
            segment_start = i * seg_len
            segment_end = (i + 1) * seg_len
            if segment_end > len(stego_audio):
                segment = np.pad(stego_audio[segment_start:], (0, segment_end - len(stego_audio)), mode='constant')
            else:
                segment = stego_audio[segment_start:segment_end]
                
            fft_segment = np.fft.fft(segment)
            extracted_phase = np.angle(fft_segment)
            phase_data = extracted_phase[start_idx : start_idx + bits_per_seg]
            extracted_bits = (phase_data < 0).astype(np.uint8)
            extracted_bits_list.extend(extracted_bits)
            
        all_bits = np.array(extracted_bits_list, dtype=np.uint8)
        all_bytes = _bits_to_bytes(all_bits)
        
        delimiter = b"||DATA_END||"
        delimiter_index = all_bytes.find(delimiter)
        
        if delimiter_index != -1:
            content = all_bytes[:delimiter_index]
            
            # --- NHAN DIEN FILE ---
            if content.startswith(b'\xff\xd8\xff'): return {'type': 'image', 'data': content, 'ext': '.jpg'}
            if content.startswith(b'\x89\x50\x4e\x47'): return {'type': 'image', 'data': content, 'ext': '.png'}
            if content.startswith(b'PK\x03\x04'): return {'type': 'archive', 'data': content, 'ext': '.zip'}
            
            try:
                text = content.decode('utf-8')
                return {'type': 'text', 'data': content, 'ext': '.txt'}
            except:
                return {'type': 'binary', 'data': content, 'ext': '.bin'}
        else:
            return {'type': 'error', 'message': "Khong tim thay dau hieu ket thuc (Phase)."}

    except Exception as e:
        return {'type': 'error', 'message': str(e)}

def process_batch(input_dir, secret_input, **kwargs):
    """
    Chay Phase Coding hang loat (Khong luu file).
    """
    results = []
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
    total_files = len(files)
    
    if total_files == 0: return []
    print(f"[Phase Batch] Tim thay {total_files} file. Bat dau...")

    # Chuan bi du lieu
    data_bytes = _get_data_bytes(secret_input)
    msg_bits = _bytes_to_bits(data_bytes)
    msg_len = len(msg_bits)

    for idx, filename in enumerate(files):
        filepath = os.path.join(input_dir, filename)
        try:
            # 1. Doc Audio (Float)
            rate, audio = _read_audio_float(filepath)
            original_audio_copy = audio.copy()
            
            # 2. Tinh tham so
            seg_len, seg_num, bits_per_seg = _calculate_segment_params(len(audio), rate)
            capacity = bits_per_seg * seg_num
            
            if msg_len > capacity:
                print(f"  [Bo qua] {filename}: Khong du dung luong (Can {msg_len}, co {capacity}).")
                continue

            # 3. Xu ly Phase (Trong RAM)
            target_len = seg_num * seg_len
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
            
            segs = audio.reshape((seg_num, seg_len))
            fft_segs = np.fft.fft(segs)
            M = np.abs(fft_segs)
            P = np.angle(fft_segs)
            M += 1e-12 
            
            PHASE_0 = np.pi / 4
            PHASE_1 = -np.pi / 4
            phase_values = np.where(msg_bits == 0, PHASE_0, PHASE_1)
            
            seg_mid = seg_len // 2
            start_idx = int(seg_mid * 0.1)
            current_msg_idx = 0
            
            for i in range(seg_num):
                bits_to_embed = min(bits_per_seg, msg_len - current_msg_idx)
                if bits_to_embed <= 0: break
                
                segment_phases = phase_values[current_msg_idx : current_msg_idx + bits_to_embed]
                embed_start = start_idx
                embed_end = start_idx + bits_to_embed
                
                P[i, embed_start:embed_end] = segment_phases
                P[i, seg_len - embed_end + 1 : seg_len - embed_start + 1] = -segment_phases[::-1]
                current_msg_idx += bits_to_embed

            # 4. Tai tao Audio (IFFT)
            modified_fft_segs = M * np.exp(1j * P)
            stego_audio = np.fft.ifft(modified_fft_segs).real.ravel()
            
            # 5. Danh gia ngay lap tuc
            mse, rmse, psnr, snr = calculate_metrics_float(original_audio_copy, stego_audio)
            
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