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

# --- CAC HAM CHINH ---

def encode(cover_path, secret_input, output_path):
    try:
        # 1. Doc du lieu
        rate, audio = _read_audio_float(cover_path)
        original_audio_copy = audio.copy() # Luu ban sao de so sanh
        
        data_bytes = _get_data_bytes(secret_input)
        msg_bits = _bytes_to_bits(data_bytes)
        msg_len = len(msg_bits)
        
        seg_len, seg_num, bits_per_seg = _calculate_segment_params(len(audio), rate)
        
        capacity = bits_per_seg * seg_num
        if msg_len > capacity:
            raise ValueError(f"Khong du dung luong. Can {msg_len} bits, co {capacity} bits.")
        
        print(f"   [Phase Process] Dang nhung {msg_len} bits vao {seg_num} phan doan...")

        # Padding
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

        modified_fft_segs = M * np.exp(1j * P)
        stego_audio = np.fft.ifft(modified_fft_segs).real.ravel()
        
        # Ghi file
        _write_audio_float(output_path, rate, stego_audio)
        
        # 2. Danh gia chat luong
        print(f"   [DANH GIA] Dang tinh toan chi so (Mien thoi gian)...")
        mse, rmse, psnr, snr = calculate_metrics_float(original_audio_copy, stego_audio)
        
        print("-" * 40)
        print(f"   BANG KET QUA (PHASE CODING):")
        print(f"   [-] MSE  : {mse:.6f}")
        print(f"   [-] RMSE : {rmse:.6f}")
        print(f"   [+] SNR  : {snr:.2f} dB")
        print(f"   [+] PSNR : {psnr:.2f} dB")
        print("-" * 40)
        
        return output_path

    except Exception as e:
        raise RuntimeError(f"Loi Encode Phase: {e}")

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
        
        delimiter_index = all_bytes.find(b"||DATA_END||")
        
        if delimiter_index != -1:
            content_bytes = all_bytes[:delimiter_index]
            try:
                return f"[Van ban]: {content_bytes.decode('utf-8')}"
            except:
                out_file = stego_path + ".extracted_phase.bin"
                with open(out_file, "wb") as f:
                    f.write(content_bytes)
                return f"[FILE] Da luu tai: {out_file}"
        else:
            return "[THAT BAI] Khong tim thay chuoi ket thuc."

    except Exception as e:
        raise RuntimeError(f"Loi Decode Phase: {e}")