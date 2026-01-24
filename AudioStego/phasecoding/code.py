import os
import numpy as np
import scipy.io.wavfile as wavfile
import math
import platform
import subprocess

# --- 1. CAC HAM HO TRO (HELPERS) ---

def _open_file_os(filepath):
    """Tu dong mo file bang trinh mac dinh"""
    try:
        if platform.system() == 'Windows':
            os.startfile(filepath)
        elif platform.system() == 'Darwin':
            subprocess.call(('open', filepath))
        else:
            subprocess.call(('xdg-open', filepath))
    except Exception:
        pass

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
    seg_len = 2 ** int(np.log2(sample_rate * 1.5)) # Doan 1.5s
    seg_len = min(seg_len, 131072)
    seg_len = max(seg_len, 8192)
    seg_num = int(np.ceil(audio_len / seg_len))
    usable_bins = int((seg_len // 2) * 0.8) 
    return seg_len, seg_num, usable_bins

def calculate_metrics_float(original, stego):
    """Tinh toan metrics cho du lieu float (-1.0 den 1.0)"""
    min_len = min(len(original), len(stego))
    orig = original[:min_len].astype(np.float64)
    mod = stego[:min_len].astype(np.float64)
    
    diff = orig - mod
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    
    if mse == 0:
        return 0, float('inf'), float('inf'), float('inf')
    
    max_val = 1.0
    psnr = 20 * np.log10(max_val / rmse)
    
    signal_power = np.sum(orig ** 2)
    noise_power = np.sum(diff ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
            
    return mse, rmse, psnr, snr


# --- 2. ENCODE (PHASE CODING) ---
def encode(cover_path, secret_input, output_path, **kwargs):
    # **kwargs de nhan cac tham so thua tu main.py ma khong bao loi
    try:
        rate, audio = _read_audio_float(cover_path)
        original_copy = audio.copy()
        
        data_bytes = _get_data_bytes(secret_input)
        msg_bits = _bytes_to_bits(data_bytes)
        msg_len = len(msg_bits)
        
        seg_len, seg_num, bits_per_seg = _calculate_segment_params(len(audio), rate)
        capacity = bits_per_seg * seg_num
        
        if msg_len > capacity:
            raise ValueError(f"File qua lon! Can {msg_len} bits, nhung chi chua duoc {capacity} bits.")
        
        print(f"   [Phase] Dang nhung {msg_len} bits...")
        
        # --- LOGIC NHUNG PHASE ---
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
            
            # Thay doi pha tan so duong
            P[i, embed_start:embed_start+bits_here] = seg_phases
            # Doi xung pha tan so am (de tin hieu la so thuc)
            P[i, seg_len - (embed_start+bits_here) + 1 : seg_len - embed_start + 1] = -seg_phases[::-1]
            
            curr += bits_here
            
        mod_fft = M * np.exp(1j * P)
        stego_audio = np.fft.ifft(mod_fft).real.ravel()
        
        _write_audio_float(output_path, rate, stego_audio)
        
        mse, rmse, psnr, snr = calculate_metrics_float(original_copy, stego_audio)
        
        print("-" * 40)
        print(f"   BANG KET QUA (PHASE):")
        print(f"   [-] MSE  : {mse:.6f}")
        print(f"   [+] SNR  : {snr:.2f} dB")
        print(f"   [+] PSNR : {psnr:.2f} dB")
        print("-" * 40)

        return {
            "status": "success",
            "output_path": output_path,
            "mse": mse, "psnr": psnr, "snr": snr,
            "capacity": capacity
        }
    except Exception as e:
        raise RuntimeError(f"Loi Encode Phase: {e}")


# --- 3. DECODE (PHASE CODING + AUTO DETECT) ---
def decode(stego_path, output_folder="outputs", **kwargs):
    try:
        # Chuan bi
        stego_path = os.path.abspath(stego_path)
        if not os.path.exists(stego_path):
            return {'status': 'error', 'message': 'File not found'}
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"   [Phase Decode] Dang xu ly: {os.path.basename(stego_path)}")

        # Doc Audio
        rate, stego_audio = _read_audio_float(stego_path)
        seg_len, seg_num, bits_per_seg = _calculate_segment_params(len(stego_audio), rate)
        
        extracted_bits_list = []
        seg_mid = seg_len // 2
        start_idx = int(seg_mid * 0.1)
        
        # --- LOGIC TRICH XUAT ---
        for i in range(seg_num):
            segment_start = i * seg_len
            segment_end = (i + 1) * seg_len
            
            if segment_end > len(stego_audio):
                segment = np.pad(stego_audio[segment_start:], (0, segment_end - len(stego_audio)), mode='constant')
            else:
                segment = stego_audio[segment_start:segment_end]
                
            fft_segment = np.fft.fft(segment)
            extracted_phase = np.angle(fft_segment)
            
            # Lay pha tai vi tri da giau
            phase_data = extracted_phase[start_idx : start_idx + bits_per_seg]
            
            # Quy uoc: Pha < 0 la bit 1, Pha > 0 la bit 0
            extracted_bits = (phase_data < 0).astype(np.uint8)
            extracted_bits_list.extend(extracted_bits)
            
        all_bits = np.array(extracted_bits_list, dtype=np.uint8)
        all_bytes = _bits_to_bytes(all_bits)
        
        # Tim dau hieu ket thuc
        delimiter = b"||DATA_END||"
        delimiter_index = all_bytes.find(delimiter)
        
        if delimiter_index == -1:
            return {'status': 'error', 'message': "Khong tim thay dau hieu ket thuc (Co the file chua duoc giau tin)."}
            
        content = all_bytes[:delimiter_index]
        result = {'status': 'success', 'data': content, 'data_len': len(content)}
        
        print("\n" + "="*20 + " KET QUA GIAI MA " + "="*20)

        # --- NHAN DIEN HEADER (TUONG TU LSB) ---
        sigs = {
            b'\xff\xd8\xff': ('.jpg', 'image'),
            b'\x89\x50\x4e\x47': ('.png', 'image'),
            b'BM': ('.bmp', 'image'),
            b'RIFF': ('.wav', 'audio'),
            b'PK\x03\x04': ('.zip', 'archive'),
            b'%PDF': ('.pdf', 'doc')
        }

        detected = False
        for sig, (ext, ftype) in sigs.items():
            if content.startswith(sig):
                if sig == b'RIFF' and content[8:12] != b'WAVE': continue
                
                print(f"[LOAI FILE]: {ftype.upper()} ({ext})")
                filename = f"extracted_{os.path.basename(stego_path)}{ext}"
                save_path = os.path.join(output_folder, filename)
                
                with open(save_path, 'wb') as f: f.write(content)
                print(f"[DA LUU TAI]: {save_path}")
                print(f"[HANH DONG]: Dang mo file...")
                _open_file_os(save_path)
                
                result.update({'type': ftype, 'ext': ext, 'save_path': save_path})
                detected = True
                break
        
        if not detected:
            try:
                text = content.decode('utf-8')
                if all(c.isprintable() or c.isspace() for c in text) and len(text) > 0:
                    print(f"[NOI DUNG VAN BAN]:\n{text}")
                    result.update({'type': 'text', 'content_text': text})
                else:
                    raise ValueError("Binary")
            except:
                print(f"[LOAI FILE]: KHONG XAC DINH (.bin)")
                filename = f"extracted_{os.path.basename(stego_path)}.bin"
                save_path = os.path.join(output_folder, filename)
                with open(save_path, 'wb') as f: f.write(content)
                result.update({'type': 'binary', 'ext': '.bin', 'save_path': save_path})

        print("="*57 + "\n")
        return result

    except Exception as e:
        return {'status': 'error', 'message': str(e)}


# --- 4. BATCH PROCESSING ---
def process_batch(input_dir, secret_input, output_dir, **kwargs):
    results = []
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
    total_files = len(files)
    
    if total_files == 0: 
        print("[Phase Batch] Khong tim thay file .wav")
        return []
        
    print(f"[Phase Batch] Tim thay {total_files} file. Bat dau xu ly...")

    data_bytes = _get_data_bytes(secret_input)
    msg_bits = _bytes_to_bits(data_bytes)
    msg_len = len(msg_bits)

    success_count = 0
    fail_count = 0
    skip_count = 0
    sum_psnr = 0
    sum_snr = 0

    for idx, filename in enumerate(files):
        print(f"\r[Phase Running] {idx+1}/{total_files} ... ", end="")
        filepath = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)
        
        try:
            # Reuse logic from encode function part manually to avoid overhead or file I/O locks
            rate, audio = _read_audio_float(filepath)
            original_copy = audio.copy()
            
            seg_len, seg_num, bits_per_seg = _calculate_segment_params(len(audio), rate)
            capacity = bits_per_seg * seg_num
            
            if msg_len > capacity:
                skip_count += 1
                results.append({"Filename": filename, "Status": "Skipped (Oversize)"})
                continue

            # Encode Logic (Simplified)
            target_len = seg_num * seg_len
            if len(audio) < target_len: audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
            
            segs = audio.reshape((seg_num, seg_len))
            fft_segs = np.fft.fft(segs)
            M, P = np.abs(fft_segs), np.angle(fft_segs)
            
            PHASE_0, PHASE_1 = np.pi/4, -np.pi/4
            phase_val = np.where(msg_bits == 0, PHASE_0, PHASE_1)
            
            curr = 0
            seg_mid = seg_len // 2
            start_idx = int(seg_mid * 0.1)
            
            for i in range(seg_num):
                bits_here = min(bits_per_seg, msg_len - curr)
                if bits_here <= 0: break
                seg_phs = phase_val[curr:curr+bits_here]
                emb_st = start_idx
                P[i, emb_st:emb_st+bits_here] = seg_phs
                P[i, seg_len-(emb_st+bits_here)+1 : seg_len-emb_st+1] = -seg_phs[::-1]
                curr += bits_here
                
            mod_fft = M * np.exp(1j * P)
            stego_audio = np.fft.ifft(mod_fft).real.ravel()
            
            _write_audio_float(out_path, rate, stego_audio)
            mse, rmse, psnr, snr = calculate_metrics_float(original_copy, stego_audio)
            
            results.append({
                "Filename": filename,
                "MSE": f"{mse:.6f}", "PSNR": f"{psnr:.2f}", "SNR": f"{snr:.2f}",
                "Status": "Success"
            })
            success_count += 1
            if psnr != float('inf'): sum_psnr += psnr
            if snr != float('inf'): sum_snr += snr

        except Exception as e:
            fail_count += 1
            results.append({"Filename": filename, "Status": "Error"})
            
    print("\n" + "="*40)
    print(f"KET QUA TONG HOP PHASE CODING")
    print(f"Thanh cong   : {success_count}")
    print(f"Bo qua (day) : {skip_count}")
    print(f"Loi          : {fail_count}")
    if success_count > 0:
        print(f"PSNR Trung binh : {sum_psnr/success_count:.2f} dB")
        print(f"SNR Trung binh  : {sum_snr/success_count:.2f} dB")
    print("="*40)

    return results