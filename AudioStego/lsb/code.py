import os
import numpy as np
from scipy.io import wavfile
import platform
import subprocess


def _get_data_bytes(secret_input):
    if os.path.isfile(secret_input):
        print(f"   [LSB Info] Nhan dien du lieu la FILE: '{os.path.basename(secret_input)}'")
        with open(secret_input, 'rb') as f: return f.read()
    else:
        print(f"   [LSB Info] Nhan dien du lieu la VAN BAN.")
        return secret_input.encode('utf-8')

def _bytes_to_bitstream(data_bytes):
    # Chuyen bytes thanh chuoi bit '010101...'
    binary_data = ''.join(format(byte, '08b') for byte in data_bytes)
    delimiter_binary = ''.join(format(byte, '08b') for byte in b"||DATA_END||")
    return binary_data + delimiter_binary

def _open_file_os(filepath):
    """Tu dong mo file bang trinh mac dinh cua he dieu hanh"""
    try:
        if platform.system() == 'Windows':
            os.startfile(filepath)
        elif platform.system() == 'Darwin':  # macOS
            subprocess.call(('open', filepath))
        else:  # Linux
            subprocess.call(('xdg-open', filepath))
    except Exception:
        pass

def calculate_metrics(original, stego):
    """Tinh toan MSE, RMSE, PSNR, SNR"""
    orig = original.astype(np.float64)
    mod = stego.astype(np.float64)
    diff = orig - mod
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    
    if mse == 0: return 0, float('inf'), float('inf'), float('inf')
    
    max_val = 32767.0 
    psnr = 20 * np.log10(max_val / rmse)
    signal_power = np.sum(orig ** 2)
    snr = 10 * np.log10(signal_power / mse)
    return mse, rmse, psnr, snr


def encode(cover_path, secret_input, output_path, k=1):
    try:
        sample_rate, audio_data = wavfile.read(cover_path)
        original_shape = audio_data.shape
        
        if audio_data.dtype != np.int16:
            if np.issubdtype(audio_data.dtype, np.floating):
                audio_data = (audio_data * 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)

        audio_data_flat = audio_data.flatten()
        stego_data_flat = audio_data_flat.copy()
        
        data_bytes = _get_data_bytes(secret_input)
        bitstream = _bytes_to_bitstream(data_bytes)
        data_length = len(bitstream)
        
        if data_length > len(stego_data_flat):
            raise ValueError(f"File qua nho! Can {data_length} bits, co {len(stego_data_flat)} bits.")
            
        print(f"   [LSB Process] Dang nhung {len(data_bytes)} bytes...")
        
        # Nhung LSB (Hardcoded 1 bit theo logic cua ban)
        bits_array = np.array([int(b) for b in bitstream], dtype=np.int16)
        stego_data_flat[:data_length] &= ~1 
        stego_data_flat[:data_length] |= bits_array
            
        stego_data = stego_data_flat.reshape(original_shape)
        wavfile.write(output_path, sample_rate, stego_data)
        
        print(f"   [DANH GIA] Dang tinh toan chi so...")
        mse, rmse, psnr, snr = calculate_metrics(audio_data, stego_data)
        
        print("-" * 40)
        print(f"   BANG KET QUA (LSB):")
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
            "capacity": len(stego_data_flat)
        }

    except Exception as e:
        raise RuntimeError(f"Loi Encode LSB: {e}")


def decode(stego_path, output_folder="outputs", k=1):
    try:
        stego_path = os.path.abspath(stego_path)
        if not os.path.exists(stego_path):
            raise FileNotFoundError(f"Khong tim thay file: {stego_path}")
            
        os.makedirs(output_folder, exist_ok=True)
        print(f"   [LSB Decode] Dang xu ly: {os.path.basename(stego_path)}")

        _, stego_data = wavfile.read(stego_path)
        if stego_data.dtype != np.int16:
             if np.issubdtype(stego_data.dtype, np.floating):
                 stego_data = (stego_data * 32767).astype(np.int16)
             else:
                 stego_data = stego_data.astype(np.int16)
        
        # Trich xuat LSB (1 bit)
        stego_flat = stego_data.flatten()
        lsb_bits = stego_flat & 1
        all_bytes = np.packbits(lsb_bits).tobytes()
        
        # Tim dau hieu ket thuc
        delimiter = b"||DATA_END||"
        end_idx = all_bytes.find(delimiter)
        
        if end_idx == -1:
            return {'status': 'error', 'message': "Khong tim thay chuoi ket thuc (File sach hoac loi)."}
            
        content = all_bytes[:end_idx]
        
        result = {'status': 'success', 'data': content, 'data_len': len(content)}
        print("\n" + "="*20 + " KET QUA GIAI MA " + "="*20)

        # Nhan dien Header
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
                text_content = content.decode('utf-8')
                if all(c.isprintable() or c.isspace() for c in text_content) and len(text_content) > 0:
                    print(f"[NOI DUNG VAN BAN]:\n{text_content}")
                    result.update({'type': 'text', 'content_text': text_content})
                else:
                    raise ValueError("Binary")
            except:
                print(f"[LOAI FILE]: KHONG XAC DINH (.bin)")
                filename = f"extracted_{os.path.basename(stego_path)}.bin"
                save_path = os.path.join(output_folder, filename)
                with open(save_path, 'wb') as f: f.write(content)
                print(f"[DA LUU TAI]: {save_path}")
                result.update({'type': 'binary', 'ext': '.bin', 'save_path': save_path})

        print("="*57 + "\n")
        return result

    except Exception as e:
        print(f"   [LOI]: {str(e)}")
        return {'status': 'error', 'message': str(e)}


def process_batch(input_dir, secret_input, output_dir, k=1, password=None):
    results = []
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
    total_files = len(files)
    
    if not files:
        print("[LSB Batch] Khong tim thay file .wav nao.")
        return []

    print(f"[LSB Batch] Tim thay {total_files} file. Bat dau xu ly...")
    
    data_bytes = _get_data_bytes(secret_input)
    bitstream = _bytes_to_bitstream(data_bytes)
    data_length = len(bitstream)
    bits_array = np.array([int(b) for b in bitstream], dtype=np.int16)

    success_count = 0
    fail_count = 0
    skip_count = 0
    sum_psnr = 0
    sum_snr = 0

    for idx, filename in enumerate(files):
        print(f"\r[LSB Running] {idx+1}/{total_files} ... ", end="")
        
        filepath = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)

        try:
            rate, audio_data = wavfile.read(filepath)
            if audio_data.dtype != np.int16: 
                audio_data = (audio_data * 32767).astype(np.int16) if audio_data.dtype == np.float32 else audio_data.astype(np.int16)
                
            original_flat = audio_data.flatten()
            stego_flat = original_flat.copy()
            
            if data_length > len(stego_flat):
                skip_count += 1
                results.append({"Filename": filename, "Status": "Skipped (Oversize)"})
                continue
            
            stego_flat[:data_length] &= ~1
            stego_flat[:data_length] |= bits_array
            
            stego_data = stego_flat.reshape(audio_data.shape)
            wavfile.write(out_path, rate, stego_data)

            mse, rmse, psnr, snr = calculate_metrics(audio_data, stego_data)
            
            results.append({
                "Filename": filename,
                "MSE": f"{mse:.4f}", "PSNR": f"{psnr:.2f}", "SNR": f"{snr:.2f}",
                "Status": "Success"
            })
            
            success_count += 1
            if psnr != float('inf'): sum_psnr += psnr
            if snr != float('inf'): sum_snr += snr
            
        except Exception as e:
            fail_count += 1
            results.append({"Filename": filename, "Status": "Error"})

    print("\n" + "="*40)
    print(f"KET QUA TONG HOP LSB")
    print(f"Thanh cong   : {success_count}")
    print(f"Bo qua (day) : {skip_count}")
    print(f"Loi          : {fail_count}")
    
    if success_count > 0:
        avg_psnr = sum_psnr / success_count
        avg_snr = sum_snr / success_count
        print(f"PSNR Trung binh : {avg_psnr:.2f} dB")
        print(f"SNR Trung binh  : {avg_snr:.2f} dB")
    print("="*40)
            
    return results