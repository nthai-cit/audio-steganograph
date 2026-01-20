import os
import numpy as np
from scipy.io import wavfile

# --- CAC HAM HO TRO ---
def _get_data_bytes(secret_input):
    if os.path.isfile(secret_input):
        # Nhan dien ten file de in log cho dep
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

def encode(cover_path, secret_input, output_path):
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
    
def decode(stego_path):
    try:
        _, stego_data = wavfile.read(stego_path)
        
        # 1. Xu ly loi kieu du lieu (Fix loi Bitwise)
        if stego_data.dtype != np.int16:
             if np.issubdtype(stego_data.dtype, np.floating):
                 stego_data = (stego_data * 32767).astype(np.int16)
             else:
                 stego_data = stego_data.astype(np.int16)
            
        stego_data_flat = stego_data.flatten()
        
        # 2. Trich xuat LSB (Numpy Vectorization)
        lsb_bits = stego_data_flat & 1
        
        # 3. Gom bit thanh byte (Pack bits)
        all_bytes = np.packbits(lsb_bits).tobytes()
        
        # 4. Tim dau hieu ket thuc
        delimiter = b"||DATA_END||"
        delimiter_index = all_bytes.find(delimiter)
        
        if delimiter_index != -1:
            content = all_bytes[:delimiter_index]
            
            # --- NHAN DIEN DINH DANG FILE ---
            if content.startswith(b'\xff\xd8\xff'): return {'type': 'image', 'data': content, 'ext': '.jpg'}
            if content.startswith(b'\x89\x50\x4e\x47'): return {'type': 'image', 'data': content, 'ext': '.png'}
            if content.startswith(b'BM'): return {'type': 'image', 'data': content, 'ext': '.bmp'}
            if content.startswith(b'RIFF') and content[8:12] == b'WAVE': return {'type': 'audio', 'data': content, 'ext': '.wav'}
            if content.startswith(b'PK\x03\x04'): return {'type': 'archive', 'data': content, 'ext': '.zip'}
            
            try:
                # Thu decode text
                text = content.decode('utf-8')
                return {'type': 'text', 'data': content, 'ext': '.txt'}
            except:
                # Neu khong phai text thi la binary
                return {'type': 'binary', 'data': content, 'ext': '.bin'}
        
        else:
            return {'type': 'error', 'message': "Khong tim thay dau hieu ket thuc."}

    except Exception as e:
        return {'type': 'error', 'message': str(e)}

# --- BATCH PROCESSING (XU LY HANG LOAT) ---
def process_batch(input_dir, secret_input):
    results = []
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
    
    if not files:
        print("[LSB Batch] Khong tim thay file .wav nao.")
        return []

    print(f"[LSB Batch] Tim thay {len(files)} file. Bat dau...")
    
    # Chuan bi du lieu 1 lan de dung chung
    data_bytes = _get_data_bytes(secret_input)
    bitstream = _bytes_to_bitstream(data_bytes)
    data_length = len(bitstream)
    bits_array = np.array([int(b) for b in bitstream], dtype=np.int16)

    for idx, filename in enumerate(files):
        filepath = os.path.join(input_dir, filename)
        try:
            rate, audio_data = wavfile.read(filepath)
            
            # Chuan hoa int16
            if audio_data.dtype != np.int16: 
                audio_data = (audio_data * 32767).astype(np.int16) if audio_data.dtype == np.float32 else audio_data.astype(np.int16)
                
            original_flat = audio_data.flatten()
            stego_flat = original_flat.copy()
            
            if data_length > len(stego_flat):
                print(f"  [Bo qua] {filename}: Qua nho.")
                continue
            
            # Embed nhanh (Vectorization)
            stego_flat[:data_length] &= ~1
            stego_flat[:data_length] |= bits_array
            
            stego_data = stego_flat.reshape(audio_data.shape)
            mse, rmse, psnr, snr = calculate_metrics(audio_data, stego_data)
            
            results.append({
                "Filename": filename,
                "MSE": mse, "PSNR": psnr, "SNR": snr,
                "Status": "Success"
            })
            print(f"  [{idx+1}/{len(files)}] {filename} -> PSNR: {psnr:.2f} dB")
            
        except Exception as e:
            print(f"  [Loi] {filename}: {e}")
            results.append({"Filename": filename, "Status": "Error"})
            
    return results