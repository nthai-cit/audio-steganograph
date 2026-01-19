import os
import numpy as np
from scipy.io import wavfile

# --- CAC HAM HO TRO ---

def _get_data_bytes(secret_input):
    if os.path.isfile(secret_input):
        print(f"   [LSB Info] Nhan dien du lieu la FILE: '{os.path.basename(secret_input)}'")
        with open(secret_input, 'rb') as f:
            return f.read()
    else:
        print(f"   [LSB Info] Nhan dien du lieu la VAN BAN.")
        return secret_input.encode('utf-8')

def _bytes_to_bitstream(data_bytes):
    binary_data = ''.join(format(byte, '08b') for byte in data_bytes)
    delimiter_binary = ''.join(format(byte, '08b') for byte in b"||DATA_END||")
    return binary_data + delimiter_binary

def _bitstream_to_bytes(bit_str):
    return bytearray(int(bit_str[i:i+8], 2) for i in range(0, len(bit_str) - len(bit_str) % 8, 8))

# --- HAM DANH GIA ---
def calculate_metrics(original, stego):
    orig = original.astype(np.float64)
    mod = stego.astype(np.float64)
    diff = orig - mod
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    
    if mse == 0: return 0, float('inf'), float('inf')
    
    max_val = 32767.0 
    psnr = 20 * np.log10(max_val / rmse)
    signal_power = np.sum(orig ** 2)
    snr = 10 * np.log10(signal_power / mse)
            
    return mse, rmse, psnr, snr

# --- CAC HAM CHINH ---

def encode(cover_path, secret_input, output_path):
    try:
        sample_rate, audio_data = wavfile.read(cover_path)
        original_shape = audio_data.shape
        
        # Chuan hoa int16
        if audio_data.dtype != np.int16:
             audio_data = (audio_data * 32767).astype(np.int16) if audio_data.dtype == np.float32 else audio_data.astype(np.int16)

        audio_data_flat = audio_data.flatten()
        stego_data_flat = audio_data_flat.copy()

        data_bytes = _get_data_bytes(secret_input)
        bitstream = _bytes_to_bitstream(data_bytes)
        data_length = len(bitstream)

        if data_length > len(stego_data_flat):
            raise ValueError(f"Du lieu qua lon ({data_length} bits) so voi dung luong audio.")

        print(f"   [LSB Process] Dang nhung {len(data_bytes)} bytes du lieu...")

        for i in range(data_length):
            stego_data_flat[i] = (stego_data_flat[i] & ~1) | int(bitstream[i])

        stego_data = stego_data_flat.reshape(original_shape)
        wavfile.write(output_path, sample_rate, stego_data)
        
        print(f"   [DANH GIA] Dang tinh toan chi so...")
        mse, rmse, psnr, snr = calculate_metrics(audio_data, stego_data)
        
        print("-" * 40)
        print(f"   BANG KET QUA (LSB CO BAN):")
        print(f"   [-] MSE  : {mse:.4f}")
        print(f"   [+] SNR  : {snr:.2f} dB")
        print(f"   [+] PSNR : {psnr:.2f} dB")
        print("-" * 40)
        
        return output_path

    except Exception as e:
        raise RuntimeError(f"Loi Encode LSB: {e}")

def decode(stego_path):
    try:
        _, stego_data = wavfile.read(stego_path)
        stego_data_flat = stego_data.flatten()

        binary_extracted = ""
        delimiter_binary = ''.join(format(byte, '08b') for byte in b"||DATA_END||")
        delimiter_len = len(delimiter_binary)

        print(f"   [LSB Process] Dang quet du lieu...")

        found = False
        # Luu y: Vong lap nay co the cham voi file rat lon
        # Trong thuc te nen dung numpy vectorization, nhung de giu logic de hieu ta dung loop
        # De tang toc, ta gioi han quet neu file qua lon hoac dung numpy
        
        # --- CACH TANG TOC BANG NUMPY (Thay the vong lap for cham chap) ---
        # Lay bit LSB cua toan bo mang
        lsb_bits = stego_data_flat & 1
        # Chuyen thanh chuoi bit (packbits nhanh hon loop string)
        # Tuy nhien de tim delimiter chinh xac, ta chuyen ve bytes truoc
        all_bytes = np.packbits(lsb_bits).tobytes()
        
        # Tim chuoi ket thuc
        delimiter_bytes = b"||DATA_END||"
        delimiter_index = all_bytes.find(delimiter_bytes)
        
        if delimiter_index != -1:
            extracted_bytes = all_bytes[:delimiter_index]
            
            # --- KIEM TRA LOAI FILE (MAGIC NUMBERS) ---
            
            # 1. Header JPEG (FF D8 FF)
            if extracted_bytes.startswith(b'\xff\xd8\xff'):
                out_file = stego_path + ".extracted.jpg"
                with open(out_file, "wb") as f:
                    f.write(extracted_bytes)
                return f"[FILE ANH] Da luu tai: {out_file}"
            
            # 2. Header PNG (89 50 4E 47)
            elif extracted_bytes.startswith(b'\x89\x50\x4e\x47'):
                out_file = stego_path + ".extracted.png"
                with open(out_file, "wb") as f:
                    f.write(extracted_bytes)
                return f"[FILE ANH] Da luu tai: {out_file}"

            # 3. Thu giai ma Text
            try:
                return f"[Van ban]: {extracted_bytes.decode('utf-8')}"
            except:
                # 4. Mac dinh la Binary
                out_file = stego_path + ".extracted.bin"
                with open(out_file, "wb") as f:
                    f.write(extracted_bytes)
                return f"[FILE BINARY] Da luu tai: {out_file}"
        
        else:
            return "[THAT BAI] Khong tim thay thong diep."

    except Exception as e:
        raise RuntimeError(f"Loi Decode LSB: {e}")

def process_batch(input_dir, secret_input):
    """
    Chay LSB tren toan bo file trong thu muc.
    Tinh toan chi so ngay trong RAM, khong ghi file ra dia.
    """
    results = []
    
    # Lay danh sach file wav
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
    total_files = len(files)
    
    if total_files == 0:
        print("[LSB Batch] Khong tim thay file .wav nao trong thu muc.")
        return []

    print(f"[LSB Batch] Tim thay {total_files} file. Bat dau thuc nghiem...")
    
    # Chuan bi du lieu bi mat 1 lan
    data_bytes = _get_data_bytes(secret_input)
    bitstream = _bytes_to_bitstream(data_bytes)
    data_length = len(bitstream)

    for idx, filename in enumerate(files):
        filepath = os.path.join(input_dir, filename)
        try:
            # 1. Doc file
            rate, audio_data = wavfile.read(filepath)
            
            # Chuan hoa ve int16
            if audio_data.dtype != np.int16:
                 audio_data = (audio_data * 32767).astype(np.int16) if audio_data.dtype == np.float32 else audio_data.astype(np.int16)
            
            original_flat = audio_data.flatten()
            stego_flat = original_flat.copy()
            
            # 2. Kiem tra dung luong
            if data_length > len(stego_flat):
                print(f"  [Bo qua] {filename}: File qua nho so voi du lieu.")
                continue

            # 3. Nhung tin (Trong RAM)
            # Logic giong ham encode nhung khong ghi file
            for i in range(data_length):
                stego_flat[i] = (stego_flat[i] & ~1) | int(bitstream[i])
            
            # 4. Danh gia ngay lap tuc
            stego_data = stego_flat.reshape(audio_data.shape)
            mse, rmse, psnr, snr = calculate_metrics(audio_data, stego_data)
            
            # 5. Luu ket qua
            results.append({
                "Filename": filename,
                "MSE": mse,
                "PSNR": psnr,
                "SNR": snr,
                "Capacity_Bits": len(stego_flat),
                "Status": "Success"
            })
            
            # Hien thi tien do
            print(f"  [{idx+1}/{total_files}] {filename} -> PSNR: {psnr:.2f} dB")

        except Exception as e:
            print(f"  [Loi] {filename}: {e}")
            results.append({"Filename": filename, "Status": "Error"})
            
    return results