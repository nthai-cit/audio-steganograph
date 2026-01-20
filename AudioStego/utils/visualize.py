import os
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from PIL import Image
import io
import sys

def plot_audio_comparison(orig_path, stego_path):

    try:
        # Doc du lieu
        rate1, data1 = wavfile.read(orig_path)
        rate2, data2 = wavfile.read(stego_path)
        
        # Chuyen ve float de ve cho dep va cat cho bang nhau
        min_len = min(len(data1), len(data2))
        data1 = data1.flatten()[:min_len].astype(np.float32)
        data2 = data2.flatten()[:min_len].astype(np.float32)
        
        # Tinh sai biet (Noise)
        diff = data1 - data2
        
        # Tao cua so bieu do
        plt.figure(figsize=(12, 8))
        plt.suptitle(f"SO SANH TIN HIEU AM THANH\nGot: {os.path.basename(orig_path)} | Stego: {os.path.basename(stego_path)}", fontsize=14)
        
        # Song am Goc
        plt.subplot(3, 1, 1)
        plt.plot(data1, color='blue')
        plt.title("1. Tin hieu Goc (Cover Audio)")
        plt.grid(True, alpha=0.3)
        
        # Song am sau khi giau
        plt.subplot(3, 1, 2)
        plt.plot(data2, color='green')
        plt.title("2. Tin hieu Da giau tin (Stego Audio)")
        plt.grid(True, alpha=0.3)
        
        # Su khac biet (Tin giau nam o day)
        plt.subplot(3, 1, 3)
        plt.plot(diff, color='red', linewidth=0.5)
        plt.title("3. Sai so (Du lieu da giau)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        print(f"   [VISUALIZE] Dang bat cua so bieu do...")
        plt.show() 
        
    except Exception as e:
        print(f"[LOI VISUALIZE] Khong the ve bieu do: {e}")


def plot_batch_results(results, input_dir=None):
    """
    Ve bieu do Batch:
    - Truc X: Index 
    - Truc Y1: PSNR
    - Truc Y2: SNR
    """
    if not results:
        print("[VISUALIZE] Khong co du lieu.")
        return

    # Loc lay file thanh cong
    data = [r for r in results if r.get("Status") == "Success"]
    if not data:
        print("[VISUALIZE] Khong co file nao Success.")
        return

    if input_dir:
        for item in data:
            try:
                file_path = os.path.join(input_dir, item['Filename'])
                item['_size_bytes'] = os.path.getsize(file_path)
            except:
                item['_size_bytes'] = 0
        
        
        data.sort(key=lambda x: x['_size_bytes'])
        print("[VISUALIZE] Da sap xep bieu do theo dung luong file.")

  
    psnr_vals = [item['PSNR'] for item in data]
    snr_vals = [item['SNR'] for item in data]
    

    if len(psnr_vals) > 100:
        print(f"[VISUALIZE] Chi hien thi 100 file dau tien...")
        psnr_vals = psnr_vals[:100]
        snr_vals = snr_vals[:100]

    x = np.arange(len(psnr_vals))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('DANH GIA CHAT LUONG GIAU TIN', fontsize=16)

    ax1.plot(x, psnr_vals, marker='.', color='green', linewidth=1, linestyle='-', label='PSNR')
    ax1.set_ylabel('PSNR (dB)', fontsize=12, color='green')
    ax1.set_title('1. Peak Signal-to-Noise Ratio (PSNR)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
 
    ax2.plot(x, snr_vals, marker='.', color='blue', linewidth=1, linestyle='-', label='SNR')
    ax2.set_ylabel('SNR (dB)', fontsize=12, color='blue')
    ax2.set_title('2. Signal-to-Noise Ratio (SNR)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
   
    ax2.set_xlabel('Danh sach file Audio', fontsize=12)
    
  
    ax2.set_xticklabels([]) 
    
    plt.tight_layout()
    print("[VISUALIZE] Dang bat cua so bieu do...")
    plt.show()

def show_extracted_content(content_path):
    if not os.path.exists(content_path): return
    try:
        if content_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            print(f"   [VISUALIZE] Dang mo anh: {content_path}")
            img = Image.open(content_path)
            img.show()
        elif content_path.lower().endswith(('.wav', '.mp3')):
            print(f"   [VISUALIZE] Dang phat am thanh: {content_path}")
            os.startfile(content_path)
    except Exception as e:
        print(f"[LOI VISUALIZE] Khong the mo file: {e}")

def show_data_from_memory(data_bytes, data_type):
    try:
        if data_type == 'image':
            print(f"   [VISUALIZE] Dang hien thi anh tu RAM...")
            image_stream = io.BytesIO(data_bytes)
            img = Image.open(image_stream)
            img.show(title="Anh giai ma") 
        elif data_type == 'text':
            print(f"\n{'='*20} NOI DUNG TIN MAT {'='*20}")
            try: print(data_bytes.decode('utf-8'))
            except: print(data_bytes.decode('latin-1'))
            print(f"{'='*60}\n")
        elif data_type == 'binary':
            print(f"   [VISUALIZE] File nhi phan ({len(data_bytes)} bytes).")
    except Exception as e:
        print(f"[LOI VISUALIZE] {e}")

def save_to_downloads(source_path):
    try:
        downloads_dir = str(Path.home() / "Downloads")
        filename = os.path.basename(source_path)
        dest_path = os.path.join(downloads_dir, filename)
        shutil.copy2(source_path, dest_path)
        print(f"   [DOWNLOAD] Da tai file ve: {dest_path}")
        return dest_path
    except: return None

def launch_file(file_path):
    if not file_path or not os.path.exists(file_path): return
    try:
        if os.name == 'nt': os.startfile(file_path)
        else:
            import subprocess
            opener = 'open' if sys.platform == 'darwin' else 'xdg-open'
            subprocess.call([opener, file_path])
    except: pass