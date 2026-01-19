import os
import shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from PIL import Image
import io

def plot_audio_comparison(orig_path, stego_path):
    """
    Ve bieu do so sanh 2 file am thanh va sai so (Difference).
    """
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
        
        # 1. Song am Goc
        plt.subplot(3, 1, 1)
        plt.plot(data1, color='blue')
        plt.title("1. Tin hieu Goc (Cover Audio)")
        plt.grid(True, alpha=0.3)
        
        # 2. Song am sau khi giau
        plt.subplot(3, 1, 2)
        plt.plot(data2, color='green')
        plt.title("2. Tin hieu Da giau tin (Stego Audio)")
        plt.grid(True, alpha=0.3)
        
        # 3. Su khac biet (Tin giau nam o day)
        plt.subplot(3, 1, 3)
        plt.plot(diff, color='red', linewidth=0.5)
        plt.title("3. Sai so (Du lieu da giau)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        print(f"   [VISUALIZE] Dang bat cua so bieu do...")
        plt.show() # Lenh nay se dung chuong trinh den khi ban tat cua so
        
    except Exception as e:
        print(f"[LOI VISUALIZE] Khong the ve bieu do: {e}")
def plot_batch_results(results):
    """
    Ve bieu do danh gia tu danh sach ket qua Batch.
    """
    if not results:
        print("[VISUALIZE] Khong co du lieu de ve bieu do.")
        return

    # Loc lay cac file thanh cong
    data = [r for r in results if r.get("Status") == "Success"]
    
    if not data:
        print("[VISUALIZE] Khong co file nao chay thanh cong.")
        return

    # Lay du lieu
    filenames = [item['Filename'] for item in data]
    psnr_vals = [item['PSNR'] for item in data]
    snr_vals = [item['SNR'] for item in data]
    
    # Gioi han hien thi neu qua nhieu file
    if len(filenames) > 20:
        print(f"[VISUALIZE] Hien thi 20/{len(filenames)} file dau tien tren bieu do...")
        filenames = filenames[:20]
        psnr_vals = psnr_vals[:20]
        snr_vals = snr_vals[:20]

    x = np.arange(len(filenames))
    width = 0.35

    plt.figure(figsize=(14, 7))
    
    plt.bar(x - width/2, psnr_vals, width, label='PSNR (dB)', color='#4CAF50')
    plt.bar(x + width/2, snr_vals, width, label='SNR (dB)', color='#2196F3')
    
    plt.xlabel('File Audio')
    plt.ylabel('Gia tri (dB)')
    plt.title('THONG KE CHAT LUONG GIAU TIN (PSNR & SNR)')
    plt.xticks(x, filenames, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    print("[VISUALIZE] Dang bat cua so bieu do thong ke...")
    
    # Hien thi thong so trung binh
    avg_psnr = np.mean([item['PSNR'] for item in data])
    avg_snr = np.mean([item['SNR'] for item in data])
    plt.figtext(0.15, 0.02, f"Trung binh: PSNR={avg_psnr:.2f}dB | SNR={avg_snr:.2f}dB", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2})
    
    plt.show()

def show_extracted_content(content_path):
    """
    Tu dong mo file anh hoac phat am thanh dua tren duoi file.
    """
    if not os.path.exists(content_path):
        return

    try:
        # Neu la file Anh
        if content_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            print(f"   [VISUALIZE] Dang mo anh: {content_path}")
            img = Image.open(content_path)
            img.show() # Bat trinh xem anh mac dinh cua Windows
            
        # Neu la file Am thanh (WAV, MP3)
        elif content_path.lower().endswith(('.wav', '.mp3')):
            print(f"   [VISUALIZE] Dang phat am thanh: {content_path}")
            # Lenh mo file tren Windows
            os.startfile(content_path)
            
    except Exception as e:
        print(f"[LOI VISUALIZE] Khong the mo file: {e}")

def show_data_from_memory(data_bytes, data_type):
    """Hien thi du lieu truc tiep tu RAM."""
    try:
        if data_type == 'image':
            print(f"   [VISUALIZE] Dang hien thi anh tu RAM (Khong luu file)...")
            image_stream = io.BytesIO(data_bytes)
            img = Image.open(image_stream)
            img.show()
        elif data_type == 'text':
            print(f"   [VISUALIZE] Noi dung tin mat:\n{'-'*30}\n{data_bytes.decode('utf-8')}\n{'-'*30}")
        elif data_type == 'binary':
            print(f"   [VISUALIZE] File nhi phan ({len(data_bytes)} bytes).")
    except Exception as e:
        print(f"[LOI VISUALIZE] {e}")

def save_to_downloads(source_path):
    """Copy file sang Downloads va tra ve duong dan moi."""
    try:
        downloads_dir = str(Path.home() / "Downloads")
        filename = os.path.basename(source_path)
        dest_path = os.path.join(downloads_dir, filename)
        shutil.copy2(source_path, dest_path)
        print(f"   [DOWNLOAD] Da tai file ve: {dest_path}")
        return dest_path
    except Exception as e:
        print(f"[LOI DOWNLOAD] {e}")
        return None

def launch_file(file_path):
    """
    Kich hoat file (Tuong duong Double-Click).
    """
    if not file_path or not os.path.exists(file_path): return
    
    try:
        if os.name == 'nt': # Windows
            print(f"   [SYSTEM] Dang mo file len...")
            os.startfile(file_path)
        else: # macOS / Linux
            import subprocess
            opener = 'open' if sys.platform == 'darwin' else 'xdg-open'
            subprocess.call([opener, file_path])
    except Exception as e:
        print(f"[LOI MO FILE] Khong the tu dong mo file: {e}")