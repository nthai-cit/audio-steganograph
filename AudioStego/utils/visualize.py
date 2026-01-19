import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from PIL import Image

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