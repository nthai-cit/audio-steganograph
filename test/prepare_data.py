import os
import numpy as np
from scipy.io import wavfile
from PIL import Image

def create_inputs():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    inputs_dir = os.path.join(project_root, "inputs")
    
    if not os.path.exists(inputs_dir):
        os.makedirs(inputs_dir)
        print(f"[INFO] Da tao thu muc: {inputs_dir}")

    wav_path = os.path.join(inputs_dir, "song.wav")
    print(f"Tao {wav_path} (5 giay)...")
    
    rate = 44100
    duration = 5 
    t = np.linspace(0, duration, rate * duration) 
    
    data = (np.sin(2 * np.pi * 440 * t) * 30000).astype(np.int16)
    wavfile.write(wav_path, rate, data)

    img_path = os.path.join(inputs_dir, "image.jpg")
    print(f"Tao {img_path}...")
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    img.save(img_path)
    
    txt_path = os.path.join(inputs_dir, "secret.txt")
    print(f"Tao {txt_path}...")
    with open(txt_path, "w") as f:
        f.write("Day la mat ma bi mat")

if __name__ == "__main__":
    create_inputs()
    print("[OK] Da tao du lieu test tai thu muc Goc.")