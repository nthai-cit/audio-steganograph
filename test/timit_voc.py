import os
import sys
import time
import argparse
import numpy as np
import random
import pandas as pd
import librosa
from scipy.io import wavfile
from tqdm import tqdm
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from AudioStego.improved_lsb import code as improved_algo
except ImportError:
    print("[ERROR] Module AudioStego not found.")
    sys.exit(1)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiment: VOC into TIMIT using improved LSB")
    
    parser.add_argument("-k", "--bits", type=int, default=2, help="LSB bits (k)")
    parser.add_argument("--timit", type=str, default=os.path.join(project_root, "inputs", "timit"))
    parser.add_argument("--voc", type=str, default=os.path.join(project_root, "inputs", "pascal-voc-2012"))
    parser.add_argument("--output", type=str, default=os.path.join(current_dir, "outputs"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--password", type=str, default="MySecurePass")
 
    parser.add_argument("--save-audio", action="store_true", 
                        help="Nếu có cờ này: LƯU file .wav. Nếu không: XÓA file sau khi tính xong (tiết kiệm ổ cứng).")
    
    return parser.parse_args()

def collect_files(timit_path, voc_path, limit):
    print(f"Scanning Audio: {timit_path}")
    print(f"Scanning Images: {voc_path}")
    
    all_wavs = []
    if os.path.exists(timit_path):
        for r, _, fs in os.walk(timit_path):
            for f in fs:
                if f.lower().endswith(".wav"):
                    all_wavs.append(os.path.join(r, f))

    all_imgs = []
    if os.path.exists(voc_path):
        for r, _, fs in os.walk(voc_path):
            for f in fs:
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    all_imgs.append(os.path.join(r, f))
    
    if not all_wavs: return None, None
    if not all_imgs: all_imgs = ["Simulated text message." * 50]

    n = min(len(all_wavs), limit)
    selected_wavs = random.sample(all_wavs, n)
    selected_imgs = [random.choice(all_imgs) for _ in range(n)]
    
    return selected_wavs, selected_imgs

def run_experiment(args):
    wav_paths, img_paths = collect_files(args.timit, args.voc, args.limit)
    if not wav_paths: return

    # Tạo thư mục
    session_dir = os.path.join(args.output, f"PSR_K{args.bits}_{time.strftime('%Y%m%d_%H%M%S')}")
    out_orig = os.path.join(session_dir, "original_wavs")
    out_stego = os.path.join(session_dir, "stego_wavs")
    

    if args.save_audio:
        os.makedirs(out_orig, exist_ok=True)
        os.makedirs(out_stego, exist_ok=True)
    else:
       
        os.makedirs(session_dir, exist_ok=True)
        out_orig = session_dir # Lưu tạm vào root session
        out_stego = session_dir 
    
    results = []
    fail_count = 0
    skip_count = 0
    
    print("-" * 60)
    print(f"STARTING EXPERIMENT (Mode: {'SAVE AUDIO' if args.save_audio else 'METRICS ONLY - NO AUDIO SAVED'})")
    print(f"Files: {len(wav_paths)}")
    print(f"Bits (k): {args.bits}")
    print("-" * 60)
    
    start_total = time.time()
    pbar = tqdm(range(len(wav_paths)), desc="Processing")
    
    for i in pbar:
        audio_src = wav_paths[i]
        image_src = img_paths[i]
        filename = os.path.basename(audio_src)
        unique_name = f"{i:04d}_{filename}"
        
        path_cover = os.path.join(out_orig, unique_name)
        path_stego = os.path.join(out_stego, f"stego_{unique_name}")
        
        try:
           
            y, sr = librosa.load(audio_src, sr=None)
            wavfile.write(path_cover, sr, (y * 32767).astype(np.int16))
            
            
            t0 = time.time()
            metrics = improved_algo.encode(
                cover_path=path_cover,
                secret_input=image_src, 
                output_path=path_stego,
                k=args.bits,
                password=args.password
            )
            t1 = time.time()
            
          
            if metrics.get('status') == 'success':
                results.append({
                    "ID": i,
                    "Filename": unique_name,
                    "K_Bit": args.bits,
                    "MSE": metrics['mse'],
                    "PSNR": metrics['psnr'],
                    "SNR": metrics['snr'],
                    "Capacity_Bytes": metrics.get('capacity', 0),
                    "Enc_Time_s": t1 - t0,
                    "Status": "Success"
                })
            else:
                msg = metrics.get('message', '')
                if "Oversize" in msg: skip_count += 1
                else: fail_count += 1

        except Exception:
            fail_count += 1
            
        finally:
           
            if not args.save_audio:
                if os.path.exists(path_stego): os.remove(path_stego)
                if os.path.exists(path_cover): os.remove(path_cover)

    if results:
        df = pd.DataFrame(results)
        csv_file = os.path.join(session_dir, "report_summary.csv")
        df.to_csv(csv_file, index=False)
        
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print(f"Total Files : {len(wav_paths)}")
        print(f"Success     : {len(results)}")
        print(f"Skipped     : {skip_count}")
        print(f"Failed      : {fail_count}")
        print(f"Total Time  : {time.time() - start_total:.2f} s")
        print("-" * 60)
        
        if not df.empty:
            print("AVERAGE METRICS:")
            print(f"PSNR  : {df['PSNR'].mean():.4f} dB")
            print(f"SNR   : {df['SNR'].mean():.4f} dB")
            print(f"MSE   : {df['MSE'].mean():.8f}")
            print(f"Speed : {df['Enc_Time_s'].mean():.4f} s/file")
        
        print(f"Report File: {csv_file}")
        if not args.save_audio:
            print("Audio Files: DELETED (Space saved)")
        print("="*60)
    else:
        print("\nNO SUCCESSFUL FILES.")

if __name__ == "__main__":
    args = parse_arguments()
    run_experiment(args)