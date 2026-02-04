import os
import subprocess
import glob
import time
import argparse
import sys
import numpy as np
import random
import soundfile as sf
import shutil
import re
from datetime import datetime, timedelta, timezone

CASE_CONFIGS = {
    1: {"name": "1_NoRandom", "flags": ["--no_random"], "desc": "Sequential"},
    2: {"name": "2_Random_Fixed_DefaultSalt", "flags": [], "desc": "Rnd_Fixed"},
    3: {"name": "3_Random_Fixed_ContentSalt", "flags": ["--salt_content"], "desc": "Rnd_Fixed_Content"},
    4: {"name": "4_Random_Adaptive_DefaultSalt", "flags": ["--adaptive"], "desc": "Rnd_Adaptive"},
    5: {"name": "5_Random_Adaptive_ContentSalt", "flags": ["--adaptive", "--salt_content"], "desc": "Rnd_Adaptive_Content"},
    6: {"name": "6_LSBMR_Literature", "flags": [], "desc": "LSBMR_Literature"},
    7: {"name": "7_PhaseCoding", "flags": ["--phase"], "desc": "Phase_Coding_FFT"}
}

def get_vn_time():
    return datetime.now(timezone.utc) + timedelta(hours=7)

def calculate_metrics(cover_path, stego_path):
    try:
        if not os.path.exists(cover_path) or not os.path.exists(stego_path):
            return 0.0, 0.0, 0.0

        c_sig, sr = sf.read(cover_path)
        s_sig, _ = sf.read(stego_path)
        
        c_sig = c_sig.astype(np.float64)
        s_sig = s_sig.astype(np.float64)
        
        if c_sig.ndim == 2 and s_sig.ndim == 1:
            c_sig = c_sig[:, 0]
        elif c_sig.ndim == 1 and s_sig.ndim == 2:
            s_sig = s_sig[:, 0]

        min_len = min(len(c_sig), len(s_sig))
        c_sig = c_sig[:min_len]
        s_sig = s_sig[:min_len]
        
        diff = c_sig - s_sig
        mse = np.mean(diff ** 2)

        if mse == 0:
            return 0.0, 100.0, 100.0

        max_val = 1.0 
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
        
        signal_power = np.mean(c_sig ** 2)
        snr = 10 * np.log10(signal_power / mse)

        return mse, psnr, snr
    except Exception:
        return 0.0, 0.0, 0.0

def get_files_recursive(directory, extensions):
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, "**", f"*.{ext}"), recursive=True))
    return files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case_id', type=int, required=True)
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_base', required=True)
    parser.add_argument('--secret_dir', required=True)
    
    parser.add_argument('--max_files', type=int, default=1000)
    parser.add_argument('--limit', type=int, default=50)
    
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--password', type=str, default="PASS")
    parser.add_argument('--size', type=int, default=256)
    
    args = parser.parse_args()

    config = CASE_CONFIGS.get(args.case_id)
    if not config:
        print(f"Error: Case ID {args.case_id} not found.")
        sys.exit(1)

    print(f"[*] Scanning Cover: {args.input_dir}...")
    all_covers = get_files_recursive(args.input_dir, ["wav", "WAV"])
    print(f"[*] Scanning Secret: {args.secret_dir}...")
    all_secrets = get_files_recursive(args.secret_dir, ["jpg", "png", "txt", "wav", "mp3", "pdf"])
    
    if not all_covers or not all_secrets:
        print("Error: Files not found.")
        sys.exit(1)

    random.shuffle(all_covers)
    num_covers = min(len(all_covers), args.max_files)
    target_covers = all_covers[:num_covers]
    
    random.shuffle(all_secrets)
    num_secrets_pool = min(len(all_secrets), args.limit)
    secret_pool = all_secrets[:num_secrets_pool]

    print("-" * 50)
    print(f"[*] Mode: DATASET GENERATION (VN Time)")
    print(f"[*] Cover: {num_covers} | Secret Pool: {num_secrets_pool}")
    print(f"[*] Image Size Target: {args.size}x{args.size}")
    print("-" * 50)

    timestamp = get_vn_time().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(args.output_base, f"{config['name']}_{timestamp}")
    dir_logs = os.path.join(base_dir, "logs")
    dir_cover = os.path.join(base_dir, "cover")
    dir_stego = os.path.join(base_dir, "stego")

    os.makedirs(dir_logs, exist_ok=True)
    os.makedirs(dir_cover, exist_ok=True)
    os.makedirs(dir_stego, exist_ok=True)

    log_file = os.path.join(dir_logs, f"benchmark.csv")

    # [MỚI] Thêm cột Rate, CPU, RAM vào Header
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Timestamp,CoverFile,SecretFile,Status,Time(s),MSE,PSNR,SNR,Rate(kBps),CPU(%),RAM(MB),Info\n")

    skipped_count = 0
    success_count = 0

    for i, c_file in enumerate(target_covers):
        s_file = random.choice(secret_pool)
        
        rel_path = os.path.relpath(c_file, args.input_dir)
        final_cover_path = os.path.join(dir_cover, rel_path)
        final_stego_path = os.path.join(dir_stego, rel_path)

        os.makedirs(os.path.dirname(final_cover_path), exist_ok=True)
        os.makedirs(os.path.dirname(final_stego_path), exist_ok=True)

        cmd = [sys.executable, "benchmark_stego.py", 
               "--cover", c_file, 
               "--secret", s_file,
               "--output", final_stego_path, 
               "--case_name", config['name'], 
               "--k", str(args.k), 
               "--password", args.password,
               "--size", str(args.size)] + config['flags']

        c_name = os.path.basename(c_file)
        start_t = time.time()
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            duration = time.time() - start_t
            
            if result.returncode == 0:
                mse, psnr, snr = calculate_metrics(c_file, final_stego_path)
                
                # [MỚI] Trích xuất Rate, CPU, RAM từ output của subprocess
                output_str = result.stdout
                rate_val = 0.0
                cpu_val = 0.0
                ram_val = 0.0
                
                # Regex bắt chuỗi: [METRICS_DATA] Rate=12.5 CPU=5.2 RAM=45.6
                match = re.search(r"Rate=([0-9\.]+).*CPU=([0-9\.]+).*RAM=([0-9\.]+)", output_str)
                if match:
                    rate_val = float(match.group(1))
                    cpu_val = float(match.group(2))
                    ram_val = float(match.group(3))

                if psnr < 1.0: 
                    skipped_count += 1
                    print(f"\r[{i+1}/{num_covers}] {c_name} -> Skipped (Low PSNR)", end="", flush=True)
                    if os.path.exists(final_stego_path): os.remove(final_stego_path)
                    continue
                
                shutil.copy2(c_file, final_cover_path)

                success_count += 1
                s_name = os.path.basename(s_file)
                current_time_str = get_vn_time().strftime('%H:%M:%S')
                
                # [MỚI] Ghi log đầy đủ
                log_line = f"{current_time_str},{c_name},{s_name},Success,{duration:.4f},{mse:.6f},{psnr:.2f},{snr:.2f},{rate_val:.4f},{cpu_val:.2f},{ram_val:.2f},{config['desc']}"
                
                # In ra màn hình console
                print(f"\r[{i+1}/{num_covers}] {c_name} -> OK (PSNR:{psnr:.1f}dB | CPU:{cpu_val}% | RAM:{ram_val:.1f}MB)", end="", flush=True)
                
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(log_line + "\n")

            else:
                print(f"\r[{i+1}/{num_covers}] {c_name} -> Skipped (Error)", end="", flush=True)

        except subprocess.TimeoutExpired:
            print(f"\r[{i+1}/{num_covers}] {c_name} -> Skipped (Timeout)", end="", flush=True)
            
        except Exception:
            print(f"\r[{i+1}/{num_covers}] {c_name} -> Skipped (Crash)", end="", flush=True)
        
    print(f"\n[DONE] Finished.")
    print(f"Structure: {base_dir}/[logs, cover, stego]")
    print(f"Total: {num_covers} | Created: {success_count} | Skipped: {skipped_count}")

if __name__ == "__main__":
    main()