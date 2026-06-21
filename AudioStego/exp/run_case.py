import os
import subprocess
import glob
import time
import argparse
import sys
import numpy as np
import soundfile as sf
import re  # Thêm thư viện Regex
from datetime import datetime

# --- CẤU HÌNH CASE ---
CASE_CONFIGS = {
    1: {"name": "1_NoRandom", "flags": ["--no_random"], "desc": "Sequential"},
    2: {"name": "2_Random_Fixed_DefaultSalt", "flags": [], "desc": "Rnd_Fixed"},
    3: {"name": "3_Random_Fixed_ContentSalt", "flags": ["--salt_content"], "desc": "Rnd_Fixed_Content"},
    4: {"name": "4_Random_Adaptive_DefaultSalt", "flags": ["--adaptive"], "desc": "Rnd_Adaptive"},
    5: {"name": "5_Random_Adaptive_ContentSalt", "flags": ["--adaptive", "--salt_content"], "desc": "Rnd_Adaptive_Content"},
    6: {"name": "6_LSBMR_Literature", "flags": [], "desc": "LSBMR_Literature"},
    7: {"name": "7_PhaseCoding", "flags": ["--phase"], "desc": "Phase_Coding_FFT"},
    8: {"name": "8_Alarood2022_RandLSB", "flags": ["--alarood"], "desc": "Alarood2022_RandLSB"},
}

def calculate_metrics(cover_path, stego_path):
    """Tính MSE, PSNR, SNR"""
    try:
        if not os.path.exists(cover_path) or not os.path.exists(stego_path):
            return 0.0, 0.0, 0.0

        c_sig, sr = sf.read(cover_path)
        s_sig, _ = sf.read(stego_path)
        
        c_sig = c_sig.astype(np.float64)
        s_sig = s_sig.astype(np.float64)
        
        # Fix lỗi Stereo/Mono
        if c_sig.ndim == 2 and s_sig.ndim == 1:
            c_sig = c_sig[:, 0]
        elif c_sig.ndim == 1 and s_sig.ndim == 2:
            s_sig = s_sig[:, 0]

        if len(c_sig) == 0 or len(s_sig) == 0:
            return 0.0, 0.0, 0.0

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case_id', type=int, required=True)
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_base', required=True)
    parser.add_argument('--secret_file', type=str, required=True, help="Path to the secret file")
    
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--password', type=str, default="PASS")

    # [MỚI] Thêm cờ --no_save
    parser.add_argument('--no_save', action='store_true', help="Khong luu file output de tiet kiem dung luong (se xoa ngay sau khi chay)")
    
    args = parser.parse_args()

    if not os.path.exists(args.secret_file):
        print(f"Error: Secret file '{args.secret_file}' not found.")
        sys.exit(1)

    config = CASE_CONFIGS.get(args.case_id)
    if not config:
        print(f"Error: Case ID {args.case_id} not found.")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_folder_name = os.path.basename(os.path.normpath(args.input_dir))
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"benchmark_{input_folder_name}_{config['name']}_{timestamp}.csv")

    raw_files = glob.glob(os.path.join(args.input_dir, "**", "*.wav"), recursive=True) + \
                glob.glob(os.path.join(args.input_dir, "**", "*.WAV"), recursive=True)
    processed = {}
    for f in raw_files:
        base = os.path.splitext(os.path.basename(f))[0]
        if base not in processed:
            processed[base] = f
            
    files = sorted(list(processed.values()))

    case_output_dir = os.path.join(args.output_base, config['name'])
    os.makedirs(case_output_dir, exist_ok=True)

    # Header log (Bao gồm các chỉ số tài nguyên từ phiên bản trước)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Timestamp,Filename,Status,Time(s),MSE,PSNR,SNR,Rate(kBps),CPU(%),RAM(MB),Info\n")

    print(f"[*] Case: {config['name']} | Input: {input_folder_name}")
    print(f"[*] Secret: {args.secret_file}")
    print(f"[*] No Save Mode: {'ON' if args.no_save else 'OFF'}")
    print(f"[*] Found: {len(files)} files")

    for i, wav_file in enumerate(files):
        rel_path = os.path.relpath(wav_file, args.input_dir)
        out_path = os.path.join(case_output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        cmd = [sys.executable, "benchmark_stego.py", 
               "--cover", wav_file, 
               "--secret", args.secret_file,
               "--output", out_path, 
               "--case_name", config['name'], 
               "--k", str(args.k), 
               "--password", args.password] + config['flags']

        start_t = time.time()
        # capture_output=True để bắt log CPU/RAM từ benchmark_stego.py
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_t
        
        log_line = ""
        if result.returncode == 0:
            # Parse các thông số từ output của benchmark_stego.py
            output_str = result.stdout
            rate_val, cpu_val, ram_val = 0.0, 0.0, 0.0
            
            match = re.search(r"Rate=([0-9\.]+).*CPU=([0-9\.]+).*RAM=([0-9\.]+)", output_str)
            if match:
                rate_val = float(match.group(1))
                cpu_val = float(match.group(2))
                ram_val = float(match.group(3))

            # Xử lý Logic NO SAVE
            if args.no_save:
                # Nếu không lưu: Không tính metrics (vì file sẽ bị xóa), gán = 0
                mse, psnr, snr = 0, 0, 0
                # Xóa file ngay lập tức
                if os.path.exists(out_path):
                    os.remove(out_path)
                status_str = "Success(NoSave)"
            else:
                # Nếu lưu: Tính metrics bình thường
                mse, psnr, snr = calculate_metrics(wav_file, out_path)
                status_str = "Success"

            log_line = f"{datetime.now().strftime('%H:%M:%S')},{os.path.basename(wav_file)},{status_str},{duration:.4f},{mse:.6f},{psnr:.2f},{snr:.2f},{rate_val:.4f},{cpu_val:.2f},{ram_val:.2f},{config['desc']}"
            
            # In ra màn hình
            metric_display = "NoSave" if args.no_save else f"PSNR:{psnr:.1f}dB"
            print(f"\r[{i+1}/{len(files)}] {os.path.basename(wav_file)} -> {metric_display} | CPU:{cpu_val}% | RAM:{ram_val:.1f}MB", end="", flush=True)

        else:
            err_msg = "Error"
            log_line = f"{datetime.now().strftime('%H:%M:%S')},{os.path.basename(wav_file)},Failed,0,0,0,0,0,0,0,{err_msg}"
            print(f"\r[{i+1}/{len(files)}] {os.path.basename(wav_file)} -> FAILED", end="", flush=True)
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")

    # Dọn dẹp thư mục rỗng nếu chạy chế độ no_save
    if args.no_save:
        try:
            # Xóa thư mục output base nếu nó rỗng
            os.rmdir(case_output_dir) 
        except:
            pass 

    print(f"\n[DONE] Hoàn tất. Log tại: {log_file}")

if __name__ == "__main__":
    main()