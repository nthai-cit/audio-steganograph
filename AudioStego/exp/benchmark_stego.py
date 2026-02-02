import os
import time
import argparse
import csv
import numpy as np
import math
import soundfile as sf
import stego_core

LOG_BASE_DIR = os.path.join("AudioStego", "logs_final")

def calculate_metrics(original_path, stego_path):
    try:
        y1, r1 = sf.read(original_path)
        y2, r2 = sf.read(stego_path)
        
        # [MỚI] Lấy thời lượng file (giây) để tính Rate
        duration = len(y1) / r1
        
        y1 = y1.astype(np.float64)
        y2 = y2.astype(np.float64)

        if y1.ndim == 2 and y2.ndim == 1:
            y1 = y1[:, 0]

        min_len = min(len(y1), len(y2))
        y1 = y1[:min_len]
        y2 = y2[:min_len]

        mse = np.mean((y1 - y2) ** 2)
        if mse == 0: return 0.0, 100.0, 100.0, duration
        
        psnr = 20 * math.log10(1.0 / math.sqrt(mse))
        
        signal_power = np.mean(y1 ** 2)
        snr = 10 * math.log10(signal_power / mse)
        
        return mse, psnr, snr, duration
    except Exception as e:
        return 0, 0, 0, 0

def append_log(case_name, filename, status, time_s, mse, psnr, snr, info):
    log_dir = os.path.join(LOG_BASE_DIR, case_name)
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "benchmark.csv")
    
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Filename', 'Status', 'Time(s)', 'MSE', 'PSNR', 'SNR', 'Info'])
        writer.writerow([time.strftime("%H:%M:%S"), filename, status, f"{time_s:.4f}", f"{mse:.4f}", f"{psnr:.2f}", f"{snr:.2f}", info])

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cover', required=True)
    parser.add_argument('--secret', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--case_name', required=True)
    
    parser.add_argument('--no_random', action='store_true')
    parser.add_argument('--adaptive', action='store_true')
    parser.add_argument('--salt_content', action='store_true')
    parser.add_argument('--phase', action='store_true')
    
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--password', type=str, default="DEFAULT_PASS")
    parser.add_argument('--size', type=int, default=256)
    
    args = parser.parse_args()

    try:
        # [QUAN TRỌNG] Lấy payload thực tế để tính dung lượng
        payload = stego_core.StegoUtils.prepare_payload(args.secret, target_size=args.size)
        payload_size_bytes = len(payload)
    except Exception as e:
        print(f"Read error: {e}")
        return

    start = time.time()
    info = ""

    if args.phase:
        res = stego_core.StegoPhase.encode(args.cover, payload, args.output)
        info = "Phase_FFT"
    elif args.no_random:
        res = stego_core.StegoBasic.encode(args.cover, payload, args.output)
        info = "Seq_LSB"
    else:
        k_strat = 'adaptive' if args.adaptive else 'fixed'
        s_source = 'content' if args.salt_content else 'default'
        
        res = stego_core.StegoImproved.encode(
            args.cover, payload, args.output, 
            password=args.password,
            k_strategy=k_strat, 
            salt_source=s_source,
            fixed_k_val=args.k
        )
        k_val = res.get('k_used', args.k)
        info = f"Rnd_{k_strat}_{s_source}_K={k_val}"

    duration = time.time() - start

    if res['status'] == 'success':
        # [MỚI] Lấy thêm duration
        mse, psnr, snr, audio_duration = calculate_metrics(args.cover, args.output)
        
        # [MỚI] Tính Rate (kBps)
        rate_kbps = 0.0
        if audio_duration > 0:
            rate_kbps = (payload_size_bytes / 1024.0) / audio_duration

        # In tag đặc biệt để script cha bắt được dữ liệu này
        print(f"[METRICS_DATA] Rate={rate_kbps:.4f}")
        
        append_log(args.case_name, os.path.basename(args.cover), "Success", duration, mse, psnr, snr, info)
    else:
        append_log(args.case_name, os.path.basename(args.cover), f"Fail: {res.get('message')}", duration, 0, 0, 0, info)

if __name__ == "__main__":
    run()