"""
run_case.py
===========
Orchestrator script for batch processing audio steganography evaluations.

This script processes a directory of cover audio files, invoking a benchmark 
worker (`benchmark_stego.py`) as a subprocess for each file. It parses 
the worker's standard output to collect resource utilisation metrics and 
calculates the resulting audio quality metrics (MSE, PSNR, SNR).
Results are aggregated and written to a CSV log.
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

# Configuration mapping for supported evaluation cases.
# Defines the descriptive name, CLI flags passed to the worker, and a short label.
CASE_CONFIGURATIONS = {
    1: {"name": "1_NoRandom", "flags": ["--no_random"], "desc": "Sequential"},
    2: {"name": "2_Random_Fixed_DefaultSalt", "flags": [], "desc": "Rnd_Fixed"},
    3: {"name": "3_Random_Fixed_ContentSalt", "flags": ["--salt_content"], "desc": "Rnd_Fixed_Content"},
    4: {"name": "4_Random_Adaptive_DefaultSalt", "flags": ["--adaptive"], "desc": "Rnd_Adaptive"},
    5: {"name": "5_Random_Adaptive_ContentSalt", "flags": ["--adaptive", "--salt_content"], "desc": "Rnd_Adaptive_Content"},
    6: {"name": "6_LSBMR_Literature", "flags": [], "desc": "LSBMR_Literature"},
    7: {"name": "7_PhaseCoding", "flags": ["--phase"], "desc": "Phase_Coding_FFT"},
    8: {"name": "8_Alarood2022_RandLSB", "flags": ["--alarood"], "desc": "Alarood2022_RandLSB"},
}

PERFECT_PSNR_DB = 100.0
PERFECT_SNR_DB = 100.0
FLOAT_PEAK_AMPLITUDE = 1.0


# ---------------------------------------------------------------------------
# Metric Computation
# ---------------------------------------------------------------------------

def calculate_audio_metrics(cover_path: str, stego_path: str) -> tuple[float, float, float]:
    """
    Calculate Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), 
    and Signal-to-Noise Ratio (SNR) between cover and stego audio signals.

    Parameters
    ----------
    cover_path : str
        Path to the unmodified cover WAV file.
    stego_path : str
        Path to the stego WAV file produced by the encoder.

    Returns
    -------
    tuple[float, float, float]
        A tuple containing (MSE, PSNR, SNR). Returns (0.0, 0.0, 0.0) on failure.
    """
    try:
        if not os.path.exists(cover_path) or not os.path.exists(stego_path):
            return 0.0, 0.0, 0.0

        cover_signal, _ = sf.read(cover_path)
        stego_signal, _ = sf.read(stego_path)
        
        cover_signal = cover_signal.astype(np.float64)
        stego_signal = stego_signal.astype(np.float64)
        
        # Handle Stereo/Mono mismatches by keeping only the first channel.
        if cover_signal.ndim == 2 and stego_signal.ndim == 1:
            cover_signal = cover_signal[:, 0]
        elif cover_signal.ndim == 1 and stego_signal.ndim == 2:
            stego_signal = stego_signal[:, 0]

        if len(cover_signal) == 0 or len(stego_signal) == 0:
            return 0.0, 0.0, 0.0

        comparable_length = min(len(cover_signal), len(stego_signal))
        cover_signal = cover_signal[:comparable_length]
        stego_signal = stego_signal[:comparable_length]
        
        difference_signal = cover_signal - stego_signal
        mse = np.mean(difference_signal ** 2)

        if mse == 0:
            return 0.0, PERFECT_PSNR_DB, PERFECT_SNR_DB

        psnr = 20 * np.log10(FLOAT_PEAK_AMPLITUDE / np.sqrt(mse))
        
        signal_power = np.mean(cover_signal ** 2)
        snr = 10 * np.log10(signal_power / mse)

        return mse, psnr, snr
    except Exception:
        return 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------
# Main Execution Flow
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Parse command-line arguments, establish directories, construct execution 
    commands for the worker subprocess, and orchestrate the batch processing 
    of all identified audio files.
    """
    parser = argparse.ArgumentParser(description="Batch process audio steganography evaluations.")
    parser.add_argument('--case_id', type=int, required=True, help="Evaluation case ID mapping to configurations.")
    parser.add_argument('--input_dir', required=True, help="Directory containing cover WAV files.")
    parser.add_argument('--output_base', required=True, help="Base directory for outputting stego files.")
    parser.add_argument('--secret_file', type=str, required=True, help="Path to the secret payload file.")
    parser.add_argument('--k', type=int, default=1, help="Number of LSB planes (if applicable).")
    parser.add_argument('--password', type=str, default="PASS", help="Password for PRNG seed generation.")

    # Flag to prevent saving output files to conserve disk space.
    parser.add_argument('--no_save', action='store_true', 
                        help="Do not save output files to conserve disk space (deleted immediately after execution).")
    
    args = parser.parse_args()

    if not os.path.exists(args.secret_file):
        print(f"Error: Secret file '{args.secret_file}' not found.")
        sys.exit(1)

    case_config = CASE_CONFIGURATIONS.get(args.case_id)
    if not case_config:
        print(f"Error: Case ID {args.case_id} not found.")
        sys.exit(1)

    timestamp_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_folder_name = os.path.basename(os.path.normpath(args.input_dir))
    
    log_directory = "logs"
    os.makedirs(log_directory, exist_ok=True)
    log_file_path = os.path.join(log_directory, f"benchmark_{input_folder_name}_{case_config['name']}_{timestamp_string}.csv")

    # Discover all audio files recursively.
    raw_file_paths = glob.glob(os.path.join(args.input_dir, "**", "*.wav"), recursive=True) + \
                     glob.glob(os.path.join(args.input_dir, "**", "*.WAV"), recursive=True)
    
    processed_files_map = {}
    for filepath in raw_file_paths:
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        if base_name not in processed_files_map:
            processed_files_map[base_name] = filepath
            
    sorted_file_paths = sorted(list(processed_files_map.values()))

    case_output_directory = os.path.join(args.output_base, case_config['name'])
    os.makedirs(case_output_directory, exist_ok=True)

    # Initialise the log file with headers (includes resource metrics from the worker).
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log_file.write("Timestamp,Filename,Status,Time(s),MSE,PSNR,SNR,Rate(kBps),CPU(%),RAM(MB),Info\n")

    print(f"[*] Case: {case_config['name']} | Input: {input_folder_name}")
    print(f"[*] Secret: {args.secret_file}")
    print(f"[*] No Save Mode: {'ON' if args.no_save else 'OFF'}")
    print(f"[*] Found: {len(sorted_file_paths)} files")

    for index, cover_filepath in enumerate(sorted_file_paths):
        relative_path = os.path.relpath(cover_filepath, args.input_dir)
        output_filepath = os.path.join(case_output_directory, relative_path)
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        
        command = [
            sys.executable, "benchmark_stego.py", 
            "--cover", cover_filepath, 
            "--secret", args.secret_file,
            "--output", output_filepath, 
            "--case_name", case_config['name'], 
            "--k", str(args.k), 
            "--password", args.password
        ] + case_config['flags']

        start_time = time.time()
        # Capture stdout to extract CPU/RAM logs emitted by benchmark_stego.py
        subprocess_result = subprocess.run(command, capture_output=True, text=True)
        elapsed_duration = time.time() - start_time
        
        log_entry = ""
        if subprocess_result.returncode == 0:
            # Parse the metrics from the benchmark_stego.py output.
            worker_output = subprocess_result.stdout
            embedding_rate_kbps, cpu_percent, ram_megabytes = 0.0, 0.0, 0.0
            
            metrics_match = re.search(r"Rate=([0-9\.]+).*CPU=([0-9\.]+).*RAM=([0-9\.]+)", worker_output)
            if metrics_match:
                embedding_rate_kbps = float(metrics_match.group(1))
                cpu_percent = float(metrics_match.group(2))
                ram_megabytes = float(metrics_match.group(3))

            # Handle the 'NO SAVE' logic to conserve disk space.
            if args.no_save:
                # Skip metric calculation since the file is not persisted; assign zeroes.
                mse, psnr, snr = 0, 0, 0
                # Delete the stego file immediately.
                if os.path.exists(output_filepath):
                    os.remove(output_filepath)
                status_string = "Success(NoSave)"
            else:
                # Compute metrics normally if saving is enabled.
                mse, psnr, snr = calculate_audio_metrics(cover_filepath, output_filepath)
                status_string = "Success"

            current_timestamp = datetime.now().strftime('%H:%M:%S')
            cover_filename = os.path.basename(cover_filepath)
            
            log_entry = (
                f"{current_timestamp},{cover_filename},{status_string},{elapsed_duration:.4f},"
                f"{mse:.6f},{psnr:.2f},{snr:.2f},{embedding_rate_kbps:.4f},"
                f"{cpu_percent:.2f},{ram_megabytes:.2f},{case_config['desc']}"
            )
            
            # Print progress to console.
            metric_display = "NoSave" if args.no_save else f"PSNR:{psnr:.1f}dB"
            print(f"\r[{index+1}/{len(sorted_file_paths)}] {cover_filename} -> {metric_display} "
                  f"| CPU:{cpu_percent}% | RAM:{ram_megabytes:.1f}MB", end="", flush=True)

        else:
            error_message = "Error"
            current_timestamp = datetime.now().strftime('%H:%M:%S')
            cover_filename = os.path.basename(cover_filepath)
            
            log_entry = (
                f"{current_timestamp},{cover_filename},Failed,0,0,0,0,0,0,0,{error_message}"
            )
            print(f"\r[{index+1}/{len(sorted_file_paths)}] {cover_filename} -> FAILED", end="", flush=True)
        
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(log_entry + "\n")

    # Clean up the output directory if it is empty (applicable in no_save mode).
    if args.no_save:
        try:
            os.rmdir(case_output_directory) 
        except Exception:
            pass 

    print(f"\n[DONE] Finished. Log saved at: {log_file_path}")

if __name__ == "__main__":
    main()