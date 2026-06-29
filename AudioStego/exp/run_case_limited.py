"""
run_case_limited.py
===================
Dataset generation orchestrator with capacity-aware pairing.

This script processes a directory of audio files (covers) and matches them 
with image payloads (secrets) to build a steganography dataset. It features 
a 'capacity-aware' pairing mechanism that assigns the largest possible image 
payload to an audio cover based on its available embedding capacity, ensuring 
maximum embedding rates with a near-zero failure rate due to oversize issues.
"""

import argparse
import glob
import os
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

CASE_CONFIGURATIONS = {
    1: {"name": "1_NoRandom", "flags": ["--no_random"], "desc": "Sequential"},
    2: {"name": "2_Random_Fixed_DefaultSalt", "flags": [], "desc": "Rnd_Fixed"},
    3: {"name": "3_Random_Fixed_ContentSalt", "flags": ["--salt_content"], "desc": "Rnd_Fixed_Content"},
    4: {"name": "4_Random_Adaptive_DefaultSalt", "flags": ["--adaptive"], "desc": "Rnd_Adaptive"},
    5: {"name": "5_Random_Adaptive_ContentSalt", "flags": ["--adaptive", "--salt_content"], "desc": "Rnd_Adaptive_Content"},
    6: {"name": "6_LSBMR_Literature", "flags": [], "desc": "LSBMR_Literature"},
    7: {"name": "7_PhaseCoding", "flags": ["--phase"], "desc": "Phase_Coding_FFT"}
}

# Constants aligned with internal core limits inside `stego_core.py`.
ANCHOR_SIZE = 1024
K_MAX = 6
OVERHEAD_BYTES = 10  # Accounts for the end marker '||END||' plus a small buffer.

PERFECT_PSNR_DB = 100.0
PERFECT_SNR_DB = 100.0
FLOAT_PEAK_AMPLITUDE = 1.0


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def get_vietnam_time() -> datetime:
    """Return the current time adjusted to the Vietnam timezone (UTC+7)."""
    return datetime.now(timezone.utc) + timedelta(hours=7)


def calculate_audio_metrics(cover_path: str, stego_path: str) -> tuple[float, float, float]:
    """Calculate MSE, PSNR, and SNR between cover and stego audio signals."""
    try:
        if not os.path.exists(cover_path) or not os.path.exists(stego_path):
            return 0.0, 0.0, 0.0

        cover_signal, _ = sf.read(cover_path)
        stego_signal, _ = sf.read(stego_path)

        cover_signal = cover_signal.astype(np.float64)
        stego_signal = stego_signal.astype(np.float64)

        if cover_signal.ndim == 2 and stego_signal.ndim == 1:
            cover_signal = cover_signal[:, 0]
        elif cover_signal.ndim == 1 and stego_signal.ndim == 2:
            stego_signal = stego_signal[:, 0]

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


def get_files_recursive(directory: str, extensions: list[str]) -> list[str]:
    """Retrieve a list of all files with matching extensions in the directory tree."""
    discovered_files = []
    for ext in extensions:
        discovered_files.extend(glob.glob(os.path.join(directory, "**", f"*.{ext}"), recursive=True))
    return discovered_files


def calculate_maximum_audio_capacity_bytes(audio_filepath: str) -> float:
    """
    Calculate the maximum embedding capacity (in bytes) of an audio file at K_MAX.
    Uses the soundfile library to read metadata quickly without loading samples.
    """
    try:
        audio_info = sf.info(audio_filepath)
        total_samples = audio_info.frames * audio_info.channels
        usable_slots = max(0, total_samples - ANCHOR_SIZE)
        maximum_bits = usable_slots * K_MAX
        return (maximum_bits / 8.0) - OVERHEAD_BYTES
    except Exception:
        return 0.0


def build_capacity_aware_file_pairs(target_covers: list[str], secret_pool: list[str], log_fn=print) -> list[tuple[str, str]]:
    """
    Pair each audio cover with the largest image secret that fits within its capacity.
    This logic maximizes the average embedding rate while preventing oversize failures.

    Parameters
    ----------
    target_covers : list[str]
        List of paths to cover audio files.
    secret_pool : list[str]
        List of paths to available secret images.
    log_fn : callable
        Function used to output progress logs.

    Returns
    -------
    list[tuple[str, str]]
        A list of matched (cover_filepath, secret_filepath) pairs.
    """
    log_fn("[*] Calculating capacity for each audio cover...")
    cover_capacity_list = []
    for cover_filepath in target_covers:
        capacity_bytes = calculate_maximum_audio_capacity_bytes(cover_filepath)
        cover_capacity_list.append((cover_filepath, capacity_bytes))

    log_fn("[*] Measuring size of each secret image...")
    # Use original file size as a quick proxy. For absolute precision after JPEG 
    # re-encoding, one could substitute this with genuine `prepare_payload()` calls.
    secret_size_list = [(secret_file, os.path.getsize(secret_file)) for secret_file in secret_pool]
    secret_size_list_sorted = sorted(secret_size_list, key=lambda x: -x[1])  # Sort descending by size

    file_pairs = []
    unmatched_count = 0

    for cover_filepath, capacity in cover_capacity_list:
        chosen_secret = None
        for secret_filepath, secret_size in secret_size_list_sorted:
            if secret_size <= capacity:
                chosen_secret = secret_filepath
                break
        
        if chosen_secret is None:
            unmatched_count += 1
            # Fallback: use the smallest image if no image inherently fits.
            # The downstream code will handle skipping if it still triggers an oversize error.
            if secret_size_list_sorted:
                chosen_secret = secret_size_list_sorted[-1][0]
            else:
                continue
                
        file_pairs.append((cover_filepath, chosen_secret))

    log_fn(f"[*] Capacity-aware matching: {len(file_pairs)} pairs | "
           f"Unmatched (fallback triggered): {unmatched_count}")
    return file_pairs


# ---------------------------------------------------------------------------
# Main Execution Flow
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Configure CLI parsing, prepare directories, orchestrate file matching, 
    and invoke the subprocess worker to generate the dataset.
    """
    parser = argparse.ArgumentParser(description="Dataset generation with capacity-aware pairing.")
    parser.add_argument('--case_id', type=int, required=True, help="Evaluation case ID.")
    parser.add_argument('--input_dir', required=True, help="Directory containing cover WAV files.")
    parser.add_argument('--output_base', required=True, help="Base directory for the generated dataset.")
    parser.add_argument('--secret_dir', required=True, help="Directory containing secret image payloads.")

    parser.add_argument('--max_files', type=int, default=1000, help="Maximum number of cover files to process.")
    parser.add_argument('--limit', type=int, default=500, help="Maximum number of secret files to load into the pool.")

    parser.add_argument('--k', type=int, default=1, help="Number of LSB planes to use.")
    parser.add_argument('--password', type=str, default="PASS", help="Password for PRNG seed generation.")
    parser.add_argument('--size', type=int, default=256, help="Target dimension for resizing images.")

    parser.add_argument('--capacity_aware', action='store_true', default=True,
                        help="Enable capacity-aware image selection (default: ON)")
    parser.add_argument('--no_capacity_aware', dest='capacity_aware', action='store_false',
                        help="Disable capacity-aware mode, revert to purely random pairing")

    args = parser.parse_args()

    case_config = CASE_CONFIGURATIONS.get(args.case_id)
    if not case_config:
        print(f"Error: Case ID {args.case_id} not found.")
        sys.exit(1)

    print(f"[*] Scanning Cover Directory: {args.input_dir}...")
    all_cover_files = get_files_recursive(args.input_dir, ["wav", "WAV"])
    
    print(f"[*] Scanning Secret Directory: {args.secret_dir}...")
    # Ignore .png masks if any exist in the dataset structure by targeting specific extensions.
    all_secret_files = get_files_recursive(args.secret_dir, ["jpg", "JPG", "jpeg", "JPEG"])

    if not all_cover_files or not all_secret_files:
        print("Error: Required files not found.")
        sys.exit(1)

    random.shuffle(all_cover_files)
    num_target_covers = min(len(all_cover_files), args.max_files)
    target_cover_list = all_cover_files[:num_target_covers]

    random.shuffle(all_secret_files)
    num_secrets_in_pool = min(len(all_secret_files), args.limit)
    secret_payload_pool = all_secret_files[:num_secrets_in_pool]

    print("-" * 50)
    print(f"[*] Mode: DATASET GENERATION (Vietnam Time)")
    print(f"[*] Cover Count: {num_target_covers} | Secret Pool Size: {num_secrets_in_pool}")
    print(f"[*] Image Size Target: {args.size}x{args.size}")
    print(f"[*] Capacity-aware matching: {'ON' if args.capacity_aware else 'OFF (random)'}")
    print("-" * 50)

    timestamp_string = get_vietnam_time().strftime("%Y%m%d_%H%M%S")
    dataset_base_dir = os.path.join(args.output_base, f"{case_config['name']}_{timestamp_string}")
    
    logs_directory = os.path.join(dataset_base_dir, "logs")
    covers_output_directory = os.path.join(dataset_base_dir, "cover")
    stegos_output_directory = os.path.join(dataset_base_dir, "stego")

    os.makedirs(logs_directory, exist_ok=True)
    os.makedirs(covers_output_directory, exist_ok=True)
    os.makedirs(stegos_output_directory, exist_ok=True)

    log_filepath = os.path.join(logs_directory, "benchmark.csv")
    with open(log_filepath, "w", encoding="utf-8") as log_file:
        log_file.write("Timestamp,CoverFile,SecretFile,Status,Time(s),MSE,PSNR,SNR,Rate(kBps),CPU(%),RAM(MB),k_used,Info\n")
    
    count_skipped = 0
    count_success = 0

    # Build cover-secret pairs before entering the processing loop.
    if args.capacity_aware:
        execution_pairs = build_capacity_aware_file_pairs(target_cover_list, secret_payload_pool)
    else:
        execution_pairs = [(cover_file, random.choice(secret_payload_pool)) for cover_file in target_cover_list]

    for index, (cover_filepath, secret_filepath) in enumerate(execution_pairs):
        relative_filepath = os.path.relpath(cover_filepath, args.input_dir)
        final_cover_destination = os.path.join(covers_output_directory, relative_filepath)
        final_stego_destination = os.path.join(stegos_output_directory, relative_filepath)

        os.makedirs(os.path.dirname(final_cover_destination), exist_ok=True)
        os.makedirs(os.path.dirname(final_stego_destination), exist_ok=True)

        worker_command = [
            sys.executable, "benchmark_stego.py",
            "--cover", cover_filepath,
            "--secret", secret_filepath,
            "--output", final_stego_destination,
            "--case_name", case_config['name'],
            "--k", str(args.k),
            "--password", args.password,
            "--size", str(args.size)
        ] + case_config['flags']

        cover_filename = os.path.basename(cover_filepath)
        start_time = time.time()

        try:
            subprocess_result = subprocess.run(worker_command, capture_output=True, text=True, timeout=180)
            elapsed_time = time.time() - start_time

            if subprocess_result.returncode == 0:
                mse, psnr, snr = calculate_audio_metrics(cover_filepath, final_stego_destination)

                worker_stdout = subprocess_result.stdout
                embedding_rate_kbps = 0.0
                cpu_percent = 0.0
                ram_megabytes = 0.0
                k_used_val = "N/A"

                # Extract k_used and resource values from the benchmark_stego.py stdout.
                regex_match = re.search(r"Rate=([0-9\.]+).*CPU=([0-9\.]+).*RAM=([0-9\.]+).*k_used=([^\s]+)", worker_stdout)
                if regex_match:
                    embedding_rate_kbps = float(regex_match.group(1))
                    cpu_percent = float(regex_match.group(2))
                    ram_megabytes = float(regex_match.group(3))
                    k_used_val = regex_match.group(4)

                if psnr < 1.0:
                    # Fix: Use actual audio duration instead of file path string length.
                    try:
                        actual_audio_duration = sf.info(cover_filepath).duration
                    except Exception:
                        actual_audio_duration = -1.0
                        
                    stdout_tail = worker_stdout[-200:] if worker_stdout else "(empty)"
                    print(f"\n[DEBUG] {cover_filename}: PSNR={psnr:.2f}, MSE={mse:.6f}, "
                          f"Audio_duration={actual_audio_duration:.3f}s, k_used={k_used_val}, "
                          f"stdout_tail={stdout_tail}")
                    
                    count_skipped += 1
                    print(f"\r[{index+1}/{len(execution_pairs)}] {cover_filename} -> Skipped (Low PSNR)", end="", flush=True)
                    if os.path.exists(final_stego_destination): 
                        os.remove(final_stego_destination)
                    continue

                shutil.copy2(cover_filepath, final_cover_destination)

                count_success += 1
                secret_filename = os.path.basename(secret_filepath)
                current_time_str = get_vietnam_time().strftime('%H:%M:%S')

                log_entry = (
                    f"{current_time_str},{cover_filename},{secret_filename},Success,"
                    f"{elapsed_time:.4f},{mse:.6f},{psnr:.2f},{snr:.2f},{embedding_rate_kbps:.4f},"
                    f"{cpu_percent:.2f},{ram_megabytes:.2f},{k_used_val},{case_config['desc']}"
                )

                print(f"\r[{index+1}/{len(execution_pairs)}] {cover_filename} -> OK "
                      f"(PSNR:{psnr:.1f}dB | k={k_used_val} | CPU:{cpu_percent}% | RAM:{ram_megabytes:.1f}MB)", end="", flush=True)

                with open(log_filepath, "a", encoding="utf-8") as log_file:
                    log_file.write(log_entry + "\n")

            else:
                # Print stderr to reveal the actual reason for subprocess failure.
                stderr_tail = (subprocess_result.stderr or "")[-300:]
                print(f"\n[DEBUG] {cover_filename} subprocess FAILED (returncode={subprocess_result.returncode}). stderr_tail: {stderr_tail}")
                print(f"\r[{index+1}/{len(execution_pairs)}] {cover_filename} -> Skipped (Error)", end="", flush=True)

        except subprocess.TimeoutExpired:
            print(f"\r[{index+1}/{len(execution_pairs)}] {cover_filename} -> Skipped (Timeout)", end="", flush=True)

        except Exception as execution_error:
            print(f"\n[DEBUG] {cover_filename} -> Crash: {execution_error}")
            print(f"\r[{index+1}/{len(execution_pairs)}] {cover_filename} -> Skipped (Crash)", end="", flush=True)

    print(f"\n[DONE] Finished.")
    print(f"Structure: {dataset_base_dir}/[logs, cover, stego]")
    print(f"Total pairs: {len(execution_pairs)} | Created successfully: {count_success} | Skipped: {count_skipped}")

if __name__ == "__main__":
    main()