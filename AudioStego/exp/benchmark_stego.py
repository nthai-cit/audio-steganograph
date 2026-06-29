"""
benchmark_stego.py
==================
Single-file steganography benchmark worker.

This script is invoked as a subprocess by ``run_case.py`` and
``run_case_limited.py``.  It encodes one cover/secret pair using the
algorithm selected via CLI flags, computes quality metrics, and appends a
row to a per-case CSV log.

The script prints a single machine-parseable line to stdout on success:

    [METRICS_DATA] Rate=<kBps> CPU=<pct> RAM=<MB> k_used=<k>

The calling orchestrator parses this line with a regex to extract the
resource-usage values without needing a shared IPC mechanism.

Exit codes
----------
* 0 — encode succeeded and metrics were logged.
* 1 — encode failed; error reason written to stderr.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import psutil
import soundfile as sf

import stego_core


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Root directory under which per-case log sub-directories are created.
LOG_ROOT_DIRECTORY = os.path.join("AudioStego", "logs_final")

# PSNR value returned when cover and stego are identical (MSE = 0).
PSNR_PERFECT_VALUE = 100.0

# SNR value returned when cover and stego are identical (MSE = 0).
SNR_PERFECT_VALUE = 100.0

# Peak amplitude for normalised float audio (soundfile default: [−1.0, 1.0]).
# Used as the reference signal level in the PSNR formula.
FLOAT_PEAK_AMPLITUDE = 1.0


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def calculate_audio_quality_metrics(
    original_path: str,
    stego_path: str,
) -> tuple[float, float, float, float]:
    """
    Compute MSE, PSNR, SNR, and audio duration for a cover/stego pair.

    Both files are read via soundfile, which normalises integer PCM to
    [−1.0, 1.0] float64 by default.  PSNR is therefore computed with a
    peak reference of 1.0.

    Formulas
    --------
    ::

        MSE  = mean((original − stego)²)
        PSNR = 20 · log₁₀(1.0 / sqrt(MSE))      [dB]
        SNR  = 10 · log₁₀(mean(original²) / MSE) [dB]

    For 16-bit PCM: a single LSB flip introduces an error of 1/32768,
    giving PSNR ≈ 106.7 dB at approximately 2.31 % sample modification density.

    Parameters
    ----------
    original_path : str
        Path to the unmodified cover WAV file.
    stego_path : str
        Path to the stego WAV file produced by the encoder.

    Returns
    -------
    tuple[float, float, float, float]
        ``(mse, psnr_db, snr_db, duration_seconds)``.
        Returns ``(0, 0, 0, 0)`` on any read error.
    """
    try:
        original_samples, original_rate = sf.read(original_path)
        stego_samples, _ = sf.read(stego_path)

        audio_duration_seconds = len(original_samples) / original_rate

        original_samples = original_samples.astype(np.float64)
        stego_samples = stego_samples.astype(np.float64)

        # Align channel count: if the original is stereo and stego is mono,
        # compare only the first channel of the original.
        if original_samples.ndim == 2 and stego_samples.ndim == 1:
            original_samples = original_samples[:, 0]

        comparable_length = min(len(original_samples), len(stego_samples))
        original_samples = original_samples[:comparable_length]
        stego_samples = stego_samples[:comparable_length]

        mse = np.mean((original_samples - stego_samples) ** 2)
        if mse == 0:
            return 0.0, PSNR_PERFECT_VALUE, SNR_PERFECT_VALUE, audio_duration_seconds

        psnr_db = 20 * math.log10(FLOAT_PEAK_AMPLITUDE / math.sqrt(mse))

        signal_power = np.mean(original_samples ** 2)
        snr_db = 10 * math.log10(signal_power / mse)

        return mse, psnr_db, snr_db, audio_duration_seconds

    except Exception:
        return 0.0, 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

def append_result_to_csv_log(
    case_name: str,
    filename: str,
    status: str,
    elapsed_seconds: float,
    mse: float,
    psnr_db: float,
    snr_db: float,
    algorithm_info: str,
    cpu_percent: float,
    ram_megabytes: float,
    k_used: int | str,
) -> None:
    """
    Append one result row to the per-case CSV benchmark log.

    The log file is created with a header row on first write.  Subsequent
    calls append rows without re-writing the header.

    Parameters
    ----------
    case_name : str
        Name of the experiment case (used as a sub-directory name).
    filename : str
        Base filename of the processed cover audio file.
    status : str
        ``'Success'`` or a descriptive failure message.
    elapsed_seconds : float
        Wall-clock time taken by the encode step.
    mse : float
        Mean Squared Error between cover and stego audio.
    psnr_db : float
        Peak Signal-to-Noise Ratio in dB.
    snr_db : float
        Signal-to-Noise Ratio in dB.
    algorithm_info : str
        Short label identifying the algorithm variant (e.g. ``'Phase_FFT'``).
    cpu_percent : float
        CPU utilisation of this process at the time of measurement.
    ram_megabytes : float
        RSS memory usage of this process in megabytes.
    k_used : int or str
        Number of LSB planes used (or ``'N/A'`` for phase coding).
    """
    log_directory = os.path.join(LOG_ROOT_DIRECTORY, case_name)
    os.makedirs(log_directory, exist_ok=True)
    csv_path = os.path.join(log_directory, "benchmark.csv")

    log_file_is_new = not os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if log_file_is_new:
            writer.writerow([
                'Timestamp', 'Filename', 'Status', 'Time(s)',
                'MSE', 'PSNR', 'SNR', 'Info',
                'CPU(%)', 'RAM(MB)', 'k_used',
            ])
        writer.writerow([
            time.strftime("%H:%M:%S"),
            filename,
            status,
            f"{elapsed_seconds:.4f}",
            f"{mse:.4f}",
            f"{psnr_db:.2f}",
            f"{snr_db:.2f}",
            algorithm_info,
            f"{cpu_percent:.2f}",
            f"{ram_megabytes:.2f}",
            k_used,
        ])


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_argument_parser() -> argparse.ArgumentParser:
    """
    Construct and return the CLI argument parser for this script.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser ready for ``parse_args()``.
    """
    parser = argparse.ArgumentParser(
        description="Single-file steganography benchmark worker."
    )

    # Required I/O paths.
    parser.add_argument('--cover',     required=True, help="Path to the cover WAV file.")
    parser.add_argument('--secret',    required=True, help="Path to the secret file.")
    parser.add_argument('--output',    required=True, help="Destination path for the stego WAV.")
    parser.add_argument('--case_name', required=True, help="Experiment case label (used as log sub-directory).")

    # Algorithm selector flags.
    parser.add_argument('--no_random',    action='store_true', help="Use sequential LSB (Case 1).")
    parser.add_argument('--adaptive',     action='store_true', help="Use adaptive k selection (Cases 4–5).")
    parser.add_argument('--salt_content', action='store_true', help="Use content-derived salt (Cases 3, 5).")
    parser.add_argument('--phase',        action='store_true', help="Use Phase Coding (Case 7).")
    parser.add_argument('--alarood',      action='store_true', help="Use Alarood (2022) baseline (Case 8).")

    # Tuning parameters.
    parser.add_argument('--k',        type=int, default=1,              help="Fixed k value for StegoImproved.")
    parser.add_argument('--password', type=str, default="DEFAULT_PASS", help="PRNG password for randomised methods.")
    parser.add_argument('--size',     type=int, default=256,            help="Image resize dimension (pixels).")

    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run() -> None:
    """
    Parse CLI arguments, run the selected encoder, log results, and exit.

    On success, prints a machine-parseable metrics line to stdout.
    On encode failure, writes the error to stderr and exits with code 1.
    """
    parser = _build_argument_parser()
    args = parser.parse_args()

    # Start CPU measurement interval before any heavy work.
    this_process = psutil.Process(os.getpid())
    this_process.cpu_percent(interval=None)

    try:
        # Image quality (JPEG compression level) is controlled inside stego_core.py
        # via the JPEG_COMPRESSION_QUALITY constant.
        payload_bytes = stego_core.StegoUtils.prepare_payload(args.secret, target_size=args.size)
        payload_size_bytes = len(payload_bytes)
    except Exception as payload_error:
        print(f"Read error: {payload_error}")
        return

    encode_start_time = time.time()
    algorithm_info = ""
    k_used_value: int | str = "N/A"

    # --- Algorithm dispatch ---
    if args.phase:
        encode_result = stego_core.StegoPhase.encode(args.cover, payload_bytes, args.output)
        algorithm_info = "Phase_FFT"
        k_used_value = "N/A"  # Phase Coding does not use k.

    elif args.no_random:
        encode_result = stego_core.StegoBasic.encode(args.cover, payload_bytes, args.output)
        algorithm_info = "Seq_LSB"
        k_used_value = 1  # Sequential baseline always modifies 1 LSB plane.

    elif args.alarood:
        encode_result = stego_core.StegoAlarood.encode(
            args.cover, payload_bytes, args.output, password=args.password
        )
        algorithm_info = "Alarood2022_RandLSB"
        k_used_value = 1  # Alarood baseline defaults to 1-LSB.

    else:
        k_strategy = 'adaptive' if args.adaptive else 'fixed'
        salt_source = 'content' if args.salt_content else 'default'

        encode_result = stego_core.StegoImproved.encode(
            args.cover,
            payload_bytes,
            args.output,
            password=args.password,
            k_strategy=k_strategy,
            salt_source=salt_source,
            fixed_k_val=args.k,
        )
        k_used_value = encode_result.get('k_used', args.k)
        algorithm_info = f"Rnd_{k_strategy}_{salt_source}_K={k_used_value}"

    elapsed_seconds = time.time() - encode_start_time

    # Collect resource usage immediately after the encode step completes.
    cpu_percent = this_process.cpu_percent(interval=None)
    ram_megabytes = this_process.memory_info().rss / (1024 * 1024)

    if encode_result['status'] == 'success':
        mse, psnr_db, snr_db, audio_duration_seconds = calculate_audio_quality_metrics(
            args.cover, args.output
        )

        embedding_rate_kbps = 0.0
        if audio_duration_seconds > 0:
            embedding_rate_kbps = (payload_size_bytes / 1024.0) / audio_duration_seconds

        # Machine-parseable line read by the orchestrator via regex.
        print(
            f"[METRICS_DATA] Rate={embedding_rate_kbps:.4f} "
            f"CPU={cpu_percent:.2f} RAM={ram_megabytes:.2f} k_used={k_used_value}"
        )

        append_result_to_csv_log(
            case_name=args.case_name,
            filename=os.path.basename(args.cover),
            status="Success",
            elapsed_seconds=elapsed_seconds,
            mse=mse,
            psnr_db=psnr_db,
            snr_db=snr_db,
            algorithm_info=algorithm_info,
            cpu_percent=cpu_percent,
            ram_megabytes=ram_megabytes,
            k_used=k_used_value,
        )

    else:
        append_result_to_csv_log(
            case_name=args.case_name,
            filename=os.path.basename(args.cover),
            status=f"Fail: {encode_result.get('message')}",
            elapsed_seconds=elapsed_seconds,
            mse=0,
            psnr_db=0,
            snr_db=0,
            algorithm_info=algorithm_info,
            cpu_percent=cpu_percent,
            ram_megabytes=ram_megabytes,
            k_used=k_used_value,
        )
        print(f"ENCODE_FAILED: {encode_result.get('message')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run()