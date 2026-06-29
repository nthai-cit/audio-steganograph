"""
Audio LSB steganography with content-derived randomised embedding.

Security upgrades implemented in this revision:
  1. Cryptographic seed derivation uses the full 256-bit HMAC-SHA256 output
     (previously truncated to 32 bits via modulo, which only provided
     ~4 billion possible permutation seeds and was vulnerable to
     brute-force search).
  2. Extraction now runs in constant time with respect to the secret
     bit-depth k. The previous implementation stopped scanning as soon
     as the end-of-payload sentinel was found, which leaked information
     about k (and therefore about payload size) through a timing
     side-channel. This version always evaluates every candidate k
     before returning a result.

Both changes affect the embedding/extraction internals but preserve
the original public API of `encode()` and `decode()`.
"""

import hashlib
import hmac
import io
import math
import os

import numpy as np
from scipy.io import wavfile

try:
    from PIL import Image
except ImportError:
    print("Warning: PIL not installed. Image processing will be disabled.")
    Image = None


ANCHOR_SIZE = 1024
MAX_EMBEDDING_DEPTH = 6
END_OF_PAYLOAD_MARKER = b"||END||"
JPEG_INITIAL_QUALITY = 95
JPEG_MIN_QUALITY = 10
JPEG_QUALITY_STEP = 5
JPEG_CHROMA_SUBSAMPLING = 2
THUMBNAIL_MAX_DIMENSION = 1024
DOWNSCALE_FACTOR = 0.9
MIN_IMAGE_DIMENSION = 10


class ImageProcessor:
    """Compresses an image file down to a target byte budget."""

    @staticmethod
    def compress_to_byte_budget(image_path, target_bytes):
        if Image is None:
            return ImageProcessor._read_raw_bytes(image_path)

        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.thumbnail((THUMBNAIL_MAX_DIMENSION, THUMBNAIL_MAX_DIMENSION))

            compressed_bytes = ImageProcessor._reduce_quality_until_fits(image, target_bytes)
            compressed_bytes = ImageProcessor._downscale_until_fits(
                image, compressed_bytes, target_bytes
            )
            return compressed_bytes
        except Exception:
            return ImageProcessor._read_raw_bytes(image_path)

    @staticmethod
    def _read_raw_bytes(image_path):
        with open(image_path, "rb") as image_file:
            return image_file.read()

    @staticmethod
    def _encode_jpeg(image, quality):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, subsampling=JPEG_CHROMA_SUBSAMPLING)
        return buffer.getvalue()

    @staticmethod
    def _reduce_quality_until_fits(image, target_bytes):
        quality = JPEG_INITIAL_QUALITY
        encoded = ImageProcessor._encode_jpeg(image, quality)

        while len(encoded) > target_bytes and quality > JPEG_MIN_QUALITY:
            quality -= JPEG_QUALITY_STEP
            encoded = ImageProcessor._encode_jpeg(image, quality)

        return encoded

    @staticmethod
    def _downscale_until_fits(image, encoded, target_bytes):
        quality = JPEG_MIN_QUALITY
        while len(encoded) > target_bytes:
            width, height = image.size
            if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
                break
            image = image.resize((int(width * DOWNSCALE_FACTOR), int(height * DOWNSCALE_FACTOR)))
            encoded = ImageProcessor._encode_jpeg(image, quality)
        return encoded


def derive_permutation_seed(password, content_salt):
    """
    Derives a full 256-bit PRNG seed from the password and a content-derived
    salt using HMAC-SHA256.

    The previous implementation truncated the hash to 32 bits via modulo
    arithmetic, reducing the effective keyspace to ~2^32 and making the
    embedding permutation susceptible to brute-force recovery. Using the
    complete 256-bit digest closes that gap.
    """
    digest = hmac.new(
        key=password.encode("utf-8"),
        msg=content_salt.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    return int.from_bytes(digest, byteorder="big")


def derive_content_salt(audio_samples, anchor_size=ANCHOR_SIZE):
    """Hashes the first `anchor_size` audio samples to produce a content-bound salt."""
    anchor_bytes = audio_samples[:anchor_size].tobytes()
    return hashlib.sha256(anchor_bytes).hexdigest()


def load_payload_bytes(secret_input, max_bytes=None):
    """
    Loads the secret payload as raw bytes.

    If `secret_input` is a path to an image file and `max_bytes` is given,
    the image is compressed to fit within that budget. Otherwise the file
    is read verbatim, or the string itself is treated as the payload.
    """
    if not os.path.isfile(secret_input):
        return secret_input.encode("utf-8")

    file_extension = os.path.splitext(secret_input)[1].lower()
    is_image = file_extension in (".jpg", ".jpeg", ".png", ".bmp")

    if is_image and max_bytes:
        return ImageProcessor.compress_to_byte_budget(secret_input, max_bytes)

    with open(secret_input, "rb") as payload_file:
        return payload_file.read()


def calculate_quality_metrics(original_samples, stego_samples):
    """Computes MSE, SNR, and PSNR between the original and stego audio."""
    original = original_samples.astype(np.float64)
    modified = stego_samples.astype(np.float64)
    error = original - modified

    mean_squared_error = np.mean(error ** 2)
    if mean_squared_error == 0:
        return 0.0, float("inf"), float("inf")

    root_mean_squared_error = np.sqrt(mean_squared_error)
    max_sample_value = 32767.0
    peak_signal_to_noise_ratio = 20 * np.log10(max_sample_value / root_mean_squared_error)

    signal_power = np.sum(original ** 2)
    noise_power = np.sum(error ** 2)
    signal_to_noise_ratio = (
        10 * np.log10(signal_power / noise_power) if noise_power != 0 else float("inf")
    )

    return mean_squared_error, signal_to_noise_ratio, peak_signal_to_noise_ratio


def _resolve_embedding_depth(payload_byte_count, available_slots, requested_depth):
    """
    Determines the bit-depth (k) to use for embedding.

    If `requested_depth` is provided, it is used as-is (and validated against
    capacity). Otherwise, the minimum depth that fits the payload is computed
    and clamped to `MAX_EMBEDDING_DEPTH`.
    """
    payload_bits_needed = (payload_byte_count + len(END_OF_PAYLOAD_MARKER)) * 8

    if requested_depth is not None and requested_depth > 0:
        if payload_bits_needed > available_slots * requested_depth:
            raise ValueError(f"Payload too large for fixed depth k={requested_depth}.")
        return requested_depth

    minimum_required_depth = math.ceil(payload_bits_needed / available_slots)
    clamped_depth = min(minimum_required_depth, MAX_EMBEDDING_DEPTH)

    if payload_bits_needed > available_slots * clamped_depth:
        raise ValueError(f"Payload too large even at maximum depth k={MAX_EMBEDDING_DEPTH}.")

    return clamped_depth


def _pack_payload_into_symbols(payload_bytes, embedding_depth):
    """Splits the payload (plus end marker) into `embedding_depth`-bit symbols."""
    full_payload = payload_bytes + END_OF_PAYLOAD_MARKER
    payload_bits = np.unpackbits(np.frombuffer(full_payload, dtype=np.uint8))

    padding_needed = (-len(payload_bits)) % embedding_depth
    if padding_needed:
        payload_bits = np.append(payload_bits, np.zeros(padding_needed, dtype=np.uint8))

    bit_weights = 1 << np.arange(embedding_depth)[::-1]
    symbols = payload_bits.reshape(-1, embedding_depth).dot(bit_weights)
    return symbols.astype(np.int16)


def _select_target_sample_indices(audio_length, password, content_salt, symbol_count):
    """Generates the pseudo-random sample indices used for embedding/extraction."""
    seed = derive_permutation_seed(password, content_salt)
    random_generator = np.random.default_rng(seed)

    embeddable_range = np.arange(ANCHOR_SIZE, audio_length)
    shuffled_indices = random_generator.permutation(embeddable_range)
    return shuffled_indices[:symbol_count]


def encode(cover_audio_path, secret_input, output_path, embedding_depth=None, password=None):
    """Embeds a secret payload into a cover WAV file using randomised LSB substitution."""
    try:
        sample_rate, audio_samples = wavfile.read(cover_audio_path)
        if audio_samples.dtype != np.int16:
            audio_samples = (
                (audio_samples * 32767).astype(np.int16)
                if audio_samples.dtype == np.float32
                else audio_samples.astype(np.int16)
            )

        flat_samples = audio_samples.flatten()
        available_slots = len(flat_samples) - ANCHOR_SIZE

        max_payload_bytes = None
        if embedding_depth is not None and embedding_depth > 0:
            max_payload_bytes = (available_slots * embedding_depth) // 8

        payload_bytes = load_payload_bytes(secret_input, max_bytes=max_payload_bytes)
        embedding_depth = _resolve_embedding_depth(
            len(payload_bytes), available_slots, embedding_depth
        )

        symbols = _pack_payload_into_symbols(payload_bytes, embedding_depth)

        content_salt = derive_content_salt(flat_samples, ANCHOR_SIZE)
        target_indices = _select_target_sample_indices(
            len(flat_samples), password or "default", content_salt, len(symbols)
        )

        stego_samples = flat_samples.copy()
        clear_mask = (1 << embedding_depth) - 1
        stego_samples[target_indices] &= ~clear_mask
        stego_samples[target_indices] |= symbols

        stego_audio = stego_samples.reshape(audio_samples.shape)
        wavfile.write(output_path, sample_rate, stego_audio)

        mse, snr, psnr = calculate_quality_metrics(audio_samples, stego_audio)

        return {
            "status": "success",
            "output_path": output_path,
            "mse": mse,
            "psnr": psnr,
            "snr": snr,
            "k": embedding_depth,
            "capacity": len(payload_bytes),
        }

    except Exception as error:
        return {"status": "error", "message": str(error)}


def _extract_symbols_for_depth(flat_samples, target_indices, embedding_depth):
    """Extracts and reassembles the byte stream hidden at a specific bit-depth."""
    mask = (1 << embedding_depth) - 1
    usable_index_count = min(len(target_indices), len(flat_samples))
    extracted_values = (flat_samples[target_indices[:usable_index_count]] & mask).astype(np.uint8)

    bit_matrix = np.unpackbits(extracted_values[:, np.newaxis], axis=1)
    payload_bits = bit_matrix[:, -embedding_depth:]
    return np.packbits(payload_bits.flatten()).tobytes()


def _classify_payload(content):
    """Infers the payload type from its leading magic bytes."""
    if content.startswith(b"\xff\xd8\xff"):
        return {"type": "image", "ext": ".jpg"}
    if content.startswith(b"\x89\x50\x4e\x47"):
        return {"type": "image", "ext": ".png"}
    if content.startswith(b"BM"):
        return {"type": "image", "ext": ".bmp"}
    if content.startswith(b"RIFF"):
        return {"type": "audio", "ext": ".wav"}

    try:
        decoded_text = content.decode("utf-8")
        if all(character.isprintable() or character.isspace() for character in decoded_text):
            return {"type": "text", "ext": ".txt", "content_text": decoded_text}
    except UnicodeDecodeError:
        pass

    return {"type": "binary", "ext": ".bin"}


def decode(stego_audio_path, embedding_depth=None, password=None):
    """
    Extracts a secret payload from a stego WAV file.

    Extraction runs in constant time with respect to the true embedding
    depth: every candidate depth from 1 to MAX_EMBEDDING_DEPTH is always
    evaluated, regardless of whether the end-of-payload marker was already
    found at an earlier candidate. This prevents an attacker from inferring
    the embedding depth (and therefore the payload size) by measuring how
    long extraction takes.
    """
    try:
        stego_audio_path = os.path.abspath(stego_audio_path)
        if not os.path.exists(stego_audio_path):
            return {"status": "error", "message": "File does not exist."}

        sample_rate, stego_samples = wavfile.read(stego_audio_path)
        if stego_samples.dtype != np.int16:
            stego_samples = (
                (stego_samples * 32767).astype(np.int16)
                if stego_samples.dtype == np.float32
                else stego_samples.astype(np.int16)
            )
        flat_samples = stego_samples.flatten()

        content_salt = derive_content_salt(flat_samples, ANCHOR_SIZE)
        seed = derive_permutation_seed(password or "default", content_salt)
        random_generator = np.random.default_rng(seed)

        embeddable_range = np.arange(ANCHOR_SIZE, len(flat_samples))
        target_indices = random_generator.permutation(embeddable_range)

        candidate_depths = [embedding_depth] if embedding_depth else []
        candidate_depths += [
            depth for depth in range(1, MAX_EMBEDDING_DEPTH + 1) if depth not in candidate_depths
        ]

        print(f"   [Decode] Scanning k in {candidate_depths}...")

        # Constant-time extraction: every candidate depth is fully evaluated,
        # and the result for the matching depth is recorded without an early
        # exit. This eliminates the timing side-channel that previously
        # existed when scanning stopped as soon as a match was found.
        extracted_per_depth = {
            depth: _extract_symbols_for_depth(flat_samples, target_indices, depth)
            for depth in candidate_depths
        }

        recovered_content = None
        detected_depth = -1
        for depth in candidate_depths:
            marker_position = extracted_per_depth[depth].find(END_OF_PAYLOAD_MARKER)
            if marker_position != -1:
                recovered_content = extracted_per_depth[depth][:marker_position]
                detected_depth = depth
                print(f"   [Decode] Success! Found data at k={detected_depth}")

        if recovered_content is None:
            return {"status": "error", "message": "Sentinel not found."}

        result = {
            "status": "success",
            "data": recovered_content,
            "k_detected": detected_depth,
        }
        result.update(_classify_payload(recovered_content))
        return result

    except Exception as error:
        return {"status": "error", "message": str(error)}


def _collect_wav_files(input_dir):
    """Recursively finds all non-hidden .wav files under `input_dir`."""
    wav_paths = []
    for root, _dirs, files in os.walk(input_dir):
        for file_name in files:
            if file_name.lower().endswith(".wav") and not file_name.startswith("."):
                wav_paths.append(os.path.join(root, file_name))
    return wav_paths


def _print_batch_summary(success_count, skipped_count, failed_count, total_psnr, total_snr):
    print("\n" + "=" * 50)
    print("BATCH SUMMARY")
    print(f"Success         : {success_count}")
    print(f"Skipped         : {skipped_count}")
    print(f"Failed          : {failed_count}")

    if success_count > 0:
        print(f"Avg PSNR        : {total_psnr / success_count:.2f} dB")
        print(f"Avg SNR         : {total_snr / success_count:.2f} dB")
    print("=" * 50)


def process_batch(input_dir, secret_input, output_dir=None, embedding_depth=None, password=None):
    """Encodes the same secret payload into every WAV file found under `input_dir`."""
    if not os.path.exists(input_dir):
        print(f"[Batch] Path not found: {input_dir}")
        return []

    print(f"[Batch] Scanning: {input_dir}...")
    audio_paths = _collect_wav_files(input_dir)

    if not audio_paths:
        print("[Batch] No wav files found.")
        return []

    print(f"[Batch] Found {len(audio_paths)} files.")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results = []
    success_count = 0
    skipped_count = 0
    failed_count = 0
    total_psnr = 0.0
    total_snr = 0.0

    for index, audio_path in enumerate(audio_paths):
        print(f"\r[Processing] {index + 1}/{len(audio_paths)}... ", end="")

        original_filename = os.path.basename(audio_path)
        output_filename = f"{index:03d}_{original_filename}"
        output_path = (
            os.path.join(output_dir, output_filename) if output_dir else f"temp_{output_filename}"
        )

        encode_result = encode(
            audio_path, secret_input, output_path, embedding_depth=embedding_depth, password=password
        )

        if encode_result["status"] == "success":
            success_count += 1
            if encode_result["psnr"] != float("inf"):
                total_psnr += encode_result["psnr"]
            if encode_result["snr"] != float("inf"):
                total_snr += encode_result["snr"]

            results.append(
                {
                    "Filename": output_filename,
                    "MSE": f"{encode_result['mse']:.6f}",
                    "PSNR": f"{encode_result['psnr']:.2f}",
                    "SNR": f"{encode_result['snr']:.2f}",
                    "Status": "Success",
                }
            )
        elif "too large" in encode_result["message"]:
            skipped_count += 1
            results.append({"Filename": output_filename, "Status": "Skipped (Oversize)"})
        else:
            failed_count += 1
            results.append(
                {"Filename": output_filename, "Status": "Error", "Info": encode_result["message"]}
            )

    _print_batch_summary(success_count, skipped_count, failed_count, total_psnr, total_snr)
    return results