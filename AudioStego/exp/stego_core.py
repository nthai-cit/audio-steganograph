"""
stego_core.py
=============
Core audio steganography engine.

This module contains all embedding algorithms used in the scientific evaluation:

* ``StegoUtils``    — payload preparation and PRNG seed generation.
* ``StegoBasic``    — Case 1: sequential 1-bit LSB substitution.
* ``StegoImproved`` — Cases 2–5: randomised k-bit LSB with fixed or adaptive k
                      and default or content-derived salt.
* ``StegoAlarood``  — Case 8: Alarood (2022) randomised LSB baseline.
* ``StegoPhase``    — Case 7: Phase Coding via per-segment FFT manipulation.

All ``encode`` methods accept a pre-prepared ``payload_bytes`` object
(produced by ``StegoUtils.prepare_payload``) and write a stego WAV to
``output_path``.  Return value is always a dict with at minimum a
``'status'`` key (``'success'`` or ``'error'``).
"""

from __future__ import annotations

import hashlib
import io
import math
import os

import numpy as np
from scipy.io import wavfile

try:
    from PIL import Image
except ImportError:
    print("Warning: PIL not installed. Image resizing will be disabled.")
    Image = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Byte sequence appended to every payload so decoders can locate the end
# of hidden data without a stored length value.
PAYLOAD_END_MARKER = b"||END||"
PHASE_PAYLOAD_END_MARKER = b"||DATA_END||"

# Number of audio samples reserved at the start of the cover signal to
# derive a content-dependent salt.  Must match the decode side.
ANCHOR_SAMPLE_COUNT = 1024

# Maximum number of LSB planes that may be modified per sample.
K_MAX_BITS = 6

# JPEG compression quality used when preparing image payloads.
# Set to 20 to achieve the physical embedding rate target cited in the paper.
JPEG_COMPRESSION_QUALITY = 20

# Phase values that encode a single bit in the frequency domain.
# +π/4 → bit 0,  −π/4 → bit 1.
PHASE_FOR_BIT_ZERO = np.pi / 4
PHASE_FOR_BIT_ONE = -np.pi / 4

# Target segment duration (seconds) for Phase Coding FFT window selection.
PHASE_SEGMENT_TARGET_DURATION_SECONDS = 1.5

# Hard bounds on segment length (samples) to keep FFT sizes tractable.
PHASE_SEGMENT_MAX_SAMPLES = 131_072
PHASE_SEGMENT_MIN_SAMPLES = 8_192

# Fraction of the positive-frequency half-spectrum usable for embedding.
PHASE_USABLE_FREQUENCY_FRACTION = 0.8

# Fraction of the positive-frequency half-spectrum to skip from DC before
# embedding, to avoid perceptually prominent low-frequency bins.
PHASE_EMBEDDING_START_FRACTION = 0.1

# Peak amplitude of a normalised float signal (soundfile default range).
FLOAT_PEAK_AMPLITUDE = 1.0

# Salt value used when the salt source is not content-derived.
STATIC_DEFAULT_SALT = "STATIC_DEFAULT_SALT"


# ---------------------------------------------------------------------------
# StegoUtils
# ---------------------------------------------------------------------------

class StegoUtils:
    """Utility methods shared across all steganography variants."""

    @staticmethod
    def compress_image(file_path: str, target_size: int, jpeg_quality: int = 45) -> bytes:
        """
        Resize *file_path* to a square of *target_size* × *target_size* pixels
        and re-encode it as JPEG at the given quality level.

        If PIL is unavailable the raw file bytes are returned unchanged.

        Parameters
        ----------
        file_path : str
            Path to the input image file.
        target_size : int
            Target width and height in pixels.
        jpeg_quality : int
            JPEG compression quality (1–95).  Lower values produce smaller
            files at the cost of image fidelity.

        Returns
        -------
        bytes
            JPEG-compressed image bytes, or raw file bytes on failure.
        """
        if Image is None:
            with open(file_path, 'rb') as image_file:
                return image_file.read()

        try:
            img = Image.open(file_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize to exact square dimensions using high-quality Lanczos filter.
            img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)

            # Compress at a fixed quality level.  A loop-based adaptive
            # approach is deliberately avoided to keep encoding deterministic.
            jpeg_buffer = io.BytesIO()
            img.save(jpeg_buffer, format='JPEG', quality=jpeg_quality)
            return jpeg_buffer.getvalue()

        except Exception as compression_error:
            print(f"[Image Error] {compression_error}. Using raw bytes.")
            with open(file_path, 'rb') as image_file:
                return image_file.read()

    @staticmethod
    def prepare_payload(file_path: str, target_size: int = 256) -> bytes:
        """
        Load and pre-process the secret file into a byte payload ready for embedding.

        Image files are resized and JPEG-compressed to keep the payload within
        practical embedding capacity limits.  All other file types are read
        as-is in binary mode.

        Parameters
        ----------
        file_path : str
            Path to the secret file (image or arbitrary binary).
        target_size : int
            Resize dimension (pixels) applied to image inputs.

        Returns
        -------
        bytes
            Processed payload bytes.

        Raises
        ------
        FileNotFoundError
            When *file_path* does not exist on disk.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension in image_extensions:
            return StegoUtils.compress_image(file_path, target_size, jpeg_quality=JPEG_COMPRESSION_QUALITY)

        with open(file_path, 'rb') as binary_file:
            return binary_file.read()

    @staticmethod
    def generate_seed(password: str, salt: str) -> int:
        """
        Derive a deterministic PRNG seed from a password and a salt string.

        HMAC-SHA256 is used so that neither the password nor the salt alone
        is sufficient to reconstruct the seed, providing one-way key
        derivation without modular truncation (unlike a simple hash-mod).

        Parameters
        ----------
        password : str
            User-supplied password string.
        salt : str
            Salt value (either a static string or a content-derived hex digest).

        Returns
        -------
        int
            A 256-bit integer suitable for seeding ``numpy.random.default_rng``.
        """
        import hmac

        seed_bytes = hmac.new(
            key=password.encode('utf-8'),
            msg=salt.encode('utf-8'),
            digestmod=hashlib.sha256,
        ).digest()  # 32 bytes = 256 bits

        # Convert to a large integer without modular truncation to preserve
        # the full entropy of the HMAC output.
        return int.from_bytes(seed_bytes, byteorder='big')


# ---------------------------------------------------------------------------
# StegoBasic  (Case 1 — Sequential 1-bit LSB)
# ---------------------------------------------------------------------------

class StegoBasic:
    """
    Case 1: Sequential 1-bit LSB substitution.

    The payload bits are written into the least significant bit of consecutive
    samples starting at index 0, with no randomisation.  This serves as the
    simplest baseline in the comparative evaluation.
    """

    @staticmethod
    def encode(cover_path: str, payload_bytes: bytes, output_path: str) -> dict:
        """
        Embed *payload_bytes* into *cover_path* using sequential 1-bit LSB.

        Parameters
        ----------
        cover_path : str
            Path to the original (cover) WAV file.
        payload_bytes : bytes
            Pre-processed payload produced by ``StegoUtils.prepare_payload``.
        output_path : str
            Destination path for the stego WAV file.

        Returns
        -------
        dict
            ``{'status': 'success'}`` on success, or
            ``{'status': 'error', 'message': ...}`` on failure.
        """
        try:
            sample_rate, audio_data = wavfile.read(cover_path)
            if audio_data.dtype != np.int16:
                audio_data = (
                    (audio_data * 32767).astype(np.int16)
                    if audio_data.dtype == np.float32
                    else audio_data.astype(np.int16)
                )

            flat_samples = audio_data.flatten()
            stego_flat = flat_samples.copy()

            full_payload = payload_bytes + PAYLOAD_END_MARKER
            payload_bits = np.unpackbits(np.frombuffer(full_payload, dtype=np.uint8))

            if len(payload_bits) > len(stego_flat):
                return {"status": "error", "message": "Oversize"}

            # Zero the LSB of each target sample, then write the payload bit.
            stego_flat[:len(payload_bits)] &= ~1
            stego_flat[:len(payload_bits)] |= payload_bits.astype(np.int16)

            stego_data = stego_flat.reshape(audio_data.shape)
            wavfile.write(output_path, sample_rate, stego_data)
            return {"status": "success"}

        except Exception as encode_error:
            return {"status": "error", "message": str(encode_error)}


# ---------------------------------------------------------------------------
# StegoImproved  (Cases 2–5 — Randomised k-bit LSB)
# ---------------------------------------------------------------------------

class StegoImproved:
    """
    Cases 2–5: Randomised k-bit LSB substitution.

    Embedding positions are determined by a seeded PRNG so that only a
    party knowing the password can locate the hidden data.  The number of
    modified LSB planes (k) may be fixed or computed adaptively from the
    payload size.  The PRNG seed is derived from the password combined with
    either a static salt or a content-dependent salt derived from the first
    ``ANCHOR_SAMPLE_COUNT`` samples of the cover audio.
    """

    @staticmethod
    def encode(
        cover_path: str,
        payload_bytes: bytes,
        output_path: str,
        password: str,
        k_strategy: str = 'fixed',
        salt_source: str = 'default',
        fixed_k_val: int = 1,
    ) -> dict:
        """
        Embed *payload_bytes* into *cover_path* using randomised k-bit LSB.

        Parameters
        ----------
        cover_path : str
            Path to the original (cover) WAV file.
        payload_bytes : bytes
            Pre-processed payload produced by ``StegoUtils.prepare_payload``.
        output_path : str
            Destination path for the stego WAV file.
        password : str
            Password used to seed the PRNG for position permutation.
        k_strategy : str
            ``'fixed'`` uses *fixed_k_val* planes; ``'adaptive'`` computes
            the minimum k that fits the payload within the available slots.
        salt_source : str
            ``'content'`` derives the salt from the anchor bytes of the cover
            audio; ``'default'`` uses a static string.
        fixed_k_val : int
            Number of LSB planes to use when *k_strategy* is ``'fixed'``.

        Returns
        -------
        dict
            ``{'status': 'success', 'k_used': k, 'salt_used': salt}`` on
            success, or ``{'status': 'error', 'message': ...}`` on failure.
        """
        try:
            sample_rate, audio_data = wavfile.read(cover_path)
            if audio_data.dtype != np.int16:
                audio_data = (
                    (audio_data * 32767).astype(np.int16)
                    if audio_data.dtype == np.float32
                    else audio_data.astype(np.int16)
                )

            flat_samples = audio_data.flatten()
            stego_flat = flat_samples.copy()

            # Samples before the anchor are reserved for salt derivation
            # and are never overwritten by payload bits.
            embedding_slot_count = len(flat_samples) - ANCHOR_SAMPLE_COUNT

            if k_strategy == 'adaptive':
                # Compute the minimum k that fits the full payload (including
                # the end marker and a small overhead buffer) in one pass.
                required_bits = (len(payload_bytes) + 10) * 8
<<<<<<< HEAD
                calc_k = math.ceil(required_bits / num_slots)
                # k = max(1, min(calc_k, 6))
                k = min(calc_k, 6) 
=======
                calculated_k = math.ceil(required_bits / embedding_slot_count)
                k = min(calculated_k, K_MAX_BITS)
>>>>>>> chien
            else:
                k = fixed_k_val

            full_payload = payload_bytes + PAYLOAD_END_MARKER
            payload_bits = np.unpackbits(np.frombuffer(full_payload, dtype=np.uint8))

            # Pad to a multiple of k so every group maps to exactly one sample slot.
            remainder = len(payload_bits) % k
            if remainder != 0:
                payload_bits = np.append(payload_bits, [0] * (k - remainder))

            # Pack k bits into a single integer value per embedding slot.
            # The dot product with descending powers of 2 converts each k-bit
            # group to its unsigned integer representation (MSB first).
            bit_plane_powers = 1 << np.arange(k)[::-1]
            secret_values = payload_bits.reshape(-1, k).dot(bit_plane_powers).astype(np.int16)

            if len(secret_values) > embedding_slot_count:
                return {
                    "status": "error",
                    "message": f"Oversize (K={k}, Need {len(secret_values)} slots, Have {embedding_slot_count})",
                }

            if salt_source == 'content':
                # Hash the anchor samples to produce a cover-specific salt that
                # ties the PRNG seed to the audio content.
                anchor_bytes = flat_samples[:ANCHOR_SAMPLE_COUNT].tobytes()
                salt = hashlib.sha256(anchor_bytes).hexdigest()
            else:
                salt = STATIC_DEFAULT_SALT

            seed = StegoUtils.generate_seed(password, salt)
            rng = np.random.default_rng(seed)

            # Permute only the non-anchor sample indices so the anchor is
            # never disturbed and remains available for salt re-derivation.
            non_anchor_indices = np.arange(ANCHOR_SAMPLE_COUNT, len(flat_samples))
            shuffled_indices = rng.permutation(non_anchor_indices)
            target_indices = shuffled_indices[:len(secret_values)]

            # Clear k LSB planes at target positions, then write the secret values.
            k_bit_mask = (1 << k) - 1
            stego_flat[target_indices] &= ~k_bit_mask
            stego_flat[target_indices] |= secret_values

            stego_data = stego_flat.reshape(audio_data.shape)
            wavfile.write(output_path, sample_rate, stego_data)

            return {"status": "success", "k_used": k, "salt_used": salt}

        except Exception as encode_error:
            return {"status": "error", "message": str(encode_error)}


# ---------------------------------------------------------------------------
# StegoAlarood  (Case 8 — Alarood 2022 baseline)
# ---------------------------------------------------------------------------

class StegoAlarood:
    """
    Case 8: Alarood (2022) randomised LSB baseline.

    This class delegates entirely to the external ``alarood`` module so that
    the published algorithm is reproduced without modification.
    """

    @staticmethod
    def encode(
        cover_path: str,
        payload_bytes: bytes,
        output_path: str,
        password: str = "DEFAULT_PASS",
    ) -> dict:
        """
        Embed *payload_bytes* using the Alarood (2022) algorithm.

        Parameters
        ----------
        cover_path : str
            Path to the original (cover) WAV file.
        payload_bytes : bytes
            Pre-processed payload bytes.
        output_path : str
            Destination path for the stego WAV file.
        password : str
            Password forwarded to the Alarood encoder.

        Returns
        -------
        dict
            Result dict from the ``alarood`` module.
        """
        try:
            import alarood  # type: ignore[import]
        except ImportError:
            return {
                "status": "error",
                "message": "alarood_stego.py not found. Place it next to stego_core.py.",
            }
        return alarood.encode(cover_path, payload_bytes, output_path, password=password)

    @staticmethod
    def decode(stego_path: str) -> dict:
        """
        Extract the hidden payload from a stego WAV produced by :meth:`encode`.

        Parameters
        ----------
        stego_path : str
            Path to the stego WAV file.

        Returns
        -------
        dict
            Result dict from the ``alarood`` module.
        """
        try:
            import alarood  # type: ignore[import]
        except ImportError:
            return {"status": "error", "message": "alarood_stego.py not found."}
        return alarood.decode(stego_path)


# ---------------------------------------------------------------------------
# StegoPhase  (Case 7 — Phase Coding via per-segment FFT)
# ---------------------------------------------------------------------------

class StegoPhase:
    """
    Case 7: Phase Coding steganography.

    Payload bits are encoded by replacing the phase of selected FFT bins in
    short, fixed-length audio segments:

    * ``PHASE_FOR_BIT_ZERO = +π/4`` encodes bit 0.
    * ``PHASE_FOR_BIT_ONE  = −π/4`` encodes bit 1.

    After modifying positive-frequency phases the conjugate-symmetric
    negative-frequency bins are updated (``phase[N−k] = −phase[k]``) to
    guarantee that the IFFT output is purely real-valued.
    """

    @staticmethod
    def _read_audio_as_float(filepath: str) -> tuple[int, np.ndarray]:
        """
        Read *filepath* and return its first channel as a normalised float32 array.

        Integer samples are scaled to [−1.0, 1.0] using the dtype maximum.
        Multi-channel files are reduced to mono by retaining the first channel.

        Parameters
        ----------
        filepath : str
            Path to the WAV file.

        Returns
        -------
        tuple[int, np.ndarray]
            ``(sample_rate_hz, float32_audio_samples)``.
        """
        sample_rate, audio_data = wavfile.read(filepath)
        if np.issubdtype(audio_data.dtype, np.integer):
            integer_max = np.iinfo(audio_data.dtype).max
            audio_data = audio_data.astype(np.float32) / integer_max
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]  # Retain first channel only.
        return sample_rate, audio_data

    @staticmethod
    def _write_audio_float(filepath: str, sample_rate: int, audio_data: np.ndarray) -> None:
        """
        Write *audio_data* to *filepath* as float32 WAV, normalising if clipped.

        If any sample exceeds ±1.0 the entire array is scaled down to prevent
        clipping, which would corrupt the embedded phase values on decode.

        Parameters
        ----------
        filepath : str
            Destination WAV path.
        sample_rate : int
            Sample rate in Hz.
        audio_data : np.ndarray
            1-D float array of audio samples.
        """
        peak_amplitude = np.max(np.abs(audio_data))
        if peak_amplitude > FLOAT_PEAK_AMPLITUDE:
            audio_data = audio_data / peak_amplitude
        wavfile.write(filepath, sample_rate, audio_data.astype(np.float32))

    @staticmethod
    def _calculate_segment_params(audio_len: int, sample_rate: int) -> tuple[int, int, int]:
        """
        Derive FFT segment geometry from audio length and sample rate.

        The segment length is the largest power-of-two duration at or below
        ``PHASE_SEGMENT_TARGET_DURATION_SECONDS``, clamped to the hard limits.
        Usable embedding bins = 80 % of the positive-frequency half, skipping
        the bottom 10 % to avoid DC-adjacent perceptual artefacts.

        Parameters
        ----------
        audio_len : int
            Total number of samples in the audio signal.
        sample_rate : int
            Sample rate in Hz.

        Returns
        -------
        tuple[int, int, int]
            ``(segment_length, segment_count, usable_bins_per_segment)``.
        """
        segment_length = 2 ** int(np.log2(sample_rate * PHASE_SEGMENT_TARGET_DURATION_SECONDS))
        segment_length = min(segment_length, PHASE_SEGMENT_MAX_SAMPLES)
        segment_length = max(segment_length, PHASE_SEGMENT_MIN_SAMPLES)
        segment_count = int(np.ceil(audio_len / segment_length))
        usable_bins = int((segment_length // 2) * PHASE_USABLE_FREQUENCY_FRACTION)
        return segment_length, segment_count, usable_bins

    @staticmethod
    def encode(cover_path: str, payload_bytes: bytes, output_path: str) -> dict:
        """
        Embed *payload_bytes* using Phase Coding via per-segment FFT.

        Parameters
        ----------
        cover_path : str
            Path to the original (cover) WAV file.
        payload_bytes : bytes
            Pre-processed payload produced by ``StegoUtils.prepare_payload``.
        output_path : str
            Destination path for the stego WAV file.

        Returns
        -------
        dict
            ``{'status': 'success', 'info': 'Phase_FFT'}`` on success, or
            ``{'status': 'error', 'message': ...}`` on failure.
        """
        try:
            sample_rate, audio_samples = StegoPhase._read_audio_as_float(cover_path)

            full_payload = payload_bytes + PHASE_PAYLOAD_END_MARKER
            payload_bits = np.unpackbits(np.frombuffer(full_payload, dtype=np.uint8))
            total_bits = len(payload_bits)

            segment_length, segment_count, usable_bins = StegoPhase._calculate_segment_params(
                len(audio_samples), sample_rate
            )
            total_capacity = usable_bins * segment_count

            if total_bits > total_capacity:
                return {
                    "status": "error",
                    "message": f"Oversize: need {total_bits} bits, capacity {total_capacity}",
                }

            # Pad audio to an exact multiple of segment_length for reshape.
            padded_length = segment_count * segment_length
            if len(audio_samples) < padded_length:
                audio_samples = np.pad(
                    audio_samples,
                    (0, padded_length - len(audio_samples)),
                    mode='constant',
                )

            # Batch FFT over all segments simultaneously.
            segments = audio_samples.reshape((segment_count, segment_length))
            fft_of_segments = np.fft.fft(segments)

            magnitudes = np.abs(fft_of_segments)
            phases = np.angle(fft_of_segments)

            # Map each bit to its sentinel phase value (vectorised).
            phase_values = np.where(payload_bits == 0, PHASE_FOR_BIT_ZERO, PHASE_FOR_BIT_ONE)

            # Embedding begins 10 % into the positive-frequency half to avoid
            # modifying DC and very low-frequency bins.
            positive_half = segment_length // 2
            embedding_start_bin = int(positive_half * PHASE_EMBEDDING_START_FRACTION)
            bits_embedded = 0

            for segment_index in range(segment_count):
                bits_in_this_segment = min(usable_bins, total_bits - bits_embedded)
                if bits_in_this_segment <= 0:
                    break

                segment_phase_slice = phase_values[bits_embedded : bits_embedded + bits_in_this_segment]
                embedding_end_bin = embedding_start_bin + bits_in_this_segment

                # Overwrite positive-frequency bin phases.
                phases[segment_index, embedding_start_bin:embedding_end_bin] = segment_phase_slice

                # Mirror to negative-frequency bins (conjugate symmetry):
                # X[N−k] = conj(X[k])  →  phase[N−k] = −phase[k].
                mirror_end = segment_length - embedding_start_bin + 1
                mirror_start = segment_length - embedding_end_bin + 1
                phases[segment_index, mirror_start:mirror_end] = -segment_phase_slice[::-1]

                bits_embedded += bits_in_this_segment

            modified_fft = magnitudes * np.exp(1j * phases)
            stego_audio = np.fft.ifft(modified_fft).real.ravel()

            StegoPhase._write_audio_float(output_path, sample_rate, stego_audio)
            return {"status": "success", "info": "Phase_FFT"}

        except Exception as encode_error:
            return {"status": "error", "message": str(encode_error)}