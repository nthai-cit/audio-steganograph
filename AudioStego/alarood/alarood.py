"""
Reimplementation of Alarood et al. (2022):
"Audio Steganography Method Using Least Significant Bit (LSB) Encoding Technique"
IJCSNS, Vol.22 No.7, July 2022. DOI: 10.22937/IJCSNS.2022.22.7.53

Adaptation notes:
- Original paper targets MP3; this implementation uses WAV (PCM int16)
  to match the MUSDB18-HQ evaluation corpus used in the present study.
- Original paper uses an unseeded MATLAB rand() call, which precludes
  reproducibility. Here the PRNG is seeded via SHA-256(password) to
  ensure deterministic and verifiable results while preserving the
  algorithmic design of the original scheme (Kerckhoffs-compliant).
"""

import hashlib
import math
import numpy as np
import scipy.io.wavfile as wavfile


# Table 2 (Alarood et al., 2022): signature for single-bit (1-LSB) insertion
_SIGNATURE = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]


def _password_to_seed(password: str) -> int:
    """Derive a 32-bit PRNG seed from a password via SHA-256."""
    digest = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return int(digest, 16) % (2 ** 32)


def _read_wav_int16(path: str):
    """Read a WAV file and return (sample_rate, int16_flat_array, original_shape)."""
    sr, data = wavfile.read(path)
    orig_dtype = data.dtype

    # Normalise to int16
    if orig_dtype == np.float32 or orig_dtype == np.float64:
        data = (data * 32767).astype(np.int16)
    elif orig_dtype != np.int16:
        data = data.astype(np.int16)

    return sr, data.flatten(), data.shape, orig_dtype


def encode(cover_path: str,
           payload_bytes: bytes,
           output_path: str,
           password: str = "DEFAULT_PASS") -> dict:
    """
    Embed payload_bytes into a WAV cover file.

    Algorithm (Alarood et al., 2022, Section 3):
      1. Compute Espace = r - ceil(|msg_bits| / deg) - 200
      2. Draw rand ~ U(0,1) from PRNG seeded by SHA-256(password)
      3. Irand = ceil(rand * floor(Espace / 2)) + 200
      4. Build payload: SIGNATURE + msg_bits + SIGNATURE
      5. Embed payload sequentially via 1-LSB substitution from Irand

    Parameters
    ----------
    cover_path   : path to the cover WAV file
    payload_bytes: raw bytes to embed
    output_path  : path for the output stego WAV file
    password     : key used to seed the PRNG (default: "DEFAULT_PASS")

    Returns
    -------
    dict with keys: status, irand, payload_bits, psnr, mse
                    (or status, message on failure)
    """
    # ── 1. Read cover ────────────────────────────────────────────────
    try:
        sr, y_flat, orig_shape, orig_dtype = _read_wav_int16(cover_path)
    except Exception as e:
        return {"status": "error", "message": f"Cannot read cover: {e}"}

    r = len(y_flat)   # total number of samples
    deg = 1           # 1-LSB insertion

    # ── 2. Convert payload to bits ───────────────────────────────────
    msg_bits = []
    for byte in payload_bytes:
        msg_bits.extend([int(b) for b in format(byte, "08b")])

    # ── 3. Capacity check (Alarood eq.) ─────────────────────────────
    # Espace = r - ceil(|msg_bits| / deg) - 200
    espace = r - math.ceil(len(msg_bits) / deg) - 200

    full_payload = _SIGNATURE + msg_bits + _SIGNATURE
    payload_length = len(full_payload)

    if espace <= 0 or r < payload_length + 200:
        return {
            "status": "error",
            "message": (f"Capacity exceeded: need {payload_length} bits "
                        f"but Espace={espace}")
        }

    # ── 4. Compute Irand ─────────────────────────────────────────────
    # Irand = ceil(rand * floor(Espace / 2)) + 200
    seed = _password_to_seed(password)
    rng = np.random.default_rng(seed)
    rand_val = rng.random()
    irand = math.ceil(rand_val * math.floor(espace / 2)) + 200

    # Guard: ensure payload fits from irand onwards
    if irand + payload_length > r:
        return {
            "status": "error",
            "message": (f"Irand={irand} too large: "
                        f"irand + payload ({payload_length}) > r ({r})")
        }

    # ── 5. Embed ─────────────────────────────────────────────────────
    y_stego = y_flat.copy()
    for i, bit in enumerate(full_payload):
        y_stego[irand + i] = (y_stego[irand + i] & ~1) | bit

    # ── 6. Save ──────────────────────────────────────────────────────
    y_out = y_stego.reshape(orig_shape).astype(np.int16)
    wavfile.write(output_path, sr, y_out)

    # ── 7. Quality metrics ───────────────────────────────────────────
    y_f = y_flat.astype(np.float64)
    y_s = y_stego.astype(np.float64)
    mse = float(np.mean((y_f - y_s) ** 2))
    max_val = float((2 ** (8 * np.dtype(np.int16).itemsize - 1)) - 1)
    psnr = float("inf") if mse == 0 else 10 * math.log10(max_val ** 2 / mse)

    return {
        "status":       "success",
        "irand":        irand,
        "payload_bits": payload_length,
        "mse":          mse,
        "psnr":         psnr,
    }


def decode(stego_path: str) -> dict:
    """
    Extract the hidden message from a stego WAV file.

    Decoding does not require the password or Irand: it scans all LSBs
    from sample index 200 onwards, locating the start and end signatures.

    Returns
    -------
    dict with keys: status, data (bytes)
                    (or status, message on failure)
    """
    try:
        _, y_flat, _, _ = _read_wav_int16(stego_path)
    except Exception as e:
        return {"status": "error", "message": f"Cannot read stego: {e}"}

    # Extract LSBs from index 200 onwards
    lsbs = (y_flat[200:] & 1).astype(np.uint8)
    sig = np.array(_SIGNATURE, dtype=np.uint8)
    sig_len = len(sig)

    # Find start signature
    start = -1
    for i in range(len(lsbs) - sig_len):
        if np.array_equal(lsbs[i: i + sig_len], sig):
            start = i + sig_len
            break
    if start == -1:
        return {"status": "error", "message": "Start signature not found"}

    # Find end signature
    end = -1
    for i in range(start, len(lsbs) - sig_len):
        if np.array_equal(lsbs[i: i + sig_len], sig):
            end = i
            break
    if end == -1:
        return {"status": "error", "message": "End signature not found"}

    # Reconstruct bytes
    msg_bits = lsbs[start:end]
    if len(msg_bits) % 8 != 0:
        msg_bits = msg_bits[:-(len(msg_bits) % 8)]

    raw = bytearray()
    for i in range(0, len(msg_bits), 8):
        byte_val = 0
        for bit in msg_bits[i: i + 8]:
            byte_val = (byte_val << 1) | int(bit)
        raw.append(byte_val)

    return {"status": "success", "data": bytes(raw)}


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import tempfile

    sr_test = 44100
    cover_data = np.zeros(sr_test * 5, dtype=np.int16)   # 5 s silence

    with tempfile.TemporaryDirectory() as tmp:
        cover = os.path.join(tmp, "cover.wav")
        stego = os.path.join(tmp, "stego.wav")
        wavfile.write(cover, sr_test, cover_data)

        secret = b"Hello Alarood 2022"
        result = encode(cover, secret, stego, password="test_key")
        assert result["status"] == "success", result
        print(f"Encoded | Irand={result['irand']} | PSNR={result['psnr']:.2f} dB")

        dec = decode(stego)
        assert dec["status"] == "success", dec
        assert dec["data"] == secret, f"Mismatch: {dec['data']}"
        print(f"Decoded | data={dec['data']}")
        print("OK — encode/decode round-trip passed.")