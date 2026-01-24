## LSB:
# text-in-audio:

python main.py encode -m lsb -i "inputs/song.wav" -s "Chien b2203431"
    BANG KET QUA (LSB):
   [-] MSE  : 0.0000
   [+] SNR  : 198.14 dB
   [+] PSNR : 142.06 dB

python main.py decode -m lsb -i "outputs\encode\2026-01-23_20-39-42_LSB\song.wav"
[INFO] SESSION: 23/01/2026 20:51:05
[INFO] CHE DO: DECODE | PHUONG PHAP: LSB
[INFO] Thu muc lam viec: outputs\decode\2026-01-23_20-51-05_LSB
   [LSB Decode] Dang doc file: song.wav
Thong Diep: Chien b2203431
[THANH CONG] Trich xuat loai: TEXT
[INFO] Tong thoi gian: 0.1030s

# img-in-audio
python main.py encode -m lsb -i "inputs/song.wav" -s "inputs/img.jpg"
                                                                                                                                
   BANG KET QUA (LSB):
   [-] MSE  : 0.0840
   [+] SNR  : 157.15 dB
   [+] PSNR : 101.06 dB

 python main.py decode -m lsb -i "outputs\encode\2026-01-23_20-52-11_LSB\song.wav"
 [INFO] SESSION: 23/01/2026 20:57:41
[INFO] CHE DO: DECODE | PHUONG PHAP: LSB
[INFO] Thu muc lam viec: outputs\decode\2026-01-23_20-57-41_LSB
   [LSB Decode] Dang xu ly: song.wav

==================== KET QUA GIAI MA ====================
[LOAI FILE]: IMAGE (.jpg)
[DA LUU TAI]: decoded_results\extracted_song.wav_image.jpg
[HANH DONG]: Dang mo file...
=========================================================

[THANH CONG] Trich xuat loai: IMAGE
[INFO] Tong thoi gian: 0.1737s

# audio-in-audio

python main.py encode -m lsb -i "inputs/song.wav" -s "inputs/audio.wav"
----------------------------------------
   BANG KET QUA (LSB):
   [-] MSE  : 0.0948
   [+] SNR  : 156.62 dB
   [+] PSNR : 100.54 dB
----------------------------------------

python main.py decode -m lsb -i "outputs\encode\2026-01-23_20-59-04_LSB/song.wav"

[INFO] SESSION: 23/01/2026 21:02:40
[INFO] CHE DO: DECODE | PHUONG PHAP: LSB
[INFO] Thu muc lam viec: outputs\decode\2026-01-23_21-02-40_LSB
   [LSB Decode] Dang xu ly: song.wav

==================== KET QUA GIAI MA ====================
[LOAI FILE]: AUDIO (.wav)
[DA LUU TAI]: decoded_results\extracted_song.wav_audio.wav
[HANH DONG]: Dang mo file...
=========================================================

[THANH CONG] Trich xuat loai: AUDIO
[INFO] Tong thoi gian: 0.7414s
============================================================
(stego) PS D:\HK2-202502026\Github-audio-stego\audio-steganog

## Phase
# text in audio
 python main.py encode -m phase -i "inputs/song.wav" -s "Chien b2203431"

[INFO] CHE DO: ENCODE | PHUONG PHAP: PHASE
[INFO] Thu muc: outputs\encode\2026-01-23_22-04-29_PHASE_text
   [Phase Info] Nhan dien du lieu la VAN BAN.
   [Phase] Dang nhung 208 bits...
----------------------------------------
   BANG KET QUA (PHASE):
   [-] MSE  : 0.000006
   [+] SNR  : 36.04 dB
   [+] PSNR : 52.00 dB
----------------------------------------
[THANH CONG] File Stego: outputs\encode\2026-01-23_22-04-29_PHASE_text\song.wav
[INFO] Tong thoi gian: 1.2697s

python main.py encode -m phase -i "outputs\encode\2026-01-23_22-04-29_PHASE_text\song.wav

============================================================
[INFO] CHE DO: DECODE | PHUONG PHAP: PHASE
   [Phase Decode] Dang xu ly: song.wav

==================== KET QUA GIAI MA ====================
[NOI DUNG VAN BAN]:
Chien b2203431
=========================================================


[NOI DUNG]: Chien b2203431

[INFO] Tong thoi gian: 0.8101s
============================================================

# img in audio 
python main.py encode -m phase -i "inputs/song.wav" -s "inputs/img.jp
g"
============================================================
[INFO] CHE DO: ENCODE | PHUONG PHAP: PHASE
[INFO] Thu muc: outputs\encode\2026-01-23_22-46-35_PHASE_img
   [Phase Info] Nhan dien du lieu la FILE: 'img.jpg'
   [Phase] Dang nhung 2541448 bits...
----------------------------------------
   BANG KET QUA (PHASE):
   [-] MSE  : 0.002057
   [+] SNR  : 10.92 dB
   [+] PSNR : 26.87 dB
----------------------------------------
[THANH CONG] File Stego: outputs\encode\2026-01-23_22-46-35_PHASE_img\song.wav
[INFO] Tong thoi gian: 1.1210s

python main.py decode -m phase -i "outputs\encode\2026-01-23_22-46-35_PHASE_img\song.wav"
============================================================
[INFO] CHE DO: DECODE | PHUONG PHAP: PHASE
   [Phase Decode] Dang xu ly: song.wav

==================== KET QUA GIAI MA ====================
[LOAI FILE]: IMAGE (.jpg)
[DA LUU TAI]: decoded_results\extracted_song.wav.jpg
[HANH DONG]: Dang mo file...
=========================================================

[THANH CONG] Da luu: outputs\decode\2026-01-23_22-48-02_PHASE\extracted_song.wav.jpg
[INFO] Tong thoi gian: 0.9137s
============================================================

# audio in audio
(stego) PS D:\HK2-202502026\Github-audio-stego\audio-steganograph> python main.py encode -m phase -i "inputs/song.wav" -s "inputs/audio.wav"
============================================================
[INFO] CHE DO: ENCODE | PHUONG PHAP: PHASE
[INFO] Thu muc: outputs\encode\2026-01-23_22-53-44_PHASE_audio
   [Phase Info] Nhan dien du lieu la FILE: 'audio.wav'
   [Phase] Dang nhung 2867648 bits...
----------------------------------------
   BANG KET QUA (PHASE):
   [-] MSE  : 0.002262
   [+] SNR  : 10.50 dB
   [+] PSNR : 26.46 dB
----------------------------------------
[THANH CONG] File Stego: outputs\encode\2026-01-23_22-53-44_PHASE_audio\song.wav
[INFO] Tong thoi gian: 1.1390s
==================================
python main.py decode -m phase -i "outputs\encode\2026-01-23_22-53-44_PHASE_audio\song.wav"
============================================================
[INFO] CHE DO: DECODE | PHUONG PHAP: PHASE
   [Phase Decode] Dang xu ly: song.wav

==================== KET QUA GIAI MA ====================
[LOAI FILE]: AUDIO (.wav)
[DA LUU TAI]: decoded_results\extracted_song.wav.wav
[HANH DONG]: Dang mo file...
=========================================================

[THANH CONG] Da luu: outputs\decode\2026-01-23_22-54-38_PHASE\extracted_song.wav.wav
[INFO] Tong thoi gian: 1.6786s


# LSB improved

# text i audio
 python main.py encode -m improved -k 1 -p "Chien" -i "inputs/song.wav" -s "Chien b2203431"
============================================================
[INFO] CHE DO: ENCODE | PHUONG PHAP: IMPROVED
[INFO] Thu muc: outputs\encode\2026-01-23_23-04-52_IMPROVED_text
   [Improved] Dang nhung 14 bytes (k=1)...
   [DANH GIA] Dang tinh toan chi so...
----------------------------------------
   BANG KET QUA (IMPROVED LSB):
   [-] MSE  : 0.000005
   [+] SNR  : 127.25 dB
   [+] PSNR : 142.97 dB
----------------------------------------
[THANH CONG] File Stego: outputs\encode\2026-01-23_23-04-52_IMPROVED_text\song.wav
[INFO] Tong thoi gian: 1.2620s



 python main.py decode -m improved -k 1 -p "Chien" -i "outputs\encode\2026-01-23_23-06-25_IMPROVED_img\song.wav"
============================================================
[INFO] CHE DO: DECODE | PHUONG PHAP: IMPROVED
   [Improved Decode] Dang xu ly: song.wav
[THANH CONG] Da luu: outputs\decode\2026-01-23_23-09-41_IMPROVED\extracted_song.wav.jpg
[INFO] Tong thoi gian: 1.2168s
============================================================

# img in audio 
python main.py encode -m improved -k 1 -p "Chien" -i "inputs/song.wav" -s "inputs/img.jpg"
============================================================
[INFO] CHE DO: ENCODE | PHUONG PHAP: IMPROVED
[INFO] Thu muc: outputs\encode\2026-01-23_23-06-25_IMPROVED_img
   [Improved] Dang nhung 237333 bytes (k=1)...
   [DANH GIA] Dang tinh toan chi so...
----------------------------------------
   BANG KET QUA (IMPROVED LSB):
   [-] MSE  : 0.062888
   [+] SNR  : 86.61 dB
   [+] PSNR : 102.32 dB
----------------------------------------
[THANH CONG] File Stego: outputs\encode\2026-01-23_23-06-25_IMPROVED_img\song.wav
[INFO] Tong thoi gian: 1.2699s
============================================================
python main.py decode -m improved -k 1 -p "Chien" -i "outputs\encode\2026-01-23_23-06-25_IMPROVED_img\song.wav" 
============================================================
[INFO] CHE DO: DECODE | PHUONG PHAP: IMPROVED
   [Improved Decode] Dang xu ly: song.wav
[THANH CONG] Da luu: outputs\decode\2026-01-23_23-07-39_IMPROVED\extracted_song.wav.jpg
[INFO] Tong thoi gian: 1.4364s
============================================================

python main.py decode -m improved -k 1 -p "Chien" -i "outputs\encode\2026-01-23_23-08-10_IMPROVED_audio\song.wav"
============================================================
[INFO] CHE DO: DECODE | PHUONG PHAP: IMPROVED
   [Improved Decode] Dang xu ly: song.wav
[THANH CONG] Da luu: outputs\decode\2026-01-23_23-09-20_IMPROVED\extracted_song.wav.wav
[INFO] Tong thoi gian: 1.6509s
============================================================