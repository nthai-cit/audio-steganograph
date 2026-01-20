import subprocess
import re
import os
import sys

# --- CAU HINH DUONG DAN TU DONG ---
PYTHON_CMD = sys.executable 

# 1. Xac dinh vi tri file script nay (.../test/full_test_suite.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Xac dinh thu muc Goc Du An (Project Root) - Di lui 1 cap
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 3. Xac dinh duong dan tuyet doi den main.py
MAIN_PY_PATH = os.path.join(PROJECT_ROOT, "main.py")

def run_command(cmd_list):
    """
    Chay lenh va tra ve ket qua.
    QUAN TRONG: cwd=PROJECT_ROOT giup main.py chay nhu the dang o thu muc goc
    """
    # In ra lenh dang chay (rut gon duong dan cho de nhin)
    display_cmd = ' '.join(cmd_list).replace(PROJECT_ROOT, ".")
    print(f"\n[EXEC] {display_cmd}")
    
    try:
        result = subprocess.run(
            cmd_list, 
            capture_output=True, 
            text=True, 
            encoding='utf-8',
            check=True,
            cwd=PROJECT_ROOT  # <--- QUAN TRONG: Chay lenh tu Thu Muc Goc
        )
        print(result.stdout)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"[LOI] Lenh that bai:\n{e.stderr}")
        return None

def extract_stego_path(output_log):
    """Tim duong dan file stego tu log cua lenh Encode"""
    # Regex tim duong dan file .wav trong log
    match = re.search(r"File Stego: (.*wav)", output_log)
    if match:
        # Tra ve duong dan (co the la tuyet doi hoac tuong doi)
        path = match.group(1).strip()
        # Neu path la tuong doi, ghep voi project root de chac chan
        if not os.path.isabs(path):
            path = os.path.join(PROJECT_ROOT, path)
        return path
    return None

def test_scenario(name, encode_args, decode_args_template):
    print("="*60)
    print(f"TEST SCENARIO: {name}")
    print("="*60)
    
    # 1. ENCODE
    print(">>> BUOC 1: ENCODE")
    # Luu y: Dung MAIN_PY_PATH tuyet doi
    enc_cmd = [PYTHON_CMD, MAIN_PY_PATH, "encode"] + encode_args
    enc_output = run_command(enc_cmd)
    
    if not enc_output: return
    
    # Lay duong dan file vua tao
    stego_path = extract_stego_path(enc_output)
    if not stego_path:
        print("[SKIP] Khong tim thay file output tu log. Bo qua buoc Decode.")
        return
        
    print(f">>> File Stego duoc tao ra tai: {stego_path}")
    
    # 2. DECODE
    print(">>> BUOC 2: DECODE")
    dec_cmd = [PYTHON_CMD, MAIN_PY_PATH, "decode", "-i", stego_path] + decode_args_template
    run_command(dec_cmd)

def main():
    input_wav = os.path.join(PROJECT_ROOT, "inputs", "song.wav")
    
    if not os.path.exists(input_wav):
        print(f"[LOI] Khong tim thay: {input_wav}")
        print("Vui long chay 'python test/prepare_data.py' tu thu muc goc truoc!")
        return

    test_scenario(
        "1. LSB Basic (Text)",
        encode_args=["-m", "lsb", "-i", "inputs/song.wav", "-s", "inputs/secret.txt"],
        decode_args_template=["-m", "lsb"]
    )

    test_scenario(
        "2. Improved LSB (Image + Password + K=4)",
        encode_args=["-m", "improved", "-k", "4", "-p", "MySecretPass", "-i", "inputs/song.wav", "-s", "inputs/image.jpg"],
        decode_args_template=["-m", "improved", "-k", "4", "-p", "MySecretPass"]
    )

    test_scenario(
        "3. Phase Coding (Text)",
        encode_args=["-m", "phase", "-i", "inputs/song.wav", "-s", "inputs/secret.txt"],
        decode_args_template=["-m", "phase"]
    )

    print("\n" + "="*60)
    print("TEST SCENARIO: 4. Security Check (Wrong Password)")
    print("="*60)
    
    print(">>> Tao file co mat khau dung...")
    enc_cmd = [PYTHON_CMD, MAIN_PY_PATH, "encode", "-m", "improved", "-p", "TruePass", "-i", "inputs/song.wav", "-s", "inputs/secret.txt"]
    enc_out = run_command(enc_cmd)
    stego_path = extract_stego_path(enc_out)
    
    if stego_path:
        print(">>> Thu giai ma voi mat khau SAI (Expect Error)...")
        dec_cmd = [PYTHON_CMD, MAIN_PY_PATH, "decode", "-m", "improved", "-p", "WRONG_PASS", "-i", stego_path]
        run_command(dec_cmd)

    print("\n" + "="*60)
    print("TEST SCENARIO: 5. Batch Processing")
    print("="*60)
    
    # Input folder la folder inputs nam o root
    inputs_folder = os.path.join(PROJECT_ROOT, "inputs")
    secret_file = os.path.join(PROJECT_ROOT, "inputs", "secret.txt")
    
    batch_cmd = [PYTHON_CMD, MAIN_PY_PATH, "batch", "-m", "improved", "-k", "3", "-i", inputs_folder, "-s", secret_file]
    run_command(batch_cmd)

    print("\n[DONE] DA HOAN TAT TOAN BO QUA TRINH TEST.")

if __name__ == "__main__":
    main()