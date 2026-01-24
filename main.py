import argparse
import os
import sys
import time
import shutil
import csv
from datetime import datetime

# --- IMPORT HAM HO TRO TU FILE HELPERS.PY ---
try:
    from helpers import (
        pick_file_gui, 
        create_session_folder, 
        write_log_csv, 
        open_file_os, 
        detect_type
    )
except ImportError as e:
    print(f"[LOI] Khong tim thay file helpers.py: {e}")
    sys.exit(1)

# --- IMPORT MODULES XU LY ---
try:
    from AudioStego.lsb import code as lsb_algo
    from AudioStego.phasecoding import code as phase_algo
    from AudioStego.improved_lsb import code as improved_algo
    from AudioStego.utils import visualize 
except ImportError as e:
    print(f"[LOI HE THONG] Thieu module AudioStego: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="He thong Giau tin Audio Steganography")
    parser.add_argument('action', choices=['encode', 'decode', 'batch'], help="Hanh dong")
    parser.add_argument('-m', '--method', required=True, choices=['lsb', 'phase', 'improved'], help="Thuat toan")
    parser.add_argument('-i', '--input', help="Input file/folder") 
    parser.add_argument('-s', '--secret', help="Secret file/text")
    parser.add_argument('-k', type=int, default=2, help="So bit LSB")
    parser.add_argument('-p', '--password', type=str, default="default", help="Mat khau (Improved)")
    parser.add_argument('-v', '--visualize', action='store_true', help="Hien thi bieu do")
    parser.add_argument('--save-files', action='store_true', default=False, help="Luu file batch")
    
    args = parser.parse_args()

    # 1. GUI Chon File (Neu thieu input)
    if not args.input:
        if args.action == 'batch':
             print("[LOI] Che do Batch can duong dan thu muc (-i).")
             sys.exit(1)
        args.input = pick_file_gui(title="Chon File Audio Cover (.wav)", file_types=[("WAV Files", "*.wav")])

    if args.action in ['encode', 'batch'] and not args.secret:
        gui_secret = pick_file_gui(title="Chon File Bi Mat Can Giau")
        args.secret = gui_secret if gui_secret else "Du lieu bi mat mac dinh."

    # Chon module
    processors = {'lsb': lsb_algo, 'phase': phase_algo, 'improved': improved_algo}
    processor = processors[args.method]

    print("="*60)
    print(f"[INFO] CHE DO: {args.action.upper()} | PHUONG PHAP: {args.method.upper()}")
    
    start_time = time.time()
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_name = f"log_{args.method}_{current_time_str}.csv"

    try:
        # === A. ENCODE ===
        if args.action == 'encode':
            input_type = detect_type(args.secret)
            session_dir = create_session_folder(args.action, args.method, extra_tag=input_type)
            print(f"[INFO] Thu muc: {session_dir}")

            original_filename = os.path.basename(args.input)
            out_path = os.path.join(session_dir, original_filename) 
            
            # Goi ham encode (tuong thich tham so)
            kwargs = {}
            if args.method == 'improved': kwargs = {'k': args.k, 'password': args.password}
            
            metrics = processor.encode(args.input, args.secret, out_path, **kwargs)
            
            if metrics.get('status') == 'success':
                print(f"[THANH CONG] File Stego: {out_path}")
            else:
                print(f"[THAT BAI] {metrics.get('message')}")

            write_log_csv(session_dir, log_name, {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Action": "Encode",
                "Method": args.method,
                "Input_Type": input_type,
                "Secret": os.path.basename(args.secret) if os.path.isfile(args.secret) else "Text",
                "Output_File": out_path,
                "MSE": f"{metrics.get('mse', 0):.6f}",
                "PSNR": f"{metrics.get('psnr', 0):.2f}",
                "Status": metrics.get('status', 'error')
            })

        # === B. DECODE ===
        elif args.action == 'decode':
            target_input = args.input
            if os.path.isdir(target_input):
                files = [f for f in os.listdir(target_input) if f.lower().endswith('.wav')]
                if not files: sys.exit("[LOI] Khong tim thay file .wav trong thu muc.")
                target_input = os.path.join(target_input, files[0])
                print(f"[AUTO] Da chon file: {files[0]}")

            session_dir = create_session_folder(args.action, args.method)
            
            kwargs = {}
            if args.method == 'improved': kwargs = {'k': args.k, 'password': args.password}
            result = processor.decode(target_input, **kwargs)
            
            status = "Success" if result.get('status') == 'success' else "Failed"
            out_filename = "N/A"
            
            if status == "Success":
                file_type = result.get('type', 'unknown')
                if file_type == 'text':
                    print(f"\n[NOI DUNG]: {result.get('content_text', result.get('data').decode('utf-8', 'ignore'))}\n")
                    out_filename = "decoded_message.txt"
                    with open(os.path.join(session_dir, out_filename), 'wb') as f: f.write(result['data'])
                else:
                    ext = result.get('ext', '.bin')
                    out_filename = f"extracted_{os.path.basename(target_input)}{ext}"
                    out_path = os.path.join(session_dir, out_filename)
                    with open(out_path, 'wb') as f: f.write(result['data'])
                    print(f"[THANH CONG] Da luu: {out_path}")
                    open_file_os(out_path)
            else:
                print(f"[THAT BAI] {result.get('message')}")

            write_log_csv(session_dir, log_name, {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Action": "Decode",
                "Status": status,
                "Output_Type": result.get('type', 'error'),
                "Output_File": out_filename
            })

        # === C. BATCH ===
        elif args.action == 'batch':
             if not os.path.isdir(args.input):
                print(f"[LOI] Input phai la thu muc.")
             else:
                session_dir = create_session_folder(args.action, args.method)
                print(f"[BATCH] Thu muc: {session_dir}")
                
                batch_out = os.path.join(session_dir, "stego_files")
                os.makedirs(batch_out, exist_ok=True)

                batch_kwargs = {
                    'input_dir': args.input, 'secret_input': args.secret,
                    'output_dir': batch_out, 'password': args.password
                }
                if args.method != 'phase': batch_kwargs['k'] = args.k

                results = processor.process_batch(**batch_kwargs) if hasattr(processor, 'process_batch') else []

                if not args.save_files and os.path.exists(batch_out): shutil.rmtree(batch_out)

                print(f"[DONE] Hoan tat {len(results)} file.")
                if results:
                    csv_path = os.path.join(session_dir, log_name)
                    with open(csv_path, 'w', newline='') as f:
                        writer = csv.DictWriter(f, list(results[0].keys()))
                        writer.writeheader()
                        writer.writerows(results)
                    print(f"[LOG] Bao cao: {csv_path}")
                
                if args.visualize and results: visualize.plot_batch_results(results, args.input)

    except Exception as e:
        print(f"[LOI CHUONG TRINH] {e}")
        import traceback
        traceback.print_exc()
        
    print(f"[INFO] Tong thoi gian: {time.time() - start_time:.4f}s")
    print("="*60)

if __name__ == "__main__":
    main()