import argparse
import os
import sys
import time
import csv
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

try:
    from AudioStego.lsb import code as lsb_algo
    from AudioStego.phasecoding import code as phase_algo
    from AudioStego.improved_lsb import code as improved_algo
    from AudioStego.utils import visualize 
except ImportError as e:
    print(f"[LOI HE THONG] Thieu module: {e}")
    sys.exit(1)

def pick_file_gui(title="Chon file", file_types=None):
    try:
        root = tk.Tk()
        root.withdraw() 
        root.attributes('-topmost', True)
        root.update()
        if file_types is None: file_types = [("All Files", "*.*")]
        file_path = filedialog.askopenfilename(title=title, filetypes=file_types)
        root.destroy()
        if not file_path: sys.exit(0)
        return file_path
    except: sys.exit(1)

def create_session_folder(action, method):
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    parent_folder = action.lower() 
    session_name = f"{timestamp}_{method.upper()}"
    session_path = os.path.join("outputs", parent_folder, session_name)
    os.makedirs(session_path, exist_ok=True)
    return session_path

def write_log_csv(folder_path, log_filename, data_dict):
    csv_path = os.path.join(folder_path, log_filename)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data_dict.keys())
        if not file_exists: writer.writeheader()
        writer.writerow(data_dict)
    print(f"[LOG] Da luu nhat ky tai: {csv_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['encode', 'decode', 'batch'])
    parser.add_argument('-m', '--method', required=True, choices=['lsb', 'phase', 'improved'])
    parser.add_argument('-i', '--input', help="Input File") 
    parser.add_argument('-s', '--secret', help="Secret File")
    parser.add_argument('-k', type=int, default=2)
    parser.add_argument('-p', '--password', type=str, default=None)
    parser.add_argument('-v', '--visualize', action='store_true', help="Hien thi bieu do")
    args = parser.parse_args()

    if not args.input:
        if args.action == 'batch':
             print("[LOI] Batch can duong dan thu muc (-i).")
             sys.exit(1)
        args.input = pick_file_gui(title="Chon File Audio (.wav)", file_types=[("WAV Files", "*.wav")])

    if args.action in ['encode', 'batch'] and not args.secret:
        args.secret = pick_file_gui(title="Chon File Can Giau")

    if args.method == 'lsb': processor = lsb_algo
    elif args.method == 'phase': processor = phase_algo
    elif args.method == 'improved': processor = improved_algo

    print("="*60)
    print(f"[INFO] SESSION: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"[INFO] CHE DO: {args.action.upper()} | PHUONG PHAP: {args.method.upper()}")
    
    start_time = time.time()
    session_dir = create_session_folder(args.action, args.method)
    print(f"[INFO] Thu muc lam viec: {session_dir}")
    
    current_time_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    log_name = f"log_{current_time_str}.csv"

    try:
 
        if args.action == 'encode':
            original_filename = os.path.basename(args.input)
            out_path = os.path.join(session_dir, original_filename)
            
            if args.method == 'improved':
                metrics = processor.encode(args.input, args.secret, out_path, k=args.k, password=args.password)
            else:
                metrics = processor.encode(args.input, args.secret, out_path)
            
            print(f"[THANH CONG] File Stego: {out_path}")
            
            log_data = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Action": "Encode",
                "Method": args.method,
                "Input_File": original_filename,
                "Secret_File": os.path.basename(args.secret) if args.secret else "Text",
                "Output_File": original_filename,
                "K_Bit": args.k if args.method == 'improved' else "N/A",
                "Password": "Yes" if args.password else "No",
                "MSE": f"{metrics.get('mse', 0):.6f}",
                "PSNR": f"{metrics.get('psnr', 0):.2f}",
                "SNR": f"{metrics.get('snr', 0):.2f}",
                "Execution_Time": f"{time.time() - start_time:.4f}s"
            }
            write_log_csv(session_dir, log_name, log_data)
            # visualize.save_to_downloads(out_path)

        elif args.action == 'decode':
            if args.method == 'improved':
                result = processor.decode(args.input, k=args.k, password=args.password)
            else:
                result = processor.decode(args.input)
            
            status = "Success" if result['type'] != 'error' else "Failed"
            
            if status == "Success":
                print(f"[THANH CONG] Trich xuat loai: {result['type'].upper()}")
                input_base = os.path.basename(args.input)
                name_no_ext = os.path.splitext(input_base)[0]
                ext = result.get('ext', '.bin')
                out_filename = f"{name_no_ext}_extracted{ext}"
                out_path = os.path.join(session_dir, out_filename)
                
                with open(out_path, 'wb') as f: f.write(result['data'])
                print(f"[LUU TRU] Noi dung: {out_path}")
                
                if result['type'] in ['image', 'text']:
                    visualize.show_data_from_memory(result['data'], result['type'])
            else:
                print(f"[THAT BAI] {result['message']}")
                out_filename = "N/A"

            log_data = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Action": "Decode",
                "Method": args.method,
                "Input_Stego": os.path.basename(args.input),
                "Status": status,
                "Output_File": out_filename,
                "K_Bit": args.k if args.method == 'improved' else "N/A",
                "Password": args.password if args.password else "None",
                "Execution_Time": f"{time.time() - start_time:.4f}s"
            }
            write_log_csv(session_dir, log_name, log_data)

        elif args.action == 'batch':
             if not os.path.isdir(args.input):
                print(f"[LOI] Input phai la thu muc.")
             else:
                print(f"[BATCH] Xu ly hang loat...")
                if args.method == 'improved':
                    results = processor.process_batch(args.input, args.secret, k=args.k, password=args.password)
                else:
                    results = processor.process_batch(args.input, args.secret)
                
                print(f"[DONE] Hoan tat {len(results)} file.")
                if results:
                    csv_path = os.path.join(session_dir, log_name)
                    keys = results[0].keys()
                    with open(csv_path, 'w', newline='') as f:
                        dict_writer = csv.DictWriter(f, keys)
                        dict_writer.writeheader()
                        dict_writer.writerows(results)
                    print(f"[LOG] Bao cao Batch da luu tai: {csv_path}")
                
         
                visualize.plot_batch_results(results, args.input)

    except Exception as e:
        print(f"[LOI CHUONG TRINH] {e}")
        import traceback
        traceback.print_exc()
        
    print(f"[INFO] Tong thoi gian: {time.time() - start_time:.4f}s")
    print("="*60)

if __name__ == "__main__":
    main()