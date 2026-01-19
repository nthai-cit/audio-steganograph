import argparse
import os
import sys
import time
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

def generate_timestamp_filename(input_path, method, action, suffix=""):
    """
    Tao ten file voi dinh dang ngay thang ro rang hon.
    """
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    
    if action == 'encode':
    
        filename = os.path.basename(input_path)
        base_name, _ = os.path.splitext(filename)
        sub_folder = "encode"
        new_name = f"{base_name}_{method}_{timestamp}.wav"
    else: 

        sub_folder = "decode"
        new_name = f"{timestamp}_{suffix}"
    
    output_dir = os.path.join("outputs", sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, new_name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['encode', 'decode', 'batch'])
    parser.add_argument('-m', '--method', required=True, choices=['lsb', 'phase', 'improved'])
    parser.add_argument('-i', '--input', help="Input File") 
    parser.add_argument('-s', '--secret', help="Secret File")
    parser.add_argument('-o', '--output', help="Output File")
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

    print("="*50)
    print(f"[INFO] CHE DO: {args.action.upper()} | PHUONG PHAP: {args.method.upper()}")
    
    start_time = time.time()

    try:

        if args.action == 'encode':
            if not args.output:
                args.output = generate_timestamp_filename(args.input, args.method, action='encode')
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            
            if args.method == 'improved':
                processor.encode(args.input, args.secret, args.output, k=args.k, password=args.password)
            else:
                processor.encode(args.input, args.secret, args.output)
            
            print(f"[THANH CONG] File Stego luu tai: {args.output}")
            visualize.save_to_downloads(args.output)

    
        elif args.action == 'decode':
            if args.method == 'improved':
                result = processor.decode(args.input, k=args.k, password=args.password)
            else:
                result = processor.decode(args.input)
            
            if result['type'] == 'error':
                print(f"[THAT BAI] {result['message']}")
            else:
                print(f"[THANH CONG] Da trich xuat du lieu loai: {result['type'].upper()}")
                
                # Tao ten file voi Ngay-Thang ro rang
                ext = result.get('ext', '.bin')
                suffix = f"extracted{ext}"
                out_path = generate_timestamp_filename(args.input, args.method, action='decode', suffix=suffix)
                
                # Luu file
                with open(out_path, 'wb') as f:
                    f.write(result['data'])
                
                print(f"[LUU TRU] Da luu file tai: {out_path}")
                visualize.save_to_downloads(out_path)
                
                # Hien thi neu la Anh/Text
                if result['type'] in ['image', 'text']:
                    visualize.show_data_from_memory(result['data'], result['type'])

        
        elif args.action == 'batch':
             if not os.path.isdir(args.input):
                print(f"[LOI] Che do batch can Input la Thu muc.")
             else:
                print(f"[BATCH] Dang xu ly hang loat...")
                if args.method == 'improved':
                    results = processor.process_batch(args.input, args.secret, k=args.k, password=args.password)
                else:
                    results = processor.process_batch(args.input, args.secret)
                print(f"[DONE] Hoan tat {len(results)} file.")
                visualize.plot_batch_results(results)

    except Exception as e:
        print(f"[LOI CHUONG TRINH] {e}")
        import traceback
        traceback.print_exc()
        
    print(f"[INFO] Thoi gian chay: {time.time() - start_time:.4f}s")
    print("="*50)

if __name__ == "__main__":
    main()