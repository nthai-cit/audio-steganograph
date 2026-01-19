import argparse
import os
import sys
import time

try:
    from AudioStego.lsb import code as lsb_algo
    from AudioStego.phasecoding import code as phase_algo
    from AudioStego.improved_lsb import code as improved_algo

    from AudioStego.utils import visualize 
except ImportError as e:
    print("[LOI] Import: Kiem tra lai cau truc thu muc hoac file __init__.py")
    print(f"Chi tiet loi: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Tool Giau tin Audio Steganography")

    parser.add_argument('action', choices=['encode', 'decode'], help="Chon 'encode' hoac 'decode'")
    parser.add_argument('-m', '--method', required=True, choices=['lsb', 'phase', 'improved'], help="Chon phuong phap")
    parser.add_argument('-i', '--input', required=True, help="File am thanh dau vao")
    parser.add_argument('-o', '--output', help="File dau ra (Bat buoc khi encode)")
    parser.add_argument('-s', '--secret', help="File tin mat (Bat buoc khi encode)")

    parser.add_argument('-k', type=int, default=2, help="So bit LSB (Chi dung cho 'improved')")
    parser.add_argument('-p', '--password', type=str, default=None, help="Mat khau (Chi dung cho 'improved')")
    parser.add_argument('-v', '--visualize', action='store_true', help="Hien thi bieu do/anh sau khi xu ly")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[LOI] Khong tim thay file input: {args.input}")
        sys.exit(1)

    if args.method == 'lsb':
        processor = lsb_algo
    elif args.method == 'phase':
        processor = phase_algo
    elif args.method == 'improved':
        processor = improved_algo

    print("="*50)
    print(f"[INFO] CHE DO: {args.action.upper()} | PHUONG PHAP: {args.method.upper()}")
    
    start_time = time.time()

    try:
        # --- ENCODE ---
        if args.action == 'encode':
            if not args.output or not args.secret:
                print("[LOI] encode can co --output va --secret")
                sys.exit(1)
            
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            print(f"[INFO] Dang nhung tin...")
            
            # Goi ham encode
            if args.method == 'improved':
                pass_str = args.password if args.password else "default"
                print(f"[INFO] Cau hinh: K={args.k} | Password='{pass_str}'")
                result = processor.encode(args.input, args.secret, args.output, k=args.k, password=args.password)
            else:
                result = processor.encode(args.input, args.secret, args.output)

            print(f"[THANH CONG] File luu tai: {args.output}")
            
            # --- HIEN THI BIEU DO (NEU CO -v) ---
            if args.visualize:
                print("-" * 30)
                visualize.plot_audio_comparison(args.input, args.output)

        # --- DECODE ---
        elif args.action == 'decode':
            print(f"[INFO] Dang trich xuat tin...")
            
            # Goi ham decode
            if args.method == 'improved':
                pass_str = args.password if args.password else "default"
                print(f"[INFO] Cau hinh: K={args.k} | Password='{pass_str}'")
                result = processor.decode(args.input, k=args.k, password=args.password)
            else:
                result = processor.decode(args.input)

            print(f"[KET QUA] Noi dung trich xuat:\n{result}")
            
            if args.visualize:
                # Chuyen ve chu thuong de tim kiem cho de (khong lo sai chinh ta L/l)
                result_lower = result.lower()
                
                if "luu tai: " in result_lower:
                    # Cat chuoi thong minh (case-insensitive split)
                    # Tim vi tri cua chuoi "luu tai: "
                    idx = result_lower.find("luu tai: ")
                    
                    # Lay phan duong dan phia sau
                    # Do dai cua "luu tai: " la 9 ky tu
                    extracted_path = result[idx + 9:].strip()
                    
                    # Goi ham hien thi
                    visualize.show_extracted_content(extracted_path)
                else:
                    # Truong hop khong tim thay duong dan trong thong bao
                    pass

    except Exception as e:
        print(f"[LOI] Co loi xay ra: {e}")
        import traceback
        traceback.print_exc()

    print(f"[INFO] Thoi gian chay: {time.time() - start_time:.4f}s")
    print("="*50)

if __name__ == "__main__":
    main()