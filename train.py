import argparse
import os
import sys
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from Steganalysis.trainer import StegoTrainer
except ImportError as e:
    print(f"[LOI IMPORT] Khong tim thay module Steganalysis: {e}")
    print("Vui long dam bao ban dang chay o thu muc goc cua du an.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Cong cu dieu phoi Huan luyen Steganalysis (Chuyen sau)")

    group_data = parser.add_argument_group('Dataset Config')
    group_data.add_argument('--cover', required=True, help="Duong dan thu muc file Goc (Clean/Cover)")
    group_data.add_argument('--stego', required=True, help="Duong dan thu muc file Stego (Embedded)")
    group_data.add_argument('--cache_dir', default="Steganalysis/cache", help="Noi luu tru dac trung da trich xuat (.npz)")
    group_data.add_argument('--log_dir', default="Steganalysis/logs", help="Noi luu Log va Model checkpoint")


    group_model = parser.add_argument_group('Model Architecture')
    group_model.add_argument('--algo', 
                             choices=['cnn', 'bilstm', 'svm', 'rf', 'lr'], 
                             default='cnn',
                             help="Thuat toan: cnn, bilstm (Deep Learning) hoac svm, rf, lr (Machine Learning)")
    group_model.add_argument('--depth', type=int, default=5, help="Do sau cua mang CNN (So lop Conv)")
    group_model.add_argument('--filters', type=int, default=64, help="So luong filter khoi tao")
    group_model.add_argument('--use_bilstm', action='store_true', help="Bat lop BiLSTM (Ket hop C-RNN)")


    group_train = parser.add_argument_group('Training Hyperparameters')
    group_train.add_argument('--epochs', type=int, default=30, help="So vong lap huan luyen")
    group_train.add_argument('--batch_size', type=int, default=32, help="Kich thuoc batch")
    group_train.add_argument('--lr', type=float, default=0.0001, help="Toc do hoc (Learning Rate)")

    return parser.parse_args()

def main():
    args = parse_args()

    print("="*60)
    print(f" BAT DAU TIEN TRINH HUAN LUYEN STEGANALYSIS")
    print(f" Thoi gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    algo_name = args.algo
    use_bilstm_flag = args.use_bilstm

    if args.algo == 'bilstm':
        algo_name = 'cnn'
        use_bilstm_flag = True
        print(f"[INFO] Phat hien algo='bilstm' -> Chuyen thanh CNN + BiLSTM flag.")


    print(f" Dữ liệu Cover: {args.cover}")
    print(f" Dữ liệu Stego: {args.stego}")
    print(f" Mô hình: {algo_name.upper()} | Depth: {args.depth} | Filters: {args.filters} | BiLSTM: {use_bilstm_flag}")
    print(f" Hyperparams: Epochs={args.epochs} | Batch={args.batch_size} | LR={args.lr}")
    print("-" * 60)

    start_time = time.time()

    try:
     
        trainer = StegoTrainer(
            cover_dir=args.cover,
            stego_dir=args.stego,
            algo=algo_name,
            cache_dir=args.cache_dir,
            log_dir=args.log_dir
        )

        results = trainer.train(
            depth=args.depth,
            filters=args.filters,
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_bilstm=use_bilstm_flag,
            lr=args.lr
        )

        if results:
            print("\n" + "="*60)
            print(f" HUAN LUYEN HOAN TAT!")
            print(f" Accuracy : {results.get('accuracy', 0):.4f}")
            print(f" AUC      : {results.get('auc', 0):.4f}")
            print(f"Model    : {results.get('model_path', 'N/A')}")
            print("="*60)
        else:
            print("\n HUAN LUYEN THAT BAI (Khong co ket qua tra ve).")

    except KeyboardInterrupt:
        print("\n\n  DUNG CHUONG TRINH BOI NGUOI DUNG.")
    except Exception as e:
        print(f"\n LOI KHONG MONG MUON: {e}")
        import traceback
        traceback.print_exc()
    finally:
        total_time = time.time() - start_time
        print(f"\n  Tong thoi gian chay: {total_time:.2f} giay")

if __name__ == "__main__":
    main()