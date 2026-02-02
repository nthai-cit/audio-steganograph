import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime

def calculate_and_save(base_dir, filter_prefix):
    # 1. Tự động tìm các thư mục con thỏa mãn điều kiện
    print(f"\n{'='*60}")
    print(f"ĐANG QUÉT THƯ MỤC: {base_dir}")
    print(f"TỪ KHÓA TÌM KIẾM: '{filter_prefix}*'")
    
    if not os.path.exists(base_dir):
        print(f"[LỖI] Thư mục '{base_dir}' không tồn tại.")
        return

    all_items = os.listdir(base_dir)
    target_folders = []
    
    for item in all_items:
        full_path = os.path.join(base_dir, item)
        # Chỉ lấy thư mục, bắt đầu bằng prefix, và có chứa file kết quả
        if os.path.isdir(full_path) and item.startswith(filter_prefix):
            if os.path.exists(os.path.join(full_path, "experiment_results.csv")):
                target_folders.append(item)

    if len(target_folders) == 0:
        print(f"[CẢNH BÁO] Không tìm thấy thư mục nào bắt đầu bằng '{filter_prefix}' có chứa kết quả!")
        return

    print(f"-> Tìm thấy {len(target_folders)} lần chạy tương ứng.")
    print(f"{'='*60}")

    # 2. Tổng hợp dữ liệu (Thêm Time và RAM)
    metrics_data = {
        'Val_Acc': [], 
        'Val_AUC': [], 
        'Precision': [], 
        'Recall': [], 
        'F1': [],
        'Total_Time(s)': [], 
        'Max_RAM(MB)': []
    }

    for folder in target_folders:
        file_path = os.path.join(base_dir, folder, "experiment_results.csv")
        try:
            df = pd.read_csv(file_path)
      
            if not df.empty:
                last_row = df.iloc[-1] 

     
                metrics_data['Val_Acc'].append(float(last_row.get('Val_Acc', 0)))
                metrics_data['Val_AUC'].append(float(last_row.get('Val_AUC', 0)))
                metrics_data['Precision'].append(float(last_row.get('Precision', 0)))
                metrics_data['Recall'].append(float(last_row.get('Recall', 0)))
                metrics_data['F1'].append(float(last_row.get('F1', 0)))
                
        
                metrics_data['Total_Time(s)'].append(float(last_row.get('Total_Time(s)', 0)))
                metrics_data['Max_RAM(MB)'].append(float(last_row.get('Max_RAM(MB)', 0)))
                
                print(f"[ĐỌC] {folder} -> Acc: {last_row.get('Val_Acc', 0):.4f} | Time: {last_row.get('Total_Time(s)', 0)}s")
            else:
                print(f"[BỎ QUA] File rỗng: {folder}")

        except Exception as e:
            print(f"[LỖI] Không đọc được {folder}: {e}")

    summary_results = []
    print(f"\n{'='*80}")
    print(f"{'METRIC':<20} | {'MEAN':<20} | {'STD':<20}")
    print(f"{'-'*80}")

    for metric, values in metrics_data.items():
        if len(values) > 0:

            clean_values = [v for v in values if v is not None]
            
            if clean_values:
                mean_val = np.mean(clean_values)
         
                std_val = np.std(clean_values, ddof=1) if len(clean_values) > 1 else 0.0
                
                print(f"{metric:<20} | {mean_val:.4f}{' '*14} | {std_val:.4f}")
                
       
                summary_results.append({
                    'Metric': metric,
                    'Mean': mean_val,
                    'Std': std_val,
                    'Num_Runs': len(clean_values)
                })

    print(f"{'='*80}")

 
    if summary_results:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_prefix = filter_prefix.strip("_") if filter_prefix else "All"
        output_filename = f"Summary_{clean_prefix}_{timestamp}.csv"
        output_path = os.path.join(base_dir, output_filename)
        
 
        df_summary = pd.DataFrame(summary_results)
        df_summary = df_summary[['Metric', 'Mean', 'Std', 'Num_Runs']] # Sắp xếp cột
        df_summary.to_csv(output_path, index=False)
        
        print(f"\n[HOÀN TẤT] Bảng tổng hợp đã lưu tại: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Tính Mean/Std và lưu file tổng hợp từ các lần chạy thực nghiệm.")
    parser.add_argument('--dir', type=str, required=True, help="Đường dẫn đến thư mục chứa logs")
    
    parser.add_argument('--filter', type=str, default="", help="Tiền tố tên thư mục cần tính toán (vd: CNN_D5_F32)")

    args = parser.parse_args()
    
    calculate_and_save(args.dir, args.filter)

if __name__ == "__main__":
    main()