import os
import csv
import joblib
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pytz
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from .dataset import StegoDataset
from .models import build_deep_model

VN_TZ = pytz.timezone('Asia/Ho_Chi_Minh')

# --- CLASS MỚI: TÍCH HỢP VÀO LOG CHUNG ---
class ResourceMonitor(Callback):
    def __init__(self, log_file):
        super(ResourceMonitor, self).__init__()
        self.log_file = log_file
        self.process = psutil.Process(os.getpid())
        # Biến tích lũy để tính trung bình
        self.total_time = 0
        self.total_ram = 0
        self.epoch_count = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        # 1. Tính toán
        duration = time.time() - self.epoch_start_time
        ram_usage = self.process.memory_info().rss / (1024 * 1024) # MB
        
        # 2. Cập nhật vào logs (để CSVLogger tự ghi)
        logs['duration'] = duration
        logs['ram_usage'] = ram_usage
        
        # 3. Tích lũy số liệu
        self.total_time += duration
        self.total_ram += ram_usage
        self.epoch_count += 1

    def on_train_end(self, logs=None):
        # 4. Ghi dòng trung bình vào cuối file
        if self.epoch_count > 0:
            avg_time = self.total_time / self.epoch_count
            avg_ram = self.total_ram / self.epoch_count
            
            # Ghi thủ công vào file CSV
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Dòng trống để ngăn cách
                writer.writerow([]) 
                # Ghi tiêu đề phụ (optional) hoặc ghi thẳng giá trị
                writer.writerow(['AVERAGE', f"{avg_time:.4f}", f"{avg_ram:.4f}", 'N/A', 'N/A'])
                writer.writerow(['TOTAL_TIME', f"{self.total_time:.2f}", 'MAX_RAM', 'N/A', 'N/A'])
            
            print(f"\n[Log] Đã ghi số liệu trung bình vào cuối file: {self.log_file}")
# -------------------------------------------------------------

class StegoTrainer:
    def __init__(self, cover_dir, stego_dir, algo='cnn', cache_dir=None, log_dir="Steganalysis/logs"):
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        self.base_log_dir = log_dir 
        self.cache_dir = cache_dir
        self.algo = algo
        
        os.makedirs(self.base_log_dir, exist_ok=True)
        if self.cache_dir: os.makedirs(self.cache_dir, exist_ok=True)
        self.current_run_dir = None 

    def train(self, depth=3, filters=32, epochs=30, batch_size=32, use_bilstm=False, algo=None, lr=0.0001):
        if algo is not None: self.algo = algo

        print(f"\n[Trainer] Starting experiment: Algo={self.algo.upper()} | LR={lr}")
        
        timestamp = datetime.now(VN_TZ).strftime("%Y%m%d-%H%M%S")
        folder_name = f"{self.algo.upper()}_D{depth}_F{filters}_LR{lr}_{timestamp}"
        
        self.current_run_dir = os.path.join(self.base_log_dir, folder_name)
        os.makedirs(self.current_run_dir, exist_ok=True)
        print(f"[Log] Directory: {self.current_run_dir}")

        dl_algos = ['cnn', 'bilstm']
        ml_algos = ['svm', 'rf', 'lr']
        if self.algo in dl_algos: model_type = 'cnn' 
        elif self.algo in ml_algos: model_type = 'svm'
        else: raise ValueError(f"Thuật toán {self.algo} không hỗ trợ.")

        print(f"[Data] Loading data ({model_type.upper()})...")
        cache_path = None
        if self.cache_dir:
            cache_file = f"features_{model_type}.npz" 
            cache_path = os.path.join(self.cache_dir, cache_file)

        dataset = StegoDataset(self.cover_dir, self.stego_dir, model_type, cache_path)
        X_train, X_val, y_train, y_val = dataset.get_train_val_split()
        
        if len(X_train) == 0:
            print("[ERROR] No training data!")
            return None

        results = {}
        total_train_time = 0 
        max_ram_usage = 0    

        # --- DEEP LEARNING ---
        if self.algo in dl_algos:
            input_shape = X_train.shape[1:]
            model = build_deep_model(input_shape, depth, filters, use_bilstm, lr=lr)
            
            model_path = os.path.join(self.current_run_dir, "best_model.keras")
            csv_log_path = os.path.join(self.current_run_dir, "training_history.csv") # File chung
            
            # QUAN TRỌNG: ResourceMonitor phải đứng TRƯỚC CSVLogger
            callbacks = [
                ResourceMonitor(csv_log_path), # <--- Tính toán duration/ram và inject vào logs
                ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', verbose=0),
                CSVLogger(csv_log_path),       # <--- Ghi logs (bao gồm cả duration/ram vừa thêm)
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
            ]
            
            start_time = time.time()
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
            total_train_time = time.time() - start_time
            max_ram_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

            print("[Evaluate] Reloading Best Model to calculate metrics...")
            model.load_weights(model_path)
            
            y_pred_prob = model.predict(X_val)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            
            results['accuracy'] = accuracy_score(y_val, y_pred)
            results['auc'] = roc_auc_score(y_val, y_pred_prob)
            results['precision'] = precision_score(y_val, y_pred, zero_division=0)
            results['recall'] = recall_score(y_val, y_pred, zero_division=0)
            results['f1'] = f1_score(y_val, y_pred, zero_division=0)
            results['model_path'] = model_path
            
            self._plot_history(history, self.algo)
            self._plot_confusion_matrix(y_val, y_pred, self.algo)

        # --- MACHINE LEARNING (Giữ nguyên) ---
        else: 
            print(f"[ML] Training {self.algo.upper()}...")
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            
            if self.algo == 'svm':
                model = make_pipeline(StandardScaler(), SVC(probability=True, kernel='rbf'))
            elif self.algo == 'rf':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            start_time = time.time()
            model.fit(X_train_flat, y_train)
            total_train_time = time.time() - start_time
            max_ram_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            
            y_pred = model.predict(X_val_flat)
            y_prob = model.predict_proba(X_val_flat)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
            results['accuracy'] = accuracy_score(y_val, y_pred)
            results['auc'] = roc_auc_score(y_val, y_prob)
            results['precision'] = precision_score(y_val, y_pred, zero_division=0)
            results['recall'] = recall_score(y_val, y_pred, zero_division=0)
            results['f1'] = f1_score(y_val, y_pred, zero_division=0)
            
            pkl_path = os.path.join(self.current_run_dir, "model.pkl")
            joblib.dump(model, pkl_path)
            results['model_path'] = pkl_path
            
            self._plot_confusion_matrix(y_val, y_pred, self.algo)

        results['total_time'] = total_train_time
        results['ram_usage'] = max_ram_usage

        print(f"\n[Results] Acc: {results['accuracy']:.4f} | F1: {results['f1']:.4f} | Time: {total_train_time:.2f}s | RAM: {max_ram_usage:.2f}MB")
        self._save_summary_log(self.algo, depth, filters, use_bilstm, results)
        return results

    # (Các hàm _plot_history, _plot_confusion_matrix, _save_summary_log giữ nguyên như cũ)
    def _plot_history(self, history, algo):
        def plot_single_metric(train_metric, val_metric, metric_name, filename):
            epochs = range(1, len(train_metric) + 1)
            plt.figure(figsize=(10, 8))
            plt.plot(epochs, train_metric, label=f'Training {metric_name}', linewidth=3)
            plt.plot(epochs, val_metric, label=f'Validation {metric_name}', linewidth=3)
            plt.title(f'{metric_name} History - {algo.upper()}', fontsize=24, fontweight='bold')
            plt.xlabel('Epochs', fontsize=20)
            plt.ylabel(metric_name, fontsize=20)
            plt.tick_params(axis='both', labelsize=18)
            plt.legend(fontsize=18, loc='best')
            plt.grid(True, linestyle='--', alpha=0.6)
            save_path = os.path.join(self.current_run_dir, filename)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

        plot_single_metric(history.history['accuracy'], history.history['val_accuracy'], 'Accuracy', 'chart_accuracy.png')
        plot_single_metric(history.history['loss'], history.history['val_loss'], 'Loss', 'chart_loss.png')
        auc_key = next((k for k in history.history.keys() if 'auc' in k and 'val' not in k), None)
        val_auc_key = next((k for k in history.history.keys() if 'auc' in k and 'val' in k), None)
        if auc_key and val_auc_key:
            plot_single_metric(history.history[auc_key], history.history[val_auc_key], 'AUC', 'chart_auc.png')

    def _plot_confusion_matrix(self, y_true, y_pred, algo):
        cm = confusion_matrix(y_true, y_pred)
        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
        labels = [f"{v1}\n({v2})" for v1, v2 in zip(group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        plt.figure(figsize=(10, 8))
        sns.set(font_scale=1.4) 
        sns.heatmap(cm/np.sum(cm), annot=labels, fmt='', cmap='Blues', annot_kws={"size": 24, "weight": "bold"}, cbar=False)
        plt.ylabel('True', fontsize=24)
        plt.xlabel('Predicted', fontsize=24)
        plt.xticks([0.5, 1.5], ['Cover', 'Stego'], fontsize=24)
        plt.yticks([0.5, 1.5], ['Cover', 'Stego'], fontsize=24, rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.current_run_dir, "chart_cm.png"))
        plt.close()
        sns.set(font_scale=1.0)

    def _save_summary_log(self, algo, depth, filters, use_bilstm, res):
        csv_file = os.path.join(self.current_run_dir, "experiment_results.csv")
        headers = ['Timestamp', 'Algo', 'Depth', 'Filters', 'BiLSTM', 
                   'Val_Acc', 'Val_AUC', 'Precision', 'Recall', 'F1', 
                   'Total_Time(s)', 'Max_RAM(MB)', 'Model_Path'] 
        
        with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            d_val = depth if algo in ['cnn', 'bilstm'] else 'N/A'
            f_val = filters if algo in ['cnn', 'bilstm'] else 'N/A'
            lstm_val = use_bilstm if algo in ['cnn', 'bilstm'] else 'N/A'
            writer.writerow([
                datetime.now(VN_TZ).strftime("%Y-%m-%d %H:%M:%S"), 
                algo, d_val, f_val, lstm_val,
                f"{res['accuracy']:.4f}", f"{res['auc']:.4f}", f"{res['precision']:.4f}",
                f"{res['recall']:.4f}", f"{res['f1']:.4f}",
                f"{res['total_time']:.2f}", f"{res['ram_usage']:.2f}",
                res['model_path']
            ])
        print(f"[Trainer] Saved detailed metrics to: {csv_file}")