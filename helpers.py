import os
import sys
import csv
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import platform
import subprocess


def open_file_os(filepath):
    """Tu dong mo file bang trinh mac dinh cua he dieu hanh"""
    try:
        if platform.system() == 'Windows':
            os.startfile(filepath)
        elif platform.system() == 'Darwin':  # macOS
            subprocess.call(('open', filepath))
        else:  # Linux
            subprocess.call(('xdg-open', filepath))
    except Exception:
        pass

def detect_type(input_data):
    """Tra ve suffix: img, audio, text, archive, file"""
    if os.path.isfile(input_data):
        ext = os.path.splitext(input_data)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']: return "img"
        if ext in ['.wav', '.mp3', '.flac', '.m4a']: return "audio"
        if ext in ['.zip', '.rar', '.7z', '.tar']: return "archive"
        if ext in ['.txt', '.doc', '.docx', '.pdf']: return "text"
        return "file"
    return "text"


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
    except Exception:
        return None

def create_session_folder(action, method, extra_tag=""):
    """
    Tao thu muc session: YYYY-MM-DD_HH-MM-SS_METHOD_TAG
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parent_folder = action.lower() 
    
    # Them suffix neu co extra_tag
    suffix = f"_{extra_tag}" if extra_tag else ""
    session_name = f"{timestamp}_{method.upper()}{suffix}"
    
    session_path = os.path.join("outputs", parent_folder, session_name)
    os.makedirs(session_path, exist_ok=True)
    return session_path

def write_log_csv(folder_path, log_filename, data_dict):
    csv_path = os.path.join(folder_path, log_filename)
    file_exists = os.path.isfile(csv_path)
    try:
        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data_dict.keys())
            if not file_exists: writer.writeheader()
            writer.writerow(data_dict)
    except Exception as e:
        print(f"[LOI LOG] Khong the ghi file CSV: {e}")