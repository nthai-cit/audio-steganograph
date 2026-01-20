# Audio Steganography

This project provides a robust toolset for performing steganography on `.wav` audio files using Python. The system supports hiding text, images, and binary files with high security and automatic signal quality assessment.

---

## 1. Key Features

1.  **Algorithms:**
    * **LSB:** Fast processing, high capacity.
    * **Phase Coding:** Embeds data into the phase of the signal, offering higher robustness.
    * **Improved LSB (Advanced):**
        * Automatic image compression to fit audio capacity.
        * Random Scattering (Shuffling) of data bits.
        * Password-protected encryption.
2.  **Session Management:**
    * Creates a unique directory for each run based on the current timestamp.
    * Automatically saves the output file and a detailed CSV Log file.
3.  **Batch Processing:**
    * Embeds data into all `.wav` files in a directory with a single command.
    * Automatically plots quality comparison charts (PSNR, SNR).
    * Sorts visualization by file size (ascending).
4.  **Quality Assessment:**
    * Automatically calculates metrics: MSE (Mean Squared Error), PSNR (Peak Signal-to-Noise Ratio), and SNR (Signal-to-Noise Ratio).

---


## 2. Directory Structure

Ensure the project directory follows this structure:

D:.
│   main.py                 # [RUN] Main entry point
│   README.md               # Documentation
│   requirements.txt        # List of dependencies
│
├───AudioStego              # [CORE] Source code
│   ├───improved_lsb        # Improved LSB Module
│   │       code.py
│   ├───lsb                 # Standard LSB Module
│   │       code.py
│   ├───phasecoding         # Phase Coding Module
│   │       code.py
│   └───utils               # Utilities (Visualization)
│           visualize.py
│
├───inputs                  # [DATA] Input files (See Section 3)
│   │   song.wav
│   ├───audio-cat-and-dogs  # Large Dataset 1 (Audio)
│   ├───musdb-18            # Large Dataset 2 (Audio)
│   └───random-image-coco   # Large Dataset 3 (Images)
│
├───outputs                 # [RESULT] Auto-generated results
│   ├───batch
│   ├───decode
│   └───encode
│
└───test                    # [TEST] Automated test scripts
        prepare_data.py
        test.py

---

## 3. Data Preparation (Important)

The `inputs/` folder contains sample files. However, large datasets required for Batch Processing are **not hosted in this repository** due to size limits.

### How to download datasets:
1.  **Download** the dataset package from the following link:
    **[INSERT_YOUR_GOOGLE_DRIVE_LINK_HERE]**
2.  **Extract** the contents.
3.  **Copy** the following folders into the `inputs/` directory of this project:
    * `audio-cat-and-dogs`
    * `musdb-18`
    * `random-image-coco`

Structure should look like this after extraction:
inputs/
├── song.wav
├── audio-cat-and-dogs/ ...
├── musdb-18/ ...
└── random-image-coco/ ...

---

##  4. Installation

### Step 1: Clone the Repository
Open your terminal or command prompt and run the following commands to download the source code:

git clone https://github.com/nthai-cit/audio-steganograph.git
cd audio-steganography

### Step 2: Install Dependencies
Ensure you have Python 3.12 or higher installed. Install the required libraries using pip:

pip install -r requirements.txt

---

## 5. Usage Guide

General Syntax:
python main.py [action] -m [method] [options]

### Encode (Hiding Data)

* **Standard LSB (Text Hiding):**
    python main.py encode -m lsb -i inputs/song.wav -s inputs/secret.txt

* **Improved LSB (Image Hiding + Password + K=4 bits):**
    python main.py encode -m improved -k 4 -p "MySecurePass" -i inputs/song.wav -s inputs/random-image-coco/image_01.jpg

    (The system will create a folder: outputs/encode/DD-MM-YYYY_.../ containing the result and log)

### Decode (Extracting Data)

* **Decoding (Must target the encoded file):**
    python main.py decode -m improved -k 4 -p "MySecurePass" -i outputs/encode/SESSION_NAME/song.wav

### Batch Processing

This mode scans all `.wav` files in the input directory, embeds the data, and visualizes quality metrics.

python main.py batch -m improved -k 2 -i "inputs/musdb-18" -s inputs/secret.txt -v

* -i: Path to the music directory (e.g., inputs/musdb-18).
* -v: Enable visualization (charts) after processing.

---

## 6. Logging & Reporting

For every operation, the system generates a `.csv` file in the corresponding Session directory.

Example Log Content:

| Timestamp | Action | Method | Input_File | Secret | Output | MSE | PSNR | SNR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-01-20 14:00 | Encode | improved | song.wav | image.jpg | song.wav | 0.038 | 104.5 | 88.5 |

---

## 7. Automated Testing

The project includes an intelligent Test Suite that automatically locates paths and verifies 5 scenarios (LSB, Improved, Phase, Wrong Password, Batch).

1.  **Step 1: Generate Sample Data** (Run once):
    python test/prepare_data.py

2.  **Step 2: Run Full System Check:**
    python test/test.py

---

## 8. Command Line Arguments

| Argument | Short | Description | Default | Note |
| :--- | :--- | :--- | :--- | :--- |
| encode/decode/batch | N/A | Operation Mode | N/A | Required |
| --method | -m | Algorithm (lsb, phase, improved) | N/A | Required |
| --input | -i | Input Audio or Directory (batch) | GUI | Opens GUI if missing |
| --secret | -s | Secret File (Text/Image) | GUI | Required for Encode |
| --password | -p | Security Password | None | Used for improved only |
| --k_bit | -k | Number of bits to replace (1-8) | 2 | Used for improved only |
| --visualize | -v | Plot charts | False | Used for batch/encode |

---