# Audio Steganography


## 1. Overview

This project implements a complete pipeline for **Audio Steganography** (hiding data) and **Steganalysis** (detecting hidden data). It features state-of-the-art algorithms like **Improved LSB ** with content-based anchoring and includes tools for large-scale batch experimentation.

![System Architecture](workflow.png)


## 2. Key Features

| Algorithm | Type | Description | Best For |
| :--- | :--- | :--- | :--- |
| **LSB-Based** | Spatial Domain | Replaces Least Significant Bits sequentially. | High capacity, educational use. |
| **Phase Coding** | Frequency Domain | Embeds data into the phase spectrum (FFT). | Robustness against manipulation. |
| **Improved LSB** | Adaptive | Uses **Pseudo-Random Shuffling (PSR)** seeded by Password + Content Hash. | **High Security** & Data protection. |

### Highlights
* **Improved LSB :** Uses Pseudo-Random scattering seeded by a password + audio content hash.
* **Adaptive Compression:** Images are automatically compressed (JPEG 4:2:0) to fit into audio carriers.
* **Batch Experimentation:** Automated scripts to embed thousands of images into audio files, calculating **PSNR, SNR, MSE** automatically.

## 3. Directory Structure

> **Note:** Large datasets in `inputs/` and generated files in `outputs/` are excluded from version control.

D:.
│   main.py                 # [CLI] Main tool for single file encode/decode
│   README.md               # Project Documentation
│   .gitignore              # Git configuration
│
├───AudioStego              # [CORE ALGORITHMS]
│   ├───improved_lsb        # PSR LSB (Secure & Robust)
│   ├───lsb                 # Standard LSB
│   ├───phasecoding         # Phase Coding
│   └───utils               # Visualization & Helpers
│
├───google_colab            # [LOGS] Experimental logs for Steganography & Steganalysis
│
├───inputs                  # [DATASETS - LOCAL ONLY]
│   ├───musdb-18            # Music Source (High Fidelity)
│   ├───random-image-coco   # COCO Image Dataset
│   ├───audio-cat-and-dogs  # Environmental Sounds
│   ├───timit               # Speech Corpus (Reference)
│   └───pascal-voc-2012     # Image Source (Reference)
│
├───outputs                 # [RESULTS - LOCAL ONLY]
│   ├───batch               # Bulk processing results
│   ├───encode              # Single encryption outputs
│   └───decode              # Extracted payloads
│
└───test                    # [EXPERIMENTS]
    ├───timit_voc.py        # Script: Large-scale embedding experiments
    └───outputs             # Experiment specific results

### 4. Data Preparation (Đã thay đổi thứ tự ưu tiên)

```markdown
## 4. Data Preparation

To replicate the large-scale experiments, download the datasets and extract them into the `inputs/` directory.

**Recommended Datasets:**

1.  **[MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html)**
    * *Type:* High-fidelity music stems.
    * *Usage:* Ideal for testing high-capacity steganography on music files.

2.  **COCO & Audio Cats/Dogs**
    * *Type:* Large-scale image and environmental audio datasets.
    * *Usage:* Used for extended generalization tests.

**Reference Datasets (For Paper Replication):**

3.  **[TIMIT Acoustic-Phonetic Continuous Speech Corpus]**
    * *Usage:* Used as the primary speech carrier in the reported experiments.

4.  **[Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)**
    * *Usage:* Used as the standard image payload for capacity testing.

## 5. Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/nthai-cit/audio-steganograph.git](https://github.com/nthai-cit/audio-steganograph.git)
    cd audio-steganography
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 6. Usage Guide

### 6.1 Single File Operations

**Encode (Hide Image with Password):**
```bash
python main.py encode -m improved -k 2 -p "MyPass" -i "inputs/song.wav" -s "inputs/img.jpg"
```
**Decode:**
```bash
python main.py decode -m improved -k 2 -p "MyPass" -i "outputs/encode/SESSION/stego.wav"
```
### 6.2. Large-Scale Experimentation

To embed random images Pascal VOC into audio carriers TIMIT and generate a consolidated CSV report:

**Standard Run (k=8 bits):**
```bash
python test/timit_voc.py -k 8
```

**Run and Save Audio Files: (Use this flag to preserve output wav files; otherwise, they are deleted to conserve disk space.)**
````bash
python test/timit_voc.py -k 2 --save-audio
````

## 7. Command Line Arguments

| Argument | Short | Description | Default | Note |
| :--- | :--- | :--- | :--- | :--- |
| `encode`/`decode`/`batch` | N/A | Operation Mode | N/A | **Required** |
| `--method` | `-m` | Algorithm (`lsb`, `phase`, `improved`) | N/A | **Required** |
| `--input` | `-i` | Input Audio or Directory (batch) | GUI | Opens GUI if missing |
| `--secret` | `-s` | Secret File (Text/Image) | GUI | Required for Encode |
| `--password` | `-p` | Security Password | None | Used for `improved` only |
| `--k_bit` | `-k` | Number of bits to replace (1-8) | 2 | Used for `improved` only |
| `--visualize` | `-v` | Plot charts | False | Used for `batch`/`encode` |

---