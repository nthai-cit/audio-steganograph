# Audio Steganography

## 1. Overview
This project implements a comprehensive pipeline for **Audio Steganography**. The core focus is the **Improved LSB** algorithm—an adaptive LSB method utilizing Pseudo-Random Shuffling based on passwords and Content Hashes to achieve resistance against deep learning-based attacks.

The system supports large-scale evaluation of signal fidelity (**SNR**, **PSNR**) and statistical security (**AUC**, **Accuracy**) against various machine learning classifiers.

![System Architecture](workflow.svg)

## 2. Key Features
| Algorithm | Type | Description |
| :--- | :--- | :--- |
| **Standard LSB** | Spatial Domain | Replaces Least Significant Bits sequentially. |
| **Phase Coding** | Frequency Domain | Embeds data into the phase spectrum for increased robustness. |
| **Improved LSB** | Adaptive | Utilizes Pseudo-Random Shuffling PSR seeded by Password + Content Salt. |

### Highlights
* **Security:** Key-Salt mechanism protects against sequential extraction attacks and CNN-based statistical analysis.
* **Integrity:** Supports original data recovery (Lossless) without compression.
* **Analysis:** Automated evaluation of SNR/PSNR and steganalysis resistance.

## 3. Directory Structure
```text
audio-steganograph
├───main.py                 # [CLI] Main tool for single-file encode/decode
├───train.py                # [CLI] Script for CNN model training & evaluation
├───AudioStego              # [CORE ALGORITHMS]
│   ├───improved_lsb        # PSR LSB (Secure & Robust)
│   ├───lsb                 # Standard LSB implementation
│   ├───phasecoding         # Phase Coding (FFT)
│   ├───utils               # Processing and Visualization utilities
│   └───exp                 # Experimental logs (5 Cases Benchmark)
├───Steganalysis            # [DETECTION & CNN MODELS]
│   ├───Data                # Feature sets (CNN Spectrograms for 256/512 payloads)
│   └───logs                # Tuning logs (Layer depth, Filter, LR)
├───inputs                  # [DATASETS]
│   ├───timit               # Speech Corpus (Used in paper)
│   ├───musdb-18            # High-Fidelity Music (Used in paper)
│   └───pascal-voc-2012     # Image Source (Used as payload)
├───outputs                 # [RESULTS]
    ├───encode / decode     # Single operation results
    └───batch               # Batch processing results
```
## 4. Data Preparation
To replicate the large-scale experiments, please download the following datasets and extract them into the `inputs/` directory:

**Datasets:**
1. **[MUSDB18-HQ](https://zenodo.org/records/3338373)**: High-fidelity music data, ideal for testing high-capacity steganography.
2. **[Audio Cats/Dogs](https://www.kaggle.com/datasets/mmoreaux/audio-cats-and-dogs)**: Used for extended generalization tests.
3. **[TIMIT Corpus](https://www.kaggle.com/datasets/mfekadu/darpa-timit-acousticphonetic-continuous-speech)**: The primary speech carrier used in the reported experiments.
4. **[Pascal VOC 2012](https://www.kaggle.com/datasets/banuprasadb/pascal-voc-2012)**: Standard image source for secret payloads.

## 5. Installation
**Requirements:** Python 3.11 or higher and CUDA (recommended for CNN training).

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/nthai-cit/audio-steganograph.git](https://github.com/nthai-cit/audio-steganograph.git)
   cd audio-steganography
   ```
2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
## 6. Usage Guide: Steganography
The `main.py` script supports three primary actions: `encode`, `decode`, and `batch`.

| Argument | Short | Action | Description | Default / Options |
| :--- | :--- | :--- | :--- | :--- |
| `action` | N/A | All | Selects the operation mode | `encode`, `decode`, `batch` |
| `--method` | `-m` | All | Steganography algorithm to use | `lsb`, `phase`, `improved` |
| `--input` | `-i` | All | Input audio file or directory | `.wav` file or Folder |
| `--secret` | `-s` | Enc/Batch | Secret data file to hide | Text, Image, or Audio |
| `-k` | N/A | Enc/Batch | Number of LSB bits (LSB/Improved only) | `2` |
| `--password` | `-p` | All | Security password for encryption | `default` |
| `--visualize`| `-v` | Batch | Display performance charts (SNR/PSNR) | `Flag` (False) |
| `--save-files`| N/A | Batch | Save generated stego files to disk | `Flag` (False) |

**Usage Examples:**

* **Encoding:**
    ```bash
    python main.py encode -m improved -i "cover.wav" -s "secret.png" -p "MyPass123"
    ```
* **Decoding:**
    ```bash
    python main.py decode -m improved -i "stego.wav" -p "MyPass123"
    ```
* **Batch Processing with Visualization:**
    ```bash
    python main.py batch -m improved -i "./dataset" -s "Secret Mess" -v --save-files
    ```

## 8. Steganalysis Training Parameters (train.py)
The `train.py` script manages the coordination of the Steganalysis training pipeline, supporting both Deep Learning and classical Machine Learning models.

#### A. Dataset Configuration
| Argument | Description | Default |
| :--- | :--- | :--- |
| `--cover` | Path to the directory containing Clean/Cover audio files | **Required** |
| `--stego` | Path to the directory containing Embedded/Stego audio files | **Required** |
| `--cache_dir` | Storage location for extracted feature files (`.npz`) | `Steganalysis/cache` |
| `--log_dir` | Directory to save training logs and model checkpoints | `Steganalysis/logs` |

#### B. Model Architecture
| Argument | Description | Options / Default |
| :--- | :--- | :--- |
| `--algo` | Selection of detection algorithm | `cnn`, `bilstm`, `svm`, `rf`, `lr` |
| `--depth` | Network depth (Number of Convolutional layers) | `5` |
| `--filters` | Number of initial filters for the model | `64` |
| `--use_bilstm` | Enable BiLSTM layer to form a C-RNN architecture | `Flag` (False) |

#### C. Training Hyperparameters
| Argument | Description | Default Value |
| :--- | :--- | :--- |
| `--epochs` | Total number of training iterations | `30` (Optimal: `50`) |
| `--batch_size` | Number of samples per training batch | `32` (Optimal: `64`) |
| `--lr` | Learning rate for gradient descent | `0.0001` ($10^{-4}$) |



**Usage Examples:**

* **Training the Baseline CNN (5 layers):**
    ```bash
    python train.py --cover "data/cover" --stego "data/stego" --algo cnn --depth 5 --filters 64
    ```
* **Training a C-RNN (CNN + BiLSTM):**
    ```bash
    python train.py --cover "data/cover" --stego "data/stego" --algo bilstm --use_bilstm --lr 0.0001
    ```

