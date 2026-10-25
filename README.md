# 🛡️ Privacy-Preserving Fall Detection Using Deep Learning

A privacy-first computer vision and deep learning system that detects human falls from body keypoints (**MediaPipe Pose**) rather than storing identifiable raw video.

---

## 🌟 Key Highlights & Privacy Architecture

* **In-Memory Video Processing:** Video frames from the webcam or camera stream are analyzed strictly in volatile RAM. No raw video files or captured camera pictures are ever saved to disk.
* **100% Local Inference:** No video frames, coordinates, or audio streams are transmitted to external servers or cloud services.
* **Privacy Stick Figure (Skeleton Mode):** Human motion is abstracted to 33 landmark coordinates $(x, y, z, \text{visibility})$. Visual identifiers such as faces, clothing, and private bedroom/bathroom environments are completely excluded.
* **Soft Biometric Awareness:** Clearly documents that while skeleton models prevent visual identification, gait and kinetic movement patterns can still constitute soft biometrics.
* **Local Sound Alert:** Sounds an immediate local alarm when a fall occurs to notify family or nearby caregivers without external surveillance feeds.
* **Interactive Dashboard:** Complete Streamlit interface with live webcam monitoring, dataset inspection, training, and diagnostic evaluation.

---

## 🏗️ Project Architecture

```
Privacy-Preserving-Fall-Detection/
├── app/
│   └── streamlit_app.py        # Streamlit web dashboard & live webcam monitor
├── data/
│   ├── raw/                    # (Excluded by .gitignore) Benchmark video datasets
│   ├── processed/              # Extracted keypoint arrays (.npy)
│   └── synthetic/              # Synthetic kinematic test benchmark
├── docs/
│   ├── dataset_guide.md        # Instructions for UP-Fall and Le2i datasets
│   └── privacy_statement.md    # Ethical considerations & privacy guarantees
├── models/
│   ├── fall_lstm_model.keras   # Trained LSTM model weights
│   ├── scaler.pkl              # Fitted feature normalizer
│   └── evaluation_results.json # Metrics, confusion matrix, and training logs
├── src/
│   ├── config.py               # Constants, thresholds, and privacy flags
│   ├── audio_alert.py          # Local audio alert (asynchronous / cross-platform)
│   ├── pose_extractor.py       # MediaPipe 33-landmark extraction & kinematic features
│   ├── dataset_loader.py       # Dataset inspector & UP-Fall / Le2i parsers
│   ├── synthetic_data.py       # Biomechanical synthetic kinematic data generator
│   ├── preprocessor.py         # Sliding windows, imputation, and scaling
│   ├── lstm_model.py           # Stacked LSTM classifier architecture
│   ├── evaluate.py             # Accuracy, precision, recall, F1, confusion matrix
│   └── real_time_detector.py   # In-memory real-time webcam detector engine
├── tests/
│   ├── test_environment.py     # Environment & dependency checks
│   ├── test_pose_extractor.py  # Landmark extraction unit tests
│   ├── test_synthetic_data.py  # Synthetic generator unit tests
│   ├── test_preprocessor.py    # Sliding window & scaling unit tests
│   └── test_pipeline_e2e.py    # Full end-to-end training and evaluation test
├── requirements.txt            # Dependency specification
└── README.md                   # This documentation
```

---

## 📌 Phase 1 Status: Environment Setup & Dataset Inspection
Phase 1 establishes the complete foundation without training a model yet:
1. **Directory Structure:** Created modular layout separating raw datasets, in-memory processing code, documentation, and tests.
2. **Dependencies:** Configured [`requirements.txt`](file:///c:/Users/1811r/OneDrive/Documents/Desktop/Privacy-Preserving-Fall-Detection/requirements.txt) with verified compatible packages on Python 3.13 (`mediapipe>=0.10.14,<0.11`, `tensorflow>=2.16.0`, `opencv-python>=4.8.0`).
3. **Configuration:** Created [`src/config.py`](file:///c:/Users/1811r/OneDrive/Documents/Desktop/Privacy-Preserving-Fall-Detection/src/config.py) defining keypoint layout (33 landmarks x 4 coordinates + 3 kinematic features = 135-dim feature vector), temporal window sizes, and privacy policies.
4. **Dataset Inspection:** Built [`src/dataset_loader.py`](file:///c:/Users/1811r/OneDrive/Documents/Desktop/Privacy-Preserving-Fall-Detection/src/dataset_loader.py) with automated directory health checking, benchmark parsing (UP-Fall and Le2i), and in-memory keypoint extraction.
5. **Pre-Training Guarantee:** In strict adherence to scientific and engineering standards, no models are trained until the dataset is inspected and verified.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 to 3.13 (64-bit)
- A connected USB webcam or built-in laptop camera (for live detection)
- Windows / macOS / Linux

### 2. Installation
Clone this repository and install the dependencies:
```bash
git clone https://github.com/your-username/Privacy-Preserving-Fall-Detection.git
cd Privacy-Preserving-Fall-Detection
pip install -r requirements.txt
```

---

## 🏃 Running the System

### Option A: Launch Interactive Streamlit Dashboard
The dashboard provides live webcam monitoring, dataset inspection, training controls, and metrics:
```bash
streamlit run app/streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### Option B: Standalone Real-Time Webcam Demo
To run the high-performance OpenCV webcam monitor directly from terminal:
```bash
# Privacy Mode (Stick figure skeleton on pure black canvas)
python -m src.real_time_detector

# Or run with skeleton overlay on webcam stream
python -m src.real_time_detector --overlay
```
* **Hotkeys in video window:**
  * Press `p` to toggle between **Privacy Skeleton Mode** and **Webcam Overlay**.
  * Press `q` or `ESC` to stop the camera and safely exit.

### Option C: Dataset Inspection & Health Check
Verify your local data folders:
```bash
python -m src.dataset_loader
```

### Option D: Synthetic Pipeline Verification & Training
If you don't have the UP-Fall or Le2i dataset downloaded yet, verify the entire pipeline using the synthetic kinematics generator:
```bash
# 1. Generate synthetic test dataset
python -c "from src.synthetic_data import create_and_save_synthetic_dataset; create_and_save_synthetic_dataset(num_samples=400)"

# 2. Run the automated end-to-end test suite
pytest -v
```

---

## 🔬 Benchmark Datasets Supported

1. **UP-Fall Detection Dataset (Martinez-Villasenor et al., Sensors 2019):**
   * Multi-camera footage with 11 activities (6 fall types, 5 normal daily activities).
   * For instructions on setting up UP-Fall, see [`docs/dataset_guide.md`](docs/dataset_guide.md).
2. **Le2i Fall Detection Dataset (Charfi et al., 2013):**
   * Realistic single-camera video sequences across 4 domestic settings (Home, Coffee room, Office, Lecture room).
   * For details, see [`docs/dataset_guide.md`](docs/dataset_guide.md).

> ⚠️ **Important Academic & Scientific Integrity Notice:**
> The synthetic data generator is provided strictly for continuous integration, code verification, and architectural testing when physical datasets are not yet present on the host device. Synthetic results must never be cited or claimed as real-world clinical performance.

---

## 🧪 Automated Testing

Run the test suite to verify all modules:
```bash
pytest
```
Tests include:
- `test_environment.py`: Python runtime and deep learning framework verification.
- `test_pose_extractor.py`: Blank frame robustness and skeleton canvas generation.
- `test_synthetic_data.py`: Keypoint trajectory shapes and label distributions.
- `test_preprocessor.py`: Sliding windows, imputation of lost landmarks, and scaling.
- `test_prepare_up_fall.py`: UP-Fall schema parsing, normalization, and window generation.
- `test_pipeline_e2e.py`: End-to-end synthetic dataset generation, training, and evaluation.
- `test_camera_and_model_errors.py`: Camera failures (invalid index, empty frames), missing model loading error handling, and configurable alert confirmation rules.

---

## 📜 Privacy Statement & Ethics
Please consult [`docs/privacy_statement.md`](docs/privacy_statement.md) for detailed analysis of in-memory data lifetimes, absence of cloud dependencies, and discussion of soft biometric characteristics.
