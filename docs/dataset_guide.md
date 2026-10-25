# Fall Detection Dataset Guide & Setup

This guide details how to acquire, configure, and use benchmark fall detection datasets, as well as how our synthetic dataset generator serves as an end-to-end pipeline validator.

---

## 1. Supported Benchmark Datasets

### A. UP-Fall Detection Dataset
* **Source:** Universidad Panamericana (UP), Mexico.
* **Citation:** Martinez-Villasenor et al., *UP-Fall Detection Dataset: A Multimodal Approach*, Sensors 2019.
* **Content:** 17 healthy young subjects performing 11 activities (6 fall types, 5 normal everyday activities / Activities of Daily Living - ADLs) across 3 trials each. Features include multi-camera video footage, wearable accelerometer/IMU signals, EEG, and ambient infrared sensors.
* **Recommended Setup for Vision Keypoints:**
  1. Download the MP4 video clips from the official repository: [UP-Fall Detection Dataset](https://www.cs.up.edu.mx/research/up-fall-detection-dataset/)
  2. Organize video files into `data/raw/up_fall/`:
     ```
     data/raw/up_fall/
     ├── Subject1/
     │   ├── Activity1/ (Fall)
     │   │   └── trial1_cam1.mp4
     │   └── Activity6/ (Walking - ADL)
     │       └── trial1_cam1.mp4
     ```
  3. Run the feature extraction script:
     ```bash
     python -m src.dataset_loader --dataset up_fall --input_dir data/raw/up_fall --output_dir data/processed
     ```

### B. Le2i Fall Detection Dataset
* **Source:** University of Burgundy (Le2i laboratory), France.
* **Citation:** Charfi et al., *Optimized spatio-temporal descriptors for real-time fall detection: comparison of support vector machine and AdaBoost*, J. Electron. Imaging 2013.
* **Content:** Video sequences recorded with a single RGB camera across 4 realistic domestic settings: Home, Coffee room, Office, and Lecture room. Contains realistic occlusions, shadows, and varied lighting.
* **Recommended Setup:**
  1. Download video files and frame-level annotations (.txt) from the Le2i repository.
  2. Place clips into `data/raw/le2i/`:
     ```
     data/raw/le2i/
     ├── Home/
     │   ├── Videos/
     │   │   ├── video (1).avi
     │   │   └── ...
     │   └── Annotation_files/
     │       ├── video (1).txt
     │       └── ...
     ```
  3. Extract keypoints:
     ```bash
     python -m src.dataset_loader --dataset le2i --input_dir data/raw/le2i --output_dir data/processed
     ```

---

## 2. Pipeline Testing with Synthetic Data

If raw datasets are not yet downloaded on the host machine:
* The system provides a **Biomechanical Synthetic Kinematics Generator** (`src/synthetic_data.py`).
* This generator simulates realistic temporal 3D trajectory trajectories of 33 MediaPipe landmarks:
  - **ADLs (No Fall - Class 0):** Standing, walking, sitting, bending, characterized by periodic vertical oscillations and stable center of gravity.
  - **Falls (Class 1):** Rapid vertical drop in hip and torso height, accompanied by high acceleration and transition into a horizontal resting orientation.
* **Important Notice:** The synthetic dataset is strictly intended for **pipeline verification, unit testing, and architectural validation**. Synthetic benchmark results should **never** be cited or claimed as real-world clinical accuracy.
