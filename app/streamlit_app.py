"""Streamlit Interactive Dashboard for Privacy-Preserving Fall Detection.
Features real-time webcam inference, privacy toggles, dataset inspection,
model training controls, and comprehensive evaluation metrics.
"""

import time
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import cv2

# Streamlit may put only the entrypoint directory on sys.path.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import project modules
from src.config import (
    FALL_PROBABILITY_THRESHOLD,
    DEFAULT_MODEL_PATH,
    SCALER_PATH,
    METADATA_PATH,
    MODELS_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SYNTHETIC_DATA_DIR,
    CLASS_NAMES,
)
from src.dataset_loader import DatasetInspector
from src.synthetic_data import SyntheticKinematicsGenerator, create_and_save_synthetic_dataset
from src.preprocessor import KeypointPreprocessor
from src.lstm_model import train_fall_detector, load_model_for_inference
from src.evaluate import evaluate_model
from src.real_time_detector import RealTimeFallDetector
from src.audio_alert import alert_manager

# Page configuration
st.set_page_config(
    page_title="Privacy-Preserving Fall Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished UI styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #334155;
        margin-bottom: 12px;
    }
    .privacy-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .badge-active {
        background-color: #064E3B;
        color: #34D399;
        border: 1px solid #059669;
    }
    .badge-warn {
        background-color: #451A03;
        color: #FBBF24;
        border: 1px solid #D97706;
    }
    .alert-banner {
        padding: 16px;
        border-radius: 8px;
        background-color: #7F1D1D;
        color: #FEE2E2;
        border: 2px solid #EF4444;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 16px;
        animation: pulse 1s infinite alternate;
    }
    .safe-banner {
        padding: 12px;
        border-radius: 8px;
        background-color: #064E3B;
        color: #D1FAE5;
        border: 1px solid #10B981;
        font-weight: 600;
        text-align: center;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initializes persistent application session variables."""
    if "webcam_active" not in st.session_state:
        st.session_state.webcam_active = False
    if "detector" not in st.session_state:
        st.session_state.detector = None
    if "history_log" not in st.session_state:
        st.session_state.history_log = []


init_session_state()

# Sidebar Navigation & Settings
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("Fall Detection AI")
    st.caption("Privacy-First Motion Keypoints")

    st.markdown("---")
    st.markdown("### 🔒 Privacy Controls")
    privacy_mode = st.toggle(
        "Skeleton Only (Black Canvas)",
        value=True,
        help="Discards all RGB video textures in RAM; renders only stick figure keypoints.",
    )
    enable_audio = st.toggle(
        "Enable Local Audio Beep",
        value=True,
        help="Sounds local alert on the host machine when fall is detected.",
    )
    conf_threshold = st.slider(
        "Fall Threshold",
        min_value=0.10,
        max_value=0.90,
        value=FALL_PROBABILITY_THRESHOLD,
        step=0.05,
        help="Model probability decision boundary.",
    )
    confirm_frames = st.slider(
        "Alert Confirmation Rule",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Consecutive positive windows required before confirming an alert (prevents noise spikes).",
    )

    st.markdown("---")
    st.markdown("### 📋 System Health")
    inspector = DatasetInspector()
    report = inspector.inspect_all()
    st.write(f"**Data Status:** `{report['status']}`")
    model_exists = DEFAULT_MODEL_PATH.exists()
    st.write(f"**Model Ready:** {'✅ Yes (Trained UP-Fall LSTM)' if model_exists else '❌ No (Train First)'}")


# Top Header
st.title("🛡️ Privacy-Preserving Fall Detection System")
st.markdown("""
Continuous fall monitoring designed for privacy. Body keypoints are extracted strictly **in-memory** using MediaPipe Pose.
Raw RGB frames are **never stored on disk or transmitted to any external cloud services**.
""")

# Privacy Badges Header
st.markdown("""
<div>
    <span class="privacy-badge badge-active">✔ In-Memory RAM Processing</span>
    <span class="privacy-badge badge-active">✔ Zero Raw Video Stored</span>
    <span class="privacy-badge badge-active">✔ 100% Local Device Inference</span>
    <span class="privacy-badge badge-warn">⚠ Skeleton Biometrics Notice</span>
</div>
""", unsafe_allow_html=True)

# Main Application Tabs
tab_live, tab_dataset, tab_train, tab_eval, tab_privacy = st.tabs([
    "📹 Live Detection",
    "📊 Dataset Inspector",
    "🧠 Model Training",
    "📈 Evaluation Metrics",
    "🔐 Privacy Policy",
])

# ==========================================
# TAB 1: LIVE DETECTION
# ==========================================
with tab_live:
    col_cam, col_telemetry = st.columns([3, 2])

    with col_cam:
        st.subheader("Webcam Monitor")

        # Camera Permission & Controls
        c1, c2, c3 = st.columns([1, 1, 2])
        start_btn = c1.button("▶ Start Camera", use_container_width=True, type="primary")
        stop_btn = c2.button("⏹ Stop Camera", use_container_width=True)

        if start_btn:
            st.session_state.webcam_active = True
            if st.session_state.detector is None:
                st.session_state.detector = RealTimeFallDetector(
                    threshold=conf_threshold,
                    alert_confirmation_frames=confirm_frames,
                )

        if stop_btn:
            st.session_state.webcam_active = False
            if st.session_state.detector:
                st.session_state.detector.close()
                st.session_state.detector = None
            st.rerun()

        frame_slot = st.empty()
        banner_slot = st.empty()

        if not st.session_state.webcam_active:
            frame_slot.info("Camera is currently stopped. Click '**Start Camera**' above to begin in-memory keypoint detection.")
            banner_slot.empty()
        else:
            # Active camera streaming loop
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                frame_slot.error("Could not access webcam. Please verify camera hardware connection and permissions.")
                st.session_state.webcam_active = False
            else:
                try:
                    # Update dynamic parameters from sliders
                    if st.session_state.detector:
                        st.session_state.detector.threshold = conf_threshold
                        st.session_state.detector.alert_confirmation_frames = confirm_frames

                    # Loop for single-frame streaming in Streamlit
                    while st.session_state.webcam_active:
                        ret, frame = cap.read()
                        if not ret:
                            frame_slot.warning("Camera stream interrupted or frame read failed.")
                            break

                        display_frame, telemetry = st.session_state.detector.process_frame(
                            frame,
                            skeleton_only_mode=privacy_mode,
                        )

                        # Update UI banner based on configurable confirmation rule
                        if telemetry.get("is_confirmed", False):
                            banner_slot.markdown(
                                f'<div class="alert-banner">🚨 EMERGENCY ALERT: FALL CONFIRMED ({telemetry["fall_probability"]*100:.1f}%)</div>',
                                unsafe_allow_html=True,
                            )
                        elif telemetry.get("consecutive_fall_count", 0) > 0:
                            banner_slot.markdown(
                                f'<div style="padding:12px; border-radius:8px; background-color:#78350F; color:#FEF3C7; font-weight:bold; text-align:center; margin-bottom:16px;">'
                                f'⚠️ POTENTIAL FALL DETECTED: Confirming ({telemetry["consecutive_fall_count"]}/{confirm_frames}) ({telemetry["fall_probability"]*100:.1f}%)</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            banner_slot.markdown(
                                f'<div class="safe-banner">✔ Normal Movement / No Fall ({telemetry["fall_probability"]*100:.1f}%) | Buffer: {telemetry["buffer_fill"]}/30</div>',
                                unsafe_allow_html=True,
                            )

                        # Display RGB frame strictly in Streamlit RAM
                        display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                        frame_slot.image(display_rgb, channels="RGB", use_container_width=True)

                        # Log event
                        st.session_state.history_log.append({
                            "Time": time.strftime("%H:%M:%S"),
                            "Probability": f"{telemetry['fall_probability']*100:.1f}%",
                            "Fall Detected": "🚨 YES" if telemetry["is_fall"] else "No",
                            "Pose Visible": "Yes" if telemetry["pose_detected"] else "No",
                        })
                        if len(st.session_state.history_log) > 200:
                            st.session_state.history_log.pop(0)

                        time.sleep(0.03)  # Approx 30 FPS throttle
                finally:
                    cap.release()

    with col_telemetry:
        st.subheader("In-Memory Telemetry")
        if st.session_state.detector:
            det = st.session_state.detector
            prob = det.last_probability

            st.metric("Fall Risk Probability", f"{prob * 100:.1f}%")
            st.progress(float(min(max(prob, 0.0), 1.0)))

            m1, m2 = st.columns(2)
            m1.metric("Buffer Frames", f"{len(det.frame_buffer)} / 30")
            m2.metric("Pose Found", "Yes" if det.last_pose_detected else "No")

            if det.model_error:
                st.warning(f"Model status: {det.model_error}")

        st.subheader("Recent Prediction History")
        if st.session_state.history_log:
            history_df = pd.DataFrame(st.session_state.history_log[-10:][::-1])
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No live frames processed yet.")

# ==========================================
# TAB 2: DATASET INSPECTOR
# ==========================================
with tab_dataset:
    st.subheader("Dataset Health & Inspection")
    report = inspector.inspect_all()

    col_sum1, col_sum2, col_sum3 = st.columns(3)
    col_sum1.metric("Raw Video Clips", report["benchmark_datasets"]["total_raw_videos"])
    col_sum2.metric("Processed Sets", report["processed_datasets"]["total_processed_sets"])
    synth_avail = report["synthetic_datasets"].get("available", False)
    col_sum3.metric("Synthetic Test Sets", f"{report['synthetic_datasets'].get('samples', 0)} samples" if synth_avail else "Not Generated")

    st.info(f"**Diagnostic Status:** {report['status']}\n\n**Recommendation:** {report['recommendation']}")

    st.markdown("---")
    st.subheader("Synthetic Kinematics Generator (For Pipeline Verification)")
    st.markdown("""
    If UP-Fall or Le2i video datasets have not been downloaded yet, you can generate a **synthetic kinematic test dataset**
    to verify the entire end-to-end preprocessing, training, and evaluation pipeline without inventing false benchmark accuracy.
    """)

    num_samples = st.slider("Synthetic Samples to Generate", 100, 1000, 400, step=50)
    if st.button("Generate Synthetic Verification Dataset", type="primary"):
        with st.spinner("Simulating human movement keypoint trajectories..."):
            x_p, y_p, m_p = create_and_save_synthetic_dataset(num_samples=num_samples)
            st.success(f"Generated {num_samples} samples successfully! Saved to {x_p.parent}")
            st.rerun()

# ==========================================
# TAB 3: MODEL TRAINING
# ==========================================
with tab_train:
    st.subheader("LSTM Classifier Training")
    st.markdown("Train the 2-layer stacked LSTM network on temporal keypoint sequence windows.")

    synth_x = SYNTHETIC_DATA_DIR / "synthetic_X.npy"
    synth_y = SYNTHETIC_DATA_DIR / "synthetic_y.npy"

    if not synth_x.exists() and len(report["processed_datasets"]["feature_files"]) == 0:
        st.warning("No dataset found. Please generate a synthetic test dataset in Tab 2 or import UP-Fall/Le2i data.")
    else:
        # Configuration hyperparameters
        c_ep, c_bs, c_lr = st.columns(3)
        epochs = c_ep.number_input("Epochs", min_value=5, max_value=100, value=25)
        batch_size = c_bs.number_input("Batch Size", min_value=8, max_value=128, value=32)
        lr = c_lr.selectbox("Learning Rate", [0.005, 0.001, 0.0005], index=1)

        if st.button("🚀 Start Training Pipeline", type="primary"):
            with st.spinner("Training LSTM model..."):
                # Load synthetic or processed dataset
                if synth_x.exists():
                    X_data = np.load(synth_x)
                    y_data = np.load(synth_y)
                    is_synth = True
                else:
                    proc_file = PROCESSED_DATA_DIR / report["processed_datasets"]["feature_files"][0]
                    X_data = np.load(proc_file)
                    y_data = np.load(str(proc_file).replace("_X.npy", "_y.npy"))
                    is_synth = False

                preprocessor = KeypointPreprocessor()
                X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_dataset(X_data, y_data)

                # Fit and save scaler
                preprocessor.fit_scaler(X_train)
                X_train_scaled = preprocessor.transform(X_train)
                X_val_scaled = preprocessor.transform(X_val)
                X_test_scaled = preprocessor.transform(X_test)

                # Save test set for evaluation
                np.save(MODELS_DIR / "test_X.npy", X_test_scaled)
                np.save(MODELS_DIR / "test_y.npy", y_test)

                # Train
                model, meta = train_fall_detector(
                    X_train=X_train_scaled,
                    y_train=y_train,
                    X_val=X_val_scaled,
                    y_val=y_val,
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    is_synthetic_training=is_synth,
                )

                st.success("Training completed and best weights saved!")
                st.json(meta["final_metrics"])

# ==========================================
# TAB 4: EVALUATION METRICS
# ==========================================
with tab_eval:
    st.subheader("Model Evaluation & Diagnostics")

    test_x_p = MODELS_DIR / "test_X.npy"
    test_y_p = MODELS_DIR / "test_y.npy"
    eval_json_p = MODELS_DIR / "evaluation_results.json"

    if not DEFAULT_MODEL_PATH.exists() or not test_x_p.exists():
        st.info("No trained model or test data found. Please run training first in Tab 3.")
    else:
        if st.button("Run Evaluation on Test Partition"):
            with st.spinner("Computing evaluation metrics..."):
                model = load_model_for_inference()
                X_test = np.load(test_x_p)
                y_test = np.load(test_y_p)

                results = evaluate_model(model, X_test, y_test, threshold=conf_threshold, save_dir=MODELS_DIR)
                st.success("Evaluation complete!")

        if eval_json_p.exists():
            with open(eval_json_p) as f:
                res = json.load(f)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy", f"{res['accuracy']*100:.1f}%")
            m2.metric("Precision", f"{res['precision']*100:.1f}%")
            m3.metric("Recall (Sensitivity)", f"{res['recall']*100:.1f}%")
            m4.metric("Specificity", f"{res['specificity']*100:.1f}%")
            m5.metric("F1-Score", f"{res['f1_score']:.3f}")

            cm_col, roc_col = st.columns(2)
            cm_img = MODELS_DIR / "confusion_matrix.png"
            if cm_img.exists():
                cm_col.image(str(cm_img), caption="Confusion Matrix", use_container_width=True)

            roc_img = MODELS_DIR / "roc_curve.png"
            if roc_img.exists():
                roc_col.image(str(roc_img), caption="Receiver Operating Characteristic", use_container_width=True)

            st.markdown("### Detailed Classification Report")
            st.code(res["classification_report"])

# ==========================================
# TAB 5: PRIVACY POLICY & ETHICS
# ==========================================
with tab_privacy:
    st.subheader("Privacy Architecture & Ethical Nuance")
    st.markdown("""
    ### Why Keypoint-Based Fall Detection?
    Traditional video fall detection records identifiable video in private rooms (bedrooms, bathrooms, eldercare facilities).
    This creates severe privacy risks, fear of surveillance, and reluctance among seniors.

    ### How This System Protects Privacy:
    1. **Volatile In-Memory Execution:** Video frames from the camera are read into RAM, analyzed for landmarks, and discarded immediately. No frames are written to disk.
    2. **Zero External Transmission:** The deep learning model runs entirely on device. No video or telemetry is sent across the internet.
    3. **Abstract Coordinate Skeletons:** The person is abstracted into 33 numerical points $(x, y, z, v)$. Faces, room decor, personal belongings, and identities are omitted.

    ### ⚠ Honest Nuance: Skeletons are Not 100% Anonymous
    * **Gait & Movement Biometrics:** Even without skin or face pixels, unique body proportions and walking gait patterns can still theoretically allow identification (soft biometrics).
    * Skeleton coordinate streams must still be safeguarded with appropriate access controls.
    """)
