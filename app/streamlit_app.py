"""Streamlit Interactive Dashboard for Privacy-Preserving Fall Detection.
Features browser-camera inference, privacy toggles, dataset inspection,
model training controls, and comprehensive evaluation metrics.
"""

import json
import sys
import threading
from pathlib import Path
import av
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from streamlit_webrtc import webrtc_streamer

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
from src.synthetic_data import create_and_save_synthetic_dataset
from src.preprocessor import KeypointPreprocessor
from src.lstm_model import train_fall_detector, load_model_for_inference
from src.evaluate import evaluate_model
from src.real_time_detector import RealTimeFallDetector
from src.webrtc_config import build_rtc_configuration

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
    if "detector" not in st.session_state:
        st.session_state.detector = None
    if "camera_data" not in st.session_state:
        st.session_state.camera_data = {
            "telemetry": None,
            "history": [],
            "skeleton_only": True,
            "threshold": FALL_PROBABILITY_THRESHOLD,
            "confirmation_frames": 3,
        }
    if "camera_data_lock" not in st.session_state:
        st.session_state.camera_data_lock = threading.Lock()


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
        help="Only affects the video returned by the app. In hosted deployments, camera frames are still sent to the cloud server.",
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
Fall-detection research dashboard using MediaPipe Pose. In the hosted version, camera frames are sent to Streamlit Cloud
for inference; when run locally, they are sent to the local app server. The app processes frames in memory and does not
intentionally save raw frames.
""")

# Privacy Badges Header
st.markdown("""
<div>
    <span class="privacy-badge badge-warn">⚠ Hosted camera streams sent to cloud</span>
    <span class="privacy-badge badge-active">✔ In-memory app processing</span>
    <span class="privacy-badge badge-active">✔ No intentional raw-video file storage</span>
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
        camera_consent = st.checkbox(
            "I agree to stream my browser camera video to the app server for fall-detection processing.",
            value=False,
            key="cloud_camera_consent",
        )
        st.warning(
            "Camera frames leave your browser and are processed by the server running this app "
            "(the cloud server in a hosted deployment). "
            "Skeleton-only mode changes the returned display; it does not prevent sending camera frames to the server. "
            "Use Stop to end the stream. This demo is not a safety-monitoring service."
        )
        try:
            webrtc_settings = st.secrets["webrtc"]
        except (StreamlitSecretNotFoundError, KeyError):
            webrtc_settings = {}
        rtc_configuration, turn_status = build_rtc_configuration(webrtc_settings)
        if turn_status == "missing":
            st.info(
                "This app currently has STUN only. Some networks block direct WebRTC connections; "
                "a TURN relay service and its credentials must be added to this app's secrets if "
                "the connection warning persists."
            )
        elif turn_status == "incomplete":
            st.warning(
                "TURN relay settings are incomplete. Add the server URL, username, and credential "
                "together in this app's secrets."
            )
        elif turn_status == "invalid":
            st.error(
                "The TURN server URL must start with `turn:` or `turns:`. Check this app's secrets."
            )

        if camera_consent:
            if st.session_state.detector is None:
                st.session_state.detector = RealTimeFallDetector(
                    threshold=conf_threshold,
                    alert_confirmation_frames=confirm_frames,
                )

            detector = st.session_state.detector
            camera_data = st.session_state.camera_data
            camera_data_lock = st.session_state.camera_data_lock
            with camera_data_lock:
                camera_data["skeleton_only"] = privacy_mode
                camera_data["threshold"] = conf_threshold
                camera_data["confirmation_frames"] = confirm_frames

            def video_frame_callback(frame):
                image = frame.to_ndarray(format="bgr24")
                with camera_data_lock:
                    detector.threshold = camera_data["threshold"]
                    detector.alert_confirmation_frames = camera_data["confirmation_frames"]
                    skeleton_only_mode = camera_data["skeleton_only"]

                display_frame, telemetry = detector.process_frame(
                    image,
                    skeleton_only_mode=skeleton_only_mode,
                )

                with camera_data_lock:
                    camera_data["telemetry"] = telemetry
                    camera_data["history"].append({
                        "Probability": f"{telemetry['fall_probability'] * 100:.1f}%",
                        "Fall Detected": "Yes" if telemetry["is_fall"] else "No",
                        "Pose Visible": "Yes" if telemetry["pose_detected"] else "No",
                    })
                    del camera_data["history"][:-200]

                return av.VideoFrame.from_ndarray(display_frame, format="bgr24")

            webrtc_streamer(
                key="fall-detection-browser-camera",
                video_frame_callback=video_frame_callback,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 640},
                        "height": {"ideal": 480},
                        "frameRate": {"ideal": 15, "max": 20},
                    },
                    "audio": False,
                },
                rtc_configuration=rtc_configuration,
            )
            st.caption(
                "Use the camera component's Start/Stop controls. The detector's audio alert runs on the cloud host "
                "in a hosted deployment and may not be audible in your browser; fall status is shown in the processed video. "
                "Uncheck the consent box to close the detector session."
            )
        else:
            if st.session_state.detector is not None:
                st.session_state.detector.close()
                st.session_state.detector = None
                with st.session_state.camera_data_lock:
                    st.session_state.camera_data["telemetry"] = None
                    st.session_state.camera_data["history"].clear()
            st.info("Camera is off. Accept the cloud-streaming notice above before starting.")

    with col_telemetry:
        st.subheader("In-Memory Telemetry")
        if camera_consent:
            @st.fragment(run_every="1s")
            def show_camera_telemetry():
                with camera_data_lock:
                    telemetry = camera_data["telemetry"]
                    history = list(camera_data["history"])

                if telemetry is None:
                    st.caption("Waiting for browser camera frames...")
                    return

                probability = float(telemetry["fall_probability"])
                st.metric("Fall Risk Probability", f"{probability * 100:.1f}%")
                st.progress(min(max(probability, 0.0), 1.0))

                metric_col1, metric_col2 = st.columns(2)
                metric_col1.metric(
                    "Buffer Frames",
                    f"{telemetry['buffer_fill']} / {telemetry['buffer_target']}",
                )
                metric_col2.metric(
                    "Pose Found",
                    "Yes" if telemetry["pose_detected"] else "No",
                )

                if telemetry["is_confirmed"]:
                    st.error("Fall confirmed. Check the person immediately.")
                elif telemetry["consecutive_fall_count"]:
                    st.warning(
                        "Potential fall: confirming "
                        f"{telemetry['consecutive_fall_count']}/{confirm_frames}."
                    )

                st.subheader("Recent Prediction History")
                if history:
                    st.dataframe(
                        pd.DataFrame(history[-10:][::-1]),
                        use_container_width=True,
                        hide_index=True,
                    )

            show_camera_telemetry()
        else:
            st.caption("No live frames processed.")

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
