"""Configuration and global constants for Privacy-Preserving Fall Detection.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# Ensure runtime directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# MediaPipe Pose Landmark Configuration
NUM_LANDMARKS = 33
# We extract (x, y, z, visibility) for each landmark
COORDINATES_PER_LANDMARK = 4
RAW_FEATURE_DIM = NUM_LANDMARKS * COORDINATES_PER_LANDMARK  # 132 features

# Derived biomechanical features (torso angle, hip height ratio, bounding aspect ratio)
NUM_DERIVED_FEATURES = 3
TOTAL_FEATURE_DIM = RAW_FEATURE_DIM + NUM_DERIVED_FEATURES  # 135 features

# Temporal Windowing Settings
SEQUENCE_LENGTH = 30     # 30 frames window (approx 1.0 second at 30 FPS)
STRIDE_STEP = 5          # Sliding window step size for sequence generation
DEFAULT_FPS = 30

# Label Mappings (Binary Classification)
LABEL_NO_FALL = 0        # Normal Activities of Daily Living (ADLs): walking, sitting, bending
LABEL_FALL = 1           # Fall events
CLASS_NAMES = ["No Fall (ADL)", "Fall"]

# Deep Learning Hyperparameters
LSTM_UNITS = [64, 32]
DROPOUT_RATE = 0.3
DENSE_UNITS = 32
LEARNING_RATE = 0.001
BATCH_SIZE = 32
DEFAULT_EPOCHS = 35
PATIENCE = 7             # Early stopping patience

# Detection & Alert Thresholds
FALL_PROBABILITY_THRESHOLD = 0.50
ALERT_COOLDOWN_SECONDS = 3.0  # Prevent buzzer spam
ALERT_FREQUENCY_HZ = 1200     # Frequency for local audio alert
ALERT_DURATION_MS = 600       # Duration of beep in milliseconds

# Model File Paths
DEFAULT_MODEL_PATH = MODELS_DIR / "fall_lstm_model.keras"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Strict Privacy Guarantees
STRICT_PRIVACY_MODE = {
    "save_raw_video": False,
    "transmit_network": False,
    "log_coordinates_only": True,
}
