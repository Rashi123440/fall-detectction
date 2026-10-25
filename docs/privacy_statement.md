# Privacy Policy & Ethical Considerations

## Core Privacy Principles

This project adheres to **Privacy by Design** principles for continuous human monitoring and eldercare assistance.

### 1. In-Memory Video Frame Processing
* Raw video streams from webcams or video feeds are captured strictly in volatile RAM.
* Frames are processed instantaneously through the landmark detector and immediately released from memory.
* **No raw RGB video frames or pictures are saved to disk, recorded, or cached** during standard detection operations.

### 2. Edge-Only Local Computing (No Cloud Transmission)
* All deep learning inference and landmark estimation execute locally on the host device.
* No video data, audio data, or keypoint streams are transmitted to third-party APIs, remote servers, or external cloud providers.

### 3. Minimal Data Representation (Skeletons Only)
* Rather than storing high-resolution facial or environmental imagery, the pipeline reduces the human body into a coordinate representation (MediaPipe 33 keypoints: $x, y, z$, visibility).
* Background environments, personal belongings, and sensitive visual surroundings are discarded.

### 4. Honest Nuance: Skeleton Data Sensitivity
* While skeleton data eliminates raw visual identification (faces, skin tone, clothing, home surroundings), **gait analysis and spatial movement patterns can theoretically still be used for soft biometric identification or behavioral inference**.
* Keypoint datasets must still be treated with appropriate data stewardship, secure access controls, and retention policies.

### 5. Explicit User Consent & Control
* Live webcam streams require deliberate user initiation (`Start Webcam`).
* A persistent, unambiguous `Stop Camera` button is always present.
* Audio alerts are sounded locally to alert the immediate environment without external surveillance broadcasts.
