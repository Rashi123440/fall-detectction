# Privacy Policy & Ethical Considerations

## Core Privacy Principles

This project adheres to **Privacy by Design** principles for continuous human monitoring and eldercare assistance.

### 1. Video Frame Handling
* The application code processes frames in memory and does not intentionally write raw camera frames to files.
* For the hosted dashboard, frames travel over WebRTC to the cloud service and may exist in service/network buffers outside this application's control. This project makes no claim about provider-side retention.
* Do not send sensitive video to the hosted demo. Use the local standalone detector when local-only processing is required.

### 2. Processing Depends on How the App Is Run
* The standalone detector and a locally launched dashboard process video on the user's own device.
* The hosted Streamlit dashboard uses WebRTC to send browser camera frames to the Streamlit Cloud server for inference, after the viewer checks an explicit consent box.
* The hosted app does not intentionally write raw camera frames to files. Frames are nevertheless transmitted to and processed by the cloud service; skeleton-only display mode does not prevent this transmission.
* Do not use the hosted demo for sensitive spaces or where local-only processing is required. Review the hosting provider's current privacy and retention terms before use.

### 3. Minimal Data Representation (Skeletons Only)
* Rather than storing high-resolution facial or environmental imagery, the pipeline reduces the human body into a coordinate representation (MediaPipe 33 keypoints: $x, y, z$, visibility).
* Background environments, personal belongings, and sensitive visual surroundings are discarded.

### 4. Honest Nuance: Skeleton Data Sensitivity
* While skeleton data eliminates raw visual identification (faces, skin tone, clothing, home surroundings), **gait analysis and spatial movement patterns can theoretically still be used for soft biometric identification or behavioral inference**.
* Keypoint datasets must still be treated with appropriate data stewardship, secure access controls, and retention policies.

### 5. Explicit User Consent & Control
* The hosted dashboard requires an explicit camera-streaming consent checkbox before the camera component can start.
* The WebRTC component provides Start/Stop controls; stopping the stream ends camera transmission.
* The hosted detector's audio alert runs on the cloud host and may not be audible to the viewer. The processed video and dashboard display fall status.
* The standalone local detector remains available when local-only processing and local audio are required.
