# 🛡️ Desk Guardian: Code Explanation

This document provides a technical breakdown of the **Desk Guardian** implementation. The project is designed to run entirely on the OAK-D 4 hardware, fulfilling the core hackathon requirement of **"Edge AI Processing."**

---

## 1. System Architecture (`main.py`)

The application is built using a linear **DepthAI Pipeline** that stacks multiple neural networks and hardware nodes.

### A. Hardware & Sensory Input
*   **Camera Node:** A unified node that manages the high-resolution RGB sensor (`CAM_A`) and the stereo mono sensors (`CAM_B`, `CAM_C`).
*   **StereoDepth Node:** Calculates a disparity map from the mono sensors. It is aligned to the RGB camera to ensure that every pixel in the color frame has a corresponding depth value.

### B. The Multi-Model Pipeline
1.  **Stage 1: Spatial Object Detection (YOLOv6 Nano)**
    *   The `SpatialDetectionNetwork` runs a YOLOv6 model to identify objects (people, laptops, phones).
    *   **Spatial Fusion:** It combines the 2D bounding boxes from YOLO with the depth map from `StereoDepth` to output precise **(X, Y, Z)** coordinates for each object.
2.  **Stage 2: Hardware Cropping**
    *   The `FrameCropper` node takes the bounding boxes of detected "People" and crops them out of the high-res RGB stream using the camera's hardware ISP.
3.  **Stage 3: Re-Identification (OSNet)**
    *   The cropped "Person" images are fed into the OSNet model, which generates a mathematical **embedding** (a 512-dimensional vector) representing that specific individual's features.

### C. Data Synchronization
*   The `GatherData` node acts as a bridge. It waits for both the Spatial Detections (Stage 1) and the corresponding Re-ID Embeddings (Stage 3) to be ready, ensuring they are perfectly synchronized by timestamp before passing them to the logic node.

---

## 2. Intelligence & Logic (`guardian_node.py`)

The "Brain" of the system is encapsulated in a custom **`GuardianNode`**, which inherits from `dai.node.HostNode`.

### A. Enrollment (The mathematical fingerprint)
When the app starts, it enters the `ENROLLMENT` state. It collects the first 20 embeddings of the person in front of the camera and calculates their average. This average vector becomes the `owner_embedding`.

### B. Identity Matching (Cosine Similarity)
For every person detected, the node calculates the **Cosine Similarity** between their embedding and the owner's embedding:
*   **High Similarity (>0.6):** The person is labeled as **OWNER**.
*   **Low Similarity:** The person is labeled as **UNKNOWN**.

### C. Spatial Proximity ("The Touch Heuristic")
This is the core security feature. The node calculates the **3D Euclidean distance** between any `UNKNOWN` person and the mapped "Assets" (like your laptop).
*   **Formula:** $d = \sqrt{(X_p - X_a)^2 + (Y_p - Y_a)^2 + (Z_p - Z_a)^2}$
*   If a stranger gets closer than **40cm** to an asset while the owner is away, the system triggers the **ALARM**.

---

## 3. State Machine Logic

The system operates as a state machine to minimize false positives:

| State | Condition | Action |
| :--- | :--- | :--- |
| **ENROLLMENT** | Startup | Captures owner's features. |
| **AUTHORIZED** | Owner is present | Disarmed. Continuously updates asset coordinates. |
| **ARMED** | Owner is absent > 5s | Security mode active. Monitoring 3D zones. |
| **ALARM** | Stranger < 40cm to Asset | High Alert! Red labels and console warning. |

---

## 4. Performance Optimization (The Winning Edge)
*   **Hardware Cropping:** By using `FrameCropper` on the RVC4 ISP, we avoid sending raw frames to the ARM CPU for cropping, keeping CPU usage low.
*   **Zero-Host Latency:** The entire decision-making loop (Detection -> ReID -> Proximity Math) happens within the camera's internal pipeline.
*   **Spatial Awareness:** By using `X,Y,Z` coordinates instead of 2D pixels, the system ignores people walking behind the desk, focusing only on those physically interacting with the workspace.
