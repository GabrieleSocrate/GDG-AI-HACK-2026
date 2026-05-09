# Hackathon Hardware Specifications: Hardware Reality


* **1x OAK-D 4 Camera:** A standalone, all-in-one spatial AI device.
* **1x PoE+ Switch:** To power the camera and provide network connectivity.
* **Required Cabling:** All necessary ethernet and power cables.

---

## 🛠 Technical Specifications
The OAK-D 4 is more than just a camera; it is a powerful embedded computer optimized for vision tasks.

| Component | Specification |
| :--- | :--- |
| **CPU** | 6-core ARM CPU (Qualcomm 8-series) |
| **Memory** | 8GB RAM |
| **Storage** | 128GB On-board storage |
| **Operating System** | Luxonis OS (based on Linux kernel 5.15) |

---

## 🧠 AI & Computer Vision Capabilities
The device is designed to handle heavy workloads directly on the edge, reducing latency and bandwidth requirements.

### Artificial Intelligence Performance
* **DSP:** 48 TOPS (INT8) / 12 TOPS (FP16)
* **GPU:** 4 TOPS (FP16)

### On-Device Computer Vision
Using the **ImageManip node**, the hardware supports:
* Warping (undistortion)
* Resizing and Cropping
* Edge Detection
* Feature Tracking
* **Custom Functions:** Ability to run your own custom CV logic on-device.

---

## 👁 Spatial Perception & Tracking
The OAK-D 4 combines high-resolution color imagery with stereo depth to understand the 3D world.

* **Stereo Depth:** Highly configurable perception featuring filtering, post-processing, and RGB-depth alignment.
* **Object Tracking:** Support for both 2D and 3D tracking via the **ObjectTracker node**.

---

## 💻 Development & Competition Strategy
### The "Pro" Path
While you can stream data to a laptop, **the judges will specifically reward teams that write CV functions to run directly on the camera via OAK Apps.** This demonstrates a true understanding of edge computing and system optimization.



