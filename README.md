# Desk Guardian

**Desk Guardian** is an edge-AI computer vision project developed during **GDG AI HACK 2026**, a 24-hour hackathon organized by **Google Developer Group - PoliMi**.

The project was developed for the **Computer Vision challenge**, using **Luxonis / OAK cameras**.

## Overview

Desk Guardian is designed to help protect personal objects, such as laptops, in public spaces like libraries, study rooms and coworking areas.

The system recognizes the owner of the laptop and triggers an alarm if an unknown person interacts with the protected object while the owner is not present.

## Key Features

- Real-time video processing with Luxonis / OAK cameras
- Owner recognition through person re-identification
- Object monitoring for protected items such as laptops
- Spatial awareness using depth information
- Alarm trigger when an unknown person approaches or interacts with the protected object
- Designed for edge-AI use cases in public or semi-public spaces

## How It Works

The system follows four main steps:

1. **Owner enrollment**  
   The system captures visual embeddings of the owner and builds a reference identity representation.

2. **Person and object detection**  
   The camera pipeline detects people and protected objects in the scene.

3. **Identity matching**  
   Detected people are compared against the owner representation to distinguish between the owner and unknown users.

4. **Alarm logic**  
   If the owner is not present and an unknown person gets close to or interacts with the protected object, the system enters an alarm state.

## Technical Approach

The project combines multiple computer vision components:

- **RGB camera stream** for visual detection
- **Stereo depth** for spatial information
- **Spatial object detection** for 2D bounding boxes and 3D coordinates
- **Person re-identification** for owner recognition
- **Custom guardian logic** for state management and alarm triggering

The internal logic can be represented as a simple state machine:

```text
ENROLLMENT -> AUTHORIZED -> ARMED -> INTRUDER
```

Where:

- `ENROLLMENT`: the system learns the owner's identity
- `AUTHORIZED`: the owner is present
- `ARMED`: the owner is absent and the object is being monitored
- `INTRUDER`: an unknown person interacts with the protected object

## Tech Stack

- Python
- DepthAI
- Luxonis / OAK cameras
- Computer Vision
- Object Detection
- Person Re-Identification
- Real-Time Video Processing

## Event

- **Event:** GDG AI HACK 2026
- **Organizer:** Google Developer Group - PoliMi
- **Challenge:** Computer Vision
- **Duration:** 24 hours
- **Hardware:** Luxonis / OAK cameras

## Team

- Gabriele Socrate
- Lorenzo Galli
- Jacopo Signò

## Notes

This project was developed as a hackathon prototype. The focus was on building a working proof of concept in a limited amount of time, combining computer vision and depth-based spatial reasoning.
