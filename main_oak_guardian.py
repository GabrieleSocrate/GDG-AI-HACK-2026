from pathlib import Path
import argparse
import time
import shutil
import threading
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import torchreid
import depthai as dai

try:
    from insightface.app import FaceAnalysis
    HAS_FACE_RECOGNITION = True
except ImportError:
    FaceAnalysis = None
    HAS_FACE_RECOGNITION = False

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

OWNER_DIR = Path("data/owner")
OWNER_BODY_GALLERY_PATH = OWNER_DIR / "owner_body_gallery.npy"
OWNER_FACE_GALLERY_PATH = OWNER_DIR / "owner_face_gallery.npy"


def clear_owner_data():
    if OWNER_DIR.exists() and any(OWNER_DIR.iterdir()):
        print(f"[OWNER DATA] Clearing existing data in {OWNER_DIR}")

        for item in OWNER_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    OWNER_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# ARGUMENTS
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--enrollment_seconds", type=float, default=30.0)

    parser.add_argument("--body_threshold", type=float, default=0.70)
    parser.add_argument("--fused_threshold", type=float, default=0.65)

    parser.add_argument("--guard_px", type=float, default=140.0)
    parser.add_argument("--contact_px", type=float, default=25.0)
    parser.add_argument("--lurker_seconds", type=float, default=5.0)

    parser.add_argument("--detection_interval", type=float, default=0.45)
    parser.add_argument("--face_interval", type=float, default=1.20)

    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=416)
    parser.add_argument("--height", type=int, default=416)

    parser.add_argument("--yolo_imgsz", type=int, default=320)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument("--oak_device", type=str, default=None)

    return parser.parse_args()


# ---------------------------------------------------------
# EMBEDDING UTILS
# ---------------------------------------------------------

def l2_normalize(embedding: np.ndarray) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(embedding)

    if norm < 1e-8:
        return embedding

    return embedding / norm


def mean_similarity_score(embedding, gallery):
    embedding = l2_normalize(embedding)
    scores = gallery @ embedding

    mean_score = float(np.mean(scores))
    max_score = float(np.max(scores))
    min_score = float(np.min(scores))

    return mean_score, max_score, min_score


def fuse_scores(body_score, face_score, body_threshold, fused_threshold):
    if face_score is not None:
        final_score = (body_score + face_score) / 2.0
        threshold = fused_threshold
        source = "OSNet+FaceRecognition"
    else:
        final_score = body_score
        threshold = body_threshold
        source = "OSNet only"

    identity = "OWNER" if final_score >= threshold else "UNKNOWN"

    return identity, final_score, source


def save_owner_galleries(body_embeddings, face_embeddings):
    OWNER_DIR.mkdir(parents=True, exist_ok=True)

    if len(body_embeddings) == 0:
        raise RuntimeError("No owner BODY embeddings collected during enrollment.")

    body_gallery = np.stack(body_embeddings, axis=0).astype(np.float32)
    np.save(OWNER_BODY_GALLERY_PATH, body_gallery)

    print("\n[OWNER ENROLLMENT COMPLETED]")
    print(f"Saved BODY embeddings: {len(body_gallery)}")
    print(f"Body gallery path: {OWNER_BODY_GALLERY_PATH}")

    face_gallery = None

    if len(face_embeddings) > 0:
        face_gallery = np.stack(face_embeddings, axis=0).astype(np.float32)
        np.save(OWNER_FACE_GALLERY_PATH, face_gallery)

        print(f"Saved FACE embeddings: {len(face_gallery)}")
        print(f"Face gallery path: {OWNER_FACE_GALLERY_PATH}")
    else:
        print("[WARNING] No face-recognition embeddings collected. System will use OSNet only.")

    print()

    return body_gallery, face_gallery


# ---------------------------------------------------------
# OSNET BODY RE-ID
# ---------------------------------------------------------

def load_osnet(device):
    model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=1000,
        pretrained=True,
    )

    model.eval()
    model.to(device)

    return model


def preprocess_person_crop(crop_bgr):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)

    transform = transforms.Compose(
        [
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return transform(pil_img).unsqueeze(0)


@torch.no_grad()
def get_osnet_embedding(osnet_model, crop_bgr, device):
    input_tensor = preprocess_person_crop(crop_bgr).to(device)
    embedding = osnet_model(input_tensor)
    embedding = embedding.detach().cpu().numpy().reshape(-1)
    return l2_normalize(embedding)


# ---------------------------------------------------------
# FACE RECOGNITION
# ---------------------------------------------------------

def load_face_recognition_model():
    if not HAS_FACE_RECOGNITION:
        print("[FACE RECOGNITION] insightface not installed. Using OSNet only.")
        return None

    app = FaceAnalysis(
        name="buffalo_s",
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )

    app.prepare(
        ctx_id=-1,
        det_size=(320, 320),
    )

    loaded_modules = list(app.models.keys())
    print(f"[FACE RECOGNITION] Loaded modules: {loaded_modules}")

    if "recognition" not in app.models:
        print("[FACE RECOGNITION] Recognition module not loaded. Using OSNet only.")
        return None

    print("[FACE RECOGNITION] Recognition model loaded correctly.")

    return app


def detect_faces_with_embeddings(face_model, frame):
    if face_model is None:
        return []

    faces_raw = face_model.get(frame)
    faces = []

    for face in faces_raw:
        x1, y1, x2, y2 = face.bbox.astype(int)

        h, w = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        embedding = None

        if hasattr(face, "normed_embedding") and face.normed_embedding is not None:
            embedding = np.asarray(face.normed_embedding, dtype=np.float32)
        elif hasattr(face, "embedding") and face.embedding is not None:
            embedding = l2_normalize(face.embedding)

        if embedding is None:
            continue

        faces.append(
            {
                "bbox": (x1, y1, x2, y2),
                "embedding": l2_normalize(embedding),
            }
        )

    return faces


def face_belongs_to_person(face_bbox, person_bbox, tolerance=80):
    fx1, fy1, fx2, fy2 = face_bbox
    px1, py1, px2, py2 = person_bbox

    face_cx = int((fx1 + fx2) / 2)
    face_cy = int((fy1 + fy2) / 2)

    px1 -= tolerance
    py1 -= tolerance
    px2 += tolerance
    py2 += tolerance

    return px1 <= face_cx <= px2 and py1 <= face_cy <= py2


def get_face_for_person(faces, person_bbox):
    matched_faces = []

    for face in faces:
        if face_belongs_to_person(face["bbox"], person_bbox):
            x1, y1, x2, y2 = face["bbox"]
            area = (x2 - x1) * (y2 - y1)
            matched_faces.append((area, face))

    if len(matched_faces) == 0:
        return None

    matched_faces = sorted(matched_faces, key=lambda x: x[0], reverse=True)
    return matched_faces[0][1]


# ---------------------------------------------------------
# YOLO DETECTION
# ---------------------------------------------------------

def get_yolo_detections(yolo_model, frame, imgsz=320, conf=0.35):
    results = yolo_model(frame, conf=conf, imgsz=imgsz, verbose=False)[0]

    persons = []
    laptops = []

    if results.boxes is None:
        return persons, laptops

    names = yolo_model.names

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        label = names[cls_id]

        if label not in ["person", "laptop"]:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        confidence = float(box.conf[0].item())

        h, w = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        area = (x2 - x1) * (y2 - y1)

        det = {
            "label": label,
            "bbox": (x1, y1, x2, y2),
            "confidence": confidence,
            "area": area,
        }

        if label == "person":
            persons.append(det)
        elif label == "laptop":
            laptops.append(det)

    persons = sorted(persons, key=lambda p: p["area"], reverse=True)
    laptops = sorted(laptops, key=lambda p: p["area"], reverse=True)

    return persons, laptops


# ---------------------------------------------------------
# GEOMETRY
# ---------------------------------------------------------

def bbox_distance_px(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)

    return float(np.sqrt(dx * dx + dy * dy))


def expand_bbox(bbox, margin, frame_shape):
    x1, y1, x2, y2 = bbox
    h, w = frame_shape[:2]

    x1 = max(0, int(x1 - margin))
    y1 = max(0, int(y1 - margin))
    x2 = min(w - 1, int(x2 + margin))
    y2 = min(h - 1, int(y2 + margin))

    return x1, y1, x2, y2


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def point_inside_bbox(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


# ---------------------------------------------------------
# ALARM
# ---------------------------------------------------------

def trigger_alarm(reason):
    print(f"\n🚨 ALARM: {reason}\n")

    if HAS_WINSOUND:
        winsound.Beep(1300, 500)
        winsound.Beep(1700, 500)
    else:
        print("\a")


# ---------------------------------------------------------
# DRAWING
# ---------------------------------------------------------

def draw_label(frame, bbox, text, color):
    x1, y1, x2, y2 = bbox

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.putText(
        frame,
        text,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


def draw_status(frame, text, color=(255, 255, 255)):
    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
    )


def draw_distance_line(frame, person_bbox, laptop_bbox, distance_px, identity):
    person_center = bbox_center(person_bbox)
    laptop_center = bbox_center(laptop_bbox)

    color = (0, 255, 0) if identity == "OWNER" else (0, 0, 255)

    cv2.line(frame, person_center, laptop_center, color, 2)
    cv2.circle(frame, person_center, 5, color, -1)
    cv2.circle(frame, laptop_center, 5, (255, 0, 0), -1)

    mid_x = int((person_center[0] + laptop_center[0]) / 2)
    mid_y = int((person_center[1] + laptop_center[1]) / 2)

    cv2.putText(
        frame,
        f"{distance_px:.1f}px",
        (mid_x, mid_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
    )


def draw_face_box(frame, face_bbox, text="FaceRec"):
    x1, y1, x2, y2 = face_bbox

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    cv2.putText(
        frame,
        text,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )


# ---------------------------------------------------------
# INFERENCE WORKER
# ---------------------------------------------------------

def inference_worker(args, state, lock, yolo_model, osnet_model, face_model):
    last_detection_time = 0.0
    last_face_time = 0.0
    processing_started = False

    while True:
        with lock:
            running = state["running"]
            phase = state["phase"]
            latest_frame = None if state["latest_frame"] is None else state["latest_frame"].copy()
            owner_body_crops = list(state["owner_body_crops"])
            owner_face_embeddings = list(state["owner_face_embeddings"])

        if not running:
            break

        if latest_frame is None:
            time.sleep(0.01)
            continue

        now = time.time()

        # -----------------------------------------------------
        # ENROLLING
        # -----------------------------------------------------

        if phase == "enrolling":
            if now - last_detection_time >= args.detection_interval:
                last_detection_time = now

                persons, laptops = get_yolo_detections(
                    yolo_model,
                    latest_frame,
                    imgsz=args.yolo_imgsz,
                )

                faces = []

                if now - last_face_time >= args.face_interval:
                    last_face_time = now
                    faces = detect_faces_with_embeddings(face_model, latest_frame)

                with lock:
                    state["last_persons"] = persons
                    state["last_faces"] = faces if len(faces) > 0 else state["last_faces"]

                    if len(persons) > 0:
                        owner_bbox = persons[0]["bbox"]
                        state["last_owner_bbox"] = owner_bbox

                        face_msg = "no_face"

                        if len(faces) > 0:
                            matched_face = get_face_for_person(faces, owner_bbox)
                            if matched_face is not None:
                                state["owner_face_embeddings"].append(matched_face["embedding"])
                                face_msg = f"face_emb={len(state['owner_face_embeddings'])}"

                        state["last_person_results"] = [
                            {
                                "bbox": owner_bbox,
                                "label": (
                                    f"ENROLL OWNER | "
                                    f"crops={len(state['owner_body_crops'])} | {face_msg}"
                                ),
                                "color": (0, 255, 255),
                                "identity": "OWNER",
                            }
                        ]

                    if len(laptops) > 0:
                        current_laptop = laptops[0]
                        state["last_laptop_bbox"] = current_laptop["bbox"]

                        if state["protected_laptop_bbox"] is None:
                            state["protected_laptop_bbox"] = current_laptop["bbox"]
                            print("[ASSET MAPPING] Laptop mapped.")

            time.sleep(0.01)
            continue

        # -----------------------------------------------------
        # PROCESSING
        # -----------------------------------------------------

        if phase == "processing":
            if processing_started:
                time.sleep(0.05)
                continue

            processing_started = True

            print("\n[PROCESSING] Computing owner body embeddings from collected crops...")
            print(f"[PROCESSING] Number of owner crops: {len(owner_body_crops)}")

            body_embeddings = []

            for idx, crop in enumerate(owner_body_crops):
                if crop is None or crop.size == 0:
                    continue

                try:
                    emb = get_osnet_embedding(osnet_model, crop, args.device)
                    body_embeddings.append(emb)
                except Exception as e:
                    print(f"[PROCESSING WARNING] Skipped crop {idx}: {e}")

            body_gallery, face_gallery = save_owner_galleries(
                body_embeddings,
                owner_face_embeddings,
            )

            with lock:
                state["owner_body_gallery"] = body_gallery
                state["owner_face_gallery"] = face_gallery
                state["owner_body_crops"] = []
                state["phase"] = "monitoring"
                state["last_person_results"] = []
                state["status_message"] = "MONITORING"

            print("[MONITORING STARTED]\n")
            continue

        # -----------------------------------------------------
        # MONITORING
        # -----------------------------------------------------

        if phase == "monitoring":
            if now - last_detection_time < args.detection_interval:
                time.sleep(0.01)
                continue

            last_detection_time = now

            with lock:
                owner_body_gallery = state["owner_body_gallery"]
                owner_face_gallery = state["owner_face_gallery"]
                last_laptop_bbox = state["last_laptop_bbox"]
                last_faces = list(state["last_faces"])

            if owner_body_gallery is None:
                time.sleep(0.05)
                continue

            persons, laptops = get_yolo_detections(
                yolo_model,
                latest_frame,
                imgsz=args.yolo_imgsz,
            )

            if now - last_face_time >= args.face_interval:
                last_face_time = now
                faces = detect_faces_with_embeddings(face_model, latest_frame)
            else:
                faces = last_faces

            if len(laptops) > 0:
                last_laptop_bbox = laptops[0]["bbox"]

            person_results = []

            for person in persons:
                px1, py1, px2, py2 = person["bbox"]
                person_crop = latest_frame[py1:py2, px1:px2]

                if person_crop.size == 0:
                    continue

                body_embedding = get_osnet_embedding(osnet_model, person_crop, args.device)

                body_score, body_max, body_min = mean_similarity_score(
                    body_embedding,
                    owner_body_gallery,
                )

                face_score = None

                if owner_face_gallery is not None:
                    matched_face = get_face_for_person(
                        faces=faces,
                        person_bbox=person["bbox"],
                    )

                    if matched_face is not None:
                        face_score, face_max, face_min = mean_similarity_score(
                            matched_face["embedding"],
                            owner_face_gallery,
                        )

                identity, final_score, source = fuse_scores(
                    body_score=body_score,
                    face_score=face_score,
                    body_threshold=args.body_threshold,
                    fused_threshold=args.fused_threshold,
                )

                color = (0, 255, 0) if identity == "OWNER" else (0, 0, 255)

                if face_score is not None:
                    label = (
                        f"{identity} | final={final_score:.3f} | "
                        f"body={body_score:.3f} face={face_score:.3f}"
                    )
                else:
                    label = (
                        f"{identity} | final={final_score:.3f} | "
                        f"body={body_score:.3f}"
                    )

                distance_px = None
                near_laptop = False
                touching_laptop = False

                if last_laptop_bbox is not None:
                    distance_px = bbox_distance_px(person["bbox"], last_laptop_bbox)

                    person_center = bbox_center(person["bbox"])
                    guard_bbox = expand_bbox(
                        last_laptop_bbox,
                        args.guard_px,
                        latest_frame.shape,
                    )

                    near_laptop = (
                        distance_px <= args.guard_px
                        or point_inside_bbox(person_center, guard_bbox)
                    )

                    touching_laptop = distance_px <= args.contact_px

                person_results.append(
                    {
                        "bbox": person["bbox"],
                        "label": label,
                        "color": color,
                        "identity": identity,
                        "distance_px": distance_px,
                        "near_laptop": near_laptop,
                        "touching_laptop": touching_laptop,
                        "source": source,
                        "final_score": final_score,
                    }
                )

            # -------------------------------------------------
            # OWNER PRESENCE OVERRIDE
            # -------------------------------------------------
            # If the owner is visible, the system is disarmed.
            # No alarm is triggered regardless of unknown people.

            owner_present = any(
                r["identity"] == "OWNER"
                for r in person_results
            )

            if owner_present:
                any_unknown_near_laptop = False
                any_unknown_touching_laptop = False
            else:
                any_unknown_near_laptop = any(
                    r["identity"] == "UNKNOWN" and r["near_laptop"]
                    for r in person_results
                )

                any_unknown_touching_laptop = any(
                    r["identity"] == "UNKNOWN" and r["touching_laptop"]
                    for r in person_results
                )

            with lock:
                state["last_persons"] = persons
                state["last_faces"] = faces
                state["last_person_results"] = person_results
                state["last_laptop_bbox"] = last_laptop_bbox
                state["owner_present"] = owner_present

                if owner_present:
                    state["unknown_near_start_time"] = None
                    state["unknown_near_duration"] = 0.0
                    unknown_near_duration = 0.0

                elif any_unknown_near_laptop:
                    if state["unknown_near_start_time"] is None:
                        state["unknown_near_start_time"] = now

                    unknown_near_duration = now - state["unknown_near_start_time"]
                    state["unknown_near_duration"] = unknown_near_duration

                else:
                    state["unknown_near_start_time"] = None
                    state["unknown_near_duration"] = 0.0
                    unknown_near_duration = 0.0

                if not owner_present:
                    if any_unknown_touching_laptop:
                        if now - state["last_alarm_time"] >= state["alarm_cooldown_seconds"]:
                            state["alarm_reason"] = "UNKNOWN person is touching the laptop."
                            state["last_alarm_time"] = now

                    elif unknown_near_duration >= args.lurker_seconds:
                        if now - state["last_alarm_time"] >= state["alarm_cooldown_seconds"]:
                            state["alarm_reason"] = (
                                f"UNKNOWN person stayed near the laptop for "
                                f"{unknown_near_duration:.1f} seconds."
                            )
                            state["last_alarm_time"] = now

            time.sleep(0.01)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    args = parse_args()

    print("### RUNNING VERSION: THREADED VIDEO + OWNER PRESENCE OVERRIDE ###")
    print(f"[DEBUG] enrollment_seconds={args.enrollment_seconds}")
    print(f"[DEBUG] detection_interval={args.detection_interval}")
    print(f"[DEBUG] face_interval={args.face_interval}")
    print(f"[DEBUG] width={args.width}, height={args.height}, fps={args.fps}")
    print(f"[DEBUG] device={args.device}")

    clear_owner_data()

    yolo_model = YOLO("yolov8n.pt")
    osnet_model = load_osnet(args.device)
    face_model = load_face_recognition_model()

    if args.oak_device:
        oak_device = dai.Device(dai.DeviceInfo(args.oak_device))
    else:
        oak_device = dai.Device()

    platform = oak_device.getPlatform().name
    print(f"[OAK] Connected platform: {platform}")

    frame_type = (
        dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p
    )

    lock = threading.Lock()

    state = {
        "running": True,
        "phase": "enrolling",
        "latest_frame": None,
        "latest_frame_id": 0,

        "owner_body_crops": [],
        "owner_face_embeddings": [],
        "owner_body_gallery": None,
        "owner_face_gallery": None,

        "last_owner_bbox": None,
        "last_persons": [],
        "last_faces": [],
        "last_person_results": [],

        "last_laptop_bbox": None,
        "protected_laptop_bbox": None,

        "owner_present": False,
        "unknown_near_start_time": None,
        "unknown_near_duration": 0.0,

        "last_alarm_time": -999.0,
        "alarm_cooldown_seconds": 3.0,
        "alarm_reason": None,

        "status_message": "ENROLLING",
    }

    worker = threading.Thread(
        target=inference_worker,
        args=(args, state, lock, yolo_model, osnet_model, face_model),
        daemon=True,
    )
    worker.start()

    print("[OWNER ENROLLMENT WAITING FOR FIRST FRAME]")
    print(f"Enrollment duration: {args.enrollment_seconds} seconds")
    print("Only the owner should be visible during enrollment.\n")

    with dai.Pipeline(oak_device) as pipeline:
        print("[OAK] Creating live camera pipeline...")

        cam = pipeline.create(dai.node.Camera).build()
        cam_out = cam.requestOutput(
            size=(args.width, args.height),
            type=frame_type,
            fps=args.fps,
        )

        rgb_queue = cam_out.createOutputQueue(maxSize=8, blocking=False)

        print("[OAK] Pipeline created.")
        pipeline.start()

        enrollment_start_time = None
        frame_id = 0

        while pipeline.isRunning():
            frame_msg = rgb_queue.tryGet()

            if frame_msg is None:
                key = cv2.waitKey(1)
                if key == ord("q"):
                    print("[INFO] Exiting.")
                    break
                continue

            frame = frame_msg.getCvFrame()
            frame_id += 1

            if enrollment_start_time is None:
                enrollment_start_time = time.time()
                print("[OWNER ENROLLMENT STARTED FROM FIRST FRAME]\n")

            now = time.time()
            elapsed = now - enrollment_start_time

            # -----------------------------------------------------
            # Push latest frame to inference thread.
            # -----------------------------------------------------

            with lock:
                state["latest_frame"] = frame.copy()
                state["latest_frame_id"] = frame_id
                phase = state["phase"]
                last_owner_bbox = state["last_owner_bbox"]

            # -----------------------------------------------------
            # Enrollment: collect owner crops every frame using
            # the latest valid bbox.
            # -----------------------------------------------------

            if phase == "enrolling":
                if elapsed > args.enrollment_seconds:
                    with lock:
                        state["phase"] = "processing"
                        state["status_message"] = "PROCESSING OWNER EMBEDDINGS"
                    phase = "processing"

                elif last_owner_bbox is not None:
                    x1, y1, x2, y2 = last_owner_bbox
                    crop = frame[y1:y2, x1:x2]

                    if crop.size > 0:
                        with lock:
                            state["owner_body_crops"].append(crop.copy())

            # -----------------------------------------------------
            # Read current shared state for drawing.
            # -----------------------------------------------------

            with lock:
                phase = state["phase"]
                last_faces = list(state["last_faces"])
                last_person_results = [dict(r) for r in state["last_person_results"]]
                laptop_bbox_for_logic = state["last_laptop_bbox"]
                unknown_near_duration = state["unknown_near_duration"]
                alarm_reason = state["alarm_reason"]
                body_crop_count = len(state["owner_body_crops"])
                face_emb_count = len(state["owner_face_embeddings"])
                owner_present = state["owner_present"]

                if alarm_reason is not None:
                    state["alarm_reason"] = None

            # -----------------------------------------------------
            # Alarm beep.
            # -----------------------------------------------------

            if alarm_reason is not None:
                trigger_alarm(alarm_reason)

            # -----------------------------------------------------
            # Draw faces.
            # -----------------------------------------------------

            cv2.putText(
                frame,
                f"FaceRec faces: {len(last_faces)}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            for face in last_faces:
                draw_face_box(frame, face["bbox"], "FaceRec")

            # -----------------------------------------------------
            # Draw laptop + guard area.
            # -----------------------------------------------------

            if laptop_bbox_for_logic is not None:
                guard_bbox = expand_bbox(
                    laptop_bbox_for_logic,
                    args.guard_px,
                    frame.shape,
                )

                cv2.rectangle(
                    frame,
                    (guard_bbox[0], guard_bbox[1]),
                    (guard_bbox[2], guard_bbox[3]),
                    (255, 255, 0),
                    2,
                )

                draw_label(
                    frame,
                    laptop_bbox_for_logic,
                    "PROTECTED LAPTOP",
                    (255, 0, 0),
                )

            # -----------------------------------------------------
            # Draw person results.
            # -----------------------------------------------------

            for result in last_person_results:
                draw_label(
                    frame,
                    result["bbox"],
                    result["label"],
                    result["color"],
                )

                if (
                    laptop_bbox_for_logic is not None
                    and result.get("distance_px") is not None
                ):
                    draw_distance_line(
                        frame=frame,
                        person_bbox=result["bbox"],
                        laptop_bbox=laptop_bbox_for_logic,
                        distance_px=result["distance_px"],
                        identity=result["identity"],
                    )

                    px1, py1, px2, py2 = result["bbox"]

                    cv2.putText(
                        frame,
                        f"dist_to_laptop={result['distance_px']:.1f}px",
                        (px1, min(frame.shape[0] - 20, py2 + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        result["color"],
                        2,
                    )

            # -----------------------------------------------------
            # Status overlay.
            # -----------------------------------------------------

            if phase == "enrolling":
                status = (
                    f"ENROLLING OWNER: {elapsed:.1f}s / {args.enrollment_seconds:.1f}s | "
                    f"crops={body_crop_count} face={face_emb_count}"
                )
                draw_status(frame, status, (0, 255, 255))

            elif phase == "processing":
                status = (
                    f"PROCESSING OWNER EMBEDDINGS... "
                    f"crops={body_crop_count} face={face_emb_count}"
                )
                draw_status(frame, status, (0, 255, 255))

            elif phase == "monitoring":
                if owner_present:
                    draw_status(frame, "MONITORING - OWNER PRESENT: DISARMED", (0, 255, 0))
                else:
                    draw_status(frame, "MONITORING - OWNER ABSENT: ARMED", (255, 255, 255))

                if unknown_near_duration > 0 and not owner_present:
                    cv2.putText(
                        frame,
                        f"UNKNOWN NEAR LAPTOP: {unknown_near_duration:.1f}s",
                        (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 165, 255),
                        2,
                    )

                    if unknown_near_duration >= args.lurker_seconds:
                        cv2.putText(
                            frame,
                            "ALARM: UNKNOWN LURKER NEAR LAPTOP",
                            (20, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            3,
                        )

            cv2.imshow("Desk Guardian - OAK Live Demo", frame)

            key = cv2.waitKey(1)

            if key == ord("q"):
                print("[INFO] Exiting.")
                break

    with lock:
        state["running"] = False

    worker.join(timeout=2.0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
