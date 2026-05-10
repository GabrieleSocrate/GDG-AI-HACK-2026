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
import torchreid
import depthai as dai

from depthai_nodes.node import ParsingNeuralNetwork, DepthMerger

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

    parser.add_argument("--body_threshold", type=float, default=0.60)
    parser.add_argument("--fused_threshold", type=float, default=0.60)

    # Used only to disarm the system when the owner is probably present.
    parser.add_argument("--owner_presence_threshold", type=float, default=0.50)
    parser.add_argument("--owner_hold_seconds", type=float, default=5.0)

    # Real 3D thresholds in meters.
    parser.add_argument("--guard_m", type=float, default=0.30)
    parser.add_argument("--contact_m", type=float, default=0.10)
    parser.add_argument("--lurker_seconds", type=float, default=5.0)

    # Host-side identity update frequency.
    parser.add_argument("--identity_interval", type=float, default=0.45)
    parser.add_argument("--face_interval", type=float, default=1.20)

    # OAK stream.
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)

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
# SPATIAL DETECTIONS FROM DEPTHMERGER
# ---------------------------------------------------------

def safe_getattr(obj, names, default=None):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def detection_bbox_to_pixels(det, frame_shape):
    """
    Converts detection bbox to pixel coordinates.
    Handles both normalized bbox values [0,1] and already-pixel values.
    """

    h, w = frame_shape[:2]

    x1 = safe_getattr(det, ["xmin", "xMin", "x_min"], None)
    y1 = safe_getattr(det, ["ymin", "yMin", "y_min"], None)
    x2 = safe_getattr(det, ["xmax", "xMax", "x_max"], None)
    y2 = safe_getattr(det, ["ymax", "yMax", "y_max"], None)

    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None

    x1 = float(x1)
    y1 = float(y1)
    x2 = float(x2)
    y2 = float(y2)

    # Normalized coordinates.
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1 *= w
        x2 *= w
        y1 *= h
        y2 *= h

    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(0, min(w - 1, x2)))
    y2 = int(max(0, min(h - 1, y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def get_spatial_xyz_m(det):
    """
    Reads x,y,z from DepthAI spatial coordinates.
    Usually they are in millimeters, so convert to meters.
    """

    spatial = safe_getattr(det, ["spatialCoordinates", "spatial_coordinates"], None)

    if spatial is None:
        return None

    x = safe_getattr(spatial, ["x"], None)
    y = safe_getattr(spatial, ["y"], None)
    z = safe_getattr(spatial, ["z"], None)

    if x is None or y is None or z is None:
        return None

    x = float(x)
    y = float(y)
    z = float(z)

    if not np.isfinite([x, y, z]).all():
        return None

    if abs(z) < 1e-6:
        return None

    # DepthAI spatial coords are normally in millimeters.
    if max(abs(x), abs(y), abs(z)) > 20.0:
        x /= 1000.0
        y /= 1000.0
        z /= 1000.0

    return np.array([x, y, z], dtype=np.float32)


def parse_spatial_detections(spatial_msg, classes, frame_shape):
    """
    Converts DepthMerger output into simple Python dicts.
    Each detection contains:
        label, bbox, confidence, spatial_m, area
    """

    parsed = []

    if spatial_msg is None:
        return parsed

    detections = safe_getattr(spatial_msg, ["detections"], [])

    if detections is None:
        return parsed

    for det in detections:
        label_id = safe_getattr(det, ["label"], None)

        if label_id is None:
            continue

        label_id = int(label_id)

        if 0 <= label_id < len(classes):
            label = classes[label_id]
        else:
            label = str(label_id)

        if label not in ["person", "laptop"]:
            continue

        bbox = detection_bbox_to_pixels(det, frame_shape)

        if bbox is None:
            continue

        confidence = safe_getattr(det, ["confidence", "conf"], 0.0)
        confidence = float(confidence)

        spatial_m = get_spatial_xyz_m(det)

        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)

        parsed.append(
            {
                "label": label,
                "bbox": bbox,
                "confidence": confidence,
                "spatial_m": spatial_m,
                "area": area,
            }
        )

    parsed = sorted(parsed, key=lambda d: d["area"], reverse=True)

    return parsed


def spatial_distance_m(det_a, det_b):
    """
    True 3D Euclidean distance between two spatial detections in meters.
    Uses x,y,z coming from OAK StereoDepth + DepthMerger.
    """

    p_a = det_a.get("spatial_m")
    p_b = det_b.get("spatial_m")

    if p_a is None or p_b is None:
        return None

    if not np.isfinite(p_a).all() or not np.isfinite(p_b).all():
        return None

    return float(np.linalg.norm(p_a - p_b))


# ---------------------------------------------------------
# DRAWING
# ---------------------------------------------------------

def trigger_alarm(reason):
    print(f"\n🚨 ALARM: {reason}\n")

    if HAS_WINSOUND:
        winsound.Beep(1300, 500)
        winsound.Beep(1700, 500)
    else:
        print("\a")


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


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


def draw_distance_line_m(frame, person_bbox, laptop_bbox, distance_m, identity):
    person_center = bbox_center(person_bbox)
    laptop_center = bbox_center(laptop_bbox)

    color = (0, 255, 0) if identity == "OWNER" else (0, 0, 255)

    cv2.line(frame, person_center, laptop_center, color, 2)
    cv2.circle(frame, person_center, 5, color, -1)
    cv2.circle(frame, laptop_center, 5, (255, 0, 0), -1)

    mid_x = int((person_center[0] + laptop_center[0]) / 2)
    mid_y = int((person_center[1] + laptop_center[1]) / 2)

    if distance_m is None:
        text = "dist=?m"
    else:
        text = f"{distance_m:.2f}m"

    cv2.putText(
        frame,
        text,
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

def inference_worker(args, state, lock, osnet_model, face_model):
    last_identity_time = 0.0
    last_face_time = 0.0
    processing_started = False

    while True:
        with lock:
            running = state["running"]
            phase = state["phase"]
            latest_frame = None if state["latest_frame"] is None else state["latest_frame"].copy()
            latest_spatial_detections = [dict(d) for d in state["latest_spatial_detections"]]
            owner_body_crops = list(state["owner_body_crops"])
            owner_face_embeddings = list(state["owner_face_embeddings"])

        if not running:
            break

        if latest_frame is None:
            time.sleep(0.01)
            continue

        now = time.time()

        persons = [d for d in latest_spatial_detections if d["label"] == "person"]
        laptops = [d for d in latest_spatial_detections if d["label"] == "laptop"]

        persons = sorted(persons, key=lambda d: d["area"], reverse=True)
        laptops = sorted(laptops, key=lambda d: d["area"], reverse=True)

        # -----------------------------------------------------
        # ENROLLING
        # -----------------------------------------------------

        if phase == "enrolling":
            if len(persons) > 0:
                owner_det = persons[0]
                owner_bbox = owner_det["bbox"]

                face_msg = "no_face"

                if now - last_face_time >= args.face_interval:
                    last_face_time = now
                    faces = detect_faces_with_embeddings(face_model, latest_frame)
                else:
                    with lock:
                        faces = list(state["last_faces"])

                matched_face = get_face_for_person(faces, owner_bbox)

                with lock:
                    state["last_faces"] = faces

                    if matched_face is not None:
                        state["owner_face_embeddings"].append(matched_face["embedding"])
                        face_msg = f"face_emb={len(state['owner_face_embeddings'])}"

                    state["last_owner_bbox"] = owner_bbox
                    state["last_person_results"] = [
                        {
                            "bbox": owner_bbox,
                            "label": (
                                f"ENROLL OWNER | "
                                f"crops={len(state['owner_body_crops'])} | {face_msg}"
                            ),
                            "color": (0, 255, 255),
                            "identity": "OWNER",
                            "distance_m": None,
                        }
                    ]

            if len(laptops) > 0:
                laptop_det = laptops[0]

                with lock:
                    state["last_laptop_bbox"] = laptop_det["bbox"]
                    state["last_laptop_spatial_m"] = laptop_det["spatial_m"]

                    if state["protected_laptop_bbox"] is None:
                        state["protected_laptop_bbox"] = laptop_det["bbox"]
                        print("[ASSET MAPPING] Laptop mapped with spatial coordinates.")

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
            if now - last_identity_time < args.identity_interval:
                time.sleep(0.01)
                continue

            last_identity_time = now

            with lock:
                owner_body_gallery = state["owner_body_gallery"]
                owner_face_gallery = state["owner_face_gallery"]
                last_faces = list(state["last_faces"])
                last_laptop_bbox = state["last_laptop_bbox"]
                last_laptop_spatial_m = state["last_laptop_spatial_m"]

            if owner_body_gallery is None:
                time.sleep(0.05)
                continue

            if len(laptops) > 0:
                laptop_det = laptops[0]
                last_laptop_bbox = laptop_det["bbox"]
                last_laptop_spatial_m = laptop_det["spatial_m"]
            elif last_laptop_bbox is not None:
                laptop_det = {
                    "bbox": last_laptop_bbox,
                    "spatial_m": last_laptop_spatial_m,
                    "label": "laptop",
                    "area": 0,
                }
            else:
                laptop_det = None

            if now - last_face_time >= args.face_interval:
                last_face_time = now
                faces = detect_faces_with_embeddings(face_model, latest_frame)
            else:
                faces = last_faces

            person_results = []

            for person_det in persons:
                px1, py1, px2, py2 = person_det["bbox"]
                person_crop = latest_frame[py1:py2, px1:px2]

                if person_crop.size == 0:
                    continue

                body_embedding = get_osnet_embedding(
                    osnet_model,
                    person_crop,
                    args.device,
                )

                body_score, _, _ = mean_similarity_score(
                    body_embedding,
                    owner_body_gallery,
                )

                face_score = None

                if owner_face_gallery is not None:
                    matched_face = get_face_for_person(
                        faces=faces,
                        person_bbox=person_det["bbox"],
                    )

                    if matched_face is not None:
                        face_score, _, _ = mean_similarity_score(
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

                distance_m = None
                near_laptop = False
                touching_laptop = False

                if laptop_det is not None:
                    distance_m = spatial_distance_m(person_det, laptop_det)

                    if distance_m is not None:
                        near_laptop = distance_m <= args.guard_m
                        touching_laptop = distance_m <= args.contact_m

                person_results.append(
                    {
                        "bbox": person_det["bbox"],
                        "label": label,
                        "color": color,
                        "identity": identity,
                        "distance_m": distance_m,
                        "near_laptop": near_laptop,
                        "touching_laptop": touching_laptop,
                        "source": source,
                        "final_score": final_score,
                    }
                )

            # -------------------------------------------------
            # HARD OWNER DISARM
            # -------------------------------------------------

            owner_seen_now = any(
                r["identity"] == "OWNER"
                or r["final_score"] >= args.owner_presence_threshold
                for r in person_results
            )

            with lock:
                if owner_seen_now:
                    state["owner_last_seen_time"] = now

                owner_last_seen_time = state["owner_last_seen_time"]

            if owner_last_seen_time is not None:
                owner_recently_seen = (
                    now - owner_last_seen_time
                ) <= args.owner_hold_seconds
            else:
                owner_recently_seen = False

            owner_present = owner_seen_now or owner_recently_seen

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
                state["last_faces"] = faces
                state["last_person_results"] = person_results
                state["last_laptop_bbox"] = last_laptop_bbox
                state["last_laptop_spatial_m"] = last_laptop_spatial_m
                state["owner_present"] = owner_present

                if owner_present:
                    # HARD DISARM:
                    # if owner is visible/recently visible, alarm cannot exist.
                    state["unknown_near_start_time"] = None
                    state["unknown_near_duration"] = 0.0
                    state["alarm_reason"] = None
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
                            state["alarm_reason"] = (
                                f"UNKNOWN is within {args.contact_m:.2f}m of the laptop."
                            )
                            state["last_alarm_time"] = now

                    elif unknown_near_duration >= args.lurker_seconds:
                        if now - state["last_alarm_time"] >= state["alarm_cooldown_seconds"]:
                            state["alarm_reason"] = (
                                f"UNKNOWN stayed within {args.guard_m:.2f}m "
                                f"of the laptop for {unknown_near_duration:.1f}s."
                            )
                            state["last_alarm_time"] = now

            time.sleep(0.01)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    args = parse_args()

    print("### RUNNING VERSION: OAK STEREODEPTH + DEPTHMERGER + HARD OWNER DISARM ###")
    print(f"[DEBUG] enrollment_seconds={args.enrollment_seconds}")
    print(f"[DEBUG] body_threshold={args.body_threshold}")
    print(f"[DEBUG] fused_threshold={args.fused_threshold}")
    print(f"[DEBUG] owner_presence_threshold={args.owner_presence_threshold}")
    print(f"[DEBUG] owner_hold_seconds={args.owner_hold_seconds}")
    print(f"[DEBUG] guard_m={args.guard_m}")
    print(f"[DEBUG] contact_m={args.contact_m}")
    print(f"[DEBUG] identity_interval={args.identity_interval}")
    print(f"[DEBUG] face_interval={args.face_interval}")
    print(f"[DEBUG] width={args.width}, height={args.height}, fps={args.fps}")
    print(f"[DEBUG] device={args.device}")

    clear_owner_data()

    osnet_model = load_osnet(args.device)
    face_model = load_face_recognition_model()

    device = dai.Device(dai.DeviceInfo(args.oak_device)) if args.oak_device else dai.Device()
    platform = device.getPlatform().name
    print(f"[OAK] Platform: {platform}")

    available_cameras = device.getConnectedCameras()

    if len(available_cameras) < 3:
        raise ValueError(
            "Device must have 3 cameras: color, left, right. "
            "StereoDepth requires the left and right cameras."
        )

    frame_type = (
        dai.ImgFrame.Type.BGR888p if platform == "RVC2" else dai.ImgFrame.Type.BGR888i
    )

    lock = threading.Lock()

    state = {
        "running": True,
        "phase": "enrolling",

        "latest_frame": None,
        "latest_frame_id": 0,
        "latest_spatial_detections": [],

        "owner_body_crops": [],
        "owner_face_embeddings": [],
        "owner_body_gallery": None,
        "owner_face_gallery": None,

        "last_owner_bbox": None,
        "last_faces": [],
        "last_person_results": [],

        "last_laptop_bbox": None,
        "last_laptop_spatial_m": None,
        "protected_laptop_bbox": None,

        "owner_present": False,
        "owner_last_seen_time": None,
        "unknown_near_start_time": None,
        "unknown_near_duration": 0.0,

        "last_alarm_time": -999.0,
        "alarm_cooldown_seconds": 3.0,
        "alarm_reason": None,

        "status_message": "ENROLLING",
    }

    worker = threading.Thread(
        target=inference_worker,
        args=(args, state, lock, osnet_model, face_model),
        daemon=True,
    )
    worker.start()

    with dai.Pipeline(device) as pipeline:
        print("[OAK] Creating pipeline with on-device YOLO + StereoDepth + DepthMerger...")

        # -----------------------------------------------------
        # On-device object detection model
        # -----------------------------------------------------

        obj_det_model_description = dai.NNModelDescription.fromYamlFile(
            f"yolov6_nano_r2_coco.{platform}.yaml"
        )

        obj_det_nn_archive = dai.NNArchive(
            dai.getModelFromZoo(obj_det_model_description)
        )

        classes = obj_det_nn_archive.getConfig().model.heads[0].metadata.classes
        classes = list(classes)

        print("[OAK] Object classes loaded.")
        print(f"[OAK] person class id: {classes.index('person') if 'person' in classes else 'missing'}")
        print(f"[OAK] laptop class id: {classes.index('laptop') if 'laptop' in classes else 'missing'}")

        # -----------------------------------------------------
        # Cameras
        # -----------------------------------------------------

        color_camera = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_A
        )

        left_cam = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_B
        )

        right_cam = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_C
        )

        # -----------------------------------------------------
        # StereoDepth
        # -----------------------------------------------------

        stereo = pipeline.create(dai.node.StereoDepth).build(
            left=left_cam.requestOutput(
                obj_det_nn_archive.getInputSize(),
                fps=args.fps,
            ),
            right=right_cam.requestOutput(
                obj_det_nn_archive.getInputSize(),
                fps=args.fps,
            ),
            presetMode=dai.node.StereoDepth.PresetMode.HIGH_DETAIL,
        )

        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

        if platform == "RVC2":
            stereo.setOutputSize(*obj_det_nn_archive.getInputSize())

        stereo.setLeftRightCheck(True)
        stereo.setRectification(True)

        # -----------------------------------------------------
        # Color output
        # -----------------------------------------------------

        camera_output = color_camera.requestOutput(
            (args.width, args.height),
            frame_type,
            fps=args.fps,
        )

        # -----------------------------------------------------
        # Resize/manip for object detection
        # -----------------------------------------------------

        det_input_w, det_input_h = obj_det_nn_archive.getInputSize()

        obj_det_manip = pipeline.create(dai.node.ImageManip)
        obj_det_manip.initialConfig.setOutputSize(
            det_input_w,
            det_input_h,
            mode=dai.ImageManipConfig.ResizeMode.STRETCH,
        )
        obj_det_manip.initialConfig.setFrameType(frame_type)

        camera_output.link(obj_det_manip.inputImage)

        obj_det_nn: ParsingNeuralNetwork = pipeline.create(
            ParsingNeuralNetwork
        ).build(
            obj_det_manip.out,
            obj_det_nn_archive,
        )

        if platform == "RVC2":
            obj_det_nn.setNNArchive(
                obj_det_nn_archive,
                numShaves=7,
            )

        # -----------------------------------------------------
        # DepthMerger: this is the key part.
        # It merges YOLO detections with StereoDepth and calibration.
        # Output detections include spatial x,y,z coordinates.
        # -----------------------------------------------------

        detection_depth_merger = pipeline.create(DepthMerger).build(
            output2d=obj_det_nn.out,
            outputDepth=stereo.depth,
            calibData=device.readCalibration2(),
            depthAlignmentSocket=dai.CameraBoardSocket.CAM_A,
            shrinkingFactor=0.1,
        )

        # Queues
        frame_queue = camera_output.createOutputQueue(
            maxSize=8,
            blocking=False,
        )

        spatial_queue = detection_depth_merger.output.createOutputQueue(
            maxSize=8,
            blocking=False,
        )

        print("[OAK] Pipeline created.")
        pipeline.start()

        enrollment_start_time = None
        frame_id = 0
        last_spatial_detections = []

        print("[OWNER ENROLLMENT WAITING FOR FIRST FRAME]")
        print(f"Enrollment duration: {args.enrollment_seconds} seconds")
        print("Only the owner should be visible during enrollment.\n")

        while pipeline.isRunning():
            frame_msg = frame_queue.tryGet()
            spatial_msg = spatial_queue.tryGet()

            if frame_msg is None:
                key = cv2.waitKey(1)
                if key == ord("q"):
                    print("[INFO] Exiting.")
                    break
                continue

            frame = frame_msg.getCvFrame()
            frame_id += 1

            if spatial_msg is not None:
                last_spatial_detections = parse_spatial_detections(
                    spatial_msg=spatial_msg,
                    classes=classes,
                    frame_shape=frame.shape,
                )

            if enrollment_start_time is None:
                enrollment_start_time = time.time()
                print("[OWNER ENROLLMENT STARTED FROM FIRST FRAME]\n")

            now = time.time()
            elapsed = now - enrollment_start_time

            with lock:
                state["latest_frame"] = frame.copy()
                state["latest_frame_id"] = frame_id
                state["latest_spatial_detections"] = [dict(d) for d in last_spatial_detections]
                phase = state["phase"]
                last_owner_bbox = state["last_owner_bbox"]

            # Collect owner body crops every frame using latest valid bbox.
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

            # Extra safety: never beep if owner is present/recently seen.
            if alarm_reason is not None and not owner_present:
                trigger_alarm(alarm_reason)

            # -----------------------------------------------------
            # Draw faces
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
            # Draw laptop
            # -----------------------------------------------------

            if laptop_bbox_for_logic is not None:
                draw_label(
                    frame,
                    laptop_bbox_for_logic,
                    "PROTECTED LAPTOP",
                    (255, 0, 0),
                )

            # -----------------------------------------------------
            # Draw persons and metric distances
            # -----------------------------------------------------

            for result in last_person_results:
                draw_label(
                    frame,
                    result["bbox"],
                    result["label"],
                    result["color"],
                )

                if laptop_bbox_for_logic is not None:
                    draw_distance_line_m(
                        frame=frame,
                        person_bbox=result["bbox"],
                        laptop_bbox=laptop_bbox_for_logic,
                        distance_m=result.get("distance_m"),
                        identity=result["identity"],
                    )

                    px1, py1, px2, py2 = result["bbox"]

                    if result.get("distance_m") is not None:
                        dist_text = f"dist_to_laptop={result['distance_m']:.2f}m"
                    else:
                        dist_text = "dist_to_laptop=?m"

                    cv2.putText(
                        frame,
                        dist_text,
                        (px1, min(frame.shape[0] - 20, py2 + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        result["color"],
                        2,
                    )

            # -----------------------------------------------------
            # Status overlay
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
                    draw_status(
                        frame,
                        "MONITORING - OWNER PRESENT / RECENTLY SEEN: DISARMED",
                        (0, 255, 0),
                    )
                else:
                    draw_status(
                        frame,
                        f"MONITORING - OWNER ABSENT: ARMED | guard={args.guard_m:.2f}m",
                        (255, 255, 255),
                    )

                if unknown_near_duration > 0 and not owner_present:
                    cv2.putText(
                        frame,
                        f"UNKNOWN WITHIN {args.guard_m:.2f}m: {unknown_near_duration:.1f}s",
                        (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 165, 255),
                        2,
                    )

                    if unknown_near_duration >= args.lurker_seconds:
                        cv2.putText(
                            frame,
                            "ALARM: UNKNOWN TOO CLOSE TO LAPTOP",
                            (20, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            3,
                        )

            cv2.imshow("Desk Guardian - OAK Spatial Demo", frame)

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
