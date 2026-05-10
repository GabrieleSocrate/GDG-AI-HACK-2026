from pathlib import Path
import argparse
import time
import shutil
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import torchreid
import depthai as dai
from insightface.app import FaceAnalysis

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


# ---------------------------------------------------------
# PROTECTED OBJECTS
# ---------------------------------------------------------

PROTECTED_OBJECT_LABELS = [
    "laptop",
    "cell phone",
    "handbag",   # proxy per portafogli / borsa; "wallet" non è in COCO standard
]

PROTECTED_OBJECT_DISPLAY_NAMES = {
    "laptop": "LAPTOP",
    "cell phone": "PHONE",
    "handbag": "WALLET/BAG",
}


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

    parser.add_argument("--guard_px", type=float, default=250.0)
    parser.add_argument("--contact_px", type=float, default=30.0)
    parser.add_argument("--lurker_seconds", type=float, default=5.0)

    parser.add_argument(
        "--inference_every",
        type=int,
        default=3,
        help="Run YOLO/OSNet/Face Recognition every N frames.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument("--oak_device", type=str, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)

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
        raise RuntimeError(
            "Face recognition model was NOT loaded. "
            "Only face detection may be available."
        )

    print("[FACE RECOGNITION] Recognition model loaded correctly.")

    return app


def detect_faces_with_embeddings(face_model, frame):
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

def get_yolo_detections(yolo_model, frame, conf=0.35):
    """
    Detects persons and protected objects.

    Protected objects are defined in PROTECTED_OBJECT_LABELS.
    With YOLO COCO:
    - laptop works
    - cell phone works
    - handbag can be used as a proxy for wallet/bag
    """

    results = yolo_model(frame, conf=conf, verbose=False)[0]

    persons = []
    protected_objects = []

    if results.boxes is None:
        return persons, protected_objects

    names = yolo_model.names
    allowed_labels = ["person"] + PROTECTED_OBJECT_LABELS

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        label = names[cls_id]

        if label not in allowed_labels:
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
            "display_name": PROTECTED_OBJECT_DISPLAY_NAMES.get(
                label,
                label.upper(),
            ),
            "bbox": (x1, y1, x2, y2),
            "confidence": confidence,
            "area": area,
        }

        if label == "person":
            persons.append(det)
        else:
            protected_objects.append(det)

    persons = sorted(persons, key=lambda p: p["area"], reverse=True)
    protected_objects = sorted(protected_objects, key=lambda p: p["area"], reverse=True)

    return persons, protected_objects


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
        winsound.Beep(1300, 700)
        winsound.Beep(1700, 700)
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
        0.8,
        color,
        2,
    )


def draw_distance_line(frame, person_bbox, object_bbox, distance_px, identity):
    person_center = bbox_center(person_bbox)
    object_center = bbox_center(object_bbox)

    color = (0, 255, 0) if identity == "OWNER" else (0, 0, 255)

    cv2.line(frame, person_center, object_center, color, 2)
    cv2.circle(frame, person_center, 5, color, -1)
    cv2.circle(frame, object_center, 5, (255, 0, 0), -1)

    mid_x = int((person_center[0] + object_center[0]) / 2)
    mid_y = int((person_center[1] + object_center[1]) / 2)

    cv2.putText(
        frame,
        f"{distance_px:.1f}px",
        (mid_x, mid_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
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
# MAIN
# ---------------------------------------------------------

def main():
    args = parse_args()

    print("### RUNNING VERSION: MULTI-PROTECTED OBJECTS ###")
    print(f"[DEBUG] enrollment_seconds={args.enrollment_seconds}")
    print(f"[DEBUG] body_threshold={args.body_threshold}")
    print(f"[DEBUG] fused_threshold={args.fused_threshold}")
    print(f"[DEBUG] inference_every={args.inference_every}")
    print(f"[DEBUG] width={args.width}, height={args.height}, fps={args.fps}")
    print(f"[DEBUG] protected labels={PROTECTED_OBJECT_LABELS}")

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

    owner_body_embeddings = []
    owner_face_embeddings = []

    owner_body_gallery = None
    owner_face_gallery = None

    enrollment_done = False

    last_protected_object_bbox = None
    last_protected_object_label = None
    last_protected_object_name = None

    protected_object_bbox = None
    protected_object_label = None
    protected_object_name = None

    unknown_near_start_time = None

    last_alarm_time = -999.0
    alarm_cooldown_seconds = 3.0

    last_faces = []
    last_person_results = []

    print("[OWNER ENROLLMENT WAITING FOR FIRST FRAME]")
    print(f"Enrollment duration: {args.enrollment_seconds} seconds")
    print("Only the owner should be visible during this phase.\n")

    with dai.Pipeline(oak_device) as pipeline:
        print("[OAK] Creating live camera pipeline...")

        cam = pipeline.create(dai.node.Camera).build()
        cam_out = cam.requestOutput(
            size=(args.width, args.height),
            type=frame_type,
            fps=args.fps,
        )

        rgb_queue = cam_out.createOutputQueue(maxSize=4, blocking=False)

        print("[OAK] Pipeline created.")
        pipeline.start()

        enrollment_start_time = None
        frame_counter = 0

        while pipeline.isRunning():
            frame_msg = rgb_queue.tryGet()

            if frame_msg is None:
                key = cv2.waitKey(1)
                if key == ord("q"):
                    print("[INFO] Exiting.")
                    break
                continue

            frame = frame_msg.getCvFrame()
            frame_counter += 1

            run_inference = (
                frame_counter == 1
                or frame_counter % args.inference_every == 0
            )

            if enrollment_start_time is None:
                enrollment_start_time = time.time()
                print("[OWNER ENROLLMENT STARTED FROM FIRST FRAME]\n")

            now = time.time()
            elapsed = now - enrollment_start_time

            # -----------------------------------------------------
            # HEAVY INFERENCE ONLY EVERY N FRAMES
            # -----------------------------------------------------

            if run_inference:
                persons, protected_objects = get_yolo_detections(yolo_model, frame)
                faces = detect_faces_with_embeddings(face_model, frame)
                last_faces = faces

                current_protected_object = (
                    protected_objects[0] if len(protected_objects) > 0 else None
                )

                if current_protected_object is not None:
                    last_protected_object_bbox = current_protected_object["bbox"]
                    last_protected_object_label = current_protected_object["label"]
                    last_protected_object_name = current_protected_object["display_name"]

                    if protected_object_bbox is None:
                        protected_object_bbox = current_protected_object["bbox"]
                        protected_object_label = current_protected_object["label"]
                        protected_object_name = current_protected_object["display_name"]

                        print(
                            f"[ASSET MAPPING] Protected object mapped: "
                            f"{protected_object_name} at t={elapsed:.2f}s"
                        )
            else:
                persons = []
                faces = last_faces

            protected_bbox_for_logic = last_protected_object_bbox
            protected_name_for_logic = last_protected_object_name or "OBJECT"

            # -----------------------------------------------------
            # DRAW FACE BOXES
            # -----------------------------------------------------

            cv2.putText(
                frame,
                f"FaceRec faces: {len(faces)}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            for face in faces:
                draw_face_box(frame, face["bbox"], "FaceRec")

            # -----------------------------------------------------
            # DRAW PROTECTED OBJECT + GUARD AREA
            # -----------------------------------------------------

            if protected_bbox_for_logic is not None:
                guard_bbox = expand_bbox(
                    protected_bbox_for_logic,
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
                    protected_bbox_for_logic,
                    f"PROTECTED {protected_name_for_logic}",
                    (255, 0, 0),
                )

            # -----------------------------------------------------
            # PHASE A: OWNER ENROLLMENT
            # -----------------------------------------------------

            if elapsed <= args.enrollment_seconds:
                status_text = (
                    f"ENROLLING OWNER: {elapsed:.1f}s / "
                    f"{args.enrollment_seconds:.1f}s"
                )

                if run_inference and len(persons) > 0:
                    owner_person = persons[0]
                    px1, py1, px2, py2 = owner_person["bbox"]

                    person_crop = frame[py1:py2, px1:px2]

                    if person_crop.size > 0:
                        body_embedding = get_osnet_embedding(
                            osnet_model,
                            person_crop,
                            args.device,
                        )

                        owner_body_embeddings.append(body_embedding)

                        matched_face = get_face_for_person(
                            faces=faces,
                            person_bbox=owner_person["bbox"],
                        )

                        if matched_face is not None:
                            owner_face_embeddings.append(matched_face["embedding"])
                            face_msg = f"face_emb={len(owner_face_embeddings)}"
                        else:
                            face_msg = "no_face"

                        last_person_results = [
                            {
                                "bbox": owner_person["bbox"],
                                "label": (
                                    f"ENROLL OWNER | "
                                    f"body={len(owner_body_embeddings)} | {face_msg}"
                                ),
                                "color": (0, 255, 255),
                                "identity": "OWNER",
                            }
                        ]

                        print(
                            f"[ENROLLMENT] t={elapsed:.2f}s | "
                            f"body={len(owner_body_embeddings)} | "
                            f"face={len(owner_face_embeddings)}"
                        )

                for result in last_person_results:
                    draw_label(
                        frame,
                        result["bbox"],
                        result["label"],
                        result["color"],
                    )

                draw_status(frame, status_text, (0, 255, 255))

            # -----------------------------------------------------
            # PHASE B: MONITORING
            # -----------------------------------------------------

            else:
                if not enrollment_done:
                    owner_body_gallery, owner_face_gallery = save_owner_galleries(
                        owner_body_embeddings,
                        owner_face_embeddings,
                    )
                    enrollment_done = True
                    last_person_results = []

                    print("[MONITORING STARTED]")
                    print(f"Body threshold: {args.body_threshold}")
                    print(f"Fused threshold: {args.fused_threshold}")
                    print(f"Inference every: {args.inference_every} frames")
                    print(f"Guard distance in pixels: {args.guard_px}")
                    print(f"Contact distance in pixels: {args.contact_px}")
                    print(f"Lurker seconds: {args.lurker_seconds}")
                    print(f"Protected labels: {PROTECTED_OBJECT_LABELS}\n")

                if run_inference:
                    last_person_results = []

                    for person in persons:
                        px1, py1, px2, py2 = person["bbox"]
                        person_crop = frame[py1:py2, px1:px2]

                        if person_crop.size == 0:
                            continue

                        body_embedding = get_osnet_embedding(
                            osnet_model,
                            person_crop,
                            args.device,
                        )

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
                        near_protected_object = False
                        touching_protected_object = False

                        if protected_bbox_for_logic is not None:
                            distance_px = bbox_distance_px(
                                person["bbox"],
                                protected_bbox_for_logic,
                            )

                            person_center = bbox_center(person["bbox"])
                            guard_bbox = expand_bbox(
                                protected_bbox_for_logic,
                                args.guard_px,
                                frame.shape,
                            )

                            near_protected_object = (
                                distance_px <= args.guard_px
                                or point_inside_bbox(person_center, guard_bbox)
                            )

                            touching_protected_object = distance_px <= args.contact_px

                        last_person_results.append(
                            {
                                "bbox": person["bbox"],
                                "label": label,
                                "color": color,
                                "identity": identity,
                                "distance_px": distance_px,
                                "near_protected_object": near_protected_object,
                                "touching_protected_object": touching_protected_object,
                                "source": source,
                                "final_score": final_score,
                            }
                        )

                        print(
                            f"[RE-ID] {identity} | source={source} | "
                            f"final={final_score:.3f}"
                        )

                any_unknown_near_protected_object = False
                any_unknown_touching_protected_object = False

                for result in last_person_results:
                    draw_label(
                        frame,
                        result["bbox"],
                        result["label"],
                        result["color"],
                    )

                    if (
                        protected_bbox_for_logic is not None
                        and result.get("distance_px") is not None
                    ):
                        draw_distance_line(
                            frame=frame,
                            person_bbox=result["bbox"],
                            object_bbox=protected_bbox_for_logic,
                            distance_px=result["distance_px"],
                            identity=result["identity"],
                        )

                        px1, py1, px2, py2 = result["bbox"]

                        cv2.putText(
                            frame,
                            f"dist_to_{protected_name_for_logic}={result['distance_px']:.1f}px",
                            (px1, min(frame.shape[0] - 20, py2 + 25)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            result["color"],
                            2,
                        )

                    if result["identity"] == "UNKNOWN":
                        if result.get("near_protected_object", False):
                            any_unknown_near_protected_object = True

                        if result.get("touching_protected_object", False):
                            any_unknown_touching_protected_object = True

                # -------------------------------------------------
                # TOUCHING ALARM
                # -------------------------------------------------

                if any_unknown_touching_protected_object:
                    if now - last_alarm_time >= alarm_cooldown_seconds:
                        trigger_alarm(
                            f"UNKNOWN person is touching the protected "
                            f"{protected_name_for_logic}."
                        )
                        last_alarm_time = now

                    cv2.putText(
                        frame,
                        f"ALARM: UNKNOWN TOUCHING {protected_name_for_logic}",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        3,
                    )

                # -------------------------------------------------
                # LURKER ALARM
                # -------------------------------------------------

                if any_unknown_near_protected_object:
                    if unknown_near_start_time is None:
                        unknown_near_start_time = now

                    unknown_near_duration = now - unknown_near_start_time

                    cv2.putText(
                        frame,
                        f"UNKNOWN NEAR {protected_name_for_logic}: {unknown_near_duration:.1f}s",
                        (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 165, 255),
                        2,
                    )

                    if unknown_near_duration >= args.lurker_seconds:
                        if now - last_alarm_time >= alarm_cooldown_seconds:
                            trigger_alarm(
                                f"UNKNOWN person stayed near the protected "
                                f"{protected_name_for_logic} for "
                                f"{unknown_near_duration:.1f} seconds."
                            )
                            last_alarm_time = now

                        cv2.putText(
                            frame,
                            f"ALARM: UNKNOWN NEAR {protected_name_for_logic}",
                            (20, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            3,
                        )

                else:
                    unknown_near_start_time = None

                draw_status(frame, "MONITORING", (255, 255, 255))

            cv2.imshow("Desk Guardian - OAK Live Demo", frame)

            key = cv2.waitKey(1)

            if key == ord("q"):
                print("[INFO] Exiting.")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
