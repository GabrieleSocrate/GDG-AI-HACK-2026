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


def clear_owner_data():
    """
    Clears previous owner embeddings before a new enrollment phase.
    """

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

    parser.add_argument(
        "--enrollment_seconds",
        type=float,
        default=30.0,
        help="First N seconds are used to save owner embeddings.",
    )

    parser.add_argument(
        "--body_threshold",
        type=float,
        default=0.70,
        help="Threshold when only OSNet body Re-ID is available.",
    )

    parser.add_argument(
        "--fused_threshold",
        type=float,
        default=0.65,
        help="Threshold when OSNet and Face Recognition scores are averaged.",
    )

    parser.add_argument(
        "--guard_px",
        type=float,
        default=250.0,
        help="Pixel distance around the laptop used as guarded area.",
    )

    parser.add_argument(
        "--contact_px",
        type=float,
        default=30.0,
        help="Pixel distance considered as touching the laptop.",
    )

    parser.add_argument(
        "--lurker_seconds",
        type=float,
        default=5.0,
        help="Seconds an UNKNOWN person can stay near the laptop before alarm.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for OSNet: cuda or cpu.",
    )

    parser.add_argument(
        "--oak_device",
        type=str,
        default=None,
        help="Optional OAK device id/ip. Leave empty to use the first available OAK.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="OAK camera FPS.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="OAK camera output width.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="OAK camera output height.",
    )

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
    """
    Computes mean cosine similarity between one embedding and a gallery.
    """

    embedding = l2_normalize(embedding)
    scores = gallery @ embedding

    mean_score = float(np.mean(scores))
    max_score = float(np.max(scores))
    min_score = float(np.min(scores))

    return mean_score, max_score, min_score


def fuse_scores(body_score, face_score, body_threshold, fused_threshold):
    """
    If face recognition is available:
        final_score = mean(OSNet body score, Face Recognition score)

    If face recognition is NOT available:
        final_score = OSNet body score only
    """

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

    tensor = transform(pil_img).unsqueeze(0)

    return tensor


@torch.no_grad()
def get_osnet_embedding(osnet_model, crop_bgr, device):
    input_tensor = preprocess_person_crop(crop_bgr).to(device)

    embedding = osnet_model(input_tensor)

    embedding = embedding.detach().cpu().numpy().reshape(-1)
    embedding = l2_normalize(embedding)

    return embedding


# ---------------------------------------------------------
# FACE RECOGNITION
# ---------------------------------------------------------

def load_face_recognition_model():
    """
    Loads InsightFace with detection + recognition modules.

    FaceAnalysis is a wrapper.
    The recognition module produces face identity embeddings.
    """

    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )

    app.prepare(
        ctx_id=-1,
        det_size=(640, 640),
    )

    loaded_modules = list(app.models.keys())
    print(f"[FACE RECOGNITION] Loaded InsightFace modules: {loaded_modules}")

    if "recognition" not in app.models:
        raise RuntimeError(
            "Face recognition model was NOT loaded. "
            "Only face detection may be available."
        )

    print("[FACE RECOGNITION] Recognition model loaded correctly.")
    print(f"[FACE RECOGNITION] Recognition model object: {app.models['recognition']}")

    return app


def detect_faces_with_embeddings(face_model, frame):
    """
    Detects faces and returns normalized face-recognition embeddings.
    If no recognition embedding is available, the face is skipped.
    """

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
            print("[FACE WARNING] Face detected but no recognition embedding found.")
            continue

        embedding = l2_normalize(embedding)

        faces.append(
            {
                "bbox": (x1, y1, x2, y2),
                "embedding": embedding,
            }
        )

    return faces


def face_belongs_to_person(face_bbox, person_bbox, tolerance=80):
    """
    Associates a face to a person if the face center is inside,
    or close to, the person bbox.
    """

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
    """
    If multiple faces are inside one person bbox, returns the largest one.
    """

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
    Detects persons and laptops.
    COCO labels:
    - person
    - laptop
    """

    results = yolo_model(frame, conf=conf, verbose=False)[0]

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
    """
    Returns 0 if the boxes touch/overlap.
    Otherwise returns minimum Euclidean distance between boxes in pixels.
    """

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

    print("### RUNNING VERSION: OSNET + FACE RECOGNITION FUSION ###")
    print(f"[DEBUG] enrollment_seconds={args.enrollment_seconds}")
    print(f"[DEBUG] body_threshold={args.body_threshold}")
    print(f"[DEBUG] fused_threshold={args.fused_threshold}")
    print(f"[DEBUG] guard_px={args.guard_px}")
    print(f"[DEBUG] contact_px={args.contact_px}")

    clear_owner_data()

    print(f"[DEVICE] Running OSNet on laptop: {args.device}")
    print("[INFO] Input is LIVE RGB stream from OAK camera.")
    print("[INFO] YOLO detects persons and laptops.")
    print("[INFO] OSNet performs body Re-ID.")
    print("[INFO] Face Recognition performs identity recognition from face embeddings.")
    print("[INFO] If face is available: final_score = mean(OSNet_score, FaceRecognition_score).")
    print("[INFO] If face is not available: final_score = OSNet_score only.\n")

    yolo_model = YOLO("yolov8n.pt")
    osnet_model = load_osnet(args.device)
    face_model = load_face_recognition_model()

    # ---------------------------------------------------------
    # OAK LIVE CAMERA SETUP
    # ---------------------------------------------------------

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

    last_laptop_bbox = None
    protected_laptop_bbox = None

    unknown_near_start_time = None

    last_alarm_time = -999.0
    alarm_cooldown_seconds = 3.0

    print("[OWNER ENROLLMENT WAITING FOR FIRST FRAME]")
    print(f"Enrollment duration: {args.enrollment_seconds} seconds")
    print("Only the owner should be visible during this phase.")
    print("Move for OSNet, and look at the camera sometimes for Face Recognition.\n")

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

        while pipeline.isRunning():
            frame_msg = rgb_queue.tryGet()

            if frame_msg is None:
                key = cv2.waitKey(1)
                if key == ord("q"):
                    print("[INFO] Exiting.")
                    break
                continue

            frame = frame_msg.getCvFrame()

            if enrollment_start_time is None:
                enrollment_start_time = time.time()
                print("[OWNER ENROLLMENT STARTED FROM FIRST FRAME]\n")

            now = time.time()
            elapsed = now - enrollment_start_time

            persons, laptops = get_yolo_detections(yolo_model, frame)
            faces = detect_faces_with_embeddings(face_model, frame)

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

            current_laptop = laptops[0] if len(laptops) > 0 else None

            if current_laptop is not None:
                last_laptop_bbox = current_laptop["bbox"]

                if protected_laptop_bbox is None:
                    protected_laptop_bbox = current_laptop["bbox"]
                    print(f"[ASSET MAPPING] Laptop mapped at t={elapsed:.2f}s")

            laptop_bbox_for_logic = last_laptop_bbox

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
            # PHASE A: OWNER ENROLLMENT
            # -----------------------------------------------------

            if elapsed <= args.enrollment_seconds:
                status_text = (
                    f"ENROLLING OWNER: {elapsed:.1f}s / "
                    f"{args.enrollment_seconds:.1f}s"
                )

                if len(persons) > 0:
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

                        draw_label(
                            frame,
                            owner_person["bbox"],
                            (
                                f"ENROLL OWNER | "
                                f"body={len(owner_body_embeddings)} | {face_msg}"
                            ),
                            (0, 255, 255),
                        )

                        print(
                            f"[ENROLLMENT] t={elapsed:.2f}s | "
                            f"body={len(owner_body_embeddings)} | "
                            f"face={len(owner_face_embeddings)}"
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

                    print("[MONITORING STARTED]")
                    print(f"Body threshold: {args.body_threshold}")
                    print(f"Fused threshold: {args.fused_threshold}")
                    print(f"Guard distance in pixels: {args.guard_px}")
                    print(f"Contact distance in pixels: {args.contact_px}")
                    print(f"Lurker seconds: {args.lurker_seconds}\n")

                any_unknown_near_laptop = False

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

                    draw_label(frame, person["bbox"], label, color)

                    print(
                        f"[RE-ID] {identity} | source={source} | "
                        f"final={final_score:.3f} | "
                        f"body={body_score:.3f} | "
                        f"face={face_score if face_score is not None else 'NA'}"
                    )

                    if laptop_bbox_for_logic is None:
                        continue

                    distance_px = bbox_distance_px(
                        person["bbox"],
                        laptop_bbox_for_logic,
                    )

                    draw_distance_line(
                        frame=frame,
                        person_bbox=person["bbox"],
                        laptop_bbox=laptop_bbox_for_logic,
                        distance_px=distance_px,
                        identity=identity,
                    )

                    person_center = bbox_center(person["bbox"])
                    guard_bbox = expand_bbox(
                        laptop_bbox_for_logic,
                        args.guard_px,
                        frame.shape,
                    )

                    near_laptop = (
                        distance_px <= args.guard_px
                        or point_inside_bbox(person_center, guard_bbox)
                    )

                    touching_laptop = distance_px <= args.contact_px

                    cv2.putText(
                        frame,
                        f"dist_to_laptop={distance_px:.1f}px",
                        (px1, min(frame.shape[0] - 20, py2 + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                    if identity == "OWNER":
                        continue

                    if near_laptop:
                        any_unknown_near_laptop = True

                    if touching_laptop:
                        if now - last_alarm_time >= alarm_cooldown_seconds:
                            trigger_alarm("UNKNOWN person is touching the laptop.")
                            last_alarm_time = now

                        cv2.putText(
                            frame,
                            "ALARM: UNKNOWN TOUCHING LAPTOP",
                            (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            3,
                        )

                # -----------------------------------------------------
                # LURKER ALARM
                # -----------------------------------------------------

                if any_unknown_near_laptop:
                    if unknown_near_start_time is None:
                        unknown_near_start_time = now

                    unknown_near_duration = now - unknown_near_start_time

                    cv2.putText(
                        frame,
                        f"UNKNOWN NEAR LAPTOP: {unknown_near_duration:.1f}s",
                        (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 165, 255),
                        2,
                    )

                    if unknown_near_duration >= args.lurker_seconds:
                        if now - last_alarm_time >= alarm_cooldown_seconds:
                            trigger_alarm(
                                f"UNKNOWN person stayed near the laptop for "
                                f"{unknown_near_duration:.1f} seconds."
                            )
                            last_alarm_time = now

                        cv2.putText(
                            frame,
                            "ALARM: UNKNOWN LURKER NEAR LAPTOP",
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
