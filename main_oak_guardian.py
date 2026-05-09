from pathlib import Path
import argparse
import time
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import torchreid
import depthai as dai

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

OWNER_DIR = Path("data/owner")
OWNER_GALLERY_PATH = OWNER_DIR / "owner_gallery.npy"
OWNER_CENTROID_PATH = OWNER_DIR / "owner_centroid.npy"


# ---------------------------------------------------------
# ARGUMENTS
# ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--enrollment_seconds",
        type=float,
        default=10.0,
        help="First N seconds are used to save owner embeddings.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Cosine similarity threshold for OWNER / UNKNOWN.",
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
        help="Torch device: cuda or cpu.",
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


def save_owner_gallery(owner_embeddings):
    OWNER_DIR.mkdir(parents=True, exist_ok=True)

    if len(owner_embeddings) == 0:
        raise RuntimeError("No owner embeddings collected during enrollment.")

    owner_gallery = np.stack(owner_embeddings, axis=0).astype(np.float32)

    owner_centroid = owner_gallery.mean(axis=0)
    owner_centroid = l2_normalize(owner_centroid)

    np.save(OWNER_GALLERY_PATH, owner_gallery)
    np.save(OWNER_CENTROID_PATH, owner_centroid)

    print("\n[OWNER ENROLLMENT COMPLETED]")
    print(f"Saved embeddings: {len(owner_gallery)}")
    print(f"Gallery path: {OWNER_GALLERY_PATH}")
    print(f"Centroid path: {OWNER_CENTROID_PATH}\n")

    return owner_gallery, owner_centroid


def match_owner(embedding, owner_gallery, threshold):
    embedding = l2_normalize(embedding)

    scores = owner_gallery @ embedding
    best_score = float(np.max(scores))

    is_owner = best_score >= threshold

    return is_owner, best_score


# ---------------------------------------------------------
# OSNET
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
# YOLO DETECTION
# ---------------------------------------------------------

def get_yolo_detections(yolo_model, frame, conf=0.35):
    """
    Detects persons and laptops.
    YOLO COCO labels:
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
    Otherwise returns the minimum Euclidean distance between boxes in pixels.
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
        0.7,
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


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    args = parse_args()

    print(f"[DEVICE] Running YOLO/OSNet on laptop: {args.device}")
    print("[INFO] Input is LIVE RGB stream from OAK camera.")
    print("[INFO] YOLO detects persons and laptops.")
    print("[INFO] OSNet performs owner re-identification.\n")

    yolo_model = YOLO("yolov8n.pt")
    osnet_model = load_osnet(args.device)

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

    owner_embeddings = []
    owner_gallery = None
    owner_centroid = None
    enrollment_done = False

    # Laptop memory
    last_laptop_bbox = None
    protected_laptop_bbox = None

    # Lurker logic
    unknown_near_start_time = None

    # Alarm cooldown to avoid beeping every frame
    last_alarm_time = -999.0
    alarm_cooldown_seconds = 3.0

    enrollment_start_time = time.time()

    print("[OWNER ENROLLMENT STARTED]")
    print(f"First {args.enrollment_seconds} seconds are used for owner enrollment.")
    print("Only the owner should be visible during this phase.\n")

    with dai.Pipeline(oak_device) as pipeline:
        print("[OAK] Creating live camera pipeline...")

        cam = pipeline.create(dai.node.Camera).build()
        cam_out = cam.requestOutput(
            size=(args.width, args.height),
            type=frame_type,
            fps=args.fps,
        )

        print("[OAK] Pipeline created.")
        pipeline.start()

        rgb_queue = cam_out.createOutputQueue(maxSize=4, blocking=False)

        while pipeline.isRunning():
            frame_msg = rgb_queue.tryGet()

            if frame_msg is None:
                key = cv2.waitKey(1)
                if key == ord("q"):
                    print("[INFO] Exiting.")
                    break
                continue

            frame = frame_msg.getCvFrame()
            now = time.time()
            elapsed = now - enrollment_start_time

            persons, laptops = get_yolo_detections(yolo_model, frame)

            # Use the largest detected laptop as the protected computer
            current_laptop = laptops[0] if len(laptops) > 0 else None

            if current_laptop is not None:
                last_laptop_bbox = current_laptop["bbox"]

                if protected_laptop_bbox is None:
                    protected_laptop_bbox = current_laptop["bbox"]
                    print(f"[ASSET MAPPING] Laptop mapped at t={elapsed:.2f}s")

            laptop_bbox_for_logic = last_laptop_bbox

            # Draw laptop and guard area
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

                # During enrollment, use only the largest detected person.
                if len(persons) > 0:
                    owner_person = persons[0]
                    x1, y1, x2, y2 = owner_person["bbox"]

                    crop = frame[y1:y2, x1:x2]

                    if crop.size > 0:
                        embedding = get_osnet_embedding(osnet_model, crop, args.device)
                        owner_embeddings.append(embedding)

                        draw_label(
                            frame,
                            owner_person["bbox"],
                            f"ENROLLING OWNER | emb={len(owner_embeddings)}",
                            (0, 255, 255),
                        )

                        print(
                            f"[ENROLLMENT] t={elapsed:.2f}s | "
                            f"saved embedding {len(owner_embeddings)}"
                        )

                draw_status(frame, status_text, (0, 255, 255))

            # -----------------------------------------------------
            # PHASE B: MONITORING
            # -----------------------------------------------------

            else:
                if not enrollment_done:
                    owner_gallery, owner_centroid = save_owner_gallery(owner_embeddings)
                    enrollment_done = True

                    print("[MONITORING STARTED]")
                    print(f"Owner threshold: {args.threshold}")
                    print(f"Guard distance in pixels: {args.guard_px}")
                    print(f"Contact distance in pixels: {args.contact_px}")
                    print(f"Lurker seconds: {args.lurker_seconds}\n")

                any_unknown_near_laptop = False

                for person in persons:
                    x1, y1, x2, y2 = person["bbox"]
                    crop = frame[y1:y2, x1:x2]

                    if crop.size == 0:
                        continue

                    embedding = get_osnet_embedding(osnet_model, crop, args.device)

                    is_owner, score = match_owner(
                        embedding,
                        owner_gallery,
                        threshold=args.threshold,
                    )

                    if is_owner:
                        identity = "OWNER"
                        color = (0, 255, 0)
                    else:
                        identity = "UNKNOWN"
                        color = (0, 0, 255)

                    label = f"{identity} | score={score:.3f}"
                    draw_label(frame, person["bbox"], label, color)

                    # Skip all alarm logic for the owner
                    if is_owner:
                        continue

                    # If no laptop has ever been detected, cannot apply asset logic
                    if laptop_bbox_for_logic is None:
                        continue

                    distance_px = bbox_distance_px(
                        person["bbox"],
                        laptop_bbox_for_logic,
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

                    if near_laptop:
                        any_unknown_near_laptop = True

                    cv2.putText(
                        frame,
                        f"dist_to_laptop={distance_px:.1f}px",
                        (x1, min(frame.shape[0] - 20, y2 + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                    # -------------------------------------------------
                    # ALARM CASE 1:
                    # UNKNOWN touches / takes the laptop
                    # -------------------------------------------------

                    if touching_laptop:
                        if now - last_alarm_time >= alarm_cooldown_seconds:
                            trigger_alarm("UNKNOWN person is touching the laptop.")
                            last_alarm_time = now

                        cv2.putText(
                            frame,
                            "ALARM: UNKNOWN TOUCHING LAPTOP",
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            3,
                        )

                # -----------------------------------------------------
                # ALARM CASE 2:
                # UNKNOWN stays near the laptop for more than 5 seconds
                # -----------------------------------------------------

                if any_unknown_near_laptop:
                    if unknown_near_start_time is None:
                        unknown_near_start_time = now

                    unknown_near_duration = now - unknown_near_start_time

                    cv2.putText(
                        frame,
                        f"UNKNOWN NEAR LAPTOP: {unknown_near_duration:.1f}s",
                        (20, 120),
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
                            (20, 160),
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
