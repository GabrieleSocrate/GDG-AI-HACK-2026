from pathlib import Path
import argparse
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import torchreid


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

OWNER_DIR = Path("data/owner")
OWNER_GALLERY_PATH = OWNER_DIR / "owner_gallery.npy"
OWNER_CENTROID_PATH = OWNER_DIR / "owner_centroid.npy"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--media_path",
        type=str,
        required=True,
        help="Path to the recorded video.",
    )

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
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu.",
    )

    return parser.parse_args()


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


def load_osnet(device):
    """
    Loads OSNet on the laptop.
    This does NOT use OAK / DepthAI.
    """

    model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=1000,
        pretrained=True,
    )

    model.eval()
    model.to(device)

    return model


def preprocess_person_crop(crop_bgr):
    """
    OSNet expects a person crop resized to 256x128.
    Shape convention: height=256, width=128.
    """

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


def get_person_detections(yolo_model, frame, conf=0.35):
    """
    Detects persons using YOLO on the laptop.
    COCO class 0 = person.
    """

    results = yolo_model(frame, conf=conf, verbose=False)[0]

    persons = []

    if results.boxes is None:
        return persons

    for box in results.boxes:
        cls_id = int(box.cls[0].item())

        # COCO class 0 = person
        if cls_id != 0:
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

        persons.append(
            {
                "bbox": (x1, y1, x2, y2),
                "confidence": confidence,
                "area": area,
            }
        )

    persons = sorted(persons, key=lambda p: p["area"], reverse=True)

    return persons


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


def main():
    args = parse_args()

    video_path = Path(args.media_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"[VIDEO] Loading video from: {video_path}")
    print(f"[DEVICE] Running host-side models on: {args.device}")
    print("[INFO] This version does NOT use OAK / DepthAI.\n")

    # YOLO for person detection on laptop
    yolo_model = YOLO("yolov8n.pt")

    # OSNet for person Re-ID on laptop
    osnet_model = load_osnet(args.device)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    owner_embeddings = []
    owner_gallery = None
    owner_centroid = None
    enrollment_done = False

    frame_idx = 0

    print("[OWNER ENROLLMENT STARTED]")
    print(f"First {args.enrollment_seconds} seconds are used for owner enrollment.")
    print("Make sure only the owner is visible in those first seconds.\n")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[INFO] End of video.")
            break

        video_time = frame_idx / fps

        persons = get_person_detections(yolo_model, frame)

        # -----------------------------------------------------
        # PHASE A: ENROLLMENT
        # -----------------------------------------------------
        # During enrollment, we take only the largest detected person.
        # This avoids saving multiple people as owner if YOLO detects more.
        # -----------------------------------------------------

        if video_time <= args.enrollment_seconds:
            status_text = f"ENROLLING OWNER: {video_time:.1f}s / {args.enrollment_seconds:.1f}s"

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
                        f"[ENROLLMENT] t={video_time:.2f}s | "
                        f"saved embedding {len(owner_embeddings)}"
                    )

            cv2.putText(
                frame,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        # -----------------------------------------------------
        # PHASE B: MONITORING
        # -----------------------------------------------------

        else:
            if not enrollment_done:
                owner_gallery, owner_centroid = save_owner_gallery(owner_embeddings)
                enrollment_done = True

                print("[MONITORING STARTED]")
                print(f"Owner threshold: {args.threshold}\n")

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

                print(f"[RE-ID] t={video_time:.2f}s | {label}")

            cv2.putText(
                frame,
                "MONITORING",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Host-side Desk Guardian Re-ID", frame)

        key = cv2.waitKey(1)

        if key == ord("q"):
            print("[INFO] Exiting.")
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
