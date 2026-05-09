from pathlib import Path
import time
import numpy as np

import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, GatherData, FrameCropper

from utils.arguments import initialize_argparser
from utils.identification import IdentificationNode


REQ_WIDTH, REQ_HEIGHT = (
    768,
    768,
)  # we request a larger input size to keep enough resolution for the second-stage Re-ID model


# ---------------------------------------------------------
# OWNER ENROLLMENT CONFIG
# ---------------------------------------------------------
# During the first ENROLLMENT_SECONDS seconds of the video,
# the system assumes that the only visible person is the desk owner.
# It saves the OSNet embeddings of that person.
# ---------------------------------------------------------

ENROLLMENT_SECONDS = 10
SAVE_EVERY_N_EMBEDDINGS = 1
OWNER_THRESHOLD = 0.80

OWNER_DIR = Path("data/owner")
OWNER_DIR.mkdir(parents=True, exist_ok=True)

OWNER_GALLERY_PATH = OWNER_DIR / "owner_gallery.npy"
OWNER_CENTROID_PATH = OWNER_DIR / "owner_centroid.npy"


def l2_normalize(embedding: np.ndarray) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(embedding)

    if norm < 1e-8:
        return embedding

    return embedding / norm


def save_owner_gallery(owner_embeddings):
    """
    Saves all owner embeddings collected during the enrollment phase.
    Also saves the average embedding, called centroid.
    """

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


def match_owner(
    embedding: np.ndarray,
    owner_gallery: np.ndarray,
    threshold: float = OWNER_THRESHOLD,
):
    """
    Compares a new embedding with the saved owner gallery.
    The identity is OWNER if the maximum cosine similarity is above the threshold.
    """

    embedding = l2_normalize(embedding)

    scores = owner_gallery @ embedding
    best_score = float(np.max(scores))

    is_owner = best_score >= threshold

    return is_owner, best_score


def extract_embedding_from_msg(msg):
    """
    Tries to extract an embedding vector from the recognition neural network output.

    Depending on the DepthAI / depthai-nodes version, the message may expose
    the output data in slightly different ways.
    """

    data = None

    if hasattr(msg, "getFirstLayerFp16"):
        data = np.array(msg.getFirstLayerFp16(), dtype=np.float32)

    elif hasattr(msg, "getData"):
        raw = msg.getData()

        if isinstance(raw, (bytes, bytearray)):
            data = np.frombuffer(raw, dtype=np.float32)
        else:
            data = np.array(raw, dtype=np.float32)

    elif hasattr(msg, "data"):
        data = np.array(msg.data, dtype=np.float32)

    elif hasattr(msg, "embedding"):
        data = np.array(msg.embedding, dtype=np.float32)

    elif hasattr(msg, "embeddings"):
        data = np.array(msg.embeddings, dtype=np.float32)

    if data is None:
        return None

    data = np.asarray(data, dtype=np.float32).reshape(-1)

    if data.size == 0:
        return None

    return l2_normalize(data)


_, args = initialize_argparser()

visualizer = dai.RemoteConnection(httpPort=8082)
device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
platform = device.getPlatform().name

print(f"Platform: {platform}")

frame_type = (
    dai.ImgFrame.Type.BGR888i if platform == "RVC4" else dai.ImgFrame.Type.BGR888p
)

if not args.fps_limit:
    args.fps_limit = 2 if platform == "RVC2" else 10
    print(
        f"\nFPS limit set to {args.fps_limit} for {platform} platform. "
        f"If you want to set a custom FPS limit, use the --fps_limit flag.\n"
    )


with dai.Pipeline(device) as pipeline:
    print("Creating pipeline...")

    if args.identify == "pose":
        det_model_description = dai.NNModelDescription.fromYamlFile(
            f"scrfd_person_detection_25g.{platform}.yaml"
        )
        rec_model_description = dai.NNModelDescription.fromYamlFile(
            f"osnet_imagenet.{platform}.yaml"
        )
        CSIM = 0.8

    elif args.identify == "face":
        det_model_description = dai.NNModelDescription.fromYamlFile(
            f"scrfd_face_detection_10g.{platform}.yaml"
        )
        rec_model_description = dai.NNModelDescription.fromYamlFile(
            f"arcface_lfw.{platform}.yaml"
        )
        CSIM = 0.1

    else:
        raise ValueError("Unknown identify option provided.")

    if args.cos_similarity_threshold:
        CSIM = args.cos_similarity_threshold  # override default threshold

    # Detection model
    det_model_nn_archive = dai.NNArchive(dai.getModelFromZoo(det_model_description))

    # Recognition model: OSNet if identify == "pose", ArcFace if identify == "face"
    rec_nn_archive = dai.NNArchive(dai.getModelFromZoo(rec_model_description))

    # ---------------------------------------------------------
    # INPUT VIDEO REGISTRATO
    # ---------------------------------------------------------
    # For this test we do NOT use the live OAK camera stream.
    # We use a recorded video passed with --media_path.
    #
    # Example:
    # python main.py --identify pose --media_path path/to/video.mp4
    #
    # The first 10 seconds of the video must contain only the owner.
    # After 10 seconds, the system switches to monitoring mode.
    # ---------------------------------------------------------

    if not args.media_path:
        raise ValueError(
            "For this test, you must pass a recorded video with --media_path. "
            "Example: python main.py --identify pose --media_path video.mp4"
        )

    replay = pipeline.create(dai.node.ReplayVideo)
    replay.setReplayVideoFile(Path(args.media_path))
    replay.setOutFrameType(frame_type)

    # For the enrollment test, it is better not to loop the video.
    # Otherwise the video restarts and may confuse the enrollment/monitoring phases.
    replay.setLoop(False)

    if args.fps_limit:
        replay.setFps(args.fps_limit)

    replay.setSize(REQ_WIDTH, REQ_HEIGHT)

    input_node_out = replay.out

    # ---------------------------------------------------------
    # CAMERA LIVE — DISABLED FOR NOW
    # ---------------------------------------------------------
    # This block will be useful when testing with the real live OAK camera.
    #
    # cam = pipeline.create(dai.node.Camera).build()
    # cam_out = cam.requestOutput(
    #     size=(REQ_WIDTH, REQ_HEIGHT),
    #     type=frame_type,
    #     fps=args.fps_limit,
    # )
    # input_node_out = cam_out
    # ---------------------------------------------------------

    # Resize the input frame to the detection model input size
    resize_node = pipeline.create(dai.node.ImageManip)
    resize_node.setMaxOutputFrameSize(REQ_WIDTH * REQ_HEIGHT * 3)
    resize_node.initialConfig.setOutputSize(
        det_model_nn_archive.getInputWidth(),
        det_model_nn_archive.getInputHeight(),
    )
    resize_node.initialConfig.setReusePreviousImage(False)
    resize_node.inputImage.setBlocking(True)

    input_node_out.link(resize_node.inputImage)

    # Person / face detection model
    det_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        resize_node.out,
        det_model_nn_archive,
    )

    # Crop detected persons/faces from the original input frame
    crop_node = (
        pipeline.create(FrameCropper)
        .fromImgDetections(
            inputImgDetections=det_nn.out,
            outputSize=(
                rec_nn_archive.getInputWidth(),
                rec_nn_archive.getInputHeight(),
            ),
        )
        .build(
            inputImage=input_node_out,
        )
    )

    # Recognition model
    # If identify == "pose", this is OSNet and produces person Re-ID embeddings.
    rec_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
        crop_node.out,
        rec_nn_archive,
    )

    # Sync detections and recognition outputs
    gather_data_node = pipeline.create(GatherData).build(
        cameraFps=args.fps_limit,
        inputData=rec_nn.out,
        inputReference=det_nn.out,
    )

    # Existing identification node
    # This still handles the visualizer annotations.
    id_node = pipeline.create(IdentificationNode).build(
        gather_data_node.out,
        csim=CSIM,
    )

    # Visualizer
    visualizer.addTopic("Video", det_nn.passthrough, "images")
    visualizer.addTopic("Objects", id_node.out, "images")

    print("Pipeline created.")

    # Start pipeline
    pipeline.start()

    # ---------------------------------------------------------
    # HOST QUEUE FOR RECOGNITION EMBEDDINGS
    # ---------------------------------------------------------
    # We read the recognition model output on the host.
    # During the first 10 seconds, these embeddings are saved as owner embeddings.
    # After that, new embeddings are compared against the saved owner gallery.
    # ---------------------------------------------------------

    rec_queue = rec_nn.out.createOutputQueue(maxSize=30, blocking=False)

    enrollment_start_time = time.time()
    owner_embeddings = []
    enrollment_done = False
    owner_gallery = None
    owner_centroid = None
    embedding_counter = 0

    print("\n[OWNER ENROLLMENT STARTED]")
    print(f"Keep ONLY the owner visible in the first {ENROLLMENT_SECONDS} seconds of the video.")
    print("Collecting owner embeddings...\n")

    while pipeline.isRunning():
        key = visualizer.waitKey(1)

        if key == ord("q"):
            print("Got q key. Exiting...")
            break

        # Read all available OSNet / recognition outputs
        while True:
            rec_msg = rec_queue.tryGet()

            if rec_msg is None:
                break

            embedding = extract_embedding_from_msg(rec_msg)

            if embedding is None:
                continue

            elapsed = time.time() - enrollment_start_time

            # -------------------------------------------------
            # PHASE A: OWNER ENROLLMENT
            # -------------------------------------------------
            # In the first 10 seconds, every embedding is assumed
            # to belong to the owner.
            # -------------------------------------------------

            if elapsed <= ENROLLMENT_SECONDS:
                embedding_counter += 1

                if embedding_counter % SAVE_EVERY_N_EMBEDDINGS == 0:
                    owner_embeddings.append(embedding)

                    print(
                        f"[ENROLLMENT] Saved embedding {len(owner_embeddings)} "
                        f"at t={elapsed:.2f}s"
                    )

                continue

            # -------------------------------------------------
            # SAVE OWNER GALLERY ONCE
            # -------------------------------------------------

            if not enrollment_done:
                owner_gallery, owner_centroid = save_owner_gallery(owner_embeddings)
                enrollment_done = True

                print("[MONITORING STARTED]")
                print(f"Owner threshold: {OWNER_THRESHOLD}\n")

            # -------------------------------------------------
            # PHASE B: MONITORING
            # -------------------------------------------------
            # After enrollment, every new embedding is compared
            # against the saved owner gallery.
            # -------------------------------------------------

            is_owner, score = match_owner(
                embedding,
                owner_gallery,
                threshold=OWNER_THRESHOLD,
            )

            identity = "OWNER" if is_owner else "UNKNOWN"

            print(f"[RE-ID] {identity} | score={score:.3f}")
