"""
nodes.py – Owner-at-desk identification pipeline (single-file entry point).

Contains:
  - CLI argument parsing  (initialize_argparser)
  - OwnerIdentificationNode  (dai.node.HostNode subclass)
  - Pipeline construction and main loop

Pipeline overview
─────────────────
                          ┌─────────────────────────────────────────────────────┐
Camera / ReplayVideo      │  Full-resolution frames (REQ_WIDTH × REQ_HEIGHT)    │
       │                  └─────────────────────────────────────────────────────┘
       ▼
ImageManip (resize to det model input)
       │
ParsingNeuralNetwork  ← det_model_nn_archive  (SCRFD person/face detector)
       │ passthrough (det frames)
       │ out (ImgDetectionsExtended)
       ▼
ImgDetectionsBridge  ──► Script node ──────────────────────────► ImageManip (crop to det bbox)
       │                     ▲                                          │
       │             full-res frames                                    ▼
       │                                              ParsingNeuralNetwork ← rec_nn_archive
       │                                                       │            (OSNet / ArcFace)
       │                                                       │ out (NNData embeddings)
       └──────────────────────────────────────────────────────►│
                                                               ▼
                                                          GatherData
                                                               │
                                                               ▼
                                                  OwnerIdentificationNode
                                                  ┌──────────────────────────────┐
                                                  │ Phase 1 (0 … enroll_dur s):  │
                                                  │   buffer owner embeddings     │
                                                  │   label → '<base>_enrolling'  │
                                                  │                               │
                                                  │ Phase 2 (after enroll):       │
                                                  │   top-k average → owner_ref   │
                                                  │   label → 'owner' | 'unknown' │
                                                  └──────────────────────────────┘
                                                               │ out
                                                               ▼
                                                          Visualiser

Usage
─────
    # pose mode (default) – uses SCRFD person detector + OSNet
    python nodes.py

    # face mode – uses SCRFD face detector + ArcFace
    python nodes.py --identify face

    # customise enrollment window and top-k pool
    python nodes.py --enrollment_duration 45 --top_k 20

    # run on a pre-recorded video file
    python nodes.py --media_path /path/to/video.mp4

    # override cosine-similarity threshold
    python nodes.py --cos_similarity_threshold 0.75
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, GatherData, ImgDetectionsBridge
from depthai_nodes.node.utils import generate_script_content
from depthai_nodes import ImgDetectionsExtended


# ═══════════════════════════════════════════════════════════════════════════════
# § 1  CLI Arguments
# ═══════════════════════════════════════════════════════════════════════════════

def initialize_argparser():
    """Return (parser, args) for the owner-identification pipeline."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-d", "--device",
        help="Optional name, DeviceID or IP of the camera to connect to.",
        required=False, default=None, type=str,
    )
    parser.add_argument(
        "-fps", "--fps_limit",
        help="FPS limit for the model runtime.",
        required=False, default=None, type=int,
    )
    parser.add_argument(
        "-media", "--media_path",
        help=(
            "Path to the media file you aim to run the model on. "
            "If not set, the model will run on the camera input."
        ),
        required=False, default=None, type=str,
    )
    parser.add_argument(
        "-id", "--identify",
        help="Whether to run pose or face re-identification.",
        required=False, default="pose", choices=["pose", "face"], type=str,
    )
    parser.add_argument(
        "-cos", "--cos_similarity_threshold",
        help=(
            "Cosine similarity between object embeddings above which detections "
            "are considered as belonging to the owner."
        ),
        required=False, default=None, type=float,
    )
    parser.add_argument(
        "-enroll", "--enrollment_duration",
        help=(
            "Duration in seconds for the enrollment phase, during which the person "
            "in front of the camera is treated as the 'owner'. "
            "After this period, new people are classified as 'owner' or 'unknown'."
        ),
        required=False, default=30, type=int,
    )
    parser.add_argument(
        "-topk", "--top_k",
        help=(
            "Number of highest-quality embeddings (ranked by L2-norm) to average "
            "when building the owner reference embedding."
        ),
        required=False, default=10, type=int,
    )
    return parser, parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# § 2  OwnerIdentificationNode
# ═══════════════════════════════════════════════════════════════════════════════

# Default constants (can be overridden via build() / setters)
_DEFAULT_ENROLLMENT_DURATION: int = 30
_DEFAULT_TOP_K: int = 10


class OwnerIdentificationNode(dai.node.HostNode):
    """Re-identifies people relative to an auto-enrolled 'owner' reference.

    Phase 1 – ENROLLMENT (first ``enrollment_duration`` seconds)
        Every detection that arrives during this window is assumed to belong
        to the 'owner' sitting at the desk.  Embeddings are buffered and
        detections are labelled ``'<basename>_enrolling'``.

    Phase 2 – IDENTIFICATION (after enrollment)
        The top-k embeddings (ranked by L2-norm) are mean-pooled and
        L2-normalised into a single owner reference vector.  Each new
        detection is then classified:

        * ``cos_sim(embedding, owner_ref) >= csim``  →  ``"owner"``
        * otherwise                                  →  ``"unknown_N"``

    Attributes
    ----------
    _cos_sim_threshold : float
        Minimum cosine similarity to classify a detection as 'owner'.
    _label_basename : str
        Prefix for generated labels (e.g. ``"pose"`` → ``"pose_enrolling"``,
        ``"owner"``, ``"unknown_0"`` …).
    _enrollment_duration : int
        Seconds to spend in the enrollment phase.
    _top_k : int
        Size of the top-norm pool used for the owner reference.
    """

    def __init__(self) -> None:
        super().__init__()

        # ── configurable hyper-parameters ──────────────────────────────────
        self._cos_sim_threshold: float = 0.5
        self._label_basename: str = "person"
        self._enrollment_duration: int = _DEFAULT_ENROLLMENT_DURATION
        self._top_k: int = _DEFAULT_TOP_K

        # ── runtime state ───────────────────────────────────────────────────
        self._start_time: float | None = None   # lazily set on first message
        self._phase: str = "enroll"             # "enroll" | "identify"
        self._owner_embeddings: list[np.ndarray] = []
        self._owner_ref: np.ndarray | None = None  # unit-normalised reference
        self._unknown_count: int = 0            # monotonically increasing id

    # ── setters ───────────────────────────────────────────────────────────────

    def setCosSimThreshold(self, csim: float) -> None:
        if not isinstance(csim, float):
            raise TypeError("Cosine similarity threshold must be a float.")
        if not 0.0 <= csim <= 1.0:
            raise ValueError("Cosine similarity threshold must be between 0 and 1.")
        self._cos_sim_threshold = csim

    def setLabelBasename(self, label_basename: str) -> None:
        if not isinstance(label_basename, str):
            raise TypeError("Label basename must be a string.")
        self._label_basename = label_basename

    def setEnrollmentDuration(self, seconds: int) -> None:
        if not isinstance(seconds, int) or seconds <= 0:
            raise ValueError("enrollment_duration must be a positive integer.")
        self._enrollment_duration = seconds

    def setTopK(self, k: int) -> None:
        if not isinstance(k, int) or k <= 0:
            raise ValueError("top_k must be a positive integer.")
        self._top_k = k

    # ── build ─────────────────────────────────────────────────────────────────

    def build(
        self,
        gather_data_msg,
        csim: float = 0.5,
        label_basename: str = "person",
        enrollment_duration: int = _DEFAULT_ENROLLMENT_DURATION,
        top_k: int = _DEFAULT_TOP_K,
    ) -> "OwnerIdentificationNode":
        """Wire inputs and configure the node.

        Parameters
        ----------
        gather_data_msg :
            Output of a ``GatherData`` node (pairs detections + embeddings).
        csim : float
            Cosine-similarity threshold for 'owner' classification.
        label_basename : str
            Prefix for detection labels.
        enrollment_duration : int
            Seconds to collect owner embeddings (phase 1 duration).
        top_k : int
            Number of best embeddings to average into the owner reference.
        """
        self.link_args(gather_data_msg)
        self.setCosSimThreshold(csim)
        self.setLabelBasename(label_basename)
        self.setEnrollmentDuration(enrollment_duration)
        self.setTopK(top_k)
        return self

    # ── process ───────────────────────────────────────────────────────────────

    def process(self, gather_data_msg) -> None:
        """Called once per synchronised (detections, embeddings) pair."""

        # Lazily start the enrollment clock on the very first message so we
        # don't lose time while the pipeline is warming up.
        if self._start_time is None:
            self._start_time = time.monotonic()
            print(
                f"[OwnerID] Enrollment started – recording owner embeddings "
                f"for {self._enrollment_duration}s …"
            )

        # Check for phase transition before processing this batch.
        elapsed = time.monotonic() - self._start_time
        if self._phase == "enroll" and elapsed >= self._enrollment_duration:
            self._finalize_enrollment()

        # Unpack the GatherData message.
        dets_msg: ImgDetectionsExtended = gather_data_msg.reference_data
        assert isinstance(dets_msg, ImgDetectionsExtended), (
            "reference_data must be ImgDetectionsExtended"
        )
        rec_msg_list: list[dai.NNData] = gather_data_msg.gathered
        assert isinstance(rec_msg_list, list)
        assert all(isinstance(msg, dai.NNData) for msg in rec_msg_list)

        # Label each detection according to the current phase.
        for detection, rec in zip(dets_msg.detections, rec_msg_list):
            embedding: np.ndarray = rec.getTensor("output", dequantize=True)

            if self._phase == "enroll":
                self._owner_embeddings.append(embedding)
                detection.label_name = f"{self._label_basename}_enrolling"
            else:
                detection.label_name = self._classify(embedding)

        self.out.send(dets_msg)

    # ── private helpers ───────────────────────────────────────────────────────

    def _finalize_enrollment(self) -> None:
        """Build the owner reference from the top-k collected embeddings."""
        self._phase = "identify"
        n = len(self._owner_embeddings)

        if n == 0:
            print(
                "[OwnerID] WARNING: no embeddings collected during enrollment. "
                "All detections will be labelled 'unknown'."
            )
            return

        # Rank by L2-norm; higher norm correlates with richer / more confident
        # embeddings (blurry or partially-occluded crops produce lower norms).
        norms = np.array([np.linalg.norm(e) for e in self._owner_embeddings])
        k = min(self._top_k, n)
        top_indices = np.argsort(norms)[::-1][:k]              # descending
        top_embeddings = np.stack(
            [self._owner_embeddings[i] for i in top_indices], axis=0
        )                                                        # (k, D)

        mean_emb = top_embeddings.mean(axis=0)                  # (D,)
        self._owner_ref = mean_emb / (np.linalg.norm(mean_emb) + 1e-9)

        print(
            f"[OwnerID] Enrollment complete. "
            f"Owner reference built from top-{k} / {n} embeddings."
        )
        self._owner_embeddings.clear()  # free memory

    def _classify(self, embedding: np.ndarray) -> str:
        """Return 'owner' or 'unknown_N' for a single embedding vector."""
        if self._owner_ref is None:
            label = f"unknown_{self._unknown_count}"
            self._unknown_count += 1
            return label

        sim = self._cosine_similarity(embedding, self._owner_ref)
        if sim >= self._cos_sim_threshold:
            return "owner"

        label = f"unknown_{self._unknown_count}"
        self._unknown_count += 1
        return label

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two 1-D vectors; 0.0 for zero vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# ═══════════════════════════════════════════════════════════════════════════════
# § 3  Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

# Camera capture resolution – larger than the detector needs so we retain
# enough detail for the second-stage recognition crop.
REQ_WIDTH, REQ_HEIGHT = 768, 768

# Default cosine-similarity thresholds per mode.
_DEFAULT_CSIM = {"pose": 0.8, "face": 0.1}


def build_and_run() -> None:
    """Construct the DAI pipeline and run the main loop."""

    _, args = initialize_argparser()

    # ── device ────────────────────────────────────────────────────────────
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
            f"\nFPS limit set to {args.fps_limit} for {platform}. "
            "Use --fps_limit to override.\n"
        )

    # ── cosine-similarity threshold ───────────────────────────────────────
    CSIM = _DEFAULT_CSIM.get(args.identify, 0.5)
    if args.cos_similarity_threshold is not None:
        CSIM = args.cos_similarity_threshold

    with dai.Pipeline(device) as pipeline:
        print("Creating pipeline …")

        # ── model descriptions ─────────────────────────────────────────────
        if args.identify == "pose":
            det_model_description = dai.NNModelDescription.fromYamlFile(
                f"scrfd_person_detection_25g.{platform}.yaml"
            )
            rec_model_description = dai.NNModelDescription.fromYamlFile(
                f"osnet_imagenet.{platform}.yaml"
            )
        elif args.identify == "face":
            det_model_description = dai.NNModelDescription.fromYamlFile(
                f"scrfd_face_detection_10g.{platform}.yaml"
            )
            rec_model_description = dai.NNModelDescription.fromYamlFile(
                f"arcface_lfw.{platform}.yaml"
            )
        else:
            raise ValueError(f"Unknown --identify option: '{args.identify}'")

        det_model_nn_archive = dai.NNArchive(dai.getModelFromZoo(det_model_description))
        rec_nn_archive = dai.NNArchive(dai.getModelFromZoo(rec_model_description))

        # ── input: live camera or video replay ────────────────────────────
        if args.media_path:
            replay = pipeline.create(dai.node.ReplayVideo)
            replay.setReplayVideoFile(Path(args.media_path))
            replay.setOutFrameType(frame_type)
            replay.setLoop(True)
            if args.fps_limit:
                replay.setFps(args.fps_limit)
            replay.setSize(REQ_WIDTH, REQ_HEIGHT)
            input_node_out = replay.out
        else:
            cam = pipeline.create(dai.node.Camera).build()
            cam_out = cam.requestOutput(
                size=(REQ_WIDTH, REQ_HEIGHT),
                type=frame_type,
                fps=args.fps_limit,
            )
            input_node_out = cam_out

        # ── resize to detector input resolution ───────────────────────────
        resize_node = pipeline.create(dai.node.ImageManip)
        resize_node.setMaxOutputFrameSize(REQ_WIDTH * REQ_HEIGHT * 3)
        resize_node.initialConfig.setOutputSize(
            det_model_nn_archive.getInputWidth(),
            det_model_nn_archive.getInputHeight(),
        )
        resize_node.initialConfig.setReusePreviousImage(False)
        resize_node.inputImage.setBlocking(True)
        input_node_out.link(resize_node.inputImage)

        # ── stage 1: person / face detection ──────────────────────────────
        det_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
            resize_node.out, det_model_nn_archive
        )

        # ── convert detections for the Script node ────────────────────────
        # TODO: remove ImgDetectionsBridge once ImgDetectionsExtended is
        #       natively supported by the Script node routing logic.
        det_bridge = pipeline.create(ImgDetectionsBridge).build(det_nn.out)

        # ── Script node: generate ImageManip configs per detected bbox ─────
        script_node = pipeline.create(dai.node.Script)
        det_bridge.out.link(script_node.inputs["det_in"])
        input_node_out.link(script_node.inputs["preview"])
        script_node.setScript(
            generate_script_content(
                resize_width=rec_nn_archive.getInputWidth(),
                resize_height=rec_nn_archive.getInputHeight(),
            )
        )

        # ── crop each bbox to the recognition model's input size ──────────
        crop_node = pipeline.create(dai.node.ImageManip)
        crop_node.initialConfig.setOutputSize(
            rec_nn_archive.getInputWidth(),
            rec_nn_archive.getInputHeight(),
        )
        crop_node.inputConfig.setWaitForMessage(True)
        script_node.outputs["manip_cfg"].link(crop_node.inputConfig)
        script_node.outputs["manip_img"].link(crop_node.inputImage)

        # ── stage 2: recognition (embedding extraction) ───────────────────
        rec_nn: ParsingNeuralNetwork = pipeline.create(ParsingNeuralNetwork).build(
            crop_node.out, rec_nn_archive
        )

        # ── synchronise detections with their embeddings ──────────────────
        gather_data_node = pipeline.create(GatherData).build(args.fps_limit)
        rec_nn.out.link(gather_data_node.input_data)
        det_nn.out.link(gather_data_node.input_reference)

        # ── owner identification ───────────────────────────────────────────
        id_node = pipeline.create(OwnerIdentificationNode).build(
            gather_data_node.out,
            csim=CSIM,
            label_basename=args.identify,
            enrollment_duration=args.enrollment_duration,
            top_k=args.top_k,
        )

        # ── visualiser topics ──────────────────────────────────────────────
        visualizer.addTopic("Video", det_nn.passthrough, "images")
        visualizer.addTopic("Objects", id_node.out, "images")

        print(
            f"Pipeline created.\n"
            f"  Mode              : {args.identify}\n"
            f"  Cosine threshold  : {CSIM}\n"
            f"  Enrollment window : {args.enrollment_duration}s\n"
            f"  Top-k pool        : {args.top_k}\n"
            f"  FPS limit         : {args.fps_limit}\n"
        )

        pipeline.start()
        while pipeline.isRunning():
            key = visualizer.waitKey(1)
            if key == ord("q"):
                print("Got 'q'. Exiting …")
                break


# ═══════════════════════════════════════════════════════════════════════════════
# § 4  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    build_and_run()
