"""
hand_tracker.py
----------------
Stage 1 (Video Acquisition) + Stage 2 (Hand Detection & Landmark Extraction)
of the project methodology.

Wraps OpenCV video capture + Google MediaPipe Hand Landmarker to produce,
for every frame, the 21 3D hand landmarks (Ref. 14 - Zhang et al.,
MediaPipe Hands).

IMPORTANT (mediapipe version note)
-----------------------------------
Older MediaPipe tutorials (and most existing YouTube/blog references) use
the legacy API:  `mp.solutions.hands.Hands(...)`.
As of mediapipe 0.10.30+ / 1.0.x (the version `pip install mediapipe`
gives you today), that legacy `solutions` API has been REMOVED. Google
replaced it with the new unified **Tasks API**
(`mediapipe.tasks.python.vision.HandLandmarker`), which this file uses.

The Tasks API needs a small pre-trained model file, `hand_landmarker.task`
(~7-9 MB), which is downloaded automatically the first time you run the
project (see `download_model()` below / `setup.py`). You need an internet
connection once, the first time you run it; after that it's cached in
`assets/hand_landmarker.task` and works fully offline.

This keeps Stage 1/2 cleanly separated from Stage 3 (gesture
classification) and Stage 5 (game engine), matching the block diagram in
the report:

    Webcam -> Hand Landmark Detection (MediaPipe) -> Gesture Classification
           -> Command Mapping -> Game Engine -> Testing & Optimization
"""

import os
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
    "hand_landmarker.task",
)

# 21-point skeleton connections (same topology MediaPipe's legacy
# HAND_CONNECTIONS used) so we can draw the hand skeleton ourselves for
# the debug overlay, since the Tasks API doesn't ship a drawing helper.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]


def download_model(dest_path: str = DEFAULT_MODEL_PATH) -> str:
    """Download the MediaPipe hand-landmark model if it isn't already
    cached locally. Called automatically by HandTracker; can also be run
    standalone via `python setup.py`."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return dest_path

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"[SETUP] Downloading hand-landmark model to {dest_path} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, dest_path)
        print("[SETUP] Model downloaded successfully.")
    except Exception as e:
        raise RuntimeError(
            "Could not download the MediaPipe hand_landmarker.task model "
            f"({e}). You need an internet connection the first time you "
            f"run this project. You can also download it manually from:\n"
            f"  {MODEL_URL}\n"
            f"and place it at:\n  {dest_path}"
        ) from e
    return dest_path


@dataclass
class HandLandmarks:
    """21 (x, y, z) landmark points for a single detected hand, in both
    normalized (0-1) and pixel coordinates, plus which hand it is."""

    normalized: List[Tuple[float, float, float]]  # 21 x (x, y, z) in [0,1]
    pixel: List[Tuple[int, int]]                  # 21 x (x, y) in frame pixels
    handedness: str                               # "Left" or "Right"
    score: float                                  # detection confidence


class HandTracker:
    """Thin, testable wrapper around mediapipe.tasks.vision.HandLandmarker.

    Usage:
        tracker = HandTracker()
        cap = tracker.open_camera()
        frame = HandTracker.read_frame(cap)
        hands, annotated = tracker.process(frame, draw=True)
        tracker.close()
    """

    # Landmark indices referenced by name elsewhere for clarity.
    WRIST = 0
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_TIP = 20

    def __init__(
        self,
        max_num_hands: int = 1,
        detection_confidence: float = 0.6,
        tracking_confidence: float = 0.5,
        camera_index: int = 0,
        frame_width: int = 960,
        frame_height: int = 540,
        model_path: Optional[str] = None,
    ):
        model_path = model_path or download_model()

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height

    # ---------- Stage 1: Video Acquisition ----------

    def open_camera(self) -> cv2.VideoCapture:
        """Open the webcam using OpenCV, as per Stage 1 of the methodology."""
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {self.camera_index}. "
                "Check that a webcam is connected and not in use by "
                "another application."
            )
        return cap

    @staticmethod
    def read_frame(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
        """Grab one BGR frame from the capture device. Returns None on failure."""
        success, frame = cap.read()
        if not success:
            return None
        # Mirror the frame so movement feels natural to the player
        # (moving your hand right visually moves it right on screen).
        frame = cv2.flip(frame, 1)
        return frame

    # ---------- Stage 2: Hand Detection & Landmark Extraction ----------

    def process(
        self, frame_bgr: np.ndarray, draw: bool = True
    ) -> Tuple[List[HandLandmarks], np.ndarray]:
        """Run MediaPipe HandLandmarker on a BGR frame.

        Returns (list_of_hand_landmarks, annotated_bgr_frame_for_display)
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self.landmarker.detect(mp_image)

        annotated = frame_bgr  # draw directly on the BGR frame we display
        detected: List[HandLandmarks] = []

        if result.hand_landmarks:
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                normalized = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
                pixel = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

                label = "Right"
                score = 1.0
                if result.handedness and i < len(result.handedness):
                    category = result.handedness[i][0]
                    label = category.category_name
                    score = category.score

                detected.append(
                    HandLandmarks(
                        normalized=normalized,
                        pixel=pixel,
                        handedness=label,
                        score=score,
                    )
                )

                if draw:
                    self._draw_landmarks(annotated, pixel)

        return detected, annotated

    @staticmethod
    def _draw_landmarks(frame: np.ndarray, pixel_points: List[Tuple[int, int]]):
        """Manually draw the 21-point hand skeleton (Tasks API has no
        built-in drawing_utils equivalent)."""
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pixel_points[a], pixel_points[b], (0, 200, 0), 2)
        for i, (x, y) in enumerate(pixel_points):
            color = (0, 140, 255) if i == 0 else (0, 255, 255)
            cv2.circle(frame, (x, y), 4, color, -1)

    def close(self):
        self.landmarker.close()
