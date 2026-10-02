"""
setup.py
--------
Run this ONCE before your first `python main.py`:

    pip install -r requirements.txt
    python setup.py

It will:
  1. Verify OpenCV, MediaPipe, Pygame, NumPy import correctly and print
     their versions (useful to paste into your DA2 result-tabulation doc).
  2. Download the MediaPipe hand_landmarker.task model file into
     assets/ (requires internet the first time only; ~7-9 MB).
  3. Try opening your default webcam for one frame to confirm it works,
     so camera problems are caught before your demo, not during it.
"""

import sys


def check_imports():
    print("Checking dependencies...")
    versions = {}
    try:
        import cv2
        versions["opencv-python"] = cv2.__version__
    except ImportError:
        print("  [MISSING] opencv-python -> pip install opencv-python")
        sys.exit(1)

    try:
        import mediapipe as mp
        versions["mediapipe"] = mp.__version__
    except ImportError:
        print("  [MISSING] mediapipe -> pip install mediapipe")
        sys.exit(1)

    try:
        import pygame
        versions["pygame"] = pygame.ver
    except ImportError:
        print("  [MISSING] pygame -> pip install pygame")
        sys.exit(1)

    try:
        import numpy
        versions["numpy"] = numpy.__version__
    except ImportError:
        print("  [MISSING] numpy -> pip install numpy")
        sys.exit(1)

    for name, ver in versions.items():
        print(f"  [OK] {name} {ver}")
    return versions


def download_model():
    print("\nChecking / downloading MediaPipe hand-landmark model...")
    from src.hand_tracker import download_model as _dl, DEFAULT_MODEL_PATH
    path = _dl()
    print(f"  [OK] Model ready at: {path}")


def check_camera():
    print("\nChecking webcam access (index 0)...")
    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [WARN] Could not open webcam index 0.")
        print("         You can still run with --keyboard, or pass --camera N")
        print("         to main.py to try a different camera index.")
        return
    ok, frame = cap.read()
    cap.release()
    if ok and frame is not None:
        print(f"  [OK] Webcam working. Frame size: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print("  [WARN] Webcam opened but returned no frame.")


if __name__ == "__main__":
    print("=" * 60)
    print(" Gesture-Controlled Endless Runner -- Setup / Preflight Check")
    print("=" * 60)
    check_imports()
    download_model()
    check_camera()
    print("\nSetup complete. Run the game with:")
    print("    python main.py")
    print("or, without a webcam:")
    print("    python main.py --keyboard")
