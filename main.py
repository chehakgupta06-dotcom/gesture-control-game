"""
main.py
-------
Entry point that wires together every stage of the methodology:

    Webcam (Stage 1) -> MediaPipe landmarks (Stage 2)
        -> Gesture classification (Stage 3)
        -> Command mapping (Stage 4)
        -> Pygame endless runner (Stage 5)

Two windows are shown side by side conceptually:
  1. The Pygame game window (what the player watches while playing).
  2. An OpenCV debug window showing the webcam feed with hand landmarks
     drawn on it and the currently recognized gesture printed on screen
     -- this is exactly the kind of view you want on your presentation
     slide/video to *show the examiner the recognition working live*.

Run modes
---------
    python main.py                 # gesture control via webcam (default)
    python main.py --keyboard      # keyboard-only fallback (no camera needed)
    python main.py --no-debug-cam  # hide the OpenCV landmark debug window
    python main.py --camera 1      # use a different camera index

Controls (keyboard fallback / always available as backup):
    A / Left arrow   -> move left
    D / Right arrow  -> move right
    W / Up / Space   -> jump
    S / Down         -> duck
    Enter            -> start the run / resume from pause / play again
    P                -> pause (stop) the run
    R                -> restart after game over
    Esc / Q          -> quit

Gestures (webcam mode):
    Open palm raised          -> jump
    Fist lowered               -> duck
    Hand moved left / right    -> change lane
    Thumbs-up (held)           -> START / resume the run
    Peace sign (held)          -> STOP / pause the run
"""

import argparse
import sys
import time

import cv2
import pygame

from src.gesture_classifier import Gesture, GestureClassifier
from src.game import EndlessRunnerGame, SCREEN_W, SCREEN_H, FPS
from src.hand_tracker import HandTracker


def parse_args():
    p = argparse.ArgumentParser(description="Gesture-Controlled Endless Runner")
    p.add_argument("--keyboard", action="store_true", help="disable webcam, use keyboard only")
    p.add_argument("--no-debug-cam", action="store_true", help="hide OpenCV debug window")
    p.add_argument("--camera", type=int, default=0, help="webcam device index")
    p.add_argument("--seed", type=int, default=None, help="RNG seed (useful for demos)")
    return p.parse_args()


def map_gesture_to_command(gesture: Gesture, game: EndlessRunnerGame):
    """Stage 4: Gesture -> in-game command mapping.

    This is intentionally a single, small switch so it is trivial to point
    to during a viva/demo and say 'here is Stage 4 of the report'.
    """
    if gesture == Gesture.LEFT:
        game.cmd_left()
    elif gesture == Gesture.RIGHT:
        game.cmd_right()
    elif gesture == Gesture.JUMP:
        game.cmd_jump()
    elif gesture == Gesture.DUCK:
        game.cmd_duck()
    elif gesture == Gesture.START:
        game.cmd_start()
    elif gesture == Gesture.STOP:
        game.cmd_stop()


def run(args):
    pygame.init()
    pygame.display.set_caption("Gesture-Controlled Endless Runner")
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()

    game = EndlessRunnerGame(screen=screen, seed=args.seed)

    tracker = None
    classifier = None
    cap = None

    use_camera = not args.keyboard
    if use_camera:
        try:
            tracker = HandTracker(camera_index=args.camera)
            classifier = GestureClassifier()
            cap = tracker.open_camera()
            print("[INFO] Camera opened. Gesture control ACTIVE.")
        except RuntimeError as e:
            print(f"[WARN] {e}")
            print("[WARN] Falling back to keyboard-only control.")
            use_camera = False

    # Simple FPS / latency instrumentation for Stage 6 (Testing & Evaluation).
    frame_times = []
    gesture_log = []  # (timestamp, gesture) - can be dumped for the report

    running = True
    try:
        while running:
            dt_ms = clock.tick(FPS)
            loop_start = time.time()

            # ---- input: keyboard (always available) ----
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_r:
                        game.cmd_restart()
                    elif event.key in (pygame.K_a, pygame.K_LEFT):
                        game.cmd_left()
                    elif event.key in (pygame.K_d, pygame.K_RIGHT):
                        game.cmd_right()
                    elif event.key in (pygame.K_w, pygame.K_UP, pygame.K_SPACE):
                        game.cmd_jump()
                    elif event.key in (pygame.K_s, pygame.K_DOWN):
                        game.cmd_duck()
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        game.cmd_start()
                    elif event.key == pygame.K_p:
                        game.cmd_stop()

            # ---- input: gesture pipeline (Stages 1-4) ----
            if use_camera:
                frame = HandTracker.read_frame(cap)
                if frame is not None:
                    hands, annotated = tracker.process(frame, draw=not args.no_debug_cam)
                    hand = hands[0] if hands else None
                    gesture = classifier.update(hand)

                    if gesture != Gesture.NONE:
                        gesture_log.append((time.time(), gesture.value))
                        map_gesture_to_command(gesture, game)

                    if not args.no_debug_cam:
                        label = gesture.value.upper() if gesture != Gesture.NONE else "-"
                        cv2.putText(
                            annotated,
                            f"Gesture: {label}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 0),
                            2,
                        )
                        cv2.putText(
                            annotated,
                            GestureClassifier.landmarks_debug_line(hand),
                            (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (200, 200, 200),
                            1,
                        )
                        cv2.imshow("Gesture Debug (MediaPipe landmarks)", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            running = False

            # ---- update + render game ----
            game.update(dt_ms)
            game.draw(screen)
            pygame.display.flip()

            frame_times.append(time.time() - loop_start)

    finally:
        if cap is not None:
            cap.release()
        if tracker is not None:
            tracker.close()
        cv2.destroyAllWindows()
        pygame.quit()

        # ---- Stage 6 summary printed on exit: quick perf snapshot ----
        if frame_times:
            avg_ms = sum(frame_times) / len(frame_times) * 1000
            fps_est = 1000.0 / avg_ms if avg_ms > 0 else 0
            print(f"[STATS] Avg loop time: {avg_ms:.2f} ms  (~{fps_est:.1f} FPS)")
        if gesture_log:
            print(f"[STATS] Gestures recognized this session: {len(gesture_log)}")


if __name__ == "__main__":
    run(parse_args())
