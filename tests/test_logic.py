"""
tests/test_logic.py
--------------------
Automated tests for Stage 3 (gesture classification) and Stage 5 (game
engine), decoupled from the camera so they run anywhere, instantly, and
without a webcam. Use these results as evidence in your DA2
result-tabulation document (Stage 6: Testing & Evaluation).

Run with:
    python -m pytest tests/ -v
or, without pytest:
    python tests/test_logic.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gesture_classifier import GestureClassifier, Gesture
from src.hand_tracker import HandLandmarks


# ---------------------------------------------------------------------------
# Helpers to fabricate synthetic MediaPipe-style landmarks for testing
# without needing a real camera or hand in front of it.
# ---------------------------------------------------------------------------

def make_hand(wrist_x, wrist_y, open_palm=True, fingers=None):
    """fingers, when given, overrides individual finger extension for
    index/middle/ring/pinky (dict of name->bool) and the thumb mode
    ('side_out' = sideways/open-palm thumb, 'up' = vertical thumbs-up,
    'in' = curled). When `fingers` is None, falls back to the simple
    open_palm/fist behaviour the original tests used."""
    pts = [(0.0, 0.0, 0.0)] * 21
    pts[0] = (wrist_x, wrist_y, 0.0)

    if fingers is None:
        fingers = {
            "index": open_palm, "middle": open_palm,
            "ring": open_palm, "pinky": open_palm,
            "thumb": "side_out" if open_palm else "in",
        }

    finger_joints = {"index": (8, 6, 5), "middle": (12, 10, 9), "ring": (16, 14, 13), "pinky": (20, 18, 17)}
    for name, (tip, pip, mcp) in finger_joints.items():
        if fingers.get(name):
            pts[mcp] = (wrist_x, wrist_y, 0)
            pts[pip] = (wrist_x, wrist_y - 0.05, 0)
            pts[tip] = (wrist_x, wrist_y - 0.15, 0)
        else:
            pts[mcp] = (wrist_x, wrist_y, 0)
            pts[pip] = (wrist_x, wrist_y - 0.01, 0)
            pts[tip] = (wrist_x, wrist_y - 0.01, 0)

    thumb_mode = fingers.get("thumb", "in")
    if thumb_mode == "up":
        pts[2] = (wrist_x, wrist_y - 0.02, 0)
        pts[3] = (wrist_x, wrist_y - 0.08, 0)
        pts[4] = (wrist_x, wrist_y - 0.15, 0)
    elif thumb_mode == "side_out":
        pts[2] = (wrist_x - 0.02, wrist_y, 0)
        pts[3] = (wrist_x - 0.05, wrist_y, 0)
        pts[4] = (wrist_x - 0.1, wrist_y, 0)
    else:  # curled in
        pts[2] = (wrist_x - 0.01, wrist_y, 0)
        pts[3] = (wrist_x - 0.01, wrist_y, 0)
        pts[4] = (wrist_x - 0.01, wrist_y, 0)

    return HandLandmarks(normalized=pts, pixel=[(0, 0)] * 21, handedness="Right", score=0.9)


def settle_baseline(clf, x=0.5, y=0.5, open_palm=False, frames=20):
    for _ in range(frames):
        clf.update(make_hand(x, y, open_palm))


# ---------------------------------------------------------------------------
# Gesture classification tests (Stage 3)
# ---------------------------------------------------------------------------

def test_jump_gesture_fires():
    clf = GestureClassifier()
    settle_baseline(clf, open_palm=False)
    fired = [clf.update(make_hand(0.5, 0.35, open_palm=True)) for _ in range(6)]
    assert Gesture.JUMP in fired, f"expected JUMP, got {fired}"


def test_duck_gesture_fires():
    clf = GestureClassifier()
    settle_baseline(clf, open_palm=True)
    fired = [clf.update(make_hand(0.5, 0.65, open_palm=False)) for _ in range(6)]
    assert Gesture.DUCK in fired, f"expected DUCK, got {fired}"


def test_left_gesture_fires():
    clf = GestureClassifier()
    settle_baseline(clf, open_palm=False)
    fired = [clf.update(make_hand(0.3, 0.5, open_palm=False)) for _ in range(6)]
    assert Gesture.LEFT in fired, f"expected LEFT, got {fired}"


def test_right_gesture_fires():
    clf = GestureClassifier()
    settle_baseline(clf, open_palm=False)
    fired = [clf.update(make_hand(0.7, 0.5, open_palm=False)) for _ in range(6)]
    assert Gesture.RIGHT in fired, f"expected RIGHT, got {fired}"


def test_no_hand_returns_none():
    clf = GestureClassifier()
    settle_baseline(clf)
    assert clf.update(None) == Gesture.NONE


def test_thumbs_up_fires_start():
    clf = GestureClassifier()
    settle_baseline(clf, open_palm=False)
    fired = [
        clf.update(make_hand(0.5, 0.5, fingers={"thumb": "up"}))
        for _ in range(10)
    ]
    assert Gesture.START in fired, f"expected START (thumbs-up), got {fired}"


def test_peace_sign_fires_stop():
    clf = GestureClassifier()
    settle_baseline(clf, open_palm=False)
    fired = [
        clf.update(make_hand(0.5, 0.5, fingers={"index": True, "middle": True}))
        for _ in range(10)
    ]
    assert Gesture.STOP in fired, f"expected STOP (peace sign), got {fired}"


def test_start_stop_do_not_interfere_with_jump():
    """Regression check: adding the two new static-pose gestures must not
    change how the pre-existing open-palm JUMP gesture is recognized."""
    clf = GestureClassifier()
    settle_baseline(clf, open_palm=False)
    fired = [clf.update(make_hand(0.5, 0.35, open_palm=True)) for _ in range(6)]
    assert Gesture.JUMP in fired, f"expected JUMP still works, got {fired}"
    assert Gesture.START not in fired
    assert Gesture.STOP not in fired


def test_gesture_debounce_does_not_spam():
    """A held gesture should fire once, then respect the cooldown, not
    fire every single frame (this is what stops false-trigger spam)."""
    clf = GestureClassifier()
    settle_baseline(clf, open_palm=False)
    fires = 0
    for _ in range(30):
        g = clf.update(make_hand(0.5, 0.35, open_palm=True))
        if g != Gesture.NONE:
            fires += 1
    assert fires <= 2, f"gesture fired too many times without hand returning to neutral: {fires}"


# ---------------------------------------------------------------------------
# Game engine tests (Stage 5) -- headless, no display required
# ---------------------------------------------------------------------------

def _make_game(auto_start=True):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame
    from src.game import EndlessRunnerGame

    if not pygame.get_init():
        pygame.init()
    screen = pygame.display.set_mode((900, 600))
    game = EndlessRunnerGame(screen=screen, seed=42)
    if auto_start:
        game.cmd_start()  # most tests want an already-running game
    return game


def test_game_waits_for_start_gesture():
    """Before a START command, the world should not scroll, spawn, or
    accept movement commands -- this is the new waiting-to-start state."""
    game = _make_game(auto_start=False)
    assert game.started is False
    game.cmd_left()  # should be ignored -- controls inactive before start
    assert game.player.lane == 1
    game.update(16)
    assert game.distance_m == 0.0
    assert game.score == 0.0

    game.cmd_start()
    assert game.started is True
    game.cmd_left()
    assert game.player.lane == 0


def test_stop_pauses_and_start_resumes():
    game = _make_game(auto_start=True)
    for _ in range(30):
        game.update(16)
    distance_before = game.distance_m

    game.cmd_stop()
    assert game.paused is True
    for _ in range(30):
        game.update(16)
    assert game.distance_m == distance_before, "world should not advance while paused"

    game.cmd_left()
    assert game.player.lane == 1, "movement commands should be ignored while paused"

    game.cmd_start()
    assert game.paused is False
    game.update(16)
    assert game.distance_m > distance_before, "world should resume advancing after START"


def test_lane_change_and_clamping():
    game = _make_game()
    assert game.player.lane == 1
    game.cmd_left()
    assert game.player.lane == 0
    game.cmd_left()  # already at leftmost lane, should clamp
    assert game.player.lane == 0
    game.cmd_right()
    game.cmd_right()
    game.cmd_right()  # should clamp at rightmost lane (2)
    assert game.player.lane == 2


def test_jump_returns_to_running_state():
    from src.game import PlayerState

    game = _make_game()  # auto_start=True, so controls are already active
    game.cmd_jump()
    assert game.player.state == PlayerState.JUMPING
    for _ in range(60):  # simulate ~1 second at 60fps
        game.player.update(16, running=True)
    assert game.player.state == PlayerState.RUNNING


def test_game_runs_many_frames_without_crashing():
    game = _make_game()
    import random

    random.seed(1)
    for i in range(1500):
        game.update(16)
        game.draw()
        if i % 250 == 0:
            choice = random.choice(["left", "right", "jump", "duck", None])
            {
                "left": game.cmd_left,
                "right": game.cmd_right,
                "jump": game.cmd_jump,
                "duck": game.cmd_duck,
                None: lambda: None,
            }[choice]()
    assert game.score >= 0


def test_game_over_triggers_after_losing_all_lives():
    game = _make_game()
    for _ in range(game.lives):
        game._on_hit()
    assert game.game_over is True

    game.cmd_restart()
    assert game.game_over is False
    assert game.lives == 3


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
    sys.exit(1 if failed else 0)
