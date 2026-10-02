"""
gesture_classifier.py
----------------------
Stage 3 (Gesture Classification) of the project methodology.

Implements the geometric / rule-based classifier described in the report:
"relative finger positions, angles, and vertical/horizontal displacement
of the wrist" are used to classify the hand pose into one of four discrete
commands: JUMP, DUCK, LEFT, RIGHT (plus NONE / IDLE when no clear gesture
is present).

This mirrors the lightweight, CPU-friendly approach validated in
Refs. 11, 12, 17, 18 (webcam -> discrete command mapping) as an alternative
to training a custom CNN/LSTM classifier (Refs. 5, 21, 24, 26), which the
report lists as an optional upgrade path (see `docs/UPGRADE_TO_ML.md`).

Design notes
------------
- All thresholds are expressed in *normalized* MediaPipe coordinates
  (0.0-1.0 relative to frame width/height), so the logic is resolution
  independent.
- A small temporal smoothing / debounce layer sits on top of the raw
  per-frame classification so that a single noisy frame can't spam
  commands, and a gesture must be "held" briefly to fire (this directly
  addresses Stage 6's requirement to "minimize false triggers").
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, List, Optional, Tuple

from .hand_tracker import HandLandmarks


class Gesture(Enum):
    NONE = "none"
    JUMP = "jump"
    DUCK = "duck"
    LEFT = "left"
    RIGHT = "right"
    START = "start"   # thumbs-up: begin / resume the run
    STOP = "stop"      # peace-sign (index+middle only): pause the run


@dataclass
class GestureConfig:
    # --- finger-extension thresholds ---
    # A finger is "extended" if its tip is above (smaller y) its MCP joint
    # by at least this fraction of the hand's bounding-box height.
    extension_margin: float = 0.03

    # --- open-palm / fist detection ---
    open_palm_min_fingers: int = 4   # >=4 extended fingers => open palm
    fist_max_fingers: int = 1        # <=1 extended fingers => fist

    # --- static pose gestures (START / STOP) ---
    # These are pure hand-SHAPE gestures (no displacement needed), so they
    # double as deliberate, low-ambiguity "control" gestures distinct from
    # the movement gestures above.
    thumb_up_margin: float = 0.05    # how far above the wrist the thumb tip must be
    start_stop_hold_frames_required: int = 6  # held longer than move gestures on purpose

    # --- wrist displacement thresholds (relative to a rolling baseline) ---
    vertical_jump_delta: float = 0.09   # wrist moves up by this much (normalized y)
    vertical_duck_delta: float = 0.07   # open palm held low in frame
    horizontal_move_delta: float = 0.09 # wrist moves left/right by this much

    # --- temporal smoothing ---
    history_len: int = 5          # frames of history for majority vote
    hold_frames_required: int = 3 # consecutive votes needed to confirm a gesture
    cooldown_frames: int = 8      # frames to wait after firing before re-firing
    baseline_alpha: float = 0.05  # EMA smoothing factor for the neutral wrist position


class GestureClassifier:
    """Stateful classifier: call `update(hand)` once per frame with the
    detected hand (or None), get back a confirmed `Gesture` (or NONE)."""

    def __init__(self, config: Optional[GestureConfig] = None):
        self.cfg = config or GestureConfig()
        self._votes: Deque[Gesture] = deque(maxlen=self.cfg.history_len)
        self._baseline_x: Optional[float] = None
        self._baseline_y: Optional[float] = None
        self._cooldown = 0
        self._last_confirmed = Gesture.NONE
        self._hold_count = 0
        self._hold_gesture = Gesture.NONE

    def reset(self):
        self._votes.clear()
        self._baseline_x = None
        self._baseline_y = None
        self._cooldown = 0
        self._last_confirmed = Gesture.NONE
        self._hold_count = 0
        self._hold_gesture = Gesture.NONE

    # ---------- geometric helpers ----------

    @staticmethod
    def _finger_extended(pts, tip_idx: int, pip_idx: int, mcp_idx: int, margin: float) -> bool:
        """A finger counts as extended if the tip is meaningfully above
        (smaller normalized y than) both its PIP and MCP joints."""
        tip_y = pts[tip_idx][1]
        pip_y = pts[pip_idx][1]
        mcp_y = pts[mcp_idx][1]
        return (mcp_y - tip_y) > margin and (pip_y - tip_y) > margin * 0.5

    @staticmethod
    def _thumb_extended(pts, margin: float) -> bool:
        """Thumb extension is measured horizontally (thumb moves sideways,
        not up/down like the other four fingers)."""
        tip_x = pts[4][0]
        ip_x = pts[3][0]
        mcp_x = pts[2][0]
        wrist_x = pts[0][0]
        # Distance of tip from the palm center line vs. the MCP joint.
        return abs(tip_x - wrist_x) > abs(mcp_x - wrist_x) + margin

    def _finger_states(self, pts) -> dict:
        """Returns a dict of individual finger extension booleans, used both
        for the open-palm/fist counting logic and for the START/STOP static
        pose checks below."""
        cfg = self.cfg
        return {
            "thumb": self._thumb_extended(pts, cfg.extension_margin),
            "index": self._finger_extended(pts, 8, 6, 5, cfg.extension_margin),
            "middle": self._finger_extended(pts, 12, 10, 9, cfg.extension_margin),
            "ring": self._finger_extended(pts, 16, 14, 13, cfg.extension_margin),
            "pinky": self._finger_extended(pts, 20, 18, 17, cfg.extension_margin),
        }

    def _count_extended_fingers(self, pts) -> int:
        return sum(1 for v in self._finger_states(pts).values() if v)

    def _is_thumbs_up(self, pts) -> bool:
        """Thumb points clearly UPWARD (not sideways) and every other
        finger is curled into the palm. This is checked independently of
        the sideways thumb-extension test used for open-palm counting,
        because a thumbs-up thumb points vertically, not laterally."""
        states = self._finger_states(pts)
        if states["index"] or states["middle"] or states["ring"] or states["pinky"]:
            return False
        wrist_y = pts[0][1]
        thumb_tip_y = pts[4][1]
        thumb_mcp_y = pts[2][1]
        points_up = (wrist_y - thumb_tip_y) > self.cfg.thumb_up_margin
        above_mcp = (thumb_mcp_y - thumb_tip_y) > self.cfg.thumb_up_margin * 0.5
        return points_up and above_mcp

    def _is_peace_sign(self, pts) -> bool:
        """Index + middle extended, ring + pinky curled (thumb ignored) --
        the classic 'V' / peace sign, used here as the STOP gesture."""
        states = self._finger_states(pts)
        return states["index"] and states["middle"] and not states["ring"] and not states["pinky"]

    def _update_baseline(self, wrist_x: float, wrist_y: float):
        a = self.cfg.baseline_alpha
        if self._baseline_x is None:
            self._baseline_x, self._baseline_y = wrist_x, wrist_y
        else:
            self._baseline_x = (1 - a) * self._baseline_x + a * wrist_x
            self._baseline_y = (1 - a) * self._baseline_y + a * wrist_y

    # ---------- main classification ----------

    def _classify_frame(self, hand: HandLandmarks) -> Gesture:
        pts = hand.normalized  # list of (x, y, z), 0-1 range
        wrist_x, wrist_y, _ = pts[0]

        # --- Static pose gestures first: these are pure hand-SHAPE checks,
        # independent of hand displacement, so they take priority over the
        # motion-based gestures below. This keeps START/STOP deliberate and
        # avoids them ever being confused with a jump/duck/lane-change.
        if self._is_thumbs_up(pts):
            return Gesture.START
        if self._is_peace_sign(pts):
            return Gesture.STOP

        extended = self._count_extended_fingers(pts)
        is_open_palm = extended >= self.cfg.open_palm_min_fingers
        is_fist = extended <= self.cfg.fist_max_fingers

        # Displacement relative to the slow-moving baseline (the "resting"
        # hand position), so gestures are relative, not tied to absolute
        # screen position.
        if self._baseline_x is None:
            self._update_baseline(wrist_x, wrist_y)
            return Gesture.NONE

        dx = wrist_x - self._baseline_x
        dy = wrist_y - self._baseline_y
        cfg = self.cfg

        gesture = Gesture.NONE

        # Priority: vertical gestures (jump/duck) take precedence over
        # lateral gestures because in the actual game, jump/duck are
        # typically more time-critical (obstacles), matching Stage 3's
        # description of jump = "open palm raised / upward swipe" and
        # duck = "closed fist / downward hand position".
        if is_open_palm and dy < -cfg.vertical_jump_delta:
            gesture = Gesture.JUMP
        elif is_fist and dy > cfg.vertical_duck_delta:
            gesture = Gesture.DUCK
        elif dx < -cfg.horizontal_move_delta:
            gesture = Gesture.LEFT
        elif dx > cfg.horizontal_move_delta:
            gesture = Gesture.RIGHT
        else:
            # Hand is near baseline and no strong pose -> slowly recentre
            # the baseline so small drift doesn't cause stuck gestures.
            self._update_baseline(wrist_x, wrist_y)

        return gesture

    def update(self, hand: Optional[HandLandmarks]) -> Gesture:
        """Call once per frame. Returns a *confirmed* Gesture (debounced),
        or Gesture.NONE most of the time."""

        if self._cooldown > 0:
            self._cooldown -= 1

        if hand is None:
            self._votes.clear()
            self._hold_count = 0
            self._hold_gesture = Gesture.NONE
            return Gesture.NONE

        raw = self._classify_frame(hand)
        self._votes.append(raw)

        # START/STOP are deliberately required to be held a bit longer than
        # the quick movement gestures, since they have bigger consequences
        # (pausing / (re)starting the run) and should not fire accidentally.
        required_hold = (
            self.cfg.start_stop_hold_frames_required
            if raw in (Gesture.START, Gesture.STOP)
            else self.cfg.hold_frames_required
        )

        # Consecutive-hold confirmation: the same non-NONE gesture must be
        # seen `required_hold` times in a row (allows brief noise).
        if raw != Gesture.NONE and raw == self._hold_gesture:
            self._hold_count += 1
        else:
            self._hold_gesture = raw
            self._hold_count = 1 if raw != Gesture.NONE else 0

        if (
            raw != Gesture.NONE
            and self._hold_count >= required_hold
            and self._cooldown == 0
        ):
            self._cooldown = self.cfg.cooldown_frames
            self._hold_count = 0
            # Snap baseline to current wrist so the player's hand can
            # return to "center" without immediately re-triggering.
            wrist_x, wrist_y, _ = hand.normalized[0]
            self._baseline_x, self._baseline_y = wrist_x, wrist_y
            return raw

        return Gesture.NONE

    @staticmethod
    def landmarks_debug_line(hand: Optional[HandLandmarks]) -> str:
        """Small helper used by main.py to print a one-line debug string
        (useful during Stage 6 testing/tuning)."""
        if hand is None:
            return "no hand detected"
        x, y, _ = hand.normalized[0]
        return f"{hand.handedness} hand @ wrist=({x:.2f},{y:.2f}) conf={hand.score:.2f}"
