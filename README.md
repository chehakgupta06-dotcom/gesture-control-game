# Gesture-Controlled Endless Runner (Ancient-Ruins / Temple-Runner Style)

Full working implementation of the project report: a webcam hand gesture
controls a 3-lane Pygame endless runner. No keyboard/touch needed to
play (keyboard is kept as a backup/fallback control).

The game is visually styled after the "temple runner" genre (wood/stone
bridge, glowing jungle either side, gold coin gems, skull and pillar
framing) -- all built from original, procedurally-drawn shapes in
`src/game.py`, not copied art assets or sprites from any commercial
game.

```
Webcam --> MediaPipe Hand Landmarks --> Gesture Classifier --> Command Mapping --> Pygame Runner
(Stage 1)      (Stage 2)                   (Stage 3)             (Stage 4)          (Stage 5)
```

## Project structure

```
gesture_runner/
├── main.py                  # entry point, wires camera + classifier + game together
├── setup.py                 # one-time preflight: checks deps, downloads model, tests webcam
├── requirements.txt
├── src/
│   ├── hand_tracker.py      # Stage 1 & 2: OpenCV capture + MediaPipe HandLandmarker
│   ├── gesture_classifier.py# Stage 3: rule-based geometric gesture classification
│   └── game.py               # Stage 5: Pygame endless-runner engine
├── tests/
│   └── test_logic.py         # Stage 6: automated tests (gesture + game logic), no camera needed
└── assets/
    └── hand_landmarker.task  # downloaded automatically on first run (~8 MB)
```

## 1. Setup (do this once)

You need **Python 3.9–3.11** (3.12 also works) and an internet connection
for the first run only (to fetch the ~8 MB MediaPipe model file).

```bash
cd gesture_runner
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python setup.py                 # downloads model, checks webcam, prints versions
```

`setup.py` will print something like:

```
[OK] opencv-python 4.13.0
[OK] mediapipe 1.0.1
[OK] pygame 2.6.1
[OK] numpy 2.4.4
[OK] Model ready at: .../assets/hand_landmarker.task
[OK] Webcam working. Frame size: 640x480
```

Keep this output — it's useful evidence for your result-tabulation
document (software versions used, webcam resolution achieved, etc.).

> **Note on MediaPipe versions:** current `pip install mediapipe`
> installs a version that uses Google's new **Tasks API**
> (`HandLandmarker`), which this project uses. If you find an older
> tutorial online using `mp.solutions.hands`, that API has been removed
> in current MediaPipe releases — don't mix it into this codebase.

## 2. Run the game

**With gesture control (webcam):**
```bash
python main.py
```
Two windows open:
- The **game window** (what the player looks at).
- A **debug window** showing your webcam feed with the hand skeleton
  drawn on it and the currently recognized gesture printed at the top —
  keep this visible in your demo video/presentation as proof the
  recognition is live and working.

**Keyboard-only (no webcam / fallback / quick testing):**
```bash
python main.py --keyboard
```

**Other options:**
```bash
python main.py --camera 1        # use a second/external webcam
python main.py --no-debug-cam    # hide the OpenCV debug window
python main.py --seed 42         # reproducible obstacle pattern, good for repeat demos
```

## 3. Gestures

| Gesture | How to do it | In-game action |
|---|---|---|
| **Jump** | Open palm, raise your hand up from resting position | Player jumps over low (stone-block) obstacles |
| **Duck** | Make a fist, lower your hand down | Player ducks under high (wooden-beam) obstacles |
| **Move Left** | Move your (resting) hand left | Player switches to the left lane |
| **Move Right** | Move your (resting) hand right | Player switches to the right lane |
| **Start / Resume** | **Thumbs-up**, held briefly | Begins the run, resumes from a pause, or starts a fresh run after game over |
| **Stop / Pause** | **Peace sign** (index + middle finger), held briefly | Freezes the character and the world in place |

Return your hand to a relaxed, roughly-centered resting position between
the four *movement* gestures (jump/duck/left/right) — the classifier
measures those relative to your resting position, not absolute screen
position, so it adapts to where you're comfortably holding your hand.

The **Start** and **Stop** gestures work differently on purpose: they are
recognized purely from hand *shape* (thumbs-up / peace sign), regardless
of where your hand is on screen, and are held for slightly longer before
they trigger (about 0.1s longer than the movement gestures) so that
pausing or starting the game is always a deliberate action, never an
accidental one.

**Keyboard fallback (always active, even in gesture mode):**
`A`/`←` left · `D`/`→` right · `W`/`↑`/`Space` jump · `S`/`↓` duck ·
`Enter` start/resume/play-again · `P` pause · `R` restart · `Esc`/`Q` quit.

## 4. Running the automated tests (Stage 6 evidence)

These test the gesture classifier and game engine directly with
synthetic landmark data — no webcam needed, run anywhere in seconds:

```bash
python tests/test_logic.py
# or, if you have pytest installed:
python -m pytest tests/ -v
```

Expected output: `10 passed, 0 failed`.

## 5. Tuning gesture sensitivity

If gestures trigger too easily (false positives) or not easily enough
(missed gestures) under your room's lighting, adjust
`GestureConfig` in `src/gesture_classifier.py`:

- `vertical_jump_delta` / `vertical_duck_delta` — how far you must move
  your hand up/down to trigger jump/duck. Increase if jumps/ducks fire
  accidentally; decrease if they're hard to trigger.
- `horizontal_move_delta` — same idea, for left/right.
- `hold_frames_required` — how many consecutive frames a gesture must be
  held before it's confirmed (raises this to reduce false triggers from
  a single noisy frame).
- `cooldown_frames` — minimum gap between two gesture triggers.

Document whichever values you land on, plus your before/after false-
trigger rate, in your Stage 6 (Testing & Optimization) write-up — this
is literally what the report says you'll do.

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open camera index 0` | Close any other app using the webcam (Zoom/Teams/browser tab); try `--camera 1`; on Linux check `/dev/video0` permissions. |
| Model download fails (`setup.py` error) | You need internet the *first* run only. If your network blocks Google Cloud Storage, download the file manually from the URL printed in the error and place it at `assets/hand_landmarker.task`. |
| Low FPS / laggy debug window | Close the OpenCV debug window with `--no-debug-cam`; lower `frame_width`/`frame_height` in `HandTracker(...)` in `main.py`. |
| Gestures feel delayed | Lower `hold_frames_required` in `GestureConfig` (trade-off: more false positives). |
| Game runs but window doesn't appear (SSH/headless) | This project needs a real display; it won't run over a plain SSH session. Run it on the physical machine you'll present from. |

## 7. What maps to what in the DA1 report

| Report stage | File |
|---|---|
| Stage 1: Video Acquisition | `src/hand_tracker.py` → `HandTracker.open_camera`, `read_frame` |
| Stage 2: Hand Detection & Landmark Extraction | `src/hand_tracker.py` → `HandTracker.process` |
| Stage 3: Gesture Classification | `src/gesture_classifier.py` → `GestureClassifier` |
| Stage 4: Gesture-to-Command Mapping | `main.py` → `map_gesture_to_command` (now includes START/STOP) |
| Stage 5: Game Engine | `src/game.py` → `EndlessRunnerGame` (ancient-ruins visual theme + start/pause state machine) |
| Stage 6: Testing & Optimization | `tests/test_logic.py` + the `[STATS]` line printed when you quit `main.py` (avg loop time / FPS) |
