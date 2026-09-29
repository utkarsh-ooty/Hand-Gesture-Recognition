# Hand Gesture & Voice Navigation AI — Project Documentation

## 1. Project Overview

**Hand Gesture & Voice Navigation AI** is a real-time, accessibility-focused desktop application that allows users to control a graphical interface using **hand gestures** and **voice commands** — entirely without a mouse or keyboard. A webcam captures the user's hand, computer vision algorithms detect and classify gestures, and a built-in voice assistant provides audio feedback and guidance at every step.

The system follows a three-phase workflow:

1. **Dataset Collection** — The user performs predefined gestures in front of the camera; the app records landmark keypoints into a CSV dataset plus saves reference images.
2. **Model Training** — A Random Forest classifier is trained on the collected keypoint features.
3. **Live Recognition** — The trained model classifies gestures in real time, enabling touchless navigation and control.

---

## 2. Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.10+ | Core application language |
| **GUI Framework** | Tkinter (with `ttk`) | Desktop UI — dark-themed, accessibility-first interface |
| **Computer Vision** | OpenCV (`cv2`) | Camera capture, frame processing, image I/O, drawing overlays |
| **Hand Detection** | MediaPipe Hand Landmarker (Tasks API) | 21-point 3D hand landmark detection via a pre-trained `.task` model |
| **ML Classification** | scikit-learn (`RandomForestClassifier`) | Static gesture classification from keypoint features |
| **Data Handling** | pandas, NumPy | Dataset CSV read/write, feature array manipulation |
| **Model Persistence** | joblib | Serialise/deserialise the trained Random Forest model (`.pkl`) |
| **Visualisation** | matplotlib | Confusion matrix plot after training |
| **Text-to-Speech** | Edge TTS (`edge-tts`) → Windows SAPI → pyttsx3 | Multi-fallback voice output with natural Indian English voice |
| **Speech-to-Text** | SpeechRecognition + Google Web Speech API | Real-time microphone listening for voice commands |
| **Audio I/O** | PyAudio / sounddevice | Microphone capture (PyAudio primary, sounddevice fallback) |
| **Image Display** | Pillow (`PIL`) | Convert OpenCV frames to Tkinter-compatible `PhotoImage` |
| **Audio Playback** | Windows MCI (`winmm.dll` via ctypes) | Low-latency playback of generated TTS `.mp3` files |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         main.py (HandGestureApp)                    │
│  Tkinter GUI  ←→  33ms update loop  ←→  State-driven UI rendering  │
└────────┬───────────────┬────────────────┬───────────────────────────┘
         │               │                │
    ┌────▼────┐   ┌──────▼──────┐   ┌─────▼──────┐
    │ Camera  │   │ HandTracker │   │ Speech     │
    │ (cv2)   │   │ (MediaPipe) │   │ Controller │
    └────┬────┘   └──────┬──────┘   │ (STT)      │
         │               │          └─────┬──────┘
         │          ┌────▼─────┐          │
         │          │ Feature  │     ┌────▼──────────┐
         │          │Extraction│     │ Command Engine │◄── Normalised intents
         │          └────┬─────┘     └────┬──────────┘
         │               │               │
    ┌────▼────────────────▼───┐     ┌─────▼──────────┐
    │   Finger    │  Gesture  │     │ Voice Assistant │
    │  Detector   │ Classifier│     │ (TTS output)   │
    │  (rules)    │ (RF model)│     └────────────────┘
    └─────┬───────┴─────┬─────┘
          │             │
    ┌─────▼─────┐ ┌─────▼────────┐
    │Navigation │ │   Swipe      │
    │ Manager   │ │  Detector    │
    │(debounce) │ │ (trajectory) │
    └───────────┘ └──────────────┘
```

---

## 4. Module-by-Module Breakdown

### 4.1 `main.py` — Application Entry Point & GUI

- **Class:** `HandGestureApp`
- **Responsibilities:**
  - Initialises all sub-modules (camera, tracker, classifier, voice, etc.)
  - Builds the Tkinter dark-themed UI with a split-pane layout (camera feed on the left, controls/guidance on the right)
  - Runs a **33 ms update loop** (~30 FPS) that reads camera frames, processes hand landmarks, routes gestures to the navigation or recognition pipeline, and updates all UI labels
  - Manages application lifecycle (startup voice greeting, window resize, clean shutdown)
- **Key Design Decisions:**
  - Uses Windows short-path conversion (`GetShortPathNameW`) to avoid Unicode path issues with MediaPipe's C++ backend
  - The camera panel width is user-adjustable (25%–60%) via preset buttons, a slider, or voice commands

---

### 4.2 `camera.py` — Webcam Abstraction

- **Class:** `Camera`
- Wraps OpenCV's `VideoCapture` with start/stop/read lifecycle
- Captures at 640×480 by default
- **Horizontally flips** every frame (`cv2.flip(frame, 1)`) for a natural selfie-view interaction

---

### 4.3 `hand_tracker.py` — Hand Landmark Detection

- **Class:** `HandTracker`
- Uses **MediaPipe Hand Landmarker Tasks API** (not the legacy `mp.solutions.hands`)
- Loads the pre-trained model file `models/hand_landmarker.task` (~7.5 MB)
- **Algorithm:** MediaPipe's hand landmarker is a two-stage pipeline:
  1. **Palm Detection** — A BlazePalm single-shot detector locates the palm bounding box
  2. **Hand Landmark Regression** — A second network predicts **21 3D keypoints** (x, y, z normalised to [0, 1]) from the cropped palm region
- **Output Methods:**
  - `get_landmarks()` → Normalised `[id, x, y, z]` list (used for feature extraction)
  - `get_pixel_landmarks()` → Absolute pixel `[id, cx, cy]` list (used for swipe detection)
  - `get_handedness()` → `"Left"` or `"Right"` + confidence score
- **Drawing:** Manually draws joints (red circles) and bone connections (green lines) on the frame, using a hardcoded `HAND_CONNECTIONS` adjacency list

---

### 4.4 `feature_extraction.py` — Keypoint Normalisation

- **Function:** `extract_features(landmarks)`
- **Input:** 21 landmarks as `[id, x, y, z]`
- **Algorithm:**
  1. Extracts only (x, y) coordinates (ignoring z for robustness)
  2. **Translation Invariance** — Subtracts the wrist (landmark 0) position so all coordinates are relative to the wrist origin
  3. **Scale Invariance** — Divides by the maximum absolute coordinate value, normalising the hand to a unit bounding box
  4. Flattens to a **42-element feature vector** (21 points × 2 coordinates)
- This normalisation ensures the model is invariant to the hand's position and distance from the camera

---

### 4.5 `finger_detector.py` — Rule-Based Finger Counting

- **Class:** `FingerDetector`
- **Algorithm:**
  - For **index, middle, ring, pinky**: A finger is "extended" if the distance from its **tip** to the **wrist** is greater than the distance from its **PIP joint** to the wrist (plus a small threshold of 2% of palm height)
  - For **thumb**: Compares the distance from thumb tip to pinky MCP vs. thumb IP to pinky MCP — if the tip is farther, the thumb is extended
- **Output:** Total finger count (0–5) + per-finger dictionary (`{'thumb': 0/1, 'index': 0/1, ...}`)
- Also provides `check_gesture_match()` to verify if the current hand pose matches a requested gesture class (e.g., `open_palm` → 5 fingers, `closed_fist` → 0 fingers, `thumbs_up` → 1 finger + tip above wrist)

---

### 4.6 `gesture_config.py` — Configuration Constants

Defines two core mappings:

| Constant | Value | Purpose |
|----------|-------|---------|
| `GESTURES` | `["open_palm", "closed_fist", "thumbs_up", "thumbs_down", "swipe_left", "swipe_right"]` | The 6 gesture classes for ML training |
| `FINGER_MAPPING` | `{0: "BACK", 1: "NEXT", 2: "PREV", 3: "SELECT", 4: "HELP", 5: "HOME"}` | Maps finger count to navigation intents (bootstrap mode) |

---

### 4.7 `dataset_collector.py` — Data Collection Pipeline

- **Class:** `DatasetCollector`
- Records training data when the user holds a gesture:
  1. **Saves an image** (`<timestamp>.jpg`) into `dataset/<gesture_class>/`
  2. **Appends a CSV row** `[label, f1, f2, ..., f42]` to `dataset/keypoint_dataset.csv`
- Collects **50 frames per gesture** before auto-stopping
- The CSV accumulates data across multiple recording sessions

---

### 4.8 `train_model.py` — ML Model Training

- **Function:** `train_gesture_model()`
- **Algorithm: Random Forest Classifier**
  - **Ensemble of 100 Decision Trees** (`n_estimators=100`)
  - Each tree is trained on a random bootstrap sample of the data with random feature subsets (bagging + feature randomisation)
  - Final prediction is the **majority vote** across all 100 trees
- **Pipeline:**
  1. Loads `dataset/keypoint_dataset.csv`
  2. Splits 80/20 train/validation with **stratified sampling** (`stratify=y`)
  3. Trains the Random Forest on the 42-dimensional feature vectors
  4. Evaluates **validation accuracy** and generates a **confusion matrix** plot
  5. Serialises the model to `models/rf_model.pkl` using joblib
- **Why Random Forest?**
  - Works well with small-to-medium datasets (hundreds to thousands of samples)
  - Naturally handles multi-class classification
  - Fast inference (~1ms), critical for real-time use
  - Robust to overfitting compared to a single decision tree
  - Provides probability estimates (`predict_proba`) for confidence thresholding

---

### 4.9 `gesture_classifier.py` — Real-Time ML Inference

- **Class:** `GestureClassifier`
- Loads the trained `rf_model.pkl` at startup
- **Prediction Pipeline:**
  1. Takes a 42-element normalised feature vector
  2. Calls `model.predict_proba()` → class probability distribution
  3. Returns the class with the **highest probability** and its confidence score
  4. If confidence < `0.6` (threshold), returns `"Unknown"`
- In `main.py`, an additional **stability filter** requires 12 consecutive frames of the same gesture at >85% confidence before triggering an action

---

### 4.10 `swipe_detector.py` — Dynamic Gesture Detection

- **Class:** `SwipeDetector`
- Detects **horizontal swipe gestures** from wrist trajectory
- **Algorithm:**
  1. Maintains a **circular buffer** of the last 15 wrist (x, y) positions (subsampled every 2nd frame)
  2. Computes displacement `dx` and `dy` between the oldest and newest positions
  3. If `|dx| > 1.5 × |dy|` (primarily horizontal) AND `|dx| > 100 pixels`:
     - `dx > 0` → **Swipe Right**
     - `dx < 0` → **Swipe Left**
  4. Enforces a **1.5-second cooldown** between swipes to prevent rapid re-triggering

---

### 4.11 `navigation_manager.py` — Finger-Based Navigation (Bootstrap Mode)

- **Class:** `NavigationManager`
- Converts raw finger counts into navigation intents using `FINGER_MAPPING`
- **Debouncing Algorithm:**
  1. Accumulates finger counts into a history buffer of 12 frames
  2. Only triggers an intent if **all 12 frames** report the **same** finger count (stability check)
  3. Won't re-trigger the same finger count consecutively
  4. Enforces a **2-second cooldown** between triggers
- This prevents accidental or jittery inputs from being registered

---

### 4.12 `application_state.py` — Finite State Machine

- **Enum:** `ApplicationState`
- **States:**

```
STARTUP → WAITING_FOR_HAND → MAIN_MENU → DATASET_MENU → DATASET_COLLECTING
                                       → TRAINING_CONFIRMATION → TRAINING → MODEL_READY
                                       → RECOGNITION
                                       → HELP / SETTINGS / ERROR
```

- `StateManager` holds the current state and provides `get_state()` / `set_state()` methods
- The state determines which input mode is active (finger navigation vs. ML recognition) and what the UI displays

---

### 4.13 `command_engine.py` — Central Intent Router

- **Class:** `CommandEngine`
- The **brain** of the application — receives normalised intents (e.g., `"NEXT"`, `"SELECT"`, `"BACK"`, `"HOME"`) from three sources:
  1. `NavigationManager` (finger gestures)
  2. `SpeechController` (voice commands)
  3. `GestureClassifier` / `SwipeDetector` (ML recognition)
- Routes intents based on the current `ApplicationState`:
  - In `MAIN_MENU`: NEXT/PREV cycle through menu items, SELECT executes
  - In `DATASET_COLLECTING`: manages the sequential gesture recording workflow
  - In `TRAINING_CONFIRMATION`: SELECT starts training, BACK cancels
  - In `RECOGNITION`: swipes map to NEXT/PREV, fist maps to BACK
- Also handles **global voice commands** (start/stop camera, adjust camera size, exit)
- Provides `say()` to queue voice assistant messages
- Provides `get_option_description()` for the UI's action-explanation card

---

### 4.14 `voice_assistant.py` — Text-to-Speech Engine

- **Class:** `VoiceAssistant`
- Runs a **background worker thread** with a speech queue to avoid blocking the GUI
- **Three-tier TTS fallback chain:**
  1. **Edge TTS** (primary) — Microsoft's neural TTS via the `edge-tts` library; uses the `en-IN-NeerjaNeural` voice (Indian English, female). Generates an `.mp3` file asynchronously, then plays it via Windows MCI
  2. **Windows SAPI** (fallback 1) — COM-based `SAPI.SpVoice` via `pywin32`
  3. **pyttsx3** (fallback 2) — Cross-platform offline TTS
- **Queue draining:** If messages pile up while the assistant is speaking, it skips to the latest message to avoid audio lag
- Exposes `is_speaking` property so the speech controller can pause microphone listening during TTS playback (echo prevention)

---

### 4.15 `speech_controller.py` — Voice Command Recognition

- **Class:** `SpeechController`
- Runs a **background listening thread** that continuously captures audio from the microphone
- **STT Pipeline:** Captures audio → sends to **Google Web Speech API** (`recognize_google()`) → receives transcribed text
- **Intent Parsing** (`_parse_and_dispatch`):
  - Matches transcribed text against keyword patterns using `any(w in text for w in [...])`
  - Supports ~30+ voice command variations grouped into:
    - Camera controls (`"start camera"`, `"wider camera"`, etc.)
    - Navigation (`"next"`, `"previous"`, `"select"`, `"back"`, `"home"`, `"help"`)
    - Section actions (`"dataset collection"`, `"train model"`, `"start recognition"`, `"stop recognition"`)
    - Confirmations (`"yes"` / `"no"`)
  - Dispatches resolved intents to the `CommandEngine` via callback
- **Self-hearing prevention:** Pauses listening while `VoiceAssistant.is_speaking` is `True`

---

## 5. Data Flow — End to End

### 5.1 Bootstrap Navigation Mode (No ML Model Required)

```
Camera Frame
    → HandTracker.find_hands()          [MediaPipe detects 21 landmarks]
    → FingerDetector.count_fingers()    [Rule-based: 0-5 fingers]
    → NavigationManager.process()       [12-frame stability + 2s cooldown]
    → CommandEngine.process_intent()    [State-aware routing]
    → VoiceAssistant.speak()            [Audio feedback]
    → UI Updates                        [Menu highlight, status banner]
```

### 5.2 ML Recognition Mode

```
Camera Frame
    → HandTracker.find_hands()          [MediaPipe detects landmarks]
    → extract_features(landmarks)       [Normalise to 42-dim vector]
    → GestureClassifier.predict()       [Random Forest predict_proba]
    → 12-frame stability filter         [>85% confidence for 12 frames]
    → CommandEngine.process_intent()    [Action dispatch]

    → SwipeDetector.add_position()      [Wrist trajectory tracking]
    → Swipe detected?                   [dx > 100px, horizontal]
    → CommandEngine.process_intent()    [NEXT or PREV]
```

### 5.3 Dataset Collection Pipeline

```
Camera Frame
    → HandTracker → extract_features()
    → FingerDetector.check_gesture_match()  [Verify correct gesture is shown]
    → DatasetCollector.record()
        → Save JPEG to dataset/<class>/
        → Append [label, f1..f42] to CSV
    → After 50 frames → auto-advance to next gesture
    → After all 6 gestures → return to main menu
```

---

## 6. Recognised Gestures

### 6.1 Static Gestures (ML-Classified)

| Gesture | Description | Finger Pattern |
|---------|-------------|----------------|
| `open_palm` | All fingers extended | 5 fingers |
| `closed_fist` | All fingers curled | 0 fingers |
| `thumbs_up` | Thumb extended upward | 1 finger (thumb tip above wrist) |
| `thumbs_down` | Thumb extended downward | 1 finger (thumb tip below wrist) |

### 6.2 Dynamic Gestures (Trajectory-Based)

| Gesture | Detection Method |
|---------|-----------------|
| `swipe_left` | Wrist moves >100px to the left over 15 frames |
| `swipe_right` | Wrist moves >100px to the right over 15 frames |

### 6.3 Finger Navigation Mapping

| Fingers Shown | Intent | Action |
|---------------|--------|--------|
| 0 (Fist) | `BACK` | Go back / Cancel |
| 1 | `NEXT` | Move to next option |
| 2 | `PREV` | Move to previous option |
| 3 | `SELECT` | Confirm / Select current option |
| 4 | `HELP` | Read help instructions |
| 5 (Open Palm) | `HOME` | Return to main menu |

---

## 7. Voice Commands

| Category | Example Commands |
|----------|-----------------|
| **Navigation** | `"next"`, `"previous"`, `"select"`, `"back"`, `"home"`, `"help"` |
| **Camera** | `"start camera"`, `"stop camera"`, `"wider camera"`, `"smaller camera"`, `"reset camera"` |
| **Dataset** | `"dataset collection"`, `"start recording"`, `"stop recording"` |
| **Training** | `"train model"`, `"start training"` |
| **Recognition** | `"start recognition"`, `"stop recognition"` |
| **App Control** | `"exit application"`, `"close app"` |
| **Confirmation** | `"yes"`, `"no"`, `"confirm"`, `"cancel"` |

---

## 8. Project Structure

```
Hand-Gesture-Recognition/
├── main.py                  # Application entry point & GUI
├── camera.py                # Webcam wrapper (OpenCV)
├── hand_tracker.py          # MediaPipe hand landmark detection
├── feature_extraction.py    # Keypoint normalisation (42-dim vector)
├── finger_detector.py       # Rule-based finger counting
├── gesture_classifier.py    # Random Forest inference
├── gesture_config.py        # Gesture & finger mapping constants
├── swipe_detector.py        # Trajectory-based swipe detection
├── dataset_collector.py     # Keypoint CSV + image recording
├── train_model.py           # Random Forest training pipeline
├── navigation_manager.py    # Finger-to-intent debouncing
├── command_engine.py        # Central intent router (FSM)
├── application_state.py     # Application state enum & manager
├── voice_assistant.py       # TTS engine (Edge TTS / SAPI / pyttsx3)
├── speech_controller.py     # STT engine (Google Web Speech API)
├── requirements.txt         # Python dependencies
│
├── models/
│   ├── hand_landmarker.task # MediaPipe pre-trained hand model (~7.5 MB)
│   ├── rf_model.pkl         # Trained Random Forest model (~2 MB)
│   └── confusion_matrix.png # Training evaluation plot
│
└── dataset/
    ├── keypoint_dataset.csv # Collected landmark features (label + 42 features)
    ├── open_palm/           # Reference images for open_palm
    ├── closed_fist/         # Reference images for closed_fist
    ├── thumbs_up/           # Reference images for thumbs_up
    ├── thumbs_down/         # Reference images for thumbs_down
    ├── swipe_left/          # Reference images for swipe_left
    └── swipe_right/         # Reference images for swipe_right
```

---

## 9. Algorithms Summary

| Algorithm | Where Used | Type | Details |
|-----------|-----------|------|---------|
| **BlazePalm + Hand Landmark Regression** | `hand_tracker.py` | Deep Learning (CNN) | MediaPipe's two-stage pipeline: palm detection → 21 3D keypoint regression |
| **Wrist-Relative Normalisation** | `feature_extraction.py` | Feature Engineering | Translation + scale invariance via wrist subtraction and max-abs normalisation |
| **Euclidean Distance Finger Extension** | `finger_detector.py` | Geometric Heuristic | Compares tip-to-wrist vs. PIP-to-wrist distances to determine finger state |
| **Random Forest Classifier** | `train_model.py`, `gesture_classifier.py` | Ensemble ML | 100 decision trees with bagging; majority-vote classification with probability output |
| **Sliding Window Swipe Detection** | `swipe_detector.py` | Signal Processing | 15-frame circular buffer; horizontal displacement thresholding (>100px) |
| **Temporal Stability Filtering** | `navigation_manager.py`, `main.py` | Debouncing | Requires N consecutive consistent readings before triggering (12 frames + cooldown) |
| **Keyword-Pattern Intent Parsing** | `speech_controller.py` | NLP (rule-based) | Substring matching against predefined keyword lists for voice command classification |
| **Finite State Machine** | `command_engine.py`, `application_state.py` | Control Flow | State-driven intent routing ensures actions are contextually appropriate |

---

## 10. Installation & Setup

### Prerequisites
- Python 3.10 or later
- A working webcam
- A working microphone (for voice commands)
- Windows OS (for SAPI TTS fallback and MCI audio playback)

### Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd Hand-Gesture-Recognition

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python main.py
```

---

## 11. Usage Workflow

1. **Launch** → The app greets you with a voice welcome message and activates the camera
2. **Show your hand** → The system detects your hand and transitions to the main menu
3. **Navigate the menu** using finger gestures:
   - ☝️ 1 finger → Next option
   - ✌️ 2 fingers → Previous option
   - 🤟 3 fingers → Select
   - ✊ Fist → Go back
   - 🖐️ 5 fingers → Home
4. **Collect Dataset** → The app guides you through recording 50 samples for each of the 6 gestures
5. **Train Model** → One-click training with voice progress updates and accuracy reporting
6. **Live Recognition** → The trained model classifies your gestures in real time; swipe left/right to navigate

---

## 12. Key Design Principles

- **Accessibility First** — Every action is accompanied by voice narration; no visual-only feedback
- **Zero-Keyboard Operation** — The entire application can be operated hands-free using gestures and/or voice
- **Graceful Degradation** — TTS has 3 fallback engines; microphone has PyAudio + sounddevice fallbacks
- **Self-Hearing Prevention** — Microphone automatically pauses during TTS playback to avoid echo loops
- **Stability Over Speed** — 12-frame consistency requirements prevent accidental gesture triggers
- **Modular Architecture** — Each concern (camera, detection, classification, navigation, voice) is encapsulated in its own module

---

## 13. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | ≥ 4.8.0 | Camera capture and image processing |
| `mediapipe` | ≥ 0.10.14 | Hand landmark detection |
| `numpy` | ≥ 1.24.0 | Array operations |
| `scikit-learn` | ≥ 1.3.0 | Random Forest classifier |
| `joblib` | ≥ 1.3.0 | Model serialisation |
| `pandas` | ≥ 2.0.0 | CSV dataset handling |
| `matplotlib` | ≥ 3.7.0 | Confusion matrix visualisation |
| `pillow` | ≥ 10.0.0 | Image format conversion for Tkinter |
| `pyttsx3` | ≥ 2.90 | Offline TTS fallback |
| `SpeechRecognition` | ≥ 3.10.0 | Microphone audio capture and Google STT |
| `PyAudio` | ≥ 0.2.13 | Primary microphone backend |
| `edge-tts` | ≥ 0.4.5 | Neural TTS (Microsoft Edge voices) |
| `pywin32` | ≥ 306 | Windows SAPI COM interface |
| `sounddevice` | ≥ 0.4.6 | Fallback microphone backend |

---

*Document generated for the Hand Gesture & Voice Navigation AI project.*
