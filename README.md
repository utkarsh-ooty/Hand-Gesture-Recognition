# Hand Gesture Recognition System

A real-time hand gesture recognition prototype built using Python, OpenCV, MediaPipe Hands, NumPy, and Scikit-learn.

## Features
- **Real-Time Hand Tracking**: Uses MediaPipe to track 21 hand landmarks robustly across different environments.
- **Gesture Classification**: Uses a Random Forest classifier trained on normalized landmarks for static gesture prediction.
- **Dynamic Swipe Detection**: A temporal buffer accurately tracks Hand position changes over consecutive frames to detect left and right swipes while prioritizing cooldowns.
- **Dataset Collection Tool**: Integrated Tkinter GUI allows easily collecting new hand gesture samples in various backgrounds right from your webcam to retrain the algorithm.
- **One-Click Model Training**: Click a button within the UI to instantly parse all collected data, train a scikit-learn model, output evaluation metrics and instantly reload it into your real-time camera feed.

## Supported Gestures
- Open Palm
- Closed Fist
- Thumbs Up
- Thumbs Down
- Swipe Left
- Swipe Right

## Setup & Installation

1. **Prerequisites**: Python 3.10 – 3.12 recommended (MediaPipe may not support newer Python versions yet).

2. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/Hand-Gesture-Recognition.git
   cd Hand-Gesture-Recognition
   ```

3. **Create a virtual environment** (recommended):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

> **Note**: The `hand_landmarker.task` MediaPipe model file is included in the `models/` directory. No additional downloads are needed.

## Usage

Start the system by running the graphical application:
```bash
python main.py
```

### Steps to Run and Test
1. Press **Start Camera**. You should see the video feed and MediaPipe drawing the skeletal tracking over your hand.
2. In order to detect gestures, you must first collect data and train the Scikit-Learn Model since it ships without predefined data. 
3. **Data Collection**: 
   - Select a static gesture from the dropdown (e.g. `open_palm`)
   - Click **Start Recording**. Move your hand around to provide variation (scale, slight rotations, different lighting).
   - Click **Stop Recording**. 
   - Repeat this for at least the core static gestures (`open_palm`, `closed_fist`, `thumbs_up`, `thumbs_down`). Collect ~100 frames per class.
4. **Training**:
   - Once data is collected, hit **Train Model**. This will extract the `dataset/keypoint_dataset.csv`, build the scikit-learn classifier, and save the binary model and a confusion matrix chart out to the `models/` directory.
5. **Inference**:
   - Hit **Start Recognition**. You should now see predictions appearing next to the `Detected Gesture` label in the control panel and written directly onto the video feed.
   - Do a swiping motion across the camera to test Dynamic Swipes (Swipe Left / Swipe Right). These use temporal positional logic rather than the Static Random Forest classifier.

### Project Structure
```text
project/
├── main.py                - Main Tkinter UI and primary loop
├── camera.py              - Easy to use OpenCV webcam handler
├── hand_tracker.py        - Wrapper around MediaPipe Hands
├── feature_extraction.py  - Code for normalizing 21 (x,y) hand landmarks
├── gesture_classifier.py  - Scikit-learn Random Forest Inference Handler
├── swipe_detector.py      - Positional History tracking for dynamic swipe determination
├── dataset_collector.py   - Captures frames and persists CSV features
├── train_model.py         - Trains a Scikit Learn RF and evaluates the performance
├── requirements.txt       - PyPi packages needed
├── models/                - Output directory for Scikit learn model pickles
├── dataset/               - Output directory for saved images and dataset CSV
└── README.md
```
