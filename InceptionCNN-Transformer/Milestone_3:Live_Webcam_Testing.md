# Milestone 3: Live Webcam Testing (Real-Time Inference)

Build a real-time Python script that opens the Mac's webcam, extracts skeletal keypoints using MediaPipe, and feeds them into our trained `ASLTransformer` to predict the signed word live.

---

## Goal

Transition the model from offline evaluation on a static dataset to **real-time, real-world inference**. This will serve as our primary prototype and validation step before moving to Android development (Milestone 4).

## User Review Required

> [!WARNING]
> **Performance vs Accuracy Tradeoff in Real-Time**
> 
> During training, we processed the *entire* video at once, allowing us to use sophisticated techniques like Cubic Spline interpolation to perfectly bridge gaps where the hands disappeared. 
> 
> In a live webcam, we can only see the *past*, not the future. To keep the model accurate, we will maintain a **rolling window of the last 60 frames** (~2 seconds of video at 30fps). On every single frame, we will run the entire normalization pipeline on those 60 frames and pass them to the model. Because our model is very lightweight (~4.3M parameters), this will easily run at 30+ FPS on your Mac without lag.

## Open Questions

> [!IMPORTANT]
> **Should we require a "trigger" gesture to start predicting?**
> Right now, the plan is for the model to predict continuously on every frame (always guessing what the last 2 seconds of motion was). This might result in random guesses when you are just sitting still. 
> 
> *Question for you:* Would you prefer it predicts continuously, or should we wait for you to press the `Spacebar` to trigger a prediction on the last 60 frames?

## Proposed Changes

### `live_test.py`

#### [NEW] [live_test.py](file:///Users/lalith/code/projects/Sing_AR/live_test.py)
A self-contained Python script to handle the live webcam feed and inference.

**Core Components:**
1. **Model Loader:** 
   - Loads the `class_map.json` to know the 100 gloss names.
   - Instantiates `ASLTransformer` and loads weights from `model/checkpoints/best_model.pt`.
   - Sets the model to `.eval()` mode.

2. **MediaPipe Pipeline:**
   - Initializes `mp.solutions.holistic`.
   - Captures webcam frames using `cv2.VideoCapture(0)`.
   - Extracts the exact same 85 keypoints (14 pose, 21 left hand, 21 right hand).

3. **Rolling Buffer (The 60-Frame Window):**
   - Uses a `collections.deque(maxlen=60)` to store the raw `(85, 3)` keypoints from the most recent 60 frames.
   - If a hand is not detected in a frame, it appends an array of `np.nan` (matching the extraction script logic).

4. **Real-Time Preprocessing:**
   - Once the deque has 60 frames, it converts it to a numpy array `(60, 85, 3)`.
   - Runs `fill_missing_frames(x)` (1D linear interpolation to fill NaNs).
   - Runs `normalize_shoulders(x)` (centers on shoulders and scales by shoulder width).
   - Flattens to `(60, 255)`.

5. **Inference & UI:**
   - Converts the flattened array to a PyTorch tensor `(1, 60, 255)`.
   - Passes it to the model.
   - Uses `torch.topk` to get the Top-3 predictions and their confidence percentages.
   - Overlays the MediaPipe skeleton and the Top-3 text predictions onto the OpenCV video window.

## Verification Plan

### Manual Verification
1. Run `python live_test.py`.
2. Ensure the webcam window opens and draws the skeletal landmarks smoothly without freezing.
3. Perform signs from the dataset (e.g., "APPLE", "MOTHER", "CITY1").
4. Verify that the correct gloss appears in the Top-3 predictions on the screen.
5. If the accuracy is significantly lower than the test set, identify the cause (e.g., camera angle, distance, lighting) to inform data augmentations for future training.
