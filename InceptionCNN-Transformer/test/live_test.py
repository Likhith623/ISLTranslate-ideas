"""
live_test.py — Real-Time ASL Webcam Tester + Mistral LLM Translation
======================================================================

Run from the Sing_AR/ project root:
    python test/live_test.py

Controls:
    SPACE       → Start recording the next sign (60 frames / ~2 seconds)
    ENTER       → Translate the accumulated sentence with Mistral LLM
    BACKSPACE   → Remove the last signed word from the sentence
    c           → Clear the sentence buffer
    q           → Quit

Dependencies:
    mediapipe, opencv-python, torch, numpy, pandas, scipy, mistralai, python-dotenv

API Key:
    Loaded automatically from the .env file in the project root.
    Expected format:  MISTRAL_API_KEY=your_key_here
"""

import json
import os
import sys
import threading

from dotenv import load_dotenv

# Load .env from project root (Sing_AR/.env) before anything else reads env vars
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import mediapipe as mp
from mediapipe.tasks import python as mp_python  # noqa
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mistralai.client.sdk import Mistral

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_DIR    = os.path.join(PROJECT_ROOT, "model")
PREPROC_DIR  = os.path.join(PROJECT_ROOT, "preprocessing")
MODELS_DIR   = os.path.join(PREPROC_DIR, "models")

CHECKPOINT   = os.path.join(MODEL_DIR, "checkpoints", "best_model.pt")
CLASS_MAP    = os.path.join(MODEL_DIR, "class_map.json")
FACE_MODEL   = os.path.join(MODELS_DIR, "face_landmarker.task")
HAND_MODEL   = os.path.join(MODELS_DIR, "hand_landmarker.task")
POSE_MODEL   = os.path.join(MODELS_DIR, "pose_landmarker_full.task")

sys.path.insert(0, MODEL_DIR)
from model import ASLTransformer  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# MISTRAL SETUP
# ─────────────────────────────────────────────────────────────────────────────

MISTRAL_MODEL  = "mistral-small-2506"
MISTRAL_PROMPT = """\
You are an expert American Sign Language (ASL) to English translator.
I will give you a list of ASL glosses signed in sequence.
Your job is to unscramble ASL syntax (Topic-Comment) and output a single,
grammatically perfect, natural-sounding English sentence.
Do NOT explain your reasoning. Output ONLY the translated sentence."""

def translate_with_mistral(glosses: list[str]) -> str:
    """Call Mistral API and return the translated English sentence."""
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        return "[MISTRAL_API_KEY not found in .env — add it to Sing_AR/.env]"
    client = Mistral(api_key=api_key)
    user_msg = "ASL glosses: " + ", ".join(glosses)
    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[
            {"role": "system", "content": MISTRAL_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
    )
    return response.choices[0].message.content.strip()

# ─────────────────────────────────────────────────────────────────────────────
# KEYPOINT CONFIG  (must match extract_keypoints.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

FACE_INDICES = [
    33, 133, 263, 362,
    70, 63, 105, 66, 107, 55, 65,
    300, 293, 334, 296, 336, 285, 295,
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    146, 91, 181, 84, 17, 314, 405, 321, 375,
    14,
]  # 39 total

POSE_INDICES   = [11, 12, 13, 14]
POSE_START     = len(FACE_INDICES)  # 39
L_SHOULDER_ROW = POSE_START + 0
R_SHOULDER_ROW = POSE_START + 1
NUM_KEYPOINTS  = len(FACE_INDICES) + len(POSE_INDICES) + 21 + 21  # 85
TARGET_FRAMES  = 60

# ─────────────────────────────────────────────────────────────────────────────
# COLOURS & FONTS
# ─────────────────────────────────────────────────────────────────────────────

C_WHITE  = (255, 255, 255)
C_GREEN  = (0,   220, 100)
C_RED    = (50,  50,  230)
C_YELLOW = (0,   200, 255)
C_BLUE   = (230, 160,  50)
C_DARK   = (20,  20,  20)
C_GREY   = (160, 160, 160)
FONT     = cv2.FONT_HERSHEY_SIMPLEX

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING  (identical to training pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_sequence(seq):
    out = seq.copy()
    for t in range(len(out)):
        l_sh = out[t, L_SHOULDER_ROW]
        r_sh = out[t, R_SHOULDER_ROW]
        if np.any(np.isnan(l_sh)) or np.any(np.isnan(r_sh)):
            continue
        midpoint = (l_sh + r_sh) / 2.0
        distance = np.linalg.norm(r_sh - l_sh)
        out[t] -= midpoint
        if distance > 1e-6:
            out[t] /= distance
    return out

def bridge_nans(seq):
    T = len(seq)
    flat = seq.reshape(T, -1)
    df = pd.DataFrame(flat)
    df = df.interpolate(method="linear", axis=0, limit_direction="both")
    df = df.fillna(0.0)
    return df.values.reshape(T, NUM_KEYPOINTS, 3).astype(np.float32)

def resample_sequence(seq, target=TARGET_FRAMES):
    from scipy.interpolate import interp1d
    T = len(seq)
    if T == target:
        return seq
    old_t  = np.linspace(0, 1, T)
    new_t  = np.linspace(0, 1, target)
    flat   = seq.reshape(T, -1)
    kind   = "cubic" if T >= 4 else "linear"
    interp = interp1d(old_t, flat, kind=kind, axis=0)
    return interp(new_t).reshape(target, NUM_KEYPOINTS, 3).astype(np.float32)

def preprocess(frames):
    seq = np.stack(frames, axis=0)
    seq = normalize_sequence(seq)
    seq = bridge_nans(seq)
    seq = resample_sequence(seq, TARGET_FRAMES)
    seq = seq.reshape(TARGET_FRAMES, -1)
    return torch.from_numpy(seq).unsqueeze(0)  # (1, 60, 255)

# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE
# ─────────────────────────────────────────────────────────────────────────────

def create_landmarkers():
    face_lm = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_MODEL),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
    )
    hand_lm = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL),
            num_hands=2,
        )
    )
    pose_lm = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODEL),
            num_poses=1,
        )
    )
    return face_lm, hand_lm, pose_lm

def extract_frame_keypoints(frame_rgb, face_lm, hand_lm, pose_lm):
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    kps    = np.full((NUM_KEYPOINTS, 3), np.nan, dtype=np.float32)
    cursor = 0

    face_res = face_lm.detect(mp_img)
    if face_res.face_landmarks:
        lms = face_res.face_landmarks[0]
        for idx in FACE_INDICES:
            if idx < len(lms):
                lm = lms[idx]
                kps[cursor] = [lm.x, lm.y, lm.z]
            cursor += 1
    else:
        cursor += len(FACE_INDICES)

    pose_res = pose_lm.detect(mp_img)
    if pose_res.pose_landmarks:
        lms = pose_res.pose_landmarks[0]
        for i, idx in enumerate(POSE_INDICES):
            if idx < len(lms):
                lm = lms[idx]
                kps[cursor + i] = [lm.x, lm.y, lm.z]
    cursor += len(POSE_INDICES)

    hand_res = hand_lm.detect(mp_img)
    if hand_res.hand_landmarks:
        for i, handedness in enumerate(hand_res.handedness):
            label = handedness[0].category_name
            arr   = np.array([[lm.x, lm.y, lm.z]
                               for lm in hand_res.hand_landmarks[i]],
                              dtype=np.float32)
            if label == "Left":
                kps[cursor:cursor + 21] = arr
            else:
                kps[cursor + 21:cursor + 42] = arr
    return kps

# ─────────────────────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def draw_skeleton(frame, kps, h, w):
    for i in range(NUM_KEYPOINTS):
        pt = kps[i]
        if not np.any(np.isnan(pt)):
            cv2.circle(frame, (int(pt[0] * w), int(pt[1] * h)), 3, C_GREEN, -1)

def draw_sentence_bar(frame, sentence_buffer, h, w):
    """Top bar showing accumulated glosses."""
    bar_h = 50
    cv2.rectangle(frame, (0, 0), (w, bar_h), C_DARK, -1)
    words = "  ›  ".join(sentence_buffer) if sentence_buffer else "—"
    label = f"Sentence: {words}"
    # Truncate if too long
    max_chars = (w // 11)
    if len(label) > max_chars:
        label = "…" + label[-(max_chars - 1):]
    cv2.putText(frame, label, (12, 33), FONT, 0.65, C_YELLOW, 1)
    # Hint
    hint = "ENTER=translate  BKSP=undo  c=clear"
    (hw, _), _ = cv2.getTextSize(hint, FONT, 0.45, 1)
    cv2.putText(frame, hint, (w - hw - 10, 33), FONT, 0.45, C_GREY, 1)

def draw_idle(frame, h, w):
    text = "Press SPACE to sign"
    scale, thick = 0.9, 2
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    x, y = (w - tw) // 2, h - 60
    cv2.rectangle(frame, (x - 10, y - th - 10), (x + tw + 10, y + 10), C_DARK, -1)
    cv2.putText(frame, text, (x, y), FONT, scale, C_WHITE, thick)

def draw_recording(frame, count, total, h, w):
    rec_text = f"  REC  [{count}/{total}]"
    cv2.putText(frame, rec_text, (20, 90), FONT, 1.1, C_RED, 2)
    bar_w = int((count / total) * (w - 40))
    cv2.rectangle(frame, (20, 108), (w - 20, 128), C_DARK, -1)
    cv2.rectangle(frame, (20, 108), (20 + bar_w, 128), C_RED, -1)

def draw_word_result(frame, top_results, h, w):
    """Small bottom panel showing Top-3 predictions for the last word."""
    panel_h = 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), C_DARK, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, "Last word predictions:", (12, h - panel_h + 22),
                FONT, 0.6, C_YELLOW, 1)
    for rank, (gloss, conf) in enumerate(top_results):
        y_b   = h - panel_h + 48 + rank * 26
        color = C_GREEN if rank == 0 else C_WHITE
        bar_l = int(conf * (w - 160))
        cv2.rectangle(frame, (12, y_b - 10), (12 + bar_l, y_b + 5),
                      color if rank == 0 else (70, 70, 70), -1)
        cv2.putText(frame, f"#{rank+1} {gloss:<22} {conf*100:5.1f}%",
                    (15, y_b), FONT, 0.58, color, 1 if rank > 0 else 2)

def draw_translation(frame, translation, h, w):
    """Full-width banner showing the LLM translation result."""
    panel_h = 100
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), (10, 60, 10), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.putText(frame, "Translation:", (12, h - panel_h + 28),
                FONT, 0.65, C_YELLOW, 1)
    # Word-wrap translation text
    words  = translation.split()
    line   = ""
    lines  = []
    for word in words:
        test = (line + " " + word).strip()
        if cv2.getTextSize(test, FONT, 0.72, 2)[0][0] < w - 24:
            line = test
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    for i, l in enumerate(lines[:2]):
        cv2.putText(frame, l, (12, h - panel_h + 58 + i * 32),
                    FONT, 0.72, C_GREEN, 2)

def draw_translating_spinner(frame, h, w):
    text = "Translating with Mistral..."
    (tw, _), _ = cv2.getTextSize(text, FONT, 0.8, 2)
    cv2.putText(frame, text, ((w - tw) // 2, h - 60), FONT, 0.8, C_BLUE, 2)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Class map ────────────────────────────────────────────────────────────
    with open(CLASS_MAP) as f:
        class_map = json.load(f)
    id_to_gloss = {v: k for k, v in class_map.items()}
    num_classes  = len(class_map)
    print(f"[ClassMap] {num_classes} classes loaded")

    # ── Model ────────────────────────────────────────────────────────────────
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[Device] {device}")
    model = ASLTransformer(num_classes=num_classes)
    ckpt  = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"[Model] Loaded epoch {ckpt.get('epoch','?')} — val acc {ckpt.get('val_acc','?')}")

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    print("[MediaPipe] Loading...")
    face_lm, hand_lm, pose_lm = create_landmarkers()
    print("[MediaPipe] Ready\n")

    # ── API key check ─────────────────────────────────────────────────────────
    if not os.environ.get("MISTRAL_API_KEY"):
        print("[WARNING] MISTRAL_API_KEY is not set. Translation will show a placeholder.")

    # ── Webcam ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return
    print("Running! Controls: SPACE=sign  ENTER=translate  BKSP=undo  c=clear  q=quit\n")

    # ── State machine ─────────────────────────────────────────────────────────
    STATE_IDLE        = "idle"
    STATE_RECORDING   = "recording"
    STATE_WORD_RESULT = "word_result"
    STATE_TRANSLATING = "translating"
    STATE_TRANSLATED  = "translated"

    state           = STATE_IDLE
    buffer          = []       # raw frames for current sign
    word_results    = []       # top-3 for last sign
    sentence_buffer = []       # accumulated glosses for full sentence
    translation     = ""       # LLM output

    # Translation runs on a background thread so the webcam doesn't freeze
    translation_lock   = threading.Lock()
    translation_thread = None

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_bgr = cv2.flip(frame_bgr, 1)
        h, w      = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        kps = extract_frame_keypoints(frame_rgb, face_lm, hand_lm, pose_lm)
        draw_skeleton(frame_bgr, kps, h, w)
        draw_sentence_bar(frame_bgr, sentence_buffer, h, w)

        # ── State rendering & logic ───────────────────────────────────────────
        if state == STATE_IDLE:
            draw_idle(frame_bgr, h, w)

        elif state == STATE_RECORDING:
            buffer.append(kps)
            draw_recording(frame_bgr, len(buffer), TARGET_FRAMES, h, w)
            if len(buffer) >= TARGET_FRAMES:
                with torch.no_grad():
                    tensor = preprocess(buffer).to(device)
                    logits = model(tensor)
                    probs  = F.softmax(logits, dim=-1)[0]
                    top_vals, top_idxs = torch.topk(probs, 3)
                    word_results = [
                        (id_to_gloss[i.item()], top_vals[r].item())
                        for r, i in enumerate(top_idxs)
                    ]
                # Auto-accept top-1 into sentence buffer
                top_word = word_results[0][0]
                sentence_buffer.append(top_word)
                print(f"[Word] {top_word}  ({word_results[0][1]*100:.1f}%) → Sentence: {sentence_buffer}")
                state = STATE_WORD_RESULT

        elif state == STATE_WORD_RESULT:
            draw_word_result(frame_bgr, word_results, h, w)

        elif state == STATE_TRANSLATING:
            draw_translating_spinner(frame_bgr, h, w)
            # Check if background thread finished
            with translation_lock:
                if translation_thread and not translation_thread.is_alive():
                    state = STATE_TRANSLATED

        elif state == STATE_TRANSLATED:
            draw_translation(frame_bgr, translation, h, w)

        # ── Show frame ────────────────────────────────────────────────────────
        cv2.imshow("Sign AR — Live Test + Translation", frame_bgr)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord(" "):
            # Start fresh recording for next word
            buffer       = []
            word_results = []
            state        = STATE_RECORDING
            print("[Recording] Started — sign now!")

        elif key == 13:  # ENTER
            if sentence_buffer:
                glosses = list(sentence_buffer)
                print(f"[Translating] Glosses: {glosses}")
                state = STATE_TRANSLATING

                def _do_translate():
                    result = translate_with_mistral(glosses)
                    with translation_lock:
                        nonlocal translation
                        translation = result
                    print(f"[Translation] {result}")

                translation_thread = threading.Thread(target=_do_translate, daemon=True)
                translation_thread.start()
            else:
                print("[Translate] No words signed yet — press SPACE to sign a word first.")

        elif key == 8:  # BACKSPACE
            if sentence_buffer:
                removed = sentence_buffer.pop()
                print(f"[Undo] Removed '{removed}' → Sentence: {sentence_buffer}")
            state = STATE_IDLE

        elif key == ord("c"):
            sentence_buffer = []
            word_results    = []
            translation     = ""
            state           = STATE_IDLE
            print("[Clear] Sentence buffer cleared.")

    cap.release()
    cv2.destroyAllWindows()
    print("[Done] Webcam closed.")

if __name__ == "__main__":
    main()
