# Project Proposal: Real-Time Continuous ASL Translation System

## 1. Executive Summary
This project aims to develop a bidirectional, real-time American Sign Language (ASL) translation system optimized for edge devices (e.g., Android). Unlike traditional systems that rely on heavy 3D-CNNs processing raw video frames, this architecture leverages lightweight spatial keypoint extraction combined with a hybrid Convolutional-Transformer model. This ensures real-time performance, low battery consumption, and high accuracy. 

The system supports two primary use cases:
1. **Sign-to-Text/Speech:** Translating word-by-word ASL video sequences into grammatically correct English text.
2. **Speech/Text-to-Sign:** Translating English text into ASL glosses to animate a 3D avatar using mapped keypoints.

## 2. Core Technology Stack
*   **Vision & Feature Extraction:** Google MediaPipe (Holistic)
*   **Sequence Modeling:** 1D-CNN + Shallow Transformer (PyTorch/TensorFlow)
*   **Natural Language Processing:** LLM API (e.g., Gemini Flash or OpenAI GPT-4o-mini)
*   **Training Dataset:** WLASL (Word-Level American Sign Language)

---

## 3. Architecture Blueprint

### Stage 1: Vision & Recognition (The Hybrid Model)
The objective of this stage is to process isolated sign language gestures provided by the user word-by-word, classifying each into an ASL gloss.

*   **Feature Extraction (MediaPipe):**
    *   We extract Hand and Pose coordinates. 
    *   *Optimization:* We will discard the dense 468-point face mesh to prevent overfitting. Instead, we will extract only a targeted subset of facial keypoints (eyes, eyebrows, mouth) to capture critical Non-Manual Markers (NMMs) required for ASL grammar (e.g., raised eyebrows for questions).
*   **Data Pipeline & Normalization:**
    *   Extract keypoints at the highest available frame rate (e.g., 60 FPS).
    *   Handle motion blur and missing data (NaNs) via **Linear Interpolation**.
    *   Smooth jittery keypoints using a **Savitzky-Golay filter**.
    *   *Spatial Normalization:* Shift all coordinates relative to a central origin (e.g., center of shoulders) and scale by shoulder width to ensure scale and positional invariance.
*   **The Hybrid Sequence Model (1D-CNN + Transformer):**
    *   **1D-CNN Block:** Acts as a local motion extractor. A small kernel (size 3-5) slides over the time dimension to capture rapid, localized movements (e.g., wrist flicks) that Transformers natively struggle to detect.
    *   **Shallow Transformer Encoder (2-4 Layers):** Receives the enriched local features from the CNN. Relying on its self-attention mechanism, it analyzes the global context of the sequence to confidently classify the ASL gloss.
*   **Isolated Word-by-Word UX:**
    *   To circumvent the complexities of continuous sign segmentation and perfectly align with the WLASL dataset, the user will sign word-by-word.
    *   The app will classify each isolated sign and hold the recognized glosses in a sequence until the user explicitly signals (e.g., via a button) that their sentence is complete.
*   **Fingerspelling Fallback Mode:**
    *   To account for proper nouns, names, or obscure vocabulary absent from the WLASL dataset, the vision system will include a dedicated fingerspelling mode.
    *   This mode employs a lightweight static image classifier trained on the ASL Alphabet dataset to spell out unrecognized words letter-by-letter.

### Stage 2: Translation (NLP via LLM)
ASL syntax (Topic-Comment) differs significantly from English syntax. Traditional mapping or custom Seq2Seq NLP models require massive amounts of hard-to-find parallel data.

*   **Gloss Accumulation:** As the user signs word-by-word, the vision model accumulates the recognized glosses in a list (e.g., `["STORE", "I", "GO"]`).
*   **LLM Integration:** Once the user explicitly signals that the sentence is complete, the array of glosses is sent to a lightweight LLM API. 
*   **Prompt Engineering:** The LLM is given a strict system prompt to act as an ASL-to-English translator. It unscrambles the glosses and outputs a grammatically perfect English sentence (e.g., `"I am going to the store."`).
*   **Streaming Output:** To mitigate perceived latency and mask network delays, the LLM API requests will utilize streaming. The translated English tokens will appear sequentially on the UI as they are generated, keeping the user engaged.

### Stage 3: Reverse Translation (Text-to-Sign Avatar)
To facilitate two-way communication:
1.  English speech/text is passed to the LLM, which is prompted to translate it *back* into an ASL gloss sequence.
2.  Each ASL gloss triggers a pre-recorded sequence of MediaPipe keypoints from a database.
3.  These keypoints are mapped to a 3D avatar's skeleton using Inverse Kinematics (IK).
4.  **Animation Blending:** To prevent robotic snapping between isolated signs, a blending algorithm like Spherical Linear Interpolation (SLERP) is applied. This generates fluid transition frames between the end of one sign and the beginning of the next, resulting in seamless, natural avatar movement.

---

## 4. Key Engineering Advantages
*   **Efficiency:** By using a 1D-CNN to handle local motion, the Transformer can be kept shallow, resulting in a tiny, fast model that runs easily on Android without draining the battery.
*   **Robustness:** The data cleaning pipeline (Interpolation + Savitzky-Golay + Normalization) guarantees the model receives pristine data, even from a shaky smartphone camera in poor lighting.
*   **Scalability:** Outsourcing the complex NLP translation step to an LLM API drastically reduces development time and circumvents the massive data requirements of training a custom ASL-to-English translation model from scratch.
