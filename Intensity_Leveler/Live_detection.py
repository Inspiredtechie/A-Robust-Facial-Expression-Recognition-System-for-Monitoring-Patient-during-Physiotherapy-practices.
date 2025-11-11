import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import math
import matplotlib.pyplot as plt
import pandas as pd

# --- Configuration ---
MODEL_PATH = "models/pain_model_hybrid.pkl"
REPORT_PLOT_PATH = "output/pain_report_hybrid.png"

# Pain labels and visual meter colors
PAIN_LABELS = {0: "No Pain", 1: "Slight Pain", 2: "Mild Pain", 3: "Intense Pain"}
PAIN_COLORS = {
    0: (0, 255, 0),    # Green
    1: (0, 255, 255),  # Yellow
    2: (0, 165, 255),  # Orange
    3: (0, 0, 255)     # Red
}

# --- MediaPipe Landmark Indices (MUST be same as extractor) ---
# 
IDX = {
    "left_eye_in": 33,
    "right_eye_in": 263,
    "mouth_top": 13,
    "mouth_bottom": 14,
    "left_eye_top": 159,
    "left_eye_bottom": 145,
    "right_eye_top": 386,
    "right_eye_bottom": 374,
    "left_brow_inner": 105,
    "right_brow_inner": 334,
    "nose_tip": 1
}

# --- Feature Extraction Functions (MUST be same as extractor) ---
def euclid(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def extract_features(landmarks, image_w, image_h):
    pts = []
    for lm in landmarks:
        pts.append((lm.x * image_w, lm.y * image_h, lm.z * max(image_w, image_h)))
    
    left_eye = pts[IDX["left_eye_in"]]
    right_eye = pts[IDX["right_eye_in"]]
    iod = euclid(left_eye, right_eye)
    if iod < 1e-6:
        iod = 1.0

    mouth_top = pts[IDX["mouth_top"]]
    mouth_bottom = pts[IDX["mouth_bottom"]]
    mouth_open = euclid(mouth_top, mouth_bottom) / iod

    left_eye_top = pts[IDX["left_eye_top"]]
    left_eye_bottom = pts[IDX["left_eye_bottom"]]
    left_eye_open = euclid(left_eye_top, left_eye_bottom) / iod

    right_eye_top = pts[IDX["right_eye_top"]]
    right_eye_bottom = pts[IDX["right_eye_bottom"]]
    right_eye_open = euclid(right_eye_top, right_eye_bottom) / iod
    
    eye_open = (left_eye_open + right_eye_open) / 2.0

    left_brow = pts[IDX["left_brow_inner"]]
    right_brow = pts[IDX["right_brow_inner"]]
    left_brow_eye = abs(left_brow[1] - left_eye_top[1]) / iod
    right_brow_eye = abs(right_brow[1] - right_eye_top[1]) / iod
    brow_eye = (left_brow_eye + right_brow_eye) / 2.0

    nose = pts[IDX["nose_tip"]]
    img_center_y = image_h / 2.0
    nose_center_offset = (nose[1] - img_center_y) / image_h

    features = {
        "mouth_open": mouth_open,
        "eye_open": eye_open,
        "left_eye_open": left_eye_open,
        "right_eye_open": right_eye_open,
        "brow_eye": brow_eye,
        "nose_center_offset": nose_center_offset
    }
    return features, list(features.keys())
# --- End Helper Functions ---


# --- Load Models ---
print("Loading hybrid model...")
try:
    with open(MODEL_PATH, 'rb') as f:
        model_payload = pickle.load(f)
    model = model_payload['model']
    scaler = model_payload['scaler']
except FileNotFoundError as e:
    print(f"Error: Could not load model file. {e}")
    print(f"Please run '2_...py' and '3_...py' first to create {MODEL_PATH}")
    exit()
except Exception as e:
    print(f"An error occurred loading models: {e}")
    exit()

print("MediaPipe models and pain classifier loaded successfully.")

# --- Initialize MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# --- Initialize Webcam ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cv2.namedWindow("Hybrid Pain Detector", cv2.WINDOW_NORMAL)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# List to store pain history for the report
pain_history = []
start_time = time.time()
print("Starting real-time pain detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    
    # Convert to RGB for MediaPipe
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_image.flags.writeable = False # Performance optimization
    results = face_mesh.process(rgb_image)
    rgb_image.flags.writeable = True

    current_pain_level = -1
    current_pain_label = "No Face Detected"
    current_pain_color = (128, 128, 128)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        
        # Calculate features
        feats, feature_names = extract_features(lm, frame_width, frame_height)
        
        # Prepare features for the model
        # Ensure order is the same as when training!
        feature_vector = [feats[name] for name in feature_names]
        features_np = np.array(feature_vector).reshape(1, -1)
        features_scaled = scaler.transform(features_np)
        
        # Predict pain level
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        confidence = np.max(prediction_proba)
        
        current_pain_level = int(prediction)
        current_pain_label = PAIN_LABELS[current_pain_level]
        current_pain_color = PAIN_COLORS[current_pain_level]
        
        # Log for report
        pain_history.append(current_pain_level)
        
        # --- Draw on display_frame ---
        label_text = f"{current_pain_label} ({confidence * 100:.1f}%)"
        cv2.putText(display_frame, label_text, (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, current_pain_color, 2)
    
    else:
        # No face detected
        pain_history.append(np.nan) # Log NaN when no face is seen

    # --- Draw the Visual Pain Meter ---
    meter_height = 50
    meter_width = frame_width - 40
    meter_x = 20
    meter_y = frame_height - meter_height - 20
    
    cv2.rectangle(display_frame, (meter_x, meter_y), 
                  (meter_x + meter_width, meter_y + meter_height), (0, 0, 0), -1)
    cv2.rectangle(display_frame, (meter_x, meter_y), 
                  (meter_x + meter_width, meter_y + meter_height), (255, 255, 255), 1)
    
    if current_pain_level >= 0:
        bar_width = int(( (current_pain_level + 1) / 4.0 ) * meter_width)
    else:
        bar_width = 0
        
    cv2.rectangle(display_frame, (meter_x, meter_y), 
                  (meter_x + bar_width, meter_y + meter_height), current_pain_color, -1)
    
    cv2.putText(display_frame, "Pain Intensity", (meter_x + 5, meter_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display_frame, current_pain_label, (meter_x + meter_width - 150, meter_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Hybrid Pain Detector", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# --- Cleanup and Report Generation ---
print("\n--- Shutting Down ---")
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
session_duration = time.time() - start_time

print(f"Session lasted {session_duration:.2f} seconds.")
print(f"Total frames processed: {len(pain_history)}")

# Generate and save the report plot
try:
    df = pd.Series(pain_history).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    
    plt.figure(figsize=(15, 5))
    df.plot(color='red')
    plt.title(f"Pain Intensity Report (Session: {session_duration:.1f}s)")
    plt.xlabel("Time (frames)")
    plt.ylabel("Pain Level")
    plt.yticks([0, 1, 2, 3], ['No Pain', 'Slight', 'Mild', 'Intense'])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(REPORT_PLOT_PATH)
    print(f"Pain report plot saved to {REPORT_PLOT_PATH}")
except Exception as e:
    print(f"Could not generate report plot. Error: {e}")

print("Done.")