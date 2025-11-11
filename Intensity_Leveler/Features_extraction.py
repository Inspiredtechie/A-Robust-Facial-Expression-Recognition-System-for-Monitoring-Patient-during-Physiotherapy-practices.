import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import math
import csv

# --- Configuration ---
DATA_DIR = "data"
LABELS = ["0_no_pain", "1_slight", "2_mild", "3_intense"]
OUTPUT_CSV = "features_hybrid.csv"

# --- MediaPipe Landmark Indices ---
# 
# These are the specific landmarks we'll use for our geometric features
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

# --- Feature Extraction Functions (from your main.py) ---
def euclid(a, b):
    """Calculates Euclidean distance between two 2D or 3D points."""
    return math.hypot(a[0]-b[0], a[1]-b[1])

def extract_features(landmarks, image_w, image_h):
    """
    Calculates robust geometric features from MediaPipe landmarks.
    landmarks: sequence of mediapipe landmark objects for a single face
    returns: feature vector (dict)
    """
    # convert to pixel coords
    pts = []
    for lm in landmarks:
        pts.append((lm.x * image_w, lm.y * image_h, lm.z * max(image_w, image_h)))

    # robust normalization scale: inter-ocular distance (between inner-eye points)
    left_eye = pts[IDX["left_eye_in"]]
    right_eye = pts[IDX["right_eye_in"]]
    iod = euclid(left_eye, right_eye)
    if iod < 1e-6:
        iod = 1.0  # prevent div by zero

    # mouth open: vertical distance between upper and lower lip
    mouth_top = pts[IDX["mouth_top"]]
    mouth_bottom = pts[IDX["mouth_bottom"]]
    mouth_open = euclid(mouth_top, mouth_bottom) / iod

    # left eye opening (top-bottom)
    left_eye_top = pts[IDX["left_eye_top"]]
    left_eye_bottom = pts[IDX["left_eye_bottom"]]
    left_eye_open = euclid(left_eye_top, left_eye_bottom) / iod

    # right eye opening
    right_eye_top = pts[IDX["right_eye_top"]]
    right_eye_bottom = pts[IDX["right_eye_bottom"]]
    right_eye_open = euclid(right_eye_top, right_eye_bottom) / iod

    # average eye opening
    eye_open = (left_eye_open + right_eye_open) / 2.0

    # brow-to-eye distance (proxy for brow lowering/raising)
    left_brow = pts[IDX["left_brow_inner"]]
    right_brow = pts[IDX["right_brow_inner"]]
    # take brow inner to eye top vertical distance normalized
    left_brow_eye = abs(left_brow[1] - left_eye_top[1]) / iod
    right_brow_eye = abs(right_brow[1] - right_eye_top[1]) / iod
    brow_eye = (left_brow_eye + right_brow_eye) / 2.0

    # nose-tip y relative to face center (proxy of head movement)
    nose = pts[IDX["nose_tip"]]
    img_center_y = image_h / 2.0
    nose_center_offset = (nose[1] - img_center_y) / image_h

    # feature dict
    features = {
        "mouth_open": mouth_open,
        "eye_open": eye_open,
        "left_eye_open": left_eye_open,
        "right_eye_open": right_eye_open,
        "brow_eye": brow_eye,
        "nose_center_offset": nose_center_offset
    }
    return features

# --- Main Feature Extraction ---
print("--- Starting Hybrid Feature Extraction (MediaPipe) ---")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5)

# Define the CSV header
feature_names = [
    "mouth_open", "eye_open", "left_eye_open", 
    "right_eye_open", "brow_eye", "nose_center_offset"
]
header = feature_names + ['label']

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    total_images = 0
    extracted_features = 0

    # Loop over each label directory
    for label_id, label_name in enumerate(LABELS):
        label_dir = os.path.join(DATA_DIR, label_name)
        if not os.path.isdir(label_dir):
            print(f"Warning: Directory not found: {label_dir}")
            continue
        
        print(f"Processing directory: {label_dir}")
        
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            
            try:
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue
                
                h, w = image.shape[:2]
                # Convert to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                total_images += 1
                
                # Process with MediaPipe
                results = face_mesh.process(rgb_image)

            except Exception as e:
                print(f"Error reading or processing {img_path}: {e}")
                continue

            # Check if landmarks were found
            if results.multi_face_landmarks:
                # We assume only one face (the patient's) is in the frame
                lm = results.multi_face_landmarks[0].landmark
                
                # Calculate features
                feats = extract_features(lm, w, h)
                
                # Create data row
                row = [feats[name] for name in feature_names] + [label_id]
                
                # Write to CSV
                writer.writerow(row)
                extracted_features += 1
            else:
                print(f"Warning: No face detected in {img_path}. Skipping.")

face_mesh.close()
print("--- Feature Extraction Finished ---")
print(f"Processed {total_images} images.")
print(f"Successfully extracted features from {extracted_features} images.")
print(f"Features saved to {OUTPUT_CSV}")