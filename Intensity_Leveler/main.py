import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import math
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import seaborn as sns

# --- Configuration ---
MODEL_PATH = "models/pain_model_hybrid.pkl"
DEFAULT_REPORT_PATH = "output/analysis_report.png"

# Pain labels and visual meter colors
PAIN_LABELS = {0: "No Pain", 1: "Slight Pain", 2: "Mild Pain", 3: "Intense Pain"}
PAIN_COLORS = {
    0: (0, 0, 255),    # Green
    1: (0, 0, 255),  # Yellow
    2: (0, 0, 255),  # Orange
    3: (0, 0, 255)     # Red
}

# --- MediaPipe Landmark Indices (MUST be same as extractor) ---
IDX = {
    "left_eye_in": 33, "right_eye_in": 263, "mouth_top": 13, "mouth_bottom": 14,
    "left_eye_top": 159, "left_eye_bottom": 145, "right_eye_top": 386, "right_eye_bottom": 374,
    "left_brow_inner": 105, "right_brow_inner": 334, "nose_tip": 1
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
        "mouth_open": mouth_open, "eye_open": eye_open, "left_eye_open": left_eye_open,
        "right_eye_open": right_eye_open, "brow_eye": brow_eye, "nose_center_offset": nose_center_offset
    }
    return features, list(features.keys())
# --- End Helper Functions ---

# --- Drawing Function ---
def draw_pain_meter(image, pain_level_idx, pain_intensity_pct, frame_height, frame_width):
    """Draws the pain meter and text onto the display frame."""
    
    pain_label = PAIN_LABELS.get(pain_level_idx, "No Face")
    pain_color = PAIN_COLORS.get(pain_level_idx, (128, 128, 128))
    
    # --- Draw the Visual Pain Meter ---
    meter_height = 40
    meter_width = frame_width - 40
    meter_x = 20
    meter_y = frame_height - meter_height - 20
    
    # Black background
    cv2.rectangle(image, (meter_x, meter_y), 
                  (meter_x + meter_width, meter_y + meter_height), (0, 0, 0), -1)
    # White border
    cv2.rectangle(image, (meter_x, meter_y), 
                  (meter_x + meter_width, meter_y + meter_height), (255, 255, 255), 1)
    
    # Draw the intensity bar
    bar_width = int(pain_intensity_pct / 100.0 * meter_width)
    cv2.rectangle(image, (meter_x, meter_y), 
                  (meter_x + bar_width, meter_y + meter_height), pain_color, -1)
    
    # --- Draw Text ---
    # Main label: "Mild Pain (75.2%)"
    label_text = f"{pain_label} ({pain_intensity_pct:.1f}%)"
    cv2.putText(image, label_text, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, pain_color, 3)
    
    # Meter title
    cv2.putText(image, "Pain Intensity", (meter_x + 5, meter_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0
                ), 2)
# --- End Drawing Function ---


# --- Report Generation ---
def generate_report(pain_levels, pain_intensities, fps, report_path):
    """Generates a 2-part report graph and saves it to a file."""
    
    print(f"Generating report... Frames: {len(pain_levels)}, FPS: {fps:.2f}")
    
    if not pain_levels or fps is None or fps == 0:
        print("Error: Cannot generate report. No data or invalid FPS.")
        return
        
    # Interpolate to fill missing (No Face) values for a smoother graph
    df_levels = pd.Series(pain_levels).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    df_intensity = pd.Series(pain_intensities).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    
    if df_intensity.empty:
        print("Error: No valid pain data to plot.")
        return

    session_duration = len(pain_levels) / fps
    time_axis = np.linspace(0, session_duration, len(df_intensity))
    
    # --- Create Plot ---
    plt.figure(figsize=(15, 10))
    
    # --- Plot 1: Pain Intensity (0-100%) over Time ---
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, df_intensity, color='red', alpha=0.6, label='Frame-by-Frame Intensity')
    
    # Add a rolling average (e.g., 2-second window)
    window_size = int(fps * 2) # 2-second moving average
    if window_size < 1: window_size = 1
    df_smooth = df_intensity.rolling(window=window_size, min_periods=1, center=True).mean()
    plt.plot(time_axis, df_smooth, color='blue', linewidth=2, label='2-Second Average')
    
    plt.title("Pain Intensity Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Pain Intensity (0-100%)")
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # --- Plot 2: Pain Level Frequency Histogram ---
    plt.subplot(2, 1, 2)
    counts = df_levels.value_counts(normalize=True).sort_index()
    # Ensure all 4 labels exist in the plot, even if they have 0 count
    counts = counts.reindex([0, 1, 2, 3], fill_value=0)
    
    # Use seaborn for a nice bar plot
    sns.barplot(x=counts.index, y=counts.values * 100, palette=PAIN_COLORS.values())
    
    plt.title("Pain Level Frequency Distribution")
    plt.xlabel("Pain Level")
    plt.ylabel("Percentage of Session (%)")
    plt.xticks(ticks=[0, 1, 2, 3], labels=PAIN_LABELS.values())
    plt.ylim(0, 100)
    
    # --- Save Plot ---
    plt.tight_layout()
    plt.savefig(report_path)
    print(f"Successfully saved analysis report to {report_path}")

# --- Main Processing ---
def main(input_path, output_path, report_path):
    # --- Load Models ---
    print("Loading hybrid model...")
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_payload = pickle.load(f)
        model = model_payload['model']
        scaler = model_payload['scaler']
    except Exception as e:
        print(f"Fatal Error: Could not load model file '{MODEL_PATH}'. {e}")
        return

    # --- Initialize MediaPipe ---
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, # False is better for videos
                                      max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5)

    # --- Check if input is Image or Video ---
    ext = os.path.splitext(input_path)[1].lower()
    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv']
    is_image = ext in ['.jpg', '.jpeg', '.png', '.bmp']

    if not is_video and not is_image:
        print(f"Error: Input file '{input_path}' is not a recognized image or video format.")
        return

    # --- Process Video File ---
    if is_video:
        print(f"Processing video file: {input_path}")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_path}")
            return
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        pain_history_levels = []
        pain_history_intensity = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # End of video
            
            frame_count += 1
            display_frame = frame.copy()
            
            # Process with MediaPipe
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_image.flags.writeable = False # Performance optimization
            results = face_mesh.process(rgb_image)
            
            current_pain_level = -1
            current_intensity_pct = 0.0
            
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                
                # Calculate features
                feats, feature_names = extract_features(lm, frame_width, frame_height)
                feature_vector = [feats[name] for name in feature_names]
                features_np = np.array(feature_vector).reshape(1, -1)
                features_scaled = scaler.transform(features_np)
                
                # Predict
                prediction = model.predict(features_scaled)[0]
                prediction_proba = model.predict_proba(features_scaled)[0]
                
                current_pain_level = int(prediction)
                
                # --- NEW: Calculate 0-100% Weighted Intensity Score ---
                # (0*P_no) + (1*P_slight) + (2*P_mild) + (3*P_intense)
                # Then divide by 3 to normalize to a 0.0-1.0 scale
                weighted_sum = (prediction_proba[1] * 1) + (prediction_proba[2] * 2) + (prediction_proba[3] * 3)
                current_intensity_pct = (weighted_sum / 3.0) * 100
                
                pain_history_levels.append(current_pain_level)
                pain_history_intensity.append(current_intensity_pct)
                
            else:
                # No face detected in this frame
                pain_history_levels.append(np.nan)
                pain_history_intensity.append(np.nan)

            # Draw meter and text
            draw_pain_meter(display_frame, current_pain_level, current_intensity_pct, frame_height, frame_width)
            
            # Write the annotated frame to the output video
            out_video.write(display_frame)

            # Show progress
            if frame_count % 10 == 0:
                print(f"Processing frame {frame_count} / {total_frames} ({current_intensity_pct:.1f}%)")

        # Cleanup
        cap.release()
        out_video.release()
        face_mesh.close()
        print(f"--- Video Processing Finished ---")
        print(f"Annotated video saved to: {output_path}")

        # Generate the final report graph
        if not report_path:
            report_path = DEFAULT_REPORT_PATH
        generate_report(pain_history_levels, pain_history_intensity, fps, report_path)
    
    # --- Process Single Image File ---
    elif is_image:
        print(f"Processing image file: {input_path}")
        frame = cv2.imread(input_path)
        if frame is None:
            print(f"Error: Could not read image {input_path}")
            return

        frame_height, frame_width = frame.shape[:2]
        display_frame = frame.copy()

        # Process with MediaPipe
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            feats, feature_names = extract_features(lm, frame_width, frame_height)
            feature_vector = [feats[name] for name in feature_names]
            features_np = np.array(feature_vector).reshape(1, -1)
            features_scaled = scaler.transform(features_np)
            
            prediction = model.predict(features_scaled)[0]
            prediction_proba = model.predict_proba(features_scaled)[0]
            
            current_pain_level = int(prediction)
            weighted_sum = (prediction_proba[1] * 1) + (prediction_proba[2] * 2) + (prediction_proba[3] * 3)
            current_intensity_pct = (weighted_sum / 3.0) * 100
            
            print(f"Detection Complete:")
            print(f"  Pain Level: {PAIN_LABELS[current_pain_level]} (Class {current_pain_level})")
            print(f"  Pain Intensity: {current_intensity_pct:.2f}%")
            
            # Draw results
            draw_pain_meter(display_frame, current_pain_level, current_intensity_pct, frame_height, frame_width)
        
        else:
            print("No face detected in the image.")
            cv2.putText(display_frame, "No Face Detected", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        # Save the annotated image
        cv2.imwrite(output_path, display_frame)
        print(f"Annotated image saved to: {output_path}")
        face_mesh.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze pain from a video or image file.")
    parser.add_argument("-i", "--input", required=True, 
                        help="Path to the input video or image file.")
    parser.add_argument("-o", "--output", required=True, 
                        help="Path to save the output annotated video or image.")
    parser.add_argument("-r", "--report", 
                        help="Path to save the report graph (video only).")
    
    args = parser.parse_args()
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
    
    main(args.input, args.output, args.report)