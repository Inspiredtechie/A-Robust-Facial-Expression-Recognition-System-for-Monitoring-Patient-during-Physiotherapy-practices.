import cv2
import os
import time

# --- Configuration ---
# This script just saves images. It is identical to the first version
# as its job is just to collect raw data.
DATA_DIR = "data"
LABELS = ["0_no_pain", "1_slight", "2_mild", "3_intense"]
SAVE_COUNT = 100  # Try to capture this many images per label

# Create directories if they don't exist
for label in LABELS:
    path = os.path.join(DATA_DIR, label)
    os.makedirs(path, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("--- Pain Data Collector ---")
print(f"We will capture {SAVE_COUNT} images for each of the {len(LABELS)} labels.")
print("Press the corresponding key to capture an image for that label.")
print("Make the facial expression and then press the key.")
print("Press 'q' to quit.")
print("-" * 30)

# Create a window
cv2.namedWindow("Data Collector", cv2.WINDOW_NORMAL)

counts = {label: len(os.listdir(os.path.join(DATA_DIR, label))) for label in LABELS}
print(f"Current image counts: {counts}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Flip the frame horizontally (like a mirror)
    frame = cv2.flip(frame, 1)

    # Display instructions on the frame
    y_offset = 30
    for i, label in enumerate(LABELS):
        text = f"Press '{i}': {label} ({counts[label]}/{SAVE_COUNT})"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        y_offset += 30
    cv2.putText(frame, "Press 'q' to QUIT", (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255))

    cv2.imshow("Data Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    label_to_save = None
    if key == ord('q'):
        break
    elif key == ord('0'):
        label_to_save = LABELS[0]
    elif key == ord('1'):
        label_to_save = LABELS[1]
    elif key == ord('2'):
        label_to_save = LABELS[2]
    elif key == ord('3'):
        label_to_save = LABELS[3]

    if label_to_save:
        if counts[label_to_save] < SAVE_COUNT:
            # Save the frame
            timestamp = int(time.time() * 1000)
            filename = f"{label_to_save}_{timestamp}.png"
            save_path = os.path.join(DATA_DIR, label_to_save, filename)
            
            cv2.imwrite(save_path, frame)
            
            counts[label_to_save] += 1
            print(f"Saved: {save_path} | Total for {label_to_save}: {counts[label_to_save]}")
            
            # Show feedback
            feedback_frame = frame.copy()
            cv2.putText(feedback_frame, f"SAVED: {label_to_save}", (frame.shape[1] // 2 - 100, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            cv2.imshow("Data Collector", feedback_frame)
            cv2.waitKey(500) # Pause for 0.5 seconds to show feedback
        else:
            print(f"Note: Already have {SAVE_COUNT} images for {label_to_save}.")

print("--- Data Collection Finished ---")
print(f"Final counts: {counts}")
cap.release()
cv2.destroyAllWindows()