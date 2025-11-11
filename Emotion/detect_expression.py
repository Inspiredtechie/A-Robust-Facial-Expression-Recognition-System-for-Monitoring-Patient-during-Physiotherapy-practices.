# detect_expression.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import sys

# --- Configuration ---
MODEL_PATH = 'expression_model.h5'
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# This now comes from the training script, making it robust
CLASS_INDICES_PATH = 'class_indices.json' 
IMG_SIZE = 48 # Must match IMG_SIZE in training

def predict_expression(image_path):
    """
    Loads an image, detects faces, and predicts the expression for each face.
    """
    # Load the trained model and face detector
    try:
        model = load_model(MODEL_PATH)
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        with open(CLASS_INDICES_PATH, 'r') as f:
            # Load the emotion labels from the JSON file
            emotion_labels = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: A required file is missing. {e}")
        print("Please ensure 'expression_model.h5', 'class_indices.json', and the cascade file exist.")
        return

    # cap = cv2.VideoCapture(0)  # 0 is the default webcam
    # if not cap.isOpened():
    #     print("Error: Could not open webcam.")
    #     return
    #

    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         print("Failed to grab frame.")
    #         break

    # Load the input image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Image not found or unable to load from '{image_path}'.")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No faces detected in the image.")

    for (x, y, w, h) in faces:
        # Extract the face ROI and preprocess it
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE))
        roi_gray = roi_gray.astype('float') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        # Predict the emotion
        prediction = model.predict(roi_gray)[0]
        confidence = np.max(prediction)
        # Get emotion label from the loaded dictionary
        emotion_index = str(prediction.argmax())
        emotion_label = emotion_labels.get(emotion_index, "Unknown")
        
        # Format label text with confidence
        label_text = f"{emotion_label}: {confidence:.2f}"
        
        # Display the results
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the final image
    cv2.imshow('Facial Expression Detection', frame)
    print("Press any key to close the window.")

    # cv2.imshow('Facial Expression Detection - Live', frame)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cap.release()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def predict_expression_live():
    """
    Captures video from webcam and predicts the expression for each detected face in real-time.
    """
    # Load the trained model and face detector
    try:
        model = load_model(MODEL_PATH)
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        with open(CLASS_INDICES_PATH, 'r') as f:
            emotion_labels = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: A required file is missing. {e}")
        return

    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE))
            roi_gray = roi_gray.astype('float') / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)

            prediction = model.predict(roi_gray)[0]
            confidence = np.max(prediction)
            emotion_index = str(prediction.argmax())
            emotion_label = emotion_labels.get(emotion_index, "Unknown")
            label_text = f"{emotion_label}: {confidence:.2f}"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Facial Expression Detection - Live', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    predict_expression_live()