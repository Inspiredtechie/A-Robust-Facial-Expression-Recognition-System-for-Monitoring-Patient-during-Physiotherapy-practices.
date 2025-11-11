# preprocess_data.py
import os
import cv2
import random
import shutil
from pathlib import Path

# --- Configuration ---
# Use a consistent image size across all scripts
IMG_SIZE = 48 
RAW_DATA_PATH = Path('raw_dataset/')
PROCESSED_PATH = Path('processed_dataset/')
EMOTIONS = ["Angry", "Happy", "Normal", "Sad"]

# Use a more robust face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_images():
    """
    Detects faces, converts images to grayscale, resizes them, and splits
    them into randomized train, validation, and test sets.
    """
    # Clean up previous runs
    if PROCESSED_PATH.exists():
        shutil.rmtree(PROCESSED_PATH)
    print(f"Starting preprocessing. Images will be resized to {IMG_SIZE}x{IMG_SIZE}.")

    for emotion in EMOTIONS:
        # Create directories for train, validation, and test sets
        for split in ['train', 'validation', 'test']:
            os.makedirs(PROCESSED_PATH / split / emotion, exist_ok=True)

        source_dir = RAW_DATA_PATH / emotion
        if not source_dir.exists():
            print(f"Warning: Source directory not found: {source_dir}")
            continue

        image_files = list(source_dir.glob('*.jpg')) # Use glob for specific file types
        random.shuffle(image_files) # Shuffle for random split

        # Splitting data: 70% train, 15% validation, 15% test
        train_split = int(len(image_files) * 0.70)
        val_split = int(len(image_files) * 0.85)
        
        train_files = image_files[:train_split]
        val_files = image_files[train_split:val_split]
        test_files = image_files[val_split:]
        
        file_sets = {
            'train': (train_files, PROCESSED_PATH / 'train' / emotion),
            'validation': (val_files, PROCESSED_PATH / 'validation' / emotion),
            'test': (test_files, PROCESSED_PATH / 'test' / emotion)
        }
        
        print(f"--- Processing Emotion: {emotion} ---")
        for set_name, (files, dest_dir) in file_sets.items():
            count = 0
            for image_path in files:
                try:
                    image = cv2.imread(str(image_path))
                    if image is None:
                        print(f"Warning: Could not read {image_path}, skipping.")
                        continue
                    
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    
                    if len(faces) == 0:
                        print(f"Warning: No face detected in {image_path}, skipping.")
                        continue

                    for (x, y, w, h) in faces:
                        face_roi = gray[y:y+h, x:x+w]
                        resized_face = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
                        
                        new_filename = f"{emotion}_{count}.jpg"
                        cv2.imwrite(str(dest_dir / new_filename), resized_face)
                        count += 1
                        break # Process only the first detected face
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
            print(f"Saved {count} images to {set_name} set for {emotion}.")

if __name__ == '__main__':
    preprocess_images()