import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import pickle
import joblib # Using joblib or pickle is fine

# --- Configuration ---
FEATURES_CSV = "features_hybrid.csv"
MODEL_PATH = "models/pain_model_hybrid.pkl" # .pkl or .joblib
EVAL_REPORT_PATH = "output/evaluation_report_hybrid.txt"
CM_PLOT_PATH = "output/confusion_matrix_hybrid.png"
LABELS = ["0_no_pain", "1_slight", "2_mild", "3_intense"]

# --- Load Data ---
print(f"Loading features from {FEATURES_CSV}...")
try:
    data = pd.read_csv(FEATURES_CSV)
except FileNotFoundError:
    print(f"Error: {FEATURES_CSV} not found.")
    print("Please run '2_feature_extractor_hybrid.py' first.")
    exit()

if data.empty:
    print("Error: The features.csv file is empty. Did data collection and extraction work?")
    exit()

# Separate features (X) and labels (y)
X = data.drop('label', axis=1)
y = data['label']

print(f"Loaded {len(X)} samples.")
print(f"Number of features: {len(X.columns)}")
print(f"Features: {list(X.columns)}")

# --- Split Data ---
# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# --- Feature Scaling ---
# Scale features to have zero mean and unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train Model ---
print("Training RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)
print("Model training complete.")

# --- Save Model and Scaler ---
# We must save the scaler as well to process live data!
model_payload = {
    'model': model,
    'scaler': scaler
}

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model_payload, f)
print(f"Model and scaler saved to {MODEL_PATH}")

# --- Evaluate Model ---
print("Evaluating model on test data...")
y_pred = model.predict(X_test_scaled)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Generate Classification Report (Precision, Recall, F1-Score)
report = classification_report(y_test, y_pred, target_names=LABELS)
print("\nClassification Report:")
print(report)

# Generate Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Save text-based evaluation report
with open(EVAL_REPORT_PATH, 'w') as f:
    f.write(f"--- Model Evaluation Report (Hybrid) ---\n\n")
    f.write(f"Model: RandomForestClassifier (n_estimators=100)\n")
    f.write(f"Features: {list(X.columns)}\n")
    f.write(f"Test Set Size: {len(X_test)}\n\n")
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(np.array2string(cm))
print(f"Evaluation report saved to {EVAL_REPORT_PATH}")

# --- Plot and Save Confusion Matrix ---
try:
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=LABELS, yticklabels=LABELS)
    plt.title('Hybrid Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(CM_PLOT_PATH)
    print(f"Confusion matrix plot saved to {CM_PLOT_PATH}")
except Exception as e:
    print(f"Could not save confusion matrix plot. Error: {e}")

print("--- Training and Evaluation Finished ---")