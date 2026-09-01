import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def train_gesture_model(dataset_path=None, model_save_path=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if dataset_path is None:
        dataset_path = os.path.join(base_dir, "dataset", "keypoint_dataset.csv")
    if model_save_path is None:
        model_save_path = os.path.join(base_dir, "models", "rf_model.pkl")

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found.")
        return False
        
    print("Loading dataset...")
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"Could not read dataset: {e}")
        return False
        
    if len(df) == 0:
        print("Dataset is empty.")
        return False
        
    print(f"Dataset shape: {df.shape}")
    
    # We expect the first column to be the label (class name)
    # The remaining columns are features
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    
    # Split the dataset
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples.")
    print("Model selected: Random Forest Classifier")
    
    # Initialize and train the model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = rf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"\n--- Validation Accuracy: {acc * 100:.2f}% ---")
    
    cm = confusion_matrix(y_val, y_pred, labels=rf.classes_)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClasses mapping:")
    for idx, c in enumerate(rf.classes_):
        print(f"{idx}: {c}")
        
    # Save confusion matrix plot 
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(rf.classes_))
    plt.xticks(tick_marks, rf.classes_, rotation=45)
    plt.yticks(tick_marks, rf.classes_)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    cm_path = os.path.join(os.path.dirname(model_save_path), "confusion_matrix.png")
    plt.savefig(cm_path)
    
    print(f"\nConfusion matrix plot saved to {cm_path}")
    
    # Save the trained model
    joblib.dump(rf, model_save_path)
    print(f"Model successfully saved to {model_save_path}")
    
    return True

if __name__ == "__main__":
    train_gesture_model()
