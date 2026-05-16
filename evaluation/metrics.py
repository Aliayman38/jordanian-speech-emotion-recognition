from sklearn.metrics import classification_report, accuracy_score
import os

def save_classification_report(y_true, y_pred, classes=['Angry', 'Happy', 'Neutral', 'Sad'], save_path="outputs/logs/cnn_classification_report.txt"):
    """
    Calculates detailed metrics and saves them to a text file.
    """
    acc = accuracy_score(y_true, y_pred) * 100
    report = classification_report(y_true, y_pred, target_names=classes)
    
    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "w") as f:
        f.write("="*50 + "\n")
        f.write("           CNN MODEL EVALUATION REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Overall Accuracy : {acc:.2f}%\n\n")
        f.write("Detailed Metrics per Class:\n")
        f.write("-" * 50 + "\n")
        f.write(report)
        
    print(f"[INFO] Classification Report saved successfully to {save_path}")
    return acc