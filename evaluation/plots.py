import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def plot_confusion_matrix(y_true, y_pred, classes=['Angry', 'Happy', 'Neutral', 'Sad'], save_path="outputs/figures/cnn_confusion_matrix.png"):
    """
    Generates and saves a high-resolution confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 12})
    
    plt.title("CNN Model - Confusion Matrix", fontsize=14)
    plt.ylabel('Actual Emotion', fontsize=12)
    plt.xlabel('Predicted Emotion', fontsize=12)
    
    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Confusion Matrix plot saved successfully to {save_path}")