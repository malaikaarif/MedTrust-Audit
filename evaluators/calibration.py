import numpy as np
from sklearn.metrics import accuracy_score

def compute_ece(y_true, y_pred_probs, n_bins=10):
    confidences = np.max(y_pred_probs, axis=1)
    predictions = np.argmax(y_pred_probs, axis=1)
    
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if np.sum(mask) > 0:
            bin_acc = accuracy_score(y_true[mask], predictions[mask])
            bin_conf = np.mean(confidences[mask])
            ece += np.sum(mask) * np.abs(bin_acc - bin_conf)
    
    return round(ece / len(y_true), 4)