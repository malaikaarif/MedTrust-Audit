import numpy as np

def find_high_confidence_errors(y_true, y_pred, probs, threshold=0.9):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    
    high_conf_mask = confidences >= threshold
    wrong_mask = predictions != y_true
    
    error_mask = high_conf_mask & wrong_mask
    total_high_conf = np.sum(high_conf_mask)
    errors = np.sum(error_mask)
    
    error_rate = errors / total_high_conf if total_high_conf > 0 else 0
    
    return {
        'high_confidence_error_rate': round(float(error_rate), 4),
        'total_high_conf_predictions': int(total_high_conf),
        'high_conf_errors': int(errors)
    }