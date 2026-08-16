from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(y_true, y_pred):
    return {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred, average='weighted'), 4),
        'recall': round(recall_score(y_true, y_pred, average='weighted'), 4),
        'f1': round(f1_score(y_true, y_pred, average='weighted'), 4)
    }