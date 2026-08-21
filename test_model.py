from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

y_true = np.load('y_true.npy')
y_pred = np.load('y_pred.npy')

print("=== CONFUSION MATRIX ===")
print(confusion_matrix(y_true, y_pred))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, target_names=['glioma', 'meningioma', 'notumor', 'pituitary']))

print("\n=== RAW METRICS (unrounded) ===")
print(f"Accuracy:  {accuracy_score(y_true, y_pred)}")
print(f"Precision: {precision_score(y_true, y_pred, average='weighted')}")
print(f"Recall:    {recall_score(y_true, y_pred, average='weighted')}")
print(f"F1:        {f1_score(y_true, y_pred, average='weighted')}")

print("\n=== MACRO AVERAGE (more honest for imbalanced data) ===")
print(f"Precision (macro): {precision_score(y_true, y_pred, average='macro')}")
print(f"Recall (macro):    {recall_score(y_true, y_pred, average='macro')}")
print(f"F1 (macro):        {f1_score(y_true, y_pred, average='macro')}")