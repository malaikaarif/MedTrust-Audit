import numpy as np
from evaluators.discrimination import compute_metrics
from evaluators.calibration import compute_ece
from evaluators.high_conf_errors import find_high_confidence_errors
from cri.clinical_readiness_index import compute_cri

# Load REAL predictions
y_true = np.load('y_true.npy')
y_pred = np.load('y_pred.npy')
y_pred_probs = np.load('y_pred_probs.npy')

print("=== DISCRIMINATION ===")
disc = compute_metrics(y_true, y_pred)
print(disc)

print("\n=== CALIBRATION ===")
ece = compute_ece(y_true, y_pred_probs)
print(f"ECE: {ece}")

print("\n=== HIGH-CONFIDENCE ERRORS ===")
hce = find_high_confidence_errors(y_true, y_pred, y_pred_probs)
print(hce)

print("\n=== CLINICAL READINESS INDEX ===")
cri = compute_cri(disc['accuracy'], ece, hce['high_confidence_error_rate'])
print(cri)