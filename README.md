# MedTrust-Audit

Clinical Trust Evaluation Tool for Medical Imaging AI

## What It Does
Automatically evaluates medical imaging models across 4 trust pillars:
- **Discrimination:** Accuracy, Precision, Recall, F1
- **Calibration:** Expected Calibration Error (ECE)
- **High-Confidence Error Audit:** Silent failure detection
- **Clinical Readiness Index (CRI):** Composite deploy/review/reject score

## Based On
Our IEEE paper: "Beyond Accuracy: A Multi-Pillar Clinical Trust Framework for Brain Tumor MRI Classification"

## Sample Result
| Metric | Value |
|--------|-------|
| Accuracy | 98.48% |
| ECE | 0.011 |
| High-Conf Error Rate | 1% |
| CRI Score | 0.9876 |
| **Verdict** | **DEPLOY** |

## How to Run
```bash
pip install numpy scikit-learn
python test_model.py



Model
The trained model (217MB) exceeds GitHub's limit. Train your own using the Kaggle notebook or contact for access.


Author
Malaika Arif — COMSATS University Islamabad


### Step 3: Build FastAPI Wrapper (Optional but Impressive)
If you want a live endpoint, create `main.py`:

```python
from fastapi import FastAPI
import numpy as np
from evaluators.discrimination import compute_metrics
from evaluators.calibration import compute_ece
from evaluators.high_conf_errors import find_high_confidence_errors
from cri.clinical_readiness_index import compute_cri

app = FastAPI(title="MedTrust-Audit")

@app.get("/audit")
def audit():
    y_true = np.load('y_true.npy')
    y_pred = np.load('y_pred.npy')
    y_pred_probs = np.load('y_pred_probs.npy')
    
    disc = compute_metrics(y_true, y_pred)
    ece = compute_ece(y_true, y_pred_probs)
    hce = find_high_confidence_errors(y_true, y_pred, y_pred_probs)
    cri = compute_cri(disc['accuracy'], ece, hce['high_confidence_error_rate'])
    
    return {
        "discrimination": disc,
        "calibration": {"ece": ece},
        "high_confidence_audit": hce,
        "clinical_readiness": cri
    }

@app.get("/")
def home():
    return {"message": "MedTrust-Audit API", "endpoint": "/audit"}