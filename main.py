from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import numpy as np
from evaluators.discrimination import compute_metrics
from evaluators.calibration import compute_ece
from evaluators.high_conf_errors import find_high_confidence_errors
from cri.clinical_readiness_index import compute_cri

app = FastAPI(title="MedTrust-Audit")

def run_audit():
    y_true = np.load('y_true.npy')
    y_pred = np.load('y_pred.npy')
    y_pred_probs = np.load('y_pred_probs.npy')
    
    disc = compute_metrics(y_true, y_pred)
    ece = compute_ece(y_true, y_pred_probs)
    hce = find_high_confidence_errors(y_true, y_pred, y_pred_probs)
    cri = compute_cri(disc['accuracy'], ece, hce['high_confidence_error_rate'], generalization=1.0)
    
    return {
        "discrimination": disc,
        "calibration": {"ece": ece},
        "high_confidence_audit": hce,
        "clinical_readiness": cri
    }

@app.get("/audit")
def audit():
    return run_audit()

@app.get("/", response_class=HTMLResponse)
def dashboard():
    r = run_audit()
    color = "#22c55e" if r["clinical_readiness"]["verdict"] == "DEPLOY" else "#eab308" if r["clinical_readiness"]["verdict"] == "REVIEW" else "#ef4444"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head><title>MedTrust-Audit</title></head>
    <body style="font-family: Arial; max-width: 600px; margin: 40px auto;">
        <h1>MedTrust-Audit</h1>
        <p>Clinical Trust Evaluation for Medical Imaging AI</p>
        
        <div style="background: {color}; color: white; padding: 20px; border-radius: 8px; font-size: 24px; text-align: center;">
            <strong>CRI: {r['clinical_readiness']['cri_score']}</strong><br>
            {r['clinical_readiness']['verdict']}
        </div>
        
        <h2>Discrimination</h2>
       <p>Accuracy: {r['discrimination']['accuracy']*100:.4f}%</p>
       <p>Precision: {r['discrimination']['precision']*100:.4f}%</p>
       <p>Recall: {r['discrimination']['recall']*100:.4f}%</p>
<p>F1: {r['discrimination']['f1']*100:.4f}%</p>
        <h2>Calibration</h2>
        <p>Expected Calibration Error: {r['calibration']['ece']}</p>
        
        <h2>High-Confidence Audit</h2>
        <p>Silent Error Rate: {r['high_confidence_audit']['high_confidence_error_rate']*100:.2f}%</p>
        <p>Errors: {r['high_confidence_audit']['high_conf_errors']} / {r['high_confidence_audit']['total_high_conf_predictions']}</p>
        
        <hr>
        <p><a href="/audit">Raw JSON</a></p>
    </body>
    </html>
    """
    return html