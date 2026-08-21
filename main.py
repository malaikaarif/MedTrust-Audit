from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from evaluators.discrimination import compute_metrics
from evaluators.calibration import compute_ece
from evaluators.high_conf_errors import find_high_confidence_errors
from cri.clinical_readiness_index import compute_cri
from fastapi.staticfiles import StaticFiles
from explainability import render_explainability_section

app = FastAPI(title="MedTrust-Audit")
app.mount("/explainability", StaticFiles(directory="explainability"), name="explainability")

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']


def run_audit():
    y_true = np.load('y_true.npy')
    y_pred = np.load('y_pred.npy')
    y_pred_probs = np.load('y_pred_probs.npy')

    disc = compute_metrics(y_true, y_pred)
    ece = compute_ece(y_true, y_pred_probs)
    hce = find_high_confidence_errors(y_true, y_pred, y_pred_probs)
    cri = compute_cri(disc['accuracy'], ece, hce['high_confidence_error_rate'], generalization=1.0)

    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)

    return {
        "discrimination": disc,
        "calibration": {"ece": ece},
        "high_confidence_audit": hce,
        "clinical_readiness": cri,
        "confusion_matrix": cm,
        "per_class": {name: report[name] for name in CLASS_NAMES}
    }


@app.get("/audit")
def audit():
    return run_audit()


@app.get("/", response_class=HTMLResponse)
def dashboard():
    r = run_audit()
    verdict = r["clinical_readiness"]["verdict"]
    color = "#16a34a" if verdict == "DEPLOY" else "#ca8a04" if verdict == "REVIEW" else "#dc2626"

    cm = r["confusion_matrix"]
    n = len(CLASS_NAMES)
    max_cell = max(max(row) for row in cm)

    def cell_style(val):
        intensity = val / max_cell if max_cell else 0
        bg = f"rgba(37, 99, 235, {0.08 + intensity * 0.55})"
        weight = "700" if intensity > 0.4 else "400"
        return f"background:{bg}; font-weight:{weight};"

    cm_rows = ""
    for i, row in enumerate(cm):
        cells = "".join(f"<td style='{cell_style(v)}'>{v}</td>" for v in row)
        cm_rows += f"<tr><th>{CLASS_NAMES[i]}</th>{cells}</tr>"

    cm_header = "".join(f"<th>{c}</th>" for c in CLASS_NAMES)

    per_class_rows = ""
    for name in CLASS_NAMES:
        pc = r["per_class"][name]
        per_class_rows += f"""
        <tr>
            <td>{name}</td>
            <td>{pc['precision']*100:.2f}%</td>
            <td>{pc['recall']*100:.2f}%</td>
            <td>{pc['f1-score']*100:.2f}%</td>
            <td>{int(pc['support'])}</td>
        </tr>"""

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MedTrust-Audit</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            :root {{ --ink:#1a1a1a; --muted:#6b7280; --border:#e5e7eb; --bg:#fafafa; }}
            * {{ box-sizing: border-box; }}
            body {{
                font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
                background: var(--bg);
                color: var(--ink);
                max-width: 760px;
                margin: 48px auto;
                padding: 0 24px;
                line-height: 1.5;
            }}
            h1 {{ font-size: 28px; margin-bottom: 4px; }}
            .subtitle {{ color: var(--muted); margin-top: 0; margin-bottom: 28px; }}
            .cri-card {{
                background: {color};
                color: white;
                padding: 28px;
                border-radius: 12px;
                text-align: center;
                margin-bottom: 32px;
                box-shadow: 0 4px 14px rgba(0,0,0,0.08);
            }}
            .cri-score {{ font-size: 40px; font-weight: 800; letter-spacing: -1px; }}
            .cri-verdict {{ font-size: 16px; letter-spacing: 2px; opacity: 0.9; margin-top: 4px; }}
            .pillar {{
                background: white;
                border: 1px solid var(--border);
                border-radius: 10px;
                padding: 20px 24px;
                margin-bottom: 20px;
            }}
            .pillar h2 {{ font-size: 15px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin: 0 0 14px 0; }}
            .metric-row {{ display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #f3f4f6; }}
            .metric-row:last-child {{ border-bottom: none; }}
            .metric-label {{ color: var(--muted); }}
            .metric-value {{ font-weight: 600; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 14px; margin-top: 8px; }}
            th, td {{ padding: 8px 10px; text-align: center; border: 1px solid var(--border); }}
            th {{ background: #f3f4f6; font-size: 12px; text-transform: uppercase; color: var(--muted); }}
            td:first-child, th:first-child {{ text-align: left; font-weight: 600; }}
            .note {{
                background: #fffbeb;
                border: 1px solid #fde68a;
                border-radius: 8px;
                padding: 14px 18px;
                font-size: 14px;
                color: #78350f;
            }}
            .footer {{ margin-top: 32px; font-size: 13px; color: var(--muted); text-align: center; }}
            .footer a {{ color: var(--muted); }}
           .scope-note {{ font-size: 13px; color: var(--muted); margin-bottom: 16px; }}
.gradcam-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 14px;
}}
.gradcam-card {{
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    background: #fafafa;
}}
.gradcam-card img {{ width: 100%; display: block; }}
.gradcam-caption {{ padding: 8px 10px; font-size: 12px; }}
.gradcam-caption span {{ display: block; }}
.gc-true {{ color: var(--muted); }}
.gc-pred {{ color: #dc2626; font-weight: 600; }}
.gc-conf {{ color: var(--muted); }}
        </style>
    </head>
    <body>
        <h1>MedTrust-Audit</h1>
        <p class="subtitle">Clinical Trust Evaluation for Medical Imaging AI</p>

        <div class="cri-card">
            <div class="cri-score">{r['clinical_readiness']['cri_score']}</div>
            <div class="cri-verdict">{verdict}</div>
        </div>

        <div class="pillar">
            <h2>Discrimination</h2>
            <div class="metric-row"><span class="metric-label">Accuracy</span><span class="metric-value">{r['discrimination']['accuracy']*100:.2f}%</span></div>
            <div class="metric-row"><span class="metric-label">Precision (weighted)</span><span class="metric-value">{r['discrimination']['precision']*100:.2f}%</span></div>
            <div class="metric-row"><span class="metric-label">Recall (weighted)</span><span class="metric-value">{r['discrimination']['recall']*100:.2f}%</span></div>
            <div class="metric-row"><span class="metric-label">F1 (weighted)</span><span class="metric-value">{r['discrimination']['f1']*100:.2f}%</span></div>
        </div>

        <div class="pillar">
            <h2>Per-Class Breakdown</h2>
            <table>
                <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>N</th></tr>
                {per_class_rows}
            </table>
        </div>

        <div class="pillar">
            <h2>Confusion Matrix</h2>
            <table>
                <tr><th>True \\ Pred</th>{cm_header}</tr>
                {cm_rows}
            </table>
        </div>

        <div class="pillar">
            <h2>Calibration</h2>
            <div class="metric-row"><span class="metric-label">Expected Calibration Error (ECE)</span><span class="metric-value">{r['calibration']['ece']}</span></div>
        </div>

        <div class="pillar">
            <h2>High-Confidence Audit</h2>
            <div class="metric-row"><span class="metric-label">Silent error rate (conf &ge; 90%)</span><span class="metric-value">{r['high_confidence_audit']['high_confidence_error_rate']*100:.2f}%</span></div>
            <div class="metric-row"><span class="metric-label">Errors / High-confidence predictions</span><span class="metric-value">{r['high_confidence_audit']['high_conf_errors']} / {r['high_confidence_audit']['total_high_conf_predictions']}</span></div>
        </div>

      {render_explainability_section()}
        <div class="footer">
            <a href="/audit">Raw JSON</a> &middot;
            Implements the Clinical Readiness Index from
            "Beyond Accuracy: A Multi-Pillar Clinical Trust Framework for Brain Tumor MRI Classification"
        </div>
    </body>
    </html>
    """
    return html