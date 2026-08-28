from fastapi import FastAPI, UploadFile, File, HTTPException
import io
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
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB cap per file, generous for prediction arrays


def _safe_load_npy(upload_bytes: bytes, expected_ndim_options):
    """Load a .npy file from raw bytes with allow_pickle=False (safe),
    and enforce it's actually a plain numeric array of an expected shape."""
    try:
        arr = np.load(io.BytesIO(upload_bytes), allow_pickle=False)
    except Exception:
        raise HTTPException(status_code=400, detail="File is not a valid .npy array, or is malformed.")
 
    if arr.ndim not in expected_ndim_options:
        raise HTTPException(
            status_code=400,
            detail=f"Unexpected array shape {arr.shape}. Expected {expected_ndim_options}-D array."
        )
    return arr


def compute_reliability_diagram(y_true, y_pred_probs, n_bins=10):
    """Compute confidence vs accuracy per bin for ECE reliability diagram."""
    confidences = np.max(y_pred_probs, axis=1)
    predictions = np.argmax(y_pred_probs, axis=1)
    correct = (predictions == y_true).astype(float)
    
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_edges[:-1]
    bin_uppers = bin_edges[1:]
    
    bars = []
    for lower, upper in zip(bin_lowers, bin_uppers):
        mask = (confidences > lower) & (confidences <= upper)
        if lower == 0.0:
            mask = (confidences >= lower) & (confidences <= upper)
        if mask.sum() > 0:
            acc = correct[mask].mean()
            conf = confidences[mask].mean()
            count = int(mask.sum())
        else:
            acc = 0.0
            conf = (lower + upper) / 2
            count = 0
        bars.append({
            "lower": lower,
            "upper": upper,
            "accuracy": acc,
            "confidence": conf,
            "count": count
        })
    return bars


def run_audit():
    y_true = np.load('y_true.npy')
    y_pred = np.load('y_pred.npy')
    y_pred_probs = np.load('y_pred_probs.npy')

    disc = compute_metrics(y_true, y_pred)
    ece = compute_ece(y_true, y_pred_probs)
    hce = find_high_confidence_errors(y_true, y_pred, y_pred_probs)
    # Generalization is a placeholder — Figshare overlapped with training data (Nickparvar = Figshare + SARTAJ + Br35H)
    # Pending genuine external dataset for zero-shot validation
    cri = compute_cri(disc['accuracy'], ece, hce['high_confidence_error_rate'], generalization=1.0)
    rel_diag = compute_reliability_diagram(y_true, y_pred_probs)

    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)

    return {
        "discrimination": disc,
        "calibration": {"ece": ece, "reliability_diagram": rel_diag},
        "high_confidence_audit": hce,
        "clinical_readiness": cri,
        "confusion_matrix": cm,
        "per_class": {name: report[name] for name in CLASS_NAMES}
    }


def run_audit_from_arrays(y_true, y_pred, y_pred_probs):
    """Same computation as run_audit(), but takes arrays directly
    instead of reading from disk -- reused by both the default
    dashboard and the upload endpoint."""
    disc = compute_metrics(y_true, y_pred)
    ece = compute_ece(y_true, y_pred_probs)
    hce = find_high_confidence_errors(y_true, y_pred, y_pred_probs)
    cri = compute_cri(disc['accuracy'], ece, hce['high_confidence_error_rate'], generalization=1.0)
    rel_diag = compute_reliability_diagram(y_true, y_pred_probs)
    return {
        "discrimination": disc,
        "calibration": {"ece": ece, "reliability_diagram": rel_diag},
        "high_confidence_audit": hce,
        "clinical_readiness": cri,
    }

@app.get("/audit")
def audit():
    return run_audit()

@app.post("/audit/upload")
async def audit_upload(
    y_true_file: UploadFile = File(...),
    y_pred_file: UploadFile = File(...),
    y_pred_probs_file: UploadFile = File(...),
):
    for f in (y_true_file, y_pred_file, y_pred_probs_file):
        if not f.filename.endswith(".npy"):
            raise HTTPException(status_code=400, detail=f"{f.filename} must be a .npy file.")
 
    y_true_bytes = await y_true_file.read()
    y_pred_bytes = await y_pred_file.read()
    y_pred_probs_bytes = await y_pred_probs_file.read()
 
    for name, b in [("y_true", y_true_bytes), ("y_pred", y_pred_bytes), ("y_pred_probs", y_pred_probs_bytes)]:
        if len(b) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail=f"{name} exceeds the 10MB size limit.")
 
    y_true = _safe_load_npy(y_true_bytes, expected_ndim_options=[1])
    y_pred = _safe_load_npy(y_pred_bytes, expected_ndim_options=[1])
    y_pred_probs = _safe_load_npy(y_pred_probs_bytes, expected_ndim_options=[2])

    if not (len(y_true) == len(y_pred) == len(y_pred_probs)):
        raise HTTPException(
            status_code=400,
            detail=f"Array length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}, y_pred_probs={len(y_pred_probs)}"
        )
 
    if len(y_true) == 0:
        raise HTTPException(status_code=400, detail="Arrays are empty.")
 
    try:
        result = run_audit_from_arrays(y_true, y_pred, y_pred_probs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not compute audit: {str(e)}")
 
    return result
 


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

    # Build CRI formula breakdown
    acc = r['discrimination']['accuracy']
    ece = r['calibration']['ece']
    hce_rate = r['high_confidence_audit']['high_confidence_error_rate']
    gen = 1.0  # placeholder
    cri_score = r['clinical_readiness']['cri_score']
    
    cri_formula = (
        f"CRI = 0.40×Acc({acc:.4f}) + 0.25×(1−ECE)({1-ece:.4f}) + "
        f"0.20×(1−HCE)({1-hce_rate:.4f}) + 0.15×Gen({gen}) = {cri_score:.4f}"
    )

    # Build reliability diagram SVG
    rel_diag = r['calibration']['reliability_diagram']
    svg_width = 600
    svg_height = 220
    chart_margin = 40
    chart_w = svg_width - 2 * chart_margin
    chart_h = svg_height - 2 * chart_margin
    
    n_bins = len(rel_diag)
    bar_width = chart_w / n_bins
    
    # Find max count for scaling bar heights
    max_count = max(b['count'] for b in rel_diag) if rel_diag else 1
    if max_count == 0:
        max_count = 1
    
    bars_svg = ""
    labels_svg = ""
    for i, bar in enumerate(rel_diag):
        x = chart_margin + i * bar_width
        # Bar height proportional to accuracy (max 1.0)
        bar_h = bar['accuracy'] * chart_h
        y = chart_margin + chart_h - bar_h
        
        # Bar color: blue if above diagonal, gray if below
        bar_color = "#2563eb" if bar['accuracy'] >= bar['confidence'] else "#9ca3af"
        
        bars_svg += f'<rect x="{x+4}" y="{y}" width="{bar_width-8}" height="{bar_h}" fill="{bar_color}" opacity="0.8" rx="3"/>'
        
        # Count label below bar
        labels_svg += f'<text x="{x + bar_width/2}" y="{chart_margin + chart_h + 16}" text-anchor="middle" font-size="10" fill="#6b7280">{bar["count"]}</text>'
    
    # Perfect calibration diagonal line
    diag_svg = f'<line x1="{chart_margin}" y1="{chart_margin + chart_h}" x2="{chart_margin + chart_w}" y2="{chart_margin}" stroke="#dc2626" stroke-width="2" stroke-dasharray="5,5" opacity="0.6"/>'
    
    reliability_svg = f"""
    <svg width="100%" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}" style="margin-top:12px;">
        {diag_svg}
        {bars_svg}
        {labels_svg}
        <text x="{chart_margin}" y="{chart_margin + chart_h + 32}" font-size="11" fill="#6b7280">0.0</text>
        <text x="{chart_margin + chart_w}" y="{chart_margin + chart_h + 32}" font-size="11" fill="#6b7280" text-anchor="end">1.0</text>
        <text x="{chart_margin + chart_w/2}" y="{chart_margin + chart_h + 32}" font-size="11" fill="#6b7280" text-anchor="middle">Confidence →</text>
        <text x="12" y="{chart_margin + chart_h/2}" font-size="11" fill="#6b7280" text-anchor="middle" transform="rotate(-90, 12, {chart_margin + chart_h/2})">Accuracy</text>
    </svg>
    """

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
                margin-bottom: 12px;
                box-shadow: 0 4px 14px rgba(0,0,0,0.08);
            }}
            .cri-score {{ font-size: 40px; font-weight: 800; letter-spacing: -1px; }}
            .cri-verdict {{ font-size: 16px; letter-spacing: 2px; opacity: 0.9; margin-top: 4px; }}
            .cri-formula {{
                background: white;
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 12px 16px;
                font-size: 13px;
                color: var(--muted);
                text-align: center;
                margin-bottom: 32px;
                font-family: "SF Mono", Monaco, monospace;
            }}
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
            .metric-value.pending {{ color: #ca8a04; }}
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
        <div class="cri-formula">
            {cri_formula}<br>
            <span style="font-size:11px; opacity:0.7;">*Generalization pending external validation on non-overlapping dataset</span>
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
            <p class="scope-note">Reliability diagram: accuracy per confidence bin (blue = above perfect calibration, gray = below). Red dashed line = perfect calibration.</p>
            {reliability_svg}
        </div>

        <div class="pillar">
            <h2>High-Confidence Audit</h2>
            <div class="metric-row"><span class="metric-label">Silent error rate (conf &ge; 90%)</span><span class="metric-value">{r['high_confidence_audit']['high_confidence_error_rate']*100:.2f}%</span></div>
            <div class="metric-row"><span class="metric-label">Errors / High-confidence predictions</span><span class="metric-value">{r['high_confidence_audit']['high_conf_errors']} / {r['high_confidence_audit']['total_high_conf_predictions']}</span></div>
        </div>

        <div class="pillar">
            <h2>Generalization</h2>
            <div class="metric-row">
                <span class="metric-label">Cross-dataset accuracy</span>
                <span class="metric-value pending">Pending validation</span>
            </div>
            <p class="scope-note">
                External zero-shot test pending non-overlapping dataset.
                Initial Figshare test revealed data overlap with the Nickparvar training compilation
                (documented sources: figshare + SARTAJ + Br35H).
            </p>
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