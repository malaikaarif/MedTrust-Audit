"""
Explainability pillar — sample Grad-CAM audit.

Scope note: the paper audits ALL 1,600 test images with quantitative
edge-bias metrics (Section III-E / IV-C). This module covers a smaller,
illustrative scope: Grad-CAM heatmaps for the 6 high-confidence errors
found in the independent MobileNetV2 reproduction. It is not a
full quantitative replication of the paper's spatial bias analysis.
"""
import os
import re

GRADCAM_DIR = os.path.join(os.path.dirname(__file__), "explainability", "gradcam_samples")


def _parse_filename(fname):
    """error_1_true-glioma_pred-meningioma_conf-1.00.png -> dict"""
    m = re.match(r"error_(\d+)_true-(\w+)_pred-(\w+)_conf-([\d.]+)\.png", fname)
    if not m:
        return None
    idx, true_label, pred_label, conf = m.groups()
    return {
        "filename": fname,
        "index": int(idx),
        "true_label": true_label,
        "pred_label": pred_label,
        "confidence": float(conf),
    }


def get_gradcam_samples():
    if not os.path.isdir(GRADCAM_DIR):
        return []
    files = sorted(os.listdir(GRADCAM_DIR))
    parsed = [_parse_filename(f) for f in files if f.endswith(".png")]
    return sorted([p for p in parsed if p], key=lambda x: x["index"])


def render_explainability_section():
    samples = get_gradcam_samples()

    if not samples:
        return """
        <div class="pillar">
            <h2>Explainability</h2>
            <div class="note">
                <strong>Not yet available.</strong> Grad-CAM sample audit pending.
            </div>
        </div>
        """

    cards = ""
    for s in samples:
        cards += f"""
        <div class="gradcam-card">
            <img src="/explainability/gradcam_samples/{s['filename']}" alt="Grad-CAM heatmap" />
            <div class="gradcam-caption">
                <span class="gc-true">True: {s['true_label']}</span>
                <span class="gc-pred">Predicted: {s['pred_label']}</span>
                <span class="gc-conf">Confidence: {s['confidence']*100:.0f}%</span>
            </div>
        </div>
        """

    return f"""
    <div class="pillar">
        <h2>Explainability &mdash; Sample Grad-CAM Audit</h2>
        <p class="scope-note">
            Heatmaps for {len(samples)} high-confidence errors from the independent MobileNetV2
            reproduction. This is an illustrative sample, not the full 1,600-image quantitative
            edge-bias audit described in the paper (Section III-E).
        </p>
        <div class="gradcam-grid">
            {cards}
        </div>
    </div>
    """