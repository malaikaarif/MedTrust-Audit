# MedTrust-Audit
### Clinical Trust Evaluation for Medical Imaging AI

A brain tumor MRI classifier can be 94%+ accurate and still be dangerous — if it's confidently wrong exactly when it matters most. MedTrust-Audit is an open-source implementation of the Clinical Readiness Index (CRI), proposed in our journal submission "Toward Trustworthy AI for Brain Tumor MRI Classification: A Multi-Pillar Clinical Readiness Framework" (Safdar, Raza, Arif — submitted, Elsevier, 2026). It runs a classifier's predictions through four evaluation pillars and returns a single composite score with a DEPLOY / REVIEW / REJECT verdict — because accuracy alone doesn't tell you whether a model is safe to trust.

## The Four Pillars

| Pillar | Question it answers |
|---|---|
| Discrimination | Standard accuracy, precision, recall, F1 |
| Calibration | Does the model's stated confidence match its actual accuracy? (ECE) |
| High-Confidence Audit | Of the model's most confident predictions, what fraction are silently wrong? |
| Explainability | Does the model attend to anatomically plausible regions when confidently wrong? (Grad-CAM) |

These combine into the Clinical Readiness Index:

```
CRI = 0.40·Accuracy + 0.25·(1−ECE) + 0.20·(1−HCE) + 0.15·Generalization
```

matching Equation (4) of the paper.

## Live Results (Independent Reproduction)

An independently trained MobileNetV2 — matching the paper's architecture, Focal Loss (γ=2.0, α=0.25), and training protocol — evaluated on 1,600 held-out test images (400/class):

| Metric | This reproduction | Paper (seed 42) |
|---|---|---|
| Accuracy | 94.19% | 94.69% |
| ECE | 0.0292 | 0.0479 |
| High-confidence error rate | 2.66% (39/1,465) | 74.12%* |
| Generalization | Pending validation (see below) | 86.13% (Figshare) |
| CRI | 0.9641 → DEPLOY | 0.8186 → DEPLOY |

*Different denominator definition — see `evaluators/high_conf_errors.py`. Under either definition, this reproduction's rate is substantially lower than the paper's reported run.

This is close to, but not identical to, the paper's figures — expected seed-to-seed variance (the paper's own Table V documents this). Notably, 5 of 6 highest-confidence errors in this reproduction were glioma misclassified as meningioma/notumor — matching the paper's own documented weak point (glioma had the lowest recall in the original study too), a real cross-validation signal that both runs found the same underlying model limitation.

## Generalization: An Honest Dead End (For Now)

We attempted to measure real cross-dataset generalization using the Figshare brain tumor dataset (the same one the paper used, reporting 86.13% zero-shot accuracy). Our test returned 98.34% — higher than our own primary-dataset accuracy, which is not how generalization is supposed to behave.

Investigating why: the Kaggle training dataset (Nickparvar et al.) used for the primary model is documented as being compiled from Figshare, SARTAJ, and Br35H sources — meaning our "unseen" Figshare test set may not have been unseen at all. We verified this against multiple independent academic sources before accepting it, rather than trusting a single claim.

We discarded the 98.34% result rather than report it. The Generalization pillar remains hardcoded to 1.0 (a neutral placeholder) pending a genuinely non-overlapping external dataset. This is disclosed directly on the dashboard, not hidden.

## Explainability: Sample Audit

Grad-CAM heatmap overlays for 6 of the model's real high-confidence errors, with true/predicted/confidence labels. This is an illustrative sample — the paper's full methodology audits all 1,600 test images with quantitative edge-bias metrics; this tool currently covers a smaller, honestly-scoped sample.

## Running Locally

```
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000` for the dashboard, or `/audit` for raw JSON.

## Auditing Your Own Model

`POST /audit/upload` accepts `y_true`, `y_pred`, `y_pred_probs` as `.npy` files and returns a full CRI audit. Deliberately accepts only prediction arrays, never model files — loading arbitrary model files (`.h5`/`.keras`) is a known code-execution risk we chose not to introduce.

## Repository Contents

```
main.py                    — FastAPI app, dashboard, /audit and /audit/upload endpoints
evaluators/                — discrimination, calibration, high-confidence-error modules
cri/                       — Clinical Readiness Index computation
explainability.py          — Grad-CAM sample rendering
explainability/gradcam_samples/  — 6 real heatmap images
tests/                     — pytest suite (9 tests, including a regression test on verified numbers)
y_true.npy, y_pred.npy, y_pred_probs.npy  — the reproduction's saved predictions
```

Model weights are not committed (large binary files, excluded via `.gitignore`). Training code matching the paper's exact protocol lives in a companion repository: `brain-tumor-clinical-trust-framework`.

## Testing

```
pip install pytest
python -m pytest tests/
```

## Status & Limitations

- Generalization pillar is a placeholder pending a valid non-overlapping external dataset.
- Explainability covers 6 sample cases, not the paper's full 1,600-image audit.
- This tool evaluates one fixed set of saved predictions by default, plus any predictions uploaded via `/audit/upload`.

## Citation

Safdar, I., Raza, Z., Arif, M. "Toward Trustworthy AI for Brain Tumor MRI Classification: A Multi-Pillar Clinical Readiness Framework." Submitted, Elsevier, 2026.

## License

MIT