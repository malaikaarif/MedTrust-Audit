# MedTrust-Audit

**Clinical Trust Evaluation for Medical Imaging AI**

MedTrust-Audit is an open-source implementation of the Clinical Readiness Index (CRI) proposed in our IEEE submission, *"Beyond Accuracy: A Multi-Pillar Clinical Trust Framework for Brain Tumor MRI Classification"* (Safdar, Raza, Arif). It runs a trained medical imaging classifier's predictions through four evaluation pillars and returns a single composite score with a DEPLOY / REVIEW / REJECT verdict.

Most medical imaging papers report accuracy alone. This tool checks whether high accuracy is hiding dangerous failure modes — miscalibration, silent high-confidence errors, and poor generalization — before a model is trusted in a clinical setting.

## The Four Pillars

1. **Discrimination** — accuracy, precision, recall, F1 (standard classification performance)
2. **Calibration** — Expected Calibration Error (ECE): does the model's stated confidence match its actual accuracy?
3. **High-Confidence Audit** — of the model's most confident predictions, what fraction are silently wrong? A model can be accurate overall while still failing dangerously on the cases it claims to be sure about.
4. **Explainability** — *(not yet implemented in this dashboard)*. The published framework's spatial bias auditing (Grad-CAM, edge-bias metrics) is planned as a future addition. See "Status & Limitations" below.

These four scores combine into the **Clinical Readiness Index**:

```
CRI = 0.40·Accuracy + 0.25·(1−ECE) + 0.20·(1−HCE) + 0.15·Generalization
```

matching Equation (4) of the paper. `Generalization` defaults to 1.0 for primary-dataset evaluation (no cross-dataset test currently performed by this tool).

## Running It Locally

```bash
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000` for the dashboard, or `http://127.0.0.1:8000/audit` for raw JSON.

The tool reads `y_true.npy`, `y_pred.npy`, and `y_pred_probs.npy` from the repo root — the ground-truth labels, model predictions, and per-class softmax probabilities for a test set.

## The Model Behind the Included Sample Audit

The `.npy` files in this repo come from an independent MobileNetV2 reproduction, trained by me on the same public Brain Tumor MRI dataset (Nickparvar et al.) and matching the paper's methodology: ImageNet-pretrained MobileNetV2 with the final 50 layers unfrozen, Focal Loss (γ=2.0, α=0.25), and the paper's exact preprocessing/augmentation settings (Section III-A–D). The test split matches the paper's structure exactly: 1,600 held-out images, 400 per class.

**Reproducibility note:** this independent run landed close to — but not identical to — the paper's originally reported seed-42 result:

| Metric | This reproduction | Paper (seed 42) |
|---|---|---|
| Accuracy | 94.19% | 94.69% |
| ECE | 0.0292 | 0.0479 |
| High-confidence error rate | 2.66% | 74.12%* |
| CRI | 0.9641 | 0.8186 |

\*The paper's stated HCE figure uses a different denominator (high-confidence errors ÷ total errors) than this tool's implementation (high-confidence errors ÷ total high-confidence predictions) — see `evaluators/high_conf_errors.py`. Under either definition, this reproduction's high-confidence error rate is substantially lower than the paper's reported run.

This gap is consistent with the paper's own Table V, which documents seed-to-seed accuracy variance (σ up to 0.61% for some architectures) — this reproduction simply landed on a more favorably-calibrated instance of the same training recipe. The model weights themselves are not included in this repository (too large for git; see below).

## Obtaining the Model Weights

Model checkpoints are not committed to this repository (large binary files, excluded via `.gitignore`). To reproduce the training run yourself, the exact training script matching the paper's protocol is available on request, or can be reconstructed from the methodology described in Section III of the paper.

## Status & Limitations

- Explainability (Grad-CAM spatial bias auditing) is described in the paper but not yet implemented in this tool.
- Generalization is currently hardcoded to 1.0 (no cross-dataset evaluation implemented yet).
- This tool evaluates one fixed set of saved predictions; it does not currently accept arbitrary uploaded models.

## Citation

If referencing this work, please cite:

> Safdar, I., Raza, Z., Arif, M. "Beyond Accuracy: A Multi-Pillar Clinical Trust Framework for Brain Tumor MRI Classification." Submitted, 2026.