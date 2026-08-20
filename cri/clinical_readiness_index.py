def compute_cri(accuracy, ece, high_conf_error_rate, generalization=1.0, weights=None):
    """
    Clinical Readiness Index (CRI), matching Eq. (4) from:
    "Beyond Accuracy: A Multi-Pillar Clinical Trust Framework for
    Brain Tumor MRI Classification" (Safdar, Raza, Arif).

    CRI = 0.40*Acc + 0.25*(1-ECE) + 0.20*(1-HCE) + 0.15*Gen

    generalization=1.0 by default, matching the paper's convention for
    primary-dataset evaluation (no cross-dataset test performed).
    """
    if weights is None:
        weights = {'accuracy': 0.40, 'ece': 0.25, 'high_conf_error': 0.20, 'generalization': 0.15}

    ece_score = max(0, 1 - ece)
    error_score = max(0, 1 - high_conf_error_rate)

    cri = (weights['accuracy'] * accuracy +
           weights['ece'] * ece_score +
           weights['high_conf_error'] * error_score +
           weights['generalization'] * generalization)

    if cri >= 0.8:
        verdict = "DEPLOY"
    elif cri >= 0.5:
        verdict = "REVIEW"
    else:
        verdict = "REJECT"

    return {
        'cri_score': round(cri, 4),
        'verdict': verdict
    }