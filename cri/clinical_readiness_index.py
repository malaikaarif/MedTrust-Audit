def compute_cri(accuracy, ece, high_conf_error_rate, weights=None):
    if weights is None:
        weights = {'accuracy': 0.4, 'ece': 0.3, 'high_conf_error': 0.3}
    
    ece_score = max(0, 1 - ece)
    error_score = max(0, 1 - high_conf_error_rate)
    
    cri = (weights['accuracy'] * accuracy + 
           weights['ece'] * ece_score + 
           weights['high_conf_error'] * error_score)
    
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