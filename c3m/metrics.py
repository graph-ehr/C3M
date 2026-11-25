"""Classification metrics; test-derived operating points are descriptive only."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, roc_curve


def classification(labels, scores, predictions, loss):
    if len(np.unique(labels)) != 2:
        raise ValueError('Evaluation requires both label classes')
    result = dict(acc=float(accuracy_score(labels, predictions)),
                  f1=float(f1_score(labels, predictions, zero_division=0)),
                  auroc=float(roc_auc_score(labels, scores)),
                  auprc=float(average_precision_score(labels, scores)), ce=float(loss))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    for specificity in [0.9, 0.95]:
        idx = np.where(fpr <= 1 - specificity)[0][-1]
        positive = scores >= thresholds[idx]
        result[f'sensitivity_at_{int(100*specificity)}'] = float(tpr[idx])
        result[f'ppv_at_{int(100*specificity)}'] = float(labels[positive].mean()) if positive.any() else 0.0
    return result
