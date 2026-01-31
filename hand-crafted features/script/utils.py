import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.utils.class_weight import compute_sample_weight


def compute_scale_pos_weight(y, args):
    if args.scale_pos_weight == 'balanced':
        args.scale_pos_weight = np.sum(y == 0) / np.sum(y == 1)
        
        
def find_operation_point(y_true, y_score, score_func=f1_score, sample_weight=None):
    # get roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score[:, 1])
    
    # adjust sample weight
    if sample_weight == 'balanced':
        sample_weight = compute_sample_weight('balanced', y_true)
            
    scores = []
    for th in thresholds:
        # get new predictions
        new_pred = apply_operation_point(y_score, th)
        
        # get new score
        scores.append(score_func(y_true, new_pred, sample_weight=sample_weight))
    scores = np.array(scores)
    
    # get the optimal threshold
    optim_idx = np.argmax(scores)
    optim_th = thresholds[optim_idx]
    return optim_idx, optim_th
        
        
def apply_operation_point(y_score, threshold):
    n_samples, _ = y_score.shape
    predictions = np.zeros((n_samples, ))
    mask = y_score[:, 1] > threshold
    predictions[mask] = 1
    predictions[~mask] = 0
    return predictions
        