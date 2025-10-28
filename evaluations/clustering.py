import numpy as np
from collections import defaultdict
from sklearn import metrics


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def harmonic_purity(labels,preds):
        contingency_matrix=metrics.cluster.contingency_matrix(labels_true=labels,labels_pred=preds)
        precision = contingency_matrix / contingency_matrix.sum(axis=0).reshape(1, -1)
        recall = contingency_matrix / contingency_matrix.sum(axis=1).reshape(-1, 1)
        
        # Handle division by zero: replace inf/nan with 0
        precision = np.nan_to_num(precision)
        recall = np.nan_to_num(recall)
        
        # Calculate F1, avoiding division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            f1 = 2 * (precision * recall) / (precision + recall)
        f1 = np.nan_to_num(f1)
        
        harmonic_purity = (np.amax(f1, axis=1) * contingency_matrix.sum(axis=1)).sum() / contingency_matrix.sum()
        return harmonic_purity

def clustering_metric(labels, preds):
    metrics_func = [
        {
            'name': 'Purity',
            'method': purity_score
        },
        {
            'name': 'Harmonic_Purity',
            'method': harmonic_purity,
        },
        {
            'name': 'NMI',
            'method': metrics.cluster.normalized_mutual_info_score
        },
        {
            'name': 'ARI',
            'method': metrics.adjusted_rand_score
        },
        {
            'name': 'MIS',
            'method': metrics.normalized_mutual_info_score
        }
    ]

    results = dict()
    for func in metrics_func:
        results[func['name']] = func['method'](labels, preds)

    return results


def evaluate_clustering(theta, labels):
    preds = np.argmax(theta, axis=1)
    return clustering_metric(labels, preds)
