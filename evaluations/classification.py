import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict
import logging


def evaluate_classification(train_theta, test_theta, train_labels, test_labels, classifier='SVM', gamma='scale', tune=False):
    if tune:
        results = {
            'acc': 0,
            'macro-F1': 0
        }
        logger = logging.getLogger('main')
        if classifier=='SVM':
            for C in [0.1, 1, 10, 100, 1000]:
                for gamma in ['scale', 'auto', 10, 1, 0.1, 0.01, 0.001]:
                    '''print(f'C: {C}, gamma: {gamma}')'''
                    for kernel in ['rbf', 'linear']:
                        logger.info(f'C: {C}, gamma: {gamma}, kernel: {kernel}')
                        clf = SVC(C=C, kernel=kernel, gamma=gamma)

                        clf.fit(train_theta, train_labels)
                        preds = clf.predict(test_theta)
                        this_results = {
                            'acc': accuracy_score(test_labels, preds),
                            'macro-F1': f1_score(test_labels, preds, average='macro')
                        }
                        results = {
                            key: max(results[key], this_results[key])
                            for key in results
                        }
                        logger.info(f'Accuracy: {this_results["acc"]:.4f}, Macro-F1: {this_results["macro-F1"]:.4f}')
        elif classifier=='GaussianNB':
            for var_smoothing in [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4]:
                logger.info(f'GaussianNB - var_smoothing: {var_smoothing}')
                clf=GaussianNB(var_smoothing=var_smoothing)
                clf.fit(train_theta,train_labels)
                preds=clf.predict(test_theta)
                this_results={
                    'acc':accuracy_score(test_labels,preds),
                    'macro-F1':f1_score(test_labels,preds,average='macro')
                }
                results={
                    key:max(results[key],this_results[key]) for key in results
                }
                logger.info(f'Accuracy: {this_results["acc"]:.4f}, Macro-F1: {this_results["macro-F1"]:.4f}')
        elif classifier=='MultinomialNB':
            for alpha in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
                logger.info(f'MultinomialBN - alpha: {alpha}')
                clf=MultinomialNB(alpha=alpha)
                clf.fit(train_theta,train_labels)
                pred=clf.predict(test_theta)
                this_results={
                    'acc':accuracy_score(test_labels,preds),
                    'macro-F1':f1_score(test_labels,preds,average='macro')
                }
                results={
                    key:max(results[key],this_results[key]) for key in results
                }
                logger.info(f'Accuracy: {this_results["acc"]:.4f}, Macro-F1: {this_results["macro-F1"]:.4f}')
        else:
            raise NotImplementedError(f'Classiifier {classifier} not supported')
    else:
        if classifier == 'SVM':
            clf = SVC(gamma=gamma)
        elif classifier=='GaussianNB':
            clf=GaussianNB()
        elif classifier=='MultinomialNB':
            clf=MultinomialNB()
        else:
            raise NotImplementedError

        clf.fit(train_theta, train_labels)
        preds = clf.predict(test_theta)
        results = {
            'acc': accuracy_score(test_labels, preds),
            'macro-F1': f1_score(test_labels, preds, average='macro')
        }
    return results


def crosslingual_classification(
    train_theta_en,
    train_theta_cn,
    test_theta_en,
    test_theta_cn,
    train_labels_en,
    train_labels_cn,
    test_labels_en,
    test_labels_cn,
    classifier="SVM",
    gamma="scale"
):
    intra_en = evaluate_classification(train_theta_en, test_theta_en, train_labels_en, test_labels_en, classifier, gamma)
    intra_cn = evaluate_classification(train_theta_cn, test_theta_cn, train_labels_cn, test_labels_cn, classifier, gamma)

    cross_en = evaluate_classification(train_theta_cn, test_theta_en, train_labels_cn, test_labels_en, classifier, gamma)
    cross_cn = evaluate_classification(train_theta_en, test_theta_cn, train_labels_en, test_labels_cn, classifier, gamma)

    return {
        'intra_en': intra_en,
        'intra_cn': intra_cn,
        'cross_en': cross_en,
        'cross_cn': cross_cn
    }


def hierarchical_classification(train_theta, test_theta, train_labels, test_labels, classifier='SVM', gamma='scale'):
    num_layer = len(train_theta)
    results = defaultdict(list)

    for layer in range(num_layer):
        layer_results = evaluate_classification(train_theta[layer], test_theta[layer], train_labels, test_labels, classifier, gamma)

        for key in layer_results:
            results[key].append(layer_results[key])

    for key in results:
        results[key] = np.mean(results[key])

    return results