from sklearn.metrics import accuracy_score, average_precision_score
import numpy as np
from collections import defaultdict
from IPython import embed

def recall(output, solutions, num_classes):
    recall = defaultdict(int)
    #recall = {}
    for i in range(num_classes):
        positions = np.where(np.array(solutions)==i)
        for p in positions[0]:
            if output[p] == i:
                recall[str(i)] += 1
        recall[str(i)] /= len(positions[0])
        recall[str(i)] *= 100
    return recall

def accuracy(output, solutions):
    return np.where(np.array(output)==solutions)

def multilabel_accuracy(output, solutions):
    return output.sum(), solutions.sum()

def multilabel_recall(output, solutions):
    return output.sum(axis=0), solutions.sum(axis=0)

def multilabel_metrics(predictions, solutions, threshold):
    output = (np.array(predictions) > threshold)*1
    solutions = np.array(solutions)
    correct_matrix = np.bitwise_and(output,solutions)
    acc_p, acc_s = multilabel_accuracy(correct_matrix, solutions)
    rec_p, rec_s = multilabel_recall(correct_matrix, solutions)
    auprc = average_precision_score(solutions, np.array(predictions), average='micro')
    return acc_p/acc_s*100, rec_p/rec_s*100, auprc
