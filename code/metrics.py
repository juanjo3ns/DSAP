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
    return output.sum(), solutions.size

def multilabel_recall(output, solutions):
    return output.sum(axis=0), solutions.sum(axis=0)

def multilabel_metrics(predictions, solutions, threshold, mixup):
    solutions = np.array(solutions)
    if mixup:
        del_rows = np.where(solutions==0.5)[0]
        solutions = np.delete(solutions, del_rows, axis=0)
        predictions = np.delete(predictions, del_rows, axis=0)
    output = (np.array(predictions) > threshold)*1
    and_matrix = np.bitwise_and(output,solutions.astype(int))
    xnor_matrix = np.ones_like(predictions) - np.bitwise_xor(output,solutions.astype(int))
    acc_p, acc_s = multilabel_accuracy(xnor_matrix, solutions)
    rec_p, rec_s = multilabel_recall(and_matrix, solutions)
    auprc = average_precision_score(solutions, np.array(predictions), average='micro')
    return acc_p/acc_s*100, rec_p/rec_s*100, auprc*100
