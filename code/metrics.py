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

def multilabel_accuracy(output):
    rows_ones = np.where(output==1)
    unique_rows = np.unique(rows_ones[0])
    return len(unique_rows)

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
    xor_matrix = np.bitwise_xor(output,solutions.astype(int))
    acc_p = multilabel_accuracy(xor_matrix)
    rec_p, rec_s = multilabel_recall(and_matrix, solutions)
    auprc = average_precision_score(solutions, np.array(predictions), average='micro')
    return 100-acc_p/output.shape[0]*100, rec_p/rec_s*100, auprc*100
