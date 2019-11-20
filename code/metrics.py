import numpy as np
from collections import defaultdict

def recall(self, output, solutions, num_classes):
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

def accuracy(self, output, solutions):
    return np.where(np.array(output)==solutions)
