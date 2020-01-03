#!/usr/bin/python3

# part accuracy check

import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

classif = np.memmap(sys.argv[1], dtype=np.float32, mode='c', order ='C')
exact = np.memmap(sys.argv[2], dtype=np.float32, mode='r', order ='C')

print("Ignoring unlabelled data:")

labelled_indexes = exact != -1

labelled_classif = classif[labelled_indexes]
labelled_labels = exact[labelled_indexes]

print("Accuracy: ", accuracy_score(labelled_labels, labelled_classif))
print("Precision: ", precision_score(labelled_labels, labelled_classif, average = 'macro'))
print("Recall: ", recall_score(labelled_labels, labelled_classif, average = 'macro'))
print("Confusion Matrix: \n", confusion_matrix(labelled_labels, labelled_classif))