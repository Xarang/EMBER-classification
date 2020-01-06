import numpy as np
import os
import sys

print("[DATASET] started dataset creation using data '{}' and labels '{}'".format(sys.argv[1], sys.argv[2]))

x = np.memmap(sys.argv[1], dtype=np.float32, mode='c', order='C')
x = x.reshape(-1, 2351)
y = np.memmap(sys.argv[2], dtype=np.float32, mode='c', order='C')

print("[DATASET] mapped files into memory")

labelled_indexes = y != -1

labelled_values = x[labelled_indexes]
labelled_labels = y[labelled_indexes]

print("[DATASET] removed unlabelled data")

set_size = len(labelled_values)
validation_set_proportion = 0.1

indexes = np.arange(set_size)
np.random.shuffle(indexes)

validation_indexes = indexes[0:int(set_size * validation_set_proportion)]
training_indexes = indexes[int(set_size * validation_set_proportion):]
print("[DATASET] got indexes of Validation data and Training data")

training_data = labelled_values[training_indexes]
training_labels = labelled_labels[training_indexes]

validation_data = labelled_values[validation_indexes]
validation_labels = labelled_labels[validation_indexes]

print("[DATASET] split our data into Training set and Validation set")
print("[DATASET] Training Set length: {}".format(len(training_data)))
assert(len(training_data) == len(training_labels))
print("[DATASET] Validation Set length: {}".format(len(validation_data)))
assert(len(validation_data) == len(validation_labels))

dirname = "dataset/"
if not os.path.exists(dirname):
    os.mkdir(dirname)

training_data.tofile("{}Xtraining.dat".format(dirname))
training_labels.tofile("{}Ytraining.dat".format(dirname))

validation_data.tofile("{}Xvalidation.dat".format(dirname))
validation_labels.tofile("{}Yvalidation.dat".format(dirname))

print("[DATASET] stored in '{}'".format(dirname))