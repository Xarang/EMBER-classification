import numpy as np

x = np.memmap("Xtrain.dat", dtype=np.float32, shape=(900000, 2351), mode='c', order='C')
y = np.memmap("Ytrain.dat", dtype=np.float32, mode='c', order='C')


labelled_indexes = y != -1

labelled_values = x[labelled_indexes]
labelled_labels = y[labelled_indexes]

subsample_indexes = np.random.choice(range(len(labelled_values)), 50000, replace=False)

new_x = labelled_values[subsample_indexes]
new_y = labelled_labels[subsample_indexes]

assert(len(new_x) == len(new_y))

print(len(new_y == 0))
print(len(new_y == 1))

new_x.tofile("Xvalidation.dat")
new_y.tofile("Yvalidation.dat")