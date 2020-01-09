
import numpy as np
import os
import sys
from sklearn.preprocessing import MinMaxScaler

print("[DATASET] started dataset analysis using data '{}' and labels '{}'".format(sys.argv[1], sys.argv[2]))

VECTOR_SIZE = 2351

x = np.memmap(sys.argv[1], dtype=np.float32, mode='c', order='C')
x = x.reshape(-1, VECTOR_SIZE)
y = np.memmap(sys.argv[2], dtype=np.float32, mode='c', order='C')

"""
subset_size = 1000
sample_indexes = np.random.choice(range(len(x)), subset_size, replace=False)
subset = x[sample_indexes]
scaler = MinMaxScaler()
scaler.fit(subset)
print("[DATASET] fitted our data scaler")

transformed_x = scaler.transform(x)
print("[DATASET] transformed our dataset through scaler")

means = np.mean(transformed_x, axis=0)
print("[DATASET] computed the means")
print(len(means), means)

means = np.array([(i, means[i]) for i in range(len(means))], dtype=[('index', int), ('value', np.float32)])
print("[DATASET] indexed our means")
print(len(means), means)

means.sort(kind='quicksort', order='value')
print("[DATASET] sorted our indexed array")
print(means)

select = means[0:128]
select_indexes = [ select[i][0] for i in range(len(select)) ]
print("[DATASET] selected {} min values".format(len(select)))
np.set_printoptions(threshold=sys.maxsize)
print("select", select)
print("select_indexes", select_indexes)
"""
def get_mean_vector(x, y, label):
    labelled_indexes = y == label
    labelled_values = x[labelled_indexes]
    mean = labelled_values.mean(axis=0)
    print("[DATASET] got mean vector for label {}".format(label))
    return mean

mean_0 = get_mean_vector(x, y, 0)
mean_1 = get_mean_vector(x, y, 1)

np.set_printoptions(threshold=sys.maxsize)

max_values = np.array([0] * VECTOR_SIZE)
min_values = np.array([0] * VECTOR_SIZE)


for i in range(len(x)):
    max_values = np.maximum(max_values, x[i], dtype=np.float32)
    min_values = np.minimum(min_values, x[i], dtype=np.float32)


for i in range(VECTOR_SIZE):                                           
    print("{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}".format( \
        #index   mean 0   mean 1        mean difference     min value of feature   max value of feature
        i, mean_0[i], mean_1[i], abs(mean_0[i] - mean_1[i]), min_values[i], max_values[i])) 

