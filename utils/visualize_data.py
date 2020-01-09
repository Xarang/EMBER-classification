import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import IncrementalPCA


"""
code from pca.py
"""

PCA_SUBSET_PROPORTION = 1
VECTOR_CHUNK_SIZE = 10000
PCA_NB_COMPONENTS = 2

def get_datasets(vector_size):
    x = np.memmap(sys.argv[1], dtype=np.float32, mode='c', order='C')
    x = x.reshape(-1, vector_size)
    y = np.memmap(sys.argv[2], dtype=np.float32, mode='c', order='C')
    print("len(x):", len(x))
    print("x:", x)
    return x, y

# Returns a generator that yields chunks of size 'chunk_size'
# from arrays data and labels
def chunks(data: np.array):
    for i in range(0, len(data), VECTOR_CHUNK_SIZE):
        yield data[i:i + VECTOR_CHUNK_SIZE, :]

def pca(data, filename):
    # subset of data to fit our pca on
    sample_indexes = np.random.choice(range(len(data)), \
        int(len(data) * PCA_SUBSET_PROPORTION), replace=False)
    subset = data[sample_indexes]
    print("got subset")

    # compute our PCA by batches of 100 to avoid RAM overusage
    pca = IncrementalPCA(n_components=PCA_NB_COMPONENTS, batch_size = 2000)
    pca = pca.fit(subset)
    print("fit PCA")

    # We process our training data by smaller chunks to not overload RAM usage

    new_data = np.vstack([ \
        pca.transform(chunk) for chunk in chunks(data) ])

    means = pca.mean_
    #print("means:", means)
    components = pca.components_
    #print("components:", components)
    np.save(filename + '_components.npy', components)
    means.tofile(filename + '_means.npy', sep='\n')

    return new_data


def load_means_components(filename):
    means = np.fromfile(filename + '_means.npy', dtype=np.float32, sep='\n')
    components = np.load(filename + '_components.npy')
    print("loaded means & components")
    #print("means:", means)
    #print("components:", components)
    return means, components

#https://stackoverflow.com/questions/27668462/numpy-dot-memoryerror-my-dot-very-slow-but-works-why
def chunking_dot(big_matrix, small_matrix, chunk_size=100):
    # Make a copy if the array is not already contiguous
    small_matrix = np.ascontiguousarray(small_matrix)
    R = np.empty((big_matrix.shape[0], small_matrix.shape[1]))
    for i in range(0, R.shape[0], chunk_size):
        end = i + chunk_size
        R[i:end] = np.dot(big_matrix[i:end], small_matrix)
    return R

def apply_pca(means, components, x):
    x = x - means
    x = chunking_dot(x, components.T)
    #x = np.dot(x, components.T)
    return x

x, y = get_datasets(128)

filename='pca'
"""
reduced = pca(x, filename)
t = reduced.transpose()
print("[1]: ", t)
"""
means, components = load_means_components(filename)
reduced_2 = apply_pca(means, components, x)
t = reduced_2.transpose()
print("[2]: ", t)


#plt.xlim(0, 500000)
#plt.ylim()
plt.scatter(t[0], t[1], c=y, cmap=ListedColormap(['red', 'blue']), alpha=0.5)
plt.savefig('out.png')