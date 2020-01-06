from time import time
import numpy as np
from sklearn.decomposition import IncrementalPCA

def log(tag, message, time_start = None):
    if time_start != None:
        print("[{}] {} in {:.2f}".format(tag, message, time() - time_start))

PCA_SUBSET_PROPORTION = 0.1
VECTOR_CHUNK_SIZE = 10000
PCA_NB_COMPONENTS = 20

# Returns a generator that yields chunks of size 'chunk_size'
# from arrays data and labels
def chunks(data: np.array):
    for i in range(0, len(data), VECTOR_CHUNK_SIZE):
        yield data[i:i + VECTOR_CHUNK_SIZE, :]

def pca(training_data, validation_data):
    time_start = time()
    # subset of data to fit our pca on
    sample_indexes = np.random.choice(range(len(training_data)), \
        int(len(training_data) * PCA_SUBSET_PROPORTION), replace=False)
    subset = training_data[sample_indexes]

    # compute our PCA by batches of 100 to avoid RAM overusage
    pca = IncrementalPCA(n_components=PCA_NB_COMPONENTS, batch_size = 100)
    pca = pca.fit(subset)
    log("PCA", "computed PCA on data subset", time_start)

    # transform all validation data with the pca we just computed
    time_start = time()
    new_validation_data = pca.transform(validation_data)
    log("PCA", "transformed our validation data using PCA", time_start)

    # We process our training data by smaller chunks to not overload RAM usage

    new_training_data = np.vstack([ \
        pca.transform(chunk) for chunk in chunks(training_data) ])

    return new_training_data, new_validation_data