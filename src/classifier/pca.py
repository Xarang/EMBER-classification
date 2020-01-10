from time import time
import numpy as np
from sklearn.decomposition import IncrementalPCA

def log(tag, message, time_start = None):
    if time_start != None:
        print("[{}] {} in {:.2f} sec".format(tag, message, time() - time_start))

PCA_SUBSET_SIZE = 10000
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
        int(PCA_SUBSET_SIZE), replace=False)
    subset = training_data[sample_indexes]
    log("PCA", "Got our data subset", time_start)
    # compute our PCA by batches of 100 to avoid RAM overusage
    pca = IncrementalPCA(n_components=PCA_NB_COMPONENTS, batch_size = 500)
    pca = pca.fit(subset)
    log("PCA", "computed PCA on data subset", time_start)

    # transform all validation data with the pca we just computed
    time_start = time()
    new_validation_data = pca.transform(validation_data)
    log("PCA", "transformed our validation data using PCA", time_start)

    # We process our training data by smaller chunks to not overload RAM usage

    new_training_data = np.vstack([ \
        pca.transform(chunk) for chunk in chunks(training_data) ])
    log("PCA", "transformed our training data using PCA", time_start)
    return new_training_data, new_validation_data