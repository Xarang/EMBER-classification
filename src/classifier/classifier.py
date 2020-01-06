## @package Module classifier
#
# KMEANS classifier using sklearn
# takes 4 datasets as input (training data, training labels, validation data, validation labels)
# reduce data size using PCA on a subset of our data
# train on the PCA reduced data
# evaluates on validation set
#

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

import time

time_start = time.time()

#TODO: remove the unlabelled data stripping

## Classify
# Extracts data from xtrain, ytrain, xvalidation, yvalidations files,
# then train a model on training data and evaluate it on validation data.
# @param xtrain: training data
# @param ytrain: training labels
# @param xvalidation: validation data
# @param yvalidation: validation labels
def classify(xtrain, ytrain, xvalidation, yvalidation, classifiers = [ ("KNeighboors(5)", KNeighborsClassifier(5)) ]):


    ## Constants
    #
    # @param VECTOR_SIZE: size (in float) of a single vector
    # @param PCA_TRAINING_SUBSET: size (in vectors) of training subset
    # @param VECTOR_CHUNK_SIZE: size (in vectors) of batches to be passed to PCA at a time
    # @param PCA_NB_COMPONENTS: number of dimensions to reduce our vectors into

    VECTOR_SIZE = 2351 
    PCA_TRAINING_SUBSET = 1000000
    VECTOR_CHUNK_SIZE = 10000
    PCA_NB_COMPONENTS = 20
    time_start_classify = time.time()

    # Returns a generator that yields chunks of size 'chunk_size'
    # from arrays data and labels
    def chunks(data: np.array):
        for i in range(0, len(data), VECTOR_CHUNK_SIZE):
            yield data[i:i + VECTOR_CHUNK_SIZE, :]

    # Map all data into memory, reshape data arrays to matrixes of dimension (NB_VECTOR, VECTOR_SIZE)
    def get_data_sets(xtrain, ytrain, xvalidation, yvalidation):
        training_data = np.memmap(xtrain, dtype=np.float32, mode='c', order='C')
        training_labels = np.memmap(ytrain, dtype=np.float32, mode='c', order='C')
        validation_data = np.memmap(xvalidation, dtype=np.float32, mode='c', order='C')
        validation_labels = np.memmap(yvalidation, dtype=np.float32, mode='c', order='C')

        training_data = training_data.reshape(-1, VECTOR_SIZE)
        validation_data = validation_data.reshape(-1, VECTOR_SIZE)
        
        print("[CLASSIF] got data set. Time elapsed since start: {}".format(time.time() - time_start))
        return training_data, validation_data, training_labels, validation_labels



    ## Data pre treatment function. Main memory / CPU usage.
    # Computes PCA for our training data, reducing the dimensions of our data vectors
    # We chose arbitrarily PCA_NB_COMPONENTS as the amount of dimensions to scale our vector into.
# 
    # The PCA is computed by increments to avoid RAM overusage (increases computation time..)
# 
    # Returns all the data array passed as arguments, unlabelled data removed and PCA-reduced
    def data_pre_treatment(training_data, validation_data):
       
        # subset of data to fit our pca on
        sample_indexes = np.random.choice(range(len(training_data)), 100000, replace=False)

        subset = training_data[sample_indexes]
        subset_indexes = training_labels[sample_indexes]

        pca = IncrementalPCA(n_components=PCA_NB_COMPONENTS, batch_size = 100)
        pca = pca.fit(subset)
        print("[CLASSIF] computed PCA on data subset. Time elapsed since start: {}".format(time.time() - time_start))

        # transform all validation data with the pca we just computed

        new_validation_data = pca.transform(validation_data)
        print("[CLASSIF] transformed our validation data using PCA. Time elapsed since start: {}".format(time.time() - time_start))

        # We process our training data by smaller chunks to not overload RAM usage

        new_training_data = np.vstack([ \
            pca.transform(chunk) for chunk in chunks(training_data) ])
        print("[CLASSIF] Transformed datasets using PCA. Training Data: {} vectors; Validation Data: {} vectors".format(len(new_training_data), len(new_validation_data)))
        print("[CLASSIF] Time elapsed since start: {}".format(time.time() - time_start))

        return new_training_data, new_validation_data

    def evaluate(classifier, validation_data, validation_labels):
        """
        Evaluates our classifier on validation data and labels.

        Returns a list of scores as well as a confusion matrix to visualize results
        """
        results = classifier.predict(validation_data)
        conf_matrix = confusion_matrix(validation_labels, results, labels=[0, 1])
        scores = precision_recall_fscore_support(validation_labels, results, average='macro', labels=[0, 1])
        return scores, conf_matrix
            
    # 0. start !

    print("[CLASSIF] starting classification process.")

    # 1. mmap data

    training_data, validation_data, training_labels, validation_labels = get_data_sets(xtrain, ytrain, xvalidation, yvalidation)

    # 2. pre process data

    training_data, validation_data = data_pre_treatment(training_data, validation_data)

    # 3. sanity check for results from pre processing

    for vector in training_data:
        assert(len(vector) == PCA_NB_COMPONENTS)
    for vector in validation_data:
        assert(len(vector) == PCA_NB_COMPONENTS)
    assert(len(training_labels) == len(training_data))
    assert(len(validation_labels) == len(validation_data))

    for (name, classifier) in classifiers:
        # 4. Build classifier and train it with Training Set
        time_clf_start = time.time()
        classifier.fit(training_data, training_labels)
        print("[CLASSIF] trained classifier {} in {} sec.".format(name, time.time() - time_clf_start))
        # 5. Evaluate classifier with Validation Set
       
        scores, matrix, = evaluate(classifier, validation_data, validation_labels)

        #accuracy = classifier.score(validation_data, validation_labels)
        print("[CLASSIF] classifier {} classified validation  {} sec.".format(name, time.time() - time_clf_start))
        print("[CLASSIF] confusion matrix:")
        print(matrix)
        print("[CLASSIF] scores:")
        print("[CLASSIF] precision: ------------ {:.2f} / 1.0".format(scores[0]))
        print("[CLASSIF] recall: --------------- {:.2f} / 1.0".format(scores[1]))
        print("[CLASSIF] F-beta score: --------- {:.2f} / 1.0".format(scores[2]))

    print("[CLASSIF] exiting program after {:.2f} seconds".format(time.time() - time_start_classify))

