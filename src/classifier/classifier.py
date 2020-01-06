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

from time import time

# our sklearn based incremental pca (src/utils/pca.py)
from pca import pca

time_start = time()

def log(tag, message, time_start = None):
    if time_start != None:
        print("[{}] {}. Time elapsed since start: {:.2f}".format(tag, message, time() - time_start))
    else:
        print("[{}] {}.".format(tag, message))

## Classify
# Extracts data from xtrain, ytrain, xvalidation, yvalidations files,
# then train a model on training data and evaluate it on validation data.
# @param xtrain: training data
# @param ytrain: training labels
# @param xvalidation: validation data
# @param yvalidation: validation labels
def classify(xtrain, ytrain, xvalidation, yvalidation):

    VECTOR_SIZE = 2351 
    time_start_classify = time()

    # Map all data into memory, reshape data arrays to matrixes of dimension (NB_VECTOR, VECTOR_SIZE)
    def get_data_sets(xtrain, ytrain, xvalidation, yvalidation):
        training_data = np.memmap(xtrain, dtype=np.float32, mode='c', order='C')
        training_labels = np.memmap(ytrain, dtype=np.float32, mode='c', order='C')
        validation_data = np.memmap(xvalidation, dtype=np.float32, mode='c', order='C')
        validation_labels = np.memmap(yvalidation, dtype=np.float32, mode='c', order='C')

        training_data = training_data.reshape(-1, VECTOR_SIZE)
        validation_data = validation_data.reshape(-1, VECTOR_SIZE)
        
        log("CLASSIF", "got data set.", time_start)
        return training_data, validation_data, training_labels, validation_labels

    ## Data pre treatment function. Main memory / CPU usage.
    # Computes PCA for our training data, reducing the dimensions of our data vectors
    # We chose arbitrarily PCA_NB_COMPONENTS (in src/utils/pca.py) as the amount of dimensions to scale our vector into.
    # The PCA is computed by increments to avoid RAM overusage (increases computation time..)
    # Returns all the data array passed as arguments, unlabelled data removed and PCA-reduced
    def data_pre_treatment(training_data, validation_data):
       
        training_data, validation_data = pca(training_data, validation_data)
        
        log("CLASSIF", "Transformed datasets using PCA.\nTraining Data: {} vectors; \
            Validation Data: {} vectors".format(len(training_data), len(validation_data)), time_start)

        return training_data, validation_data

    ## Evaluates our classifier on validation data and labels.
    # Returns a list of scores as well as a confusion matrix to visualize results
    def evaluate(classifier, validation_data, validation_labels):
        results = classifier.predict(validation_data)
        conf_matrix = confusion_matrix(validation_labels, results, labels=[0, 1])
        scores = precision_recall_fscore_support(validation_labels, results, average='macro', labels=[0, 1])
        return scores, conf_matrix
            
    # 0. start !
    log("CLASSIF", "starting classification process.", time_start)

    # 1. mmap data
    training_data, validation_data, training_labels, validation_labels = get_data_sets(xtrain, ytrain, xvalidation, yvalidation)

    # 2. pre process data
    training_data, validation_data = data_pre_treatment(training_data, validation_data)

    # 3. Build classifier and train it with Training Set
    classifier = KNeighborsClassifier(5)
    time_clf_start = time()
    classifier.fit(training_data, training_labels)
    log("CLASSIF", "trained classifier KNeighboor(5) in {:.2f} sec.".format(time() - time_clf_start))
    
    # 4. Evaluate classifier with Validation Set
    scores, matrix, = evaluate(classifier, validation_data, validation_labels)
    log("CLASSIF", "classifier KNeighboor(5) classified validation set in {:.2f} sec.".format(time() - time_clf_start))
    log("CLASSIF", "confusion matrix:")
    print(matrix)
    log("CLASSIF", "scores:")
    log("CLASSIF", "precision: ------------ {:.2f} / 1.0".format(scores[0]))
    log("CLASSIF", "recall: --------------- {:.2f} / 1.0".format(scores[1]))
    log("CLASSIF", "F-beta score: --------- {:.2f} / 1.0".format(scores[2]))
    log("CLASSIF", "Completed classification.", time_start)
