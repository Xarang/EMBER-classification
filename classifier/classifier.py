import sys
import struct
import resource
import numpy as np

from sklearn import svm
from sklearn.decomposition import IncrementalPCA

from multiprocessing import Process

import time

time_start = time.time()


def chunks(array: np.array, chunk_size: int):
    for i in range(0, len(array), chunk_size):
        yield array[i:i + chunk_size, :]


def classifier(xdatfile, ydatfile):

    # CONSTS
    SET_SIZE = 900000 # total amonut of vectors in ember dataset
    VECTOR_SIZE = 2351 # size (in float) of a single vector

    TRAINING_SET_SIZE = 800000 # amount of vectors in training set (subset of SET_SIZE)
    VALIDATION_SET_SIZE = 50000 # amount of vectors in validation set (subset of SET_SIZE)

    PCA_TRAINING_SUBSET = 1000000

    VECTOR_CHUNK_SIZE = 10000 # vectors to be transformed by tca at a time

    PCA_NB_COMPONENTS = 20
    #######

    # reshaped_data = np.reshape(data, (-1, 2351))
    # sample = reshaped_data[np.random.randint(0, 900000, 100000),:]
    # sample_indexes = np.random.randint(0, 900000, 100000)
    # sample_x = reshaped_data[sample_indexes,:]
    # sample_y = labels[sample_indexes]
    # classified_indexes = sample_y != -1
    # final_y = sample_y[classified_indexes]
    # final_x = sample_x[classified_indexes]

    def limit_memory(maxmemory):
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (maxmemory, hard))

    def get_data_sets(xfile, yfile):

        data = np.memmap(xfile, dtype=np.float32, mode='c', order='C')
        labels = np.memmap(yfile, dtype=np.float32, mode='c', order='C')

        data = data.reshape(-1, VECTOR_SIZE)

        training_data     =   data[0:TRAINING_SET_SIZE]
        training_labels   = labels[0:TRAINING_SET_SIZE]
        validation_data   =   data[TRAINING_SET_SIZE + 1: TRAINING_SET_SIZE + VALIDATION_SET_SIZE + 1]
        validation_labels = labels[TRAINING_SET_SIZE + 1: TRAINING_SET_SIZE + VALIDATION_SET_SIZE + 1]
        
        def remove_unlabelled_data(data, labels):
            labelled_indexes = labels != -1
            data = data[labelled_indexes]
            labels = labels[labelled_indexes]
            return data, labels

        training_data, training_labels = remove_unlabelled_data(training_data, training_labels)
        validation_data, validation_labels = remove_unlabelled_data(validation_data, validation_labels)

        print("[CLASSIF] got data set. Time elapsed since start: {}".format(time.time() - time_start))
        return training_data, validation_data, training_labels, validation_labels

    def data_pre_treatment(training_data, validation_data):

        #subset of data to fit our pca on
        sample_indexes = np.random.choice(range(len(training_data)), 100000, replace=False)
        subset = training_data[sample_indexes]
        
        pca = IncrementalPCA(n_components=PCA_NB_COMPONENTS, batch_size = 50)
        pca = pca.fit(subset)
        print("[CLASSIF] computed PCA on data subset. Time elapsed since start: {}".format(time.time() - time_start))
        new_validation_data = pca.transform(validation_data)
        print("[CLASSIF] transformed our validation data using PCA. Time elapsed since start: {}".format(time.time() - time_start))
        
        # We process our training data by smaller chunks to not overload RAM usage


        new_training_data = np.vstack([pca.transform(chunk) for chunk in chunks(training_data, VECTOR_CHUNK_SIZE)])

        print("[CLASSIF] Transformed datasets using PCA. Training Data: {} vectors; Validation Data: {} vectors".format(len(new_training_data), len(new_validation_data)))
        print("[CLASSIF] Time elapsed since start: {}".format(time.time() - time_start))
        return new_training_data, new_validation_data


    def build_svm(data, labels):
        clf = svm.LinearSVC()
        clf.fit(data, labels)
        print("[CLASSIF] trained classifier. Time elapsed since start: {}".format(time.time() - time_start))
        confidence = clf.decision_function(data)
        print("[CLASSIF] confidence scores for samples:")
        print(confidence)
        return clf

    # 1. mmap data

    training_data, validation_data, training_labels, validation_labels = get_data_sets(xdatfile, ydatfile)

    # 2. pre process data

    training_data, validation_data = data_pre_treatment(training_data, validation_data)

    # 3. sanity check for results from pre processing

    for vector in training_data:
        assert(len(vector) == PCA_NB_COMPONENTS)
    for vector in validation_data:
        assert(len(vector) == PCA_NB_COMPONENTS)
    assert(len(training_labels) == len(training_data))
    assert(len(validation_labels) == len(validation_data))

    # 4. Build classifier and train it with Training Set

    clf = build_svm(training_data, training_labels)

    # 5. Evaluate classifier with Validation Set

    accuracy = clf.score(validation_data, validation_labels)
    print("[CLASSIF] validated validation set with accuracy: {}. Time elapsed since start: {}".format(accuracy, time.time() - time_start))


# run this in a separate thread

xdatfile = sys.argv[1]
ydatfile = sys.argv[2]

run_classifier = Process(target=classifier, args=(xdatfile, ydatfile))

run_classifier.start()
run_classifier.join()