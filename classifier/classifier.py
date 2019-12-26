import sys
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from multiprocessing import Process

import psutil

import time

time_start = time.time()

def classify(xdatfile, ydatfile, classifiers = [ ("KNeighboors(5)", KNeighborsClassifier(5)) ]):

    # CONSTS

    SET_SIZE = 900000 # total amonut of vectors in ember dataset
    VECTOR_SIZE = 2351 # size (in float) of a single vector
    TRAINING_SET_SIZE = 850000 # amount of vectors in training set (subset of SET_SIZE)
    VALIDATION_SET_SIZE = 50000 # amount of vectors in validation set (subset of SET_SIZE)
    PCA_TRAINING_SUBSET = 1000000 # amount of vectors to generate our PCA with
    VECTOR_CHUNK_SIZE = 10000 # vectors to be transformed by tca at a time
    PCA_NB_COMPONENTS = 20 # nb of dimensions vectors to be reduced into

    time_start_classify = time.time()

    #######
    def chunks(data: np.array, labels: np.array, chunk_size: int):
        for i in range(0, len(data), chunk_size):
            yield ( data[i:i + chunk_size, :], labels[i:i + chunk_size] )

    def remove_unlabelled_data(data, labels):
        labelled_indexes = labels != -1
        data = data[labelled_indexes]
        labels = labels[labelled_indexes]
        return data, labels

    def get_data_sets(xfile, yfile):

        data = np.memmap(xfile, dtype=np.float32, mode='c', order='C')
        labels = np.memmap(yfile, dtype=np.float32, mode='c', order='C')

        data = data.reshape(-1, VECTOR_SIZE)

        training_data     =   data[0:TRAINING_SET_SIZE]
        training_labels   = labels[0:TRAINING_SET_SIZE]
        validation_data   =   data[TRAINING_SET_SIZE + 1: TRAINING_SET_SIZE + VALIDATION_SET_SIZE + 1]
        validation_labels = labels[TRAINING_SET_SIZE + 1: TRAINING_SET_SIZE + VALIDATION_SET_SIZE + 1]
        
        print("[CLASSIF] got data set. Time elapsed since start: {}".format(time.time() - time_start))
        return training_data, validation_data, training_labels, validation_labels

    def data_pre_treatment(training_data, validation_data, training_labels, validation_labels):

        # subset of data to fit our pca on
        sample_indexes = np.random.choice(range(len(training_data)), 100000, replace=False)

        subset = training_data[sample_indexes]
        subset_indexes = training_labels[sample_indexes]
        subset, subset_indexes = remove_unlabelled_data(subset, subset_indexes)

        pca = IncrementalPCA(n_components=PCA_NB_COMPONENTS, batch_size = 100)
        pca = pca.fit(subset)
        print("[CLASSIF] computed PCA on data subset. Time elapsed since start: {}".format(time.time() - time_start))

        # transform all validation data with the pca we just computed

        validation_data, validation_labels = remove_unlabelled_data(validation_data, validation_labels)
        new_validation_data = pca.transform(validation_data)
        new_validation_labels = validation_labels
        print("[CLASSIF] transformed our validation data using PCA. Time elapsed since start: {}".format(time.time() - time_start))


        # We process our training data by smaller chunks to not overload RAM usage

        new_training_data = np.vstack([ \
                pca.transform(remove_unlabelled_data(chunk_tuple[0], chunk_tuple[1])[0]) \
                for chunk_tuple in chunks(training_data, training_labels, VECTOR_CHUNK_SIZE) \
                ])
        new_training_labels = training_labels[training_labels != -1]
        print("[CLASSIF] Transformed datasets using PCA. Training Data: {} vectors; Validation Data: {} vectors".format(len(new_training_data), len(new_validation_data)))
        print("[CLASSIF] Time elapsed since start: {}".format(time.time() - time_start))

        return new_training_data, new_validation_data, new_training_labels, new_validation_labels

    def evaluate(classifier, validation_data, validation_labels):
        results = classifier.predict(validation_data)
        conf_matrix = confusion_matrix(validation_labels, results, labels=[0, 1])
        scores = precision_recall_fscore_support(validation_labels, results, average='macro', labels=[0, 1])
        return scores, conf_matrix
            
    # 0. start !

    print("[CLASSIF] starting classification process.")

    # 1. mmap data

    training_data, validation_data, training_labels, validation_labels = get_data_sets(xdatfile, ydatfile)

    # 2. pre process data

    training_data, validation_data, training_labels, validation_labels = data_pre_treatment(training_data, validation_data, training_labels, validation_labels)

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


# run this in a separate thread

if __name__ == '__main__':
    xdatfile = sys.argv[1]
    ydatfile = sys.argv[2]

    def get_resources_informations(report_id):
        memory_infos = psutil.virtual_memory()
        cpu_usage = psutil.cpu_percent()
        memory_used = memory_infos.total - memory_infos.available
        memory_used_gb = memory_used / 1024 / 1024 / 1024
        memory_used_percentage = memory_used / memory_infos.total * 100

        print("[RESOURCES] report #{}; Memory used: {:.2f} GB ({:.2f}%). CPU usage: {:.2f}%".format(report_id, memory_used_gb, memory_used_percentage, cpu_usage))

    run_classifier = Process(target=classify, args=[xdatfile, ydatfile])

    run_classifier.start()
    report_id = 0
    while run_classifier.is_alive():
        run_memory_monitor = Process(target=get_resources_informations, args=[report_id])
        run_memory_monitor.start()
        report_id += 1
        run_memory_monitor.join()
        time.sleep(10)

    run_classifier.join()