import sys
import time
import os
from time import time
from time import clock_gettime

import numpy as np

from keras.models import model_from_json
from keras import utils
from keras import backend as b

from dnn.dnn import build_dnn, compile_dnn

def log(tag, message, time_start = None):
    if time_start != None:
        print("[{}] {}. Time elapsed since start: {:.2f}".format(tag, message, time() - time_start))
    else:
        print("[{}] {}.".format(tag, message))

def triplet_loss(y_true, y_pred):
    return b.mean(b.maximum(b.constant(0),\
        b.square(y_pred[:,0,0]) - 0.5 * b.square(y_pred[:,1,0]) + b.square(y_pred[:,2,0])) + b.constant(1))

## ember_classification_dnn
# the dnn class can be used in two major ways:
# train on a data set
# load a model from files
# we provide methods below to manipulate dnn
class ember_classification_neural_network:
    training_data = None
    training_labels = None
    validation_data = None
    validation_labels = None
    model = None
    nn_type = None

    def __init__(self, nn_type = 'dnn'):

        self.VECTOR_SIZE = 2351
        self.time_start = time()
        self.nn_type = nn_type

    ## load_training_set
    # maps xtrain and ytrain files into memory
    def load_training_set(self, xtrain, ytrain):
        self.training_data = np.memmap(xtrain, dtype=np.float32, mode='c', order='C')
        self.training_data = self.training_data.reshape(-1, self.VECTOR_SIZE)
        self.training_labels = np.memmap(ytrain, dtype=np.float32, mode='c', order='C')
        assert(len(self.training_data) == len(self.training_labels))
        log("NEURAL NETWORK", "got Training set. Size: {}".format(len(self.training_data)), self.time_start)
    
    ## load_validation_set
    # maps xvalidation and yvalidation into memory
    def load_validation_set(self, xvalidation, yvalidation):
        self.validation_data = np.memmap(xvalidation, dtype=np.float32, mode='c', order='C')
        self.validation_data = self.validation_data.reshape(-1, self.VECTOR_SIZE)
        self.validation_labels = np.memmap(yvalidation, dtype=np.float32, mode='c', order='C')
        assert(len(self.validation_data) == len(self.validation_labels))
        log("NEURAL NETWORK", "got Validation set. Size: {}".format(len(self.validation_data)), self.time_start)

    ## build_dnn
    # builds our dnn and compiles it
    def build(self):
        if self.nn_type == 'dnn':
            self.model = build_dnn()
            compile_dnn(self.model)
        else:
            pass

    ## builds dnn using triplet loss function (siamese network)
    def build_siamese(self):
        pass


    ## train
    # Training function. requires a built dnn, and all datasets to be previously loaded
    # @param output_filename: base file name of model to be outputted into
    # After each epoch computation, runs an evaluation on our validation set to keep track
    # of our actual accuracy gain. Outputs trained model in 'output_filename'.json and weights in 'output_filename'.weights
    def train(self, output_filename):
        NB_TRAINING_EPOCHS = 300
        log("NEURAL NETWORK", "Starting training..", self.time_start)
        self.model.fit(self.training_data, self.training_labels, epochs=NB_TRAINING_EPOCHS, validation_data=(self.validation_data, self.validation_labels), batch_size=256)
        log("NEURAL NETWORK", "training completed !", self.time_start)
        if (output_filename == None):
            return
        nn_json = self.model.to_json()
        with open(output_filename + '.json', "w") as json_file:
            json_file.write(nn_json)
        self.model.save_weights(output_filename + '.weights')

    ## evaluation
    # Evaluation function. required a built, trained dnn and validation dataset to be previously loaded
    # displays some useful metrics regarding our results
    def evaluate(self):
        score, metrics = self.model.evaluate(self.validation_data, self.validation_labels, batch_size = 10)
        log("NEURAL NETWORK", "DNN evaluation:", self.time_start)
        log("NEURAL NETWORK", "Score: {}".format(score), self.time_start)
        log("NEURAL NETWORK", "Accuracy: {}".format(metrics), self.time_start)

    ## load
    # @param input_filename: base filename of model to load from
    # builds a dnn using 'input_filename'.json as model and 'input_filename'.weights as weights
    # then compiles it
    def load(self, input_filename):
        model_json = open(input_filename + '.json', "r")
        nn = model_json.read()
        nn = model_from_json(nn)
        model_json.close()
        nn.load_weights(input_filename + '.weights')
        self.model = nn
        log("NEURAL NETWORK", "Loaded neural network", self.time_start)
        if self.nn_type == 'dnn':
            compile_dnn(self.model)
        else:
            pass


## dnn_train_and_save
# @param xtrainfile: training data
# @param ytrainfile: training labels
# @param xvalidationfile: validation data
# @param yvalidationfile: validation labels
# Loads provided dataset, builds a dnn and train on training dataset,
# Then run evaluation once and outputs generated model in a file
# named after current time
def dnn_train_and_save(xtrainfile, ytrainfile, xvalidationfile, yvalidationfile):
    ecd = ember_classification_neural_network('dnn')
    log("[DNN]TRAIN&SAVE entered train&save procedure", ecd.time_start)
    log("[DNN]TRAIN&SAVE data sets: {}".format([xtrainfile, ytrainfile, xvalidationfile, yvalidationfile]), ecd.time_start)
    ecd.load_training_set(xtrainfile, ytrainfile)
    ecd.load_validation_set(xvalidationfile, yvalidationfile)
    if not os.path.exists('models'):
        os.mkdir('models')
    ecd.build_dnn()
    ecd.train('models/train_{}'.format(time.clock_gettime(0)))
    ecd.evaluate()
    log("[DNN]TRAIN&SAVE", "exited train&save procedure", ecd.time_start)


## dnn_load_and_evaluate
# @param model_filename: base filename of model to be loaded from. 
# Weights are contained in 'model_filename'.model whereas model is contained in 'model_filename'.json
# I provide a pre-trained model and weights in this repository, that can be found in dnn/models/
# @param xvalidationfile: validation data
# @param yvalidationfile: validation labels
def dnn_load_and_evaluate(model_filename, xvalidationfile, yvalidationfile):
    ecd = ember_classification_neural_network('dnn')
    log("[DNN]LOAD&EVALUATE", "entered load&evaluate procedure", ecd.time_start)
    log("[DNN]LOAD&EVALUATE", "model filename: {}".format(model_filename), ecd.time_start)
    log("[DNN]LOAD&EVALUATE", "data sets: {}".format([xvalidationfile, yvalidationfile]), ecd.time_start)
    ecd.load_validation_set(xvalidationfile, yvalidationfile)
    ecd.load(model_filename)
    ecd.evaluate()
    log("[DNN]LOAD&EVALUATE", "exited load&evaluate procedure", ecd.time_start)