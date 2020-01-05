import sys
import time
import os
import time

import numpy as np

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras import optimizers
from keras import utils
from keras import losses

def log(message, time_start):
    print("[DNN] {}. Time elapsed since start: {:.2f}".format(message, time.time() - time_start))

class ember_classification_dnn:
    training_data = None
    training_labels = None
    validation_data = None
    validation_labels = None

    model = None

    def __init__(self):

        self.VECTOR_SIZE = 2351
        self.time_start = time.time()

    def load_training_set(self, xtrain, ytrain):
        self.training_data = np.memmap(xtrain, dtype=np.float32, mode='c', order='C')
        self.training_data = self.training_data.reshape(-1, self.VECTOR_SIZE)
        self.training_labels = np.memmap(ytrain, dtype=np.float32, mode='c', order='C')
        assert(len(self.training_data) == len(self.training_labels))
        log("got Training set. Size: {}".format(len(self.training_data)), self.time_start)

    def load_validation_set(self, xvalidation, yvalidation):
        self.validation_data = np.memmap(xvalidation, dtype=np.float32, mode='c', order='C')
        self.validation_data = self.validation_data.reshape(-1, self.VECTOR_SIZE)
        self.validation_labels = np.memmap(yvalidation, dtype=np.float32, mode='c', order='C')
        assert(len(self.validation_data) == len(self.validation_labels))
        log("got Validation set. Size: {}".format(len(self.training_data)), self.time_start)

    def build_dnn(self):

        hidden_layer_sizes = [ 512, 128, 128, 64, 32 ]
        hidden_layer_activation_function='relu'

        model = Sequential()
        model.add(Dense(hidden_layer_sizes[0], activation='relu', input_shape=(self.VECTOR_SIZE,), name='input'))
        model.add(BatchNormalization(name='normalize_input'))

        for i in range(len(hidden_layer_sizes)):
            model.add(Dense(hidden_layer_sizes[i], activation=hidden_layer_activation_function, name='hidden_{}'.format(i)))
        model.add(Dense(1, activation='sigmoid', name='output'))
        log("Built DNN", self.time_start)
        self.model = model
        self.compile()

    def compile(self):
        self.model.compile(loss='binary_crossentropy',
                    optimizer='Adagrad',
                    metrics=['accuracy'])
        log("Compiled DNN. Time elapsed since start", self.time_start)

    def train(self, output_filename):
        NB_TRAINING_EPOCHS = 50
        log("Starting DNN training..", self.time_start)
        self.model.fit(self.training_data, self.training_labels, epochs=NB_TRAINING_EPOCHS, validation_data=(self.validation_data, self.validation_labels), batch_size=256)
        log("DNN training completed !", self.time_start)
        if (output_filename == None):
            return
        dnn_json = self.model.to_json()
        with open(output_filename + '.json', "w") as json_file:
            json_file.write(dnn_json)
        self.model.save_weights(output_filename + '.weights')

    def evaluate(self):
        score, metrics = self.model.evaluate(self.validation_data, self.validation_labels, batch_size = 10)
        log("DNN evaluation", self.time_start)
        log("Score: {}".format(score), self.time_start)
        log("Accuracy: {}".format(metrics), self.time_start)

    def load(self, input_filename):
        model_json = open(input_filename + '.json', "r")
        dnn = model_json.read()
        dnn = model_from_json(dnn)
        model_json.close()
        dnn.load_weights(input_filename + '.weights')
        self.model = dnn
        log("Loaded DNN", self.time_start)
        self.compile()


def train_and_save(xtrainfile, ytrainfile, xvalidationfile, yvalidationfile):
    ecd = ember_classification_dnn()
    log("[TRAIN&SAVE] entered train&save procedure", ecd.time_start)
    ecd.load_training_set(xtrainfile, ytrainfile)
    ecd.load_validation_set(xvalidationfile, yvalidationfile)
    if not os.path.exists('models'):
        os.mkdir('models')
    ecd.build_dnn()
    ecd.train('models/train_{}'.format(time.clock_gettime(0)))
    ecd.evaluate()
    log("[TRAIN&SAVE] exited train&save procedure", ecd.time_start)

def load_and_evaluate(model_filename, xvalidationfile, yvalidationfile):
    ecd = ember_classification_dnn()
    log("[LOAD&EVALUATE] entered load&evaluate procedure", ecd.time_start)
    ecd.load_validation_set(xvalidationfile, yvalidationfile)
    ecd.load(model_filename)
    ecd.evaluate
    log("[LOAD&EVALUATE] exited load&evaluate procedure", ecd.time_start)