import sys
import time

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import losses


xtrainfile, ytrainfile, xvalidationfile, yvalidationfile

xtrainfile = sys.argv[1]
ytrainfile = sys.argv[2]
xvalidationfile = sys.argv[3]
yvalidationfile = sys.argv[4]

VECTOR_SIZE = 2351
NB_TRAINING_EPOCHS = 5

time_start = time.time()

def log(message):
    print("[DNN] {}. Time elapsed since start: {:.2f}".format(message, time.time() - time_start))

def get_data_sets(xtrain, ytrain, xvalidation, yvalidation):

    training_data = np.memmap(xtrain, dtype=np.float32, mode='c', order='C')
    training_labels = np.memmap(ytrain, dtype=np.float32, mode='c', order='C')
    validation_data = np.memmap(xvalidation, dtype=np.float32, mode='c', order='C')
    validation_labels = np.memmap(yvalidation, dtype=np.float32, mode='c', order='C')

    training_data = training_data.reshape(-1, VECTOR_SIZE)
    validation_data = validation_data.reshape(-1, VECTOR_SIZE)
    
    log("got data set")
    return training_data, validation_data, training_labels, validation_labels

def build_dnn():
    #just some random structure
    model = Sequential()s
    model.add(Dense(input_shape=(VECTOR_SIZE,), units=VECTOR_SIZE))
    model.add(Dense(input_shape=(VECTOR_SIZE,), units=VECTOR_SIZE))
    model.add(Dense(input_shape=(VECTOR_SIZE,), units=1))
    log("Built DNN")

    # stochastic gradient descent
    sgd = optimizers.SGD(lr=0.01, decay=0.000001, momentum=0.9, nesterov=True)
    #configure learning process
    model.compile(loss=losses.mean_squared_error,
            optimizer=sgd,
            metrics=['accuracy'])
    log("Configured DNN. Time elapsed since start")
    return model

def train_dnn(dnn, xtrain, ytrain):
    log("Starting DNN training..")
    dnn.fit(xtrain, ytrain, epochs=NB_TRAINING_EPOCHS, batch_size=32)
    log("DNN training completed !")

def evaluate_dnn(dnn, xvalid, yvalid):
    loss_and_metrics = dnn.evaluate(xvalid, yvalid, batch_size = 128)
    log("DNN evaluation")
    print(loss_and_metrics)

def save_dnn(dnn, model_filename, weights_filename):
    dnn_json = dnn.to_json()
    with open(model_file, "w") as json_file:
        json_file.write(dnn_json)
    model.save_weights(weights_file)

def load_dnn(dnn, model_filename, weights_filename)
    model_json = open(model_filename, "r")
    dnn = model_json.read()
    model_json.close()
    dnn.load_weights(weights_filename)
    return dnn

training_data, validation_data, training_labels, validation_labels = get_data_sets(xtrainfile, ytrainfile, xvalidationfile, yvalidationfile)

dnn = build_dnn()

train_dnn(dnn, training_data, training_labels)

save_dnn(dnn, "model.json", "model.weights")

evaluate_dnn(dnn, validation_data, validation_labels)