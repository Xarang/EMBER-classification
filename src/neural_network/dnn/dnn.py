from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.optimizers import Adagrad

from time import time

VECTOR_SIZE = 2351

def log(tag, message, time_start = None):
    print("[{}] {} in {:.2f} sec.".format(tag, message, time() - time_start))


## create our densely connected network
def build_dnn():
    time_start = time()
    hidden_layer_sizes = [ 1024, 512, 256, 128, 64 ]
    hidden_layer_activation_function='relu'

    model = Sequential()
    model.add(Dense(hidden_layer_sizes[0], activation='relu', input_shape=(2351,), name='input'))
    model.add(BatchNormalization(name='normalize_input'))

    for i in range(len(hidden_layer_sizes)):
        model.add(Dense(hidden_layer_sizes[i], activation=hidden_layer_activation_function, name='hidden_{}'.format(i)))
    model.add(Dense(1, activation='sigmoid', name='output'))
    log("DNN", "Built DNN", time_start)
    return model

## compile network passed as argument using Adagrad optimizer and binary crossentropy loss
def compile_dnn(dnn):
    time_start = time()
    opt = Adagrad(lr=0.0007)
    dnn.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
    log("DNN", "Compiled DNN.", time_start)
    