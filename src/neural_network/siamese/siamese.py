from keras import Model
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Input, Lambda
from keras import backend as b

def triplet_loss(y_true, y_pred):
    return b.mean(b.maximum(b.constant(0),\
        b.square(y_pred[:,0,0]) - 0.5 * b.square(y_pred[:,1,0]) + b.square(y_pred[:,2,0])) + b.constant(1))

def euclidean_distance(vectors):
    v1, v2 = vectors
    return b.sqrt(b.maximum(b.sum(b.square(v1 - v2), axis=1, keepdims=True),\
                            b.epsilon()))

def build_siamese():

    def base_model():
        INPUT_NORMALIZATION_LAYER_SIZE=1024
        model = Sequential()
        model.add(Dense(INPUT_NORMALIZATION_LAYER_SIZE, input_shape=(2351,), activation='relu', name='input'))
        model.add(BatchNormalization(name='normalize_input'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1, activation='sigmoid', name='output'))
        return model


    base = base_model()


    # creates 3 inputs
    input_anchor = Input(shape=(2351,), name='input_anchor')
    input_positive = Input(shape=(2351,), name='input_positive')
    input_negative = Input(shape=(2351,), name='input_negative')

    output_anchor = base(input_anchor)
    output_positive = base(input_positive)
    output_negative = base(input_negative)

    positive_distance = Lambda(euclidean_distance, name='distance_positive')([output_anchor, output_positive])
    negative_distance = Lambda(euclidean_distance, name='distance_negative')([output_anchor, output_negative])
    tertiary_distance = Lambda(euclidean_distance, name='distance_tertiary')([output_positive, output_negative])

    stacked_distances = Lambda(lambda vects: b.stack(vects, axis=1), name='stacked_distances')([positive_distance, negative_distance, tertiary_distance])

    model = Model([input_anchor, input_positive, input_negative], stacked_distances, name='siamese')

    return model

def compile_siamese(siamese):
    siamese.compile(loss=triplet_loss, optimizer='sgd', metrics=['accuracy'])