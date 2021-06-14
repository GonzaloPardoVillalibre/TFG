import tensorflow as tf
import utils.utils as nnUtils
from tensorflow.keras import layers

def load_model(input_rows:int, input_columns:int, channels:int):

    nnUtils.restrict_to_physcial_gpu()
    nnUtils.set_memory_growth()

    model = tf.keras.Sequential([
        layers.Lambda(lambda x: x[:,:,:,0], input_shape=(*(input_rows,input_columns), 1)),
        layers.LSTM(units=256, return_sequences=True),
        layers.Dropout(0.5),
        layers.LSTM(units=128, return_sequences=True),
        layers.LSTM(units=64),
        layers.Dense(128),
        layers.Dense(9, activation = "softmax")
    ])

    # model.summary()

    model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    return model

