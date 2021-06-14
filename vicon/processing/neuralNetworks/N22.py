import tensorflow as tf
import utils.utils as nnUtils
from tensorflow.keras import layers

def load_model(input_rows:int, input_columns:int, channels:int):

    nnUtils.restrict_to_physcial_gpu()
    nnUtils.set_memory_growth()

    input_shape = (input_rows, input_columns, channels)

    model = tf.keras.Sequential([
        layers.Reshape((1,) + input_shape, input_shape=input_shape),
        layers.ConvLSTM2D(filters=8, kernel_size=(3, 3),
                     activation='relu',
                     return_sequences=True,
                     padding='same',
                     input_shape=(None, input_rows, input_columns, channels)),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(9, activation = "softmax")
    ])

    # model.summary()

    model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    return model

