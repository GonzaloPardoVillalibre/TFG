import json
from keras.models import model_from_json
import pandas as pd
import tensorflow as tf
import numpy as np

def load_model(modelPath:str):
    bestWeightsPath = modelPath + '/best_weights'
    weightsPath = modelPath + '/model.h5'
    modelPath = modelPath  + '/model.json'
    json_file = open(modelPath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(bestWeightsPath)
    return model

def file_to_inputdata(input_path, file, rows, columns, movements):
    temp = pd.read_csv(open(input_path+file,'r')) # Change this line to read any other type of file
    data = temp.values.reshape(rows,columns,1) # Convert column data to matrix like data with one channel
    activity = file.split('-')[1]
    label_classes = tf.constant(movements)
    pattern = tf.constant(activity)  # This line has changed
    for j in range(len(label_classes)):
        if label_classes[j].numpy() == pattern.numpy(): # Pattern is matched against different label_classes
            label = j
    data = np.asarray(data).reshape(-1,rows,columns,1)
    return data, label
