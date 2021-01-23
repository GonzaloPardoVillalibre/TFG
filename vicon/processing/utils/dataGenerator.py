import os, json
import tensorflow as tf 
import numpy as np
import pandas as pd
# This function load the data from the csv, labels it and generates a tensorflow tf

def tf_data_generator(input_path:str, file_list: list, batch_size: int, movements: list):
    i = 0
    while True:
        if i*batch_size >= len(file_list):  # This loop is used to run the generator indefinitely.
            i = 0
            np.random.shuffle(file_list)
        else:
            file_chunk = file_list[i*batch_size:(i+1)*batch_size] 
            data = []
            labels = []
            label_classes = tf.constant(movements)
            for file in file_chunk:
                temp = pd.read_csv(open(input_path+file,'r')) # Change this line to read any other type of file
                temp = temp.drop(temp.columns[0], axis=1)
                data.append(temp.values.reshape(128,84,1)) # Convert column data to matrix like data with one channel
                pieces = file.decode("utf-8").split('/')
                activity = pieces[-1].split('-')[1]
                pattern = tf.constant(activity)  # This line has changed
                for j in range(len(label_classes)):
                    if label_classes[j].numpy() == pattern.numpy(): # Pattern is matched against different label_classes
                        labels.append(j)  
            data = np.asarray(data).reshape(-1,128,84,1)
            labels = np.asarray(labels)
            yield data, labels
            i = i + 1