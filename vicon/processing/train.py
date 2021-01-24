import utils.utils as utils
import utils.dataGenerator as datagen
import tensorflow as tf 
import re, os, json
import importlib

main_path = os.getcwd()
final_input_path = main_path + '/vicon/pre-processing/final-dataset/orientation/'
_, _, files = next(os.walk(final_input_path))

#########################
#  Main train function  #
#########################
def train_main(cfg: json):

  rows, columns, movements, batch_size, train_steps, validation_steps, test_steps, epochs = utils.extract_info_from_config(cfg)

  train_set, validation_set, test_set = utils.split_dataset(files, cfg)

  train_dataset = tf.data.Dataset.from_generator(datagen.tf_data_generator,args= [final_input_path, train_set, batch_size, movements],output_types = (tf.float32, tf.float32), output_shapes = ((None,rows,columns,1),(None,)))
  test_dataset = tf.data.Dataset.from_generator(datagen.tf_data_generator,args= [final_input_path, test_set, batch_size, movements],output_types = (tf.float32, tf.float32), output_shapes = ((None,rows,columns,1),(None,)))
  validation_dataset = tf.data.Dataset.from_generator(datagen.tf_data_generator,args= [final_input_path, validation_set, batch_size, movements],output_types = (tf.float32, tf.float32), output_shapes = ((None,rows,columns,1),(None,)))

  nn = importlib.import_module('neuralNetworks.' + cfg["neural-network"])

  model = nn.load_model(rows, columns)

  history_callback = model.fit(train_dataset, validation_data = validation_dataset, steps_per_epoch = train_steps,
          validation_steps = validation_steps, epochs = epochs)
  
  test_loss, test_accuracy = model.evaluate(test_dataset, steps = test_steps)

  return model, test_loss, test_accuracy, history_callback