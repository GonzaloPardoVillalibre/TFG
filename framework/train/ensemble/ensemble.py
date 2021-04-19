import os, json, gc, sys
import numpy as np
from sklearn.metrics import accuracy_score
import copy
import pandas as pd
from tabulate import tabulate

main_path = os.getcwd()
# main_path = '/TFG'
train_path = main_path + '/framework/train'
outcome = main_path + '/framework/train/ensemble/ensembleOutcomes'
models_path =  main_path + '/framework/train/ensemble/models'
final_input_path = main_path + '/framework/final-dataset/orientation/'
sys.path.insert(1, train_path)
import utils.utils as utils
import ensembleUtils

# Wich K train of the K-fold to test assemble
K = 7

models_paths = []

# Read all available models
_, models_folders, files = next(os.walk(models_path))

for folder in models_folders:
    folder_path = models_path + '/' + folder
    _, models_folder, files = next(os.walk(folder_path))
    models_folder = sorted(models_folder)
    models_paths.append(folder_path + '/' + models_folder[K])

# Read configuration 
cfg = utils.loadCfgJson(models_paths[0] + '/config.json')
movements = cfg["movements"]
rows = cfg["input-rows"]
columns = cfg["input-columns"]
test_subjects = cfg["test-subjects"] 

models = []
# Load all models
for model in models_paths:
    models.append(ensembleUtils.load_model(model))

# Load test set
_, folders, files = next(os.walk(final_input_path))
test_user_regex_string = utils.build_regex_for_subjects(test_subjects)
test_files = utils.filter_files_by_regex(files, test_user_regex_string)
test_files = utils.filter_files_by_regex(test_files, r'-0.csv$')
np.random.shuffle(test_files)
# test_files = test_files[:50]

predictions = np.zeros((len(models), len(test_files),  len(movements)))
input_labels = []

# Predict movement for each file and model
file_number = 0
for file in test_files:
    model_number = 0
    test_data, test_label = ensembleUtils.file_to_inputdata(final_input_path, file, rows, columns, movements)
    for model in models:
        predictions[model_number][file_number] = model.predict(test_data) 
        model_number = model_number + 1
    file_number = file_number +1
    input_labels.append(test_label)

models_accuracy = []
for model in range(len(models)):
    models_prediction = np.argmax(predictions[model], axis=1)
    model_accuracy = accuracy_score(input_labels, models_prediction)
    print('Accuracy for ' + models_paths[model].split('/')[-2] + ' :', model_accuracy)
    models_accuracy.append(model_accuracy)

summed = np.sum(predictions, axis=0)
ensemble_prediction = np.argmax(summed, axis=1)
ensembled_accuracy = accuracy_score(input_labels, ensemble_prediction)
print('Accuracy for ensembled model:', ensembled_accuracy)

df = pd.DataFrame([])


##################
# For two models #
##################
# Combine weights for two models to get the better combination
# for w1 in range(0, 10):
#     w2 = 10 - w1
#     new_predictions = copy.deepcopy(predictions)
#     wts = [w1/10.,w2/10.]
#     new_predictions[0] = new_predictions[0]*wts[0]
#     new_predictions[1] = new_predictions[1]*wts[1]
#     summed = np.sum(new_predictions, axis=1)
#     ensemble_prediction = np.argmax(summed, axis=1)
#     ensembled_accuracy = accuracy_score(input_labels, ensemble_prediction)
#     df = df.append(pd.DataFrame({'wt1':wts[0],'wt2':wts[1], 
#                                     'acc':ensembled_accuracy*100}, index=[0]), ignore_index=True)

# max_acc_row = df.iloc[df['acc'].idxmax()]
# print("Max accuracy of ", max_acc_row[2], " obained with w1=", max_acc_row[0],
#       " w2=", max_acc_row[1])

####################
# For three models #
####################
# Combine weights for three models to get the better combination
for w1 in range(0, 10):
    x = 10 - w1
    for w2 in range(0, x):
            w3 = 10 - w1 - w2
            wts = [w1/10.,w2/10.,w3/10.]
            new_predictions = copy.deepcopy(predictions)
            new_predictions[0] = new_predictions[0]*wts[0]
            new_predictions[1] = new_predictions[1]*wts[1]
            new_predictions[2] = new_predictions[2]*wts[2]
            summed = np.sum(new_predictions, axis=0)
            ensemble_prediction = np.argmax(summed, axis=1)
            ensembled_accuracy = accuracy_score(input_labels, ensemble_prediction)
            df = df.append(pd.DataFrame({'wt1':wts[0],'wt2':wts[1], 
                                         'wt3':wts[2], 'acc':ensembled_accuracy*100}, index=[0]), ignore_index=True)
            
max_acc_row = df.iloc[df['acc'].idxmax()]
print("Max accuracy of ", max_acc_row[3], " obained with w1=", max_acc_row[0],
      " w2=", max_acc_row[1], " and w3=", max_acc_row[2])         

print(tabulate(df, headers='keys', tablefmt='psql'))