import train
import os, json, gc
import numpy as np
import utils.utils as utils

# main_path = os.getcwd()
main_path = '/TFG'
cfg_files_path = main_path + '/framework/train/toTrain'
outcome = main_path + '/framework/train/trainOutcomes'
_, folders, files = next(os.walk(cfg_files_path))
files.remove('.gitkeep')

############################################################
#  AUXILIAR FUNCION TO MANAGE KFOLD OR INDEPENDENT TRAINS  #
############################################################
def train_all_files(input_directory_path:str, outcome_directory_path:str, file:str):
    # Clean garbage & load configuration file
    gc.collect()
    outcome_path = utils.create_folder(outcome_directory_path)
    cfg = utils.loadCfgJson(input_directory_path + '/' + file)

    # Train phase
    model, test_loss, test_accuracy, prediction, history_callback = train.train_main(cfg, outcome_path)

    # Generate report
    utils.save_model_and_weights(outcome_path, model)
    utils.create_confusion_matrix(prediction, outcome_path, cfg["movements"])
    utils.create_outcome_file(outcome_path, model, test_loss, test_accuracy, history_callback, cfg["comments"])
    utils.create_config_output_file(outcome_path, cfg)

##########
#  Main  #
##########
# K-Fold training directories. IMPORTANT DISCLAIMER:
# Configuration files in each folder can only differ in train, test & validation subjects.
for folder in folders:
    kFold_input = cfg_files_path + '/' + folder
    kFold_outcome = utils.create_folder(outcome, folder)
    _, folders, kFoldFiles = next(os.walk(kFold_input))
    for file in kFoldFiles:
        train_all_files(kFold_input, kFold_outcome, file)
    utils.build_average_confusion_matrix(kFold_outcome)
    
# Independent training configurations.
for file in files:
   train_all_files(cfg_files_path, outcome, file)