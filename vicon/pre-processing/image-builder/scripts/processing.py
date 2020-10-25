from toolz import interleave
import os
import pandas as pd
import json
import shutil 

# Porject main directory path
main_path = os.getcwd()

# Loads configuration file
cfg_filename = main_path + '/vicon/pre-processing/config.json'
with open(cfg_filename) as f:
  full_config = json.load(f)
  dt_cfg = full_config["1"]["in-dt"]
  cfg = full_config["1"]["im-bu"]

# Path to _output file
input_path = main_path + '/vicon/pre-processing/interleaved-dataframe/_output/'
zippedinput_path = main_path + '/vicon/pre-processing/interleaved-dataframe/_output-zipped/'
output_path = main_path + '/vicon/pre-processing/image-builder/_output/'

##################################################################
# Methods                                                        #
##################################################################
def build_output_directory():
  #Clean output directory
  shutil.rmtree(output_path+'position/')
  shutil.rmtree(output_path+'orientation/')

  #Create postion directories:
  os.mkdir(output_path+'position/')
  for movement in dt_cfg["movements"]["list"]:
    os.mkdir(output_path+'position/'+movement+'/')

  #Create orientation directories:
  os.mkdir(output_path+'orientation/')
  for movement in dt_cfg["movements"]["list"]:
    os.mkdir(output_path+'orientation/'+movement+'/')

def path_to_save(file: str, sensorType:str):
  for movement in dt_cfg["movements"]["list"]:
    if movement in file:
      return output_path + sensorType + movement + '/'

def load_files():
  _, _, files = next(os.walk(input_path))
  files.remove('config.json')
  return files

def calculate_sample_size():
  global orientation_sample_size
  global position_sample_size
  sensors_size =  len(dt_cfg["orientationSensors"]["list"])
  batch_size = cfg["images"]["batch-size"]
  orientation_sample_size = sensors_size*batch_size
  sensors_size =  len(dt_cfg["positionSensors"]["list"])
  position_sample_size = sensors_size*batch_size

def build_position_images(file: str):
  print("Building images for posititon file:" + file)
  df = pd.read_fwf(input_path + file, header=None)
  df = df[0].str.split(',', expand=True)
  # Calculate image sizing
  df_len = len(df)
  images_number = df_len//position_sample_size
  # Build individual images
  for i in range(images_number):
    final_df = df.iloc[(i*position_sample_size)+1:position_sample_size*(i+1)+1]
    del final_df[0]
    final_df.columns = ['3D vecotr', '0', '1', '2']
    final_df[['0', '1', '2']] = final_df[['0', '1', '2']].astype(float)
    path = path_to_save(file, 'position/')
    image_name = path + file[:-4] +'-' + str(i+1) + '.csv'
    final_df.to_csv(image_name)

def build_orientation_images(file:str):
  print("Building images for orientation file: " + file)
  df = pd.read_fwf(input_path + file, header=None)
  df = df[0].str.split(',', expand=True)
  # Calculate image sizing
  df_len = len(df)
  images_number = df_len//orientation_sample_size
  # Build individual images
  for i in range(images_number):
    final_df = df.iloc[(i*orientation_sample_size)+1:orientation_sample_size*(i+1)+1]
    del final_df[0]
    final_df.columns = ['quat', '0', '1', '2', '3']
    final_df[['0', '1', '2', '3']] = final_df[['0', '1', '2', '3']].astype(dtype=float, errors="ignore")
    path = path_to_save(file, 'orientation/')
    image_name= path + file[:-4] +'-' + str(i+1) + '.csv'
    final_df.to_csv(image_name )

def build_images(files : str):
  if 'Position' in files and cfg["positionSensors"]["enabled"]:
    build_position_images(file)
  if 'Orientation' in files and cfg["orientationSensors"]["enabled"]:
    build_orientation_images(file)

#########################
# Main                  #
#########################
if cfg["enabled"]:
  if cfg["interleavedOutput"]["useOlder"]["enabled"]:
    # TO DO unzip old file in /tmp to use
    # input_path = this tmp folder
    # cfg = stored_CFG
    print('Using zipped output')
  else:
    print('Using non zipped output')
    files = load_files()
  
  # Prepare output directory
  build_output_directory()

  # Calculate sample size
  calculate_sample_size()

  # Build images
  for file in files:
      build_images(file)

  print("\nIMAGE BUILDING FINISHED")
else:
  print("\nIMAGE BUILDING DISABLED")
  