### We have created a bunch of helpful functions during Machine learning experiments
### Storing them here so they're easily accessible and no need to write again and again.

import tensorflow as tf
import zipfile
import os
import datetime
import matplotlib.pyplot as plt

# Create a function which can unzip a zipfile into current working directory
# since we're going to download and unzip files many times.
def unzip_data(filename):
  """
  Unzip filename into the current working directory
  
  Args:
    filename(str): a filepath to a target zip folder to be unzipped.
  """
  zip_ref = zipfile.ZipFile(filename, 'r')
  zip_ref.extractall()
  zip_ref.close()
  
  
# Walk through an image classification directory and find out how many files (images)  
# are in each subdirectory
def walk_through_dir(dir_path):
  """
  Walk through the dir_path returning its contents.
  
  Args:
    dir_path(str): target directory
  Returns:
    A print out of:
      number of subdirectories in dir_path
      number of images (files) in each subdirectories
      name of each subdirectory
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} files in '{dirpath}'")
    
## Create Tensorboard Callback function to track the experiments while training our model
def create_tensorboard_callback(dir_name, experiment_name):
  """
  Create Tensorboard Callbacks to track the experiments while training our model in form
  of log files in the passed directory.
  
  Args:
    dir_name(str): Directory name to store the log files
    experiment_name(str): Name of the experiment
  Return:
    tensorboard callback instance
  """
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir = log_dir
  )
  print(f"Saving Tensorboard log files to: {log_dir}")
  return tensorboard_callback
 
## Create a function which will plot the loss curves and accuracy curves between
## training and validation set
def plot_loss_curves(history):
  """
  It plots two figures, one contain loss curves plotting between training and
  validation set.
  Another one is the accuracy curves between training and validation set of 
  the model.
  
  Args:
    history: model fit history
  Return:
    It plots directly into the notebook
  """
  
  loss = history.history["loss"]
  val_loss = history.history["val_loss"]
  
  accuracy = history.history["accuracy"]
  val_accuracy = history.history["val_accuracy"]
  
  epochs = range(len(loss))
  
  plt.plot(epochs, loss, label='train_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.xlabel('Epochs')
  plt.legend()
  
  plt.figure()
  
  plt.plot(epochs, accuracy, label='train_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.xlabel('Epochs')
  plt.legend();
