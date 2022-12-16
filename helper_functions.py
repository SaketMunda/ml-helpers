### We have created a bunch of helpful functions during Machine learning experiments
### Storing them here so they're easily accessible and no need to write again and again.

import tensorflow as tf
import zipfile
import os

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
 
