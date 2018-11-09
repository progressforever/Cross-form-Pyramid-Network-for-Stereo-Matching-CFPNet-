import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

  #left_fold  = 'colored_0/'
  #right_fold = 'colored_1/'
  #disp_occ   = 'disp_occ/'

  #image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]

  #train = image[:163]
  #val   = image[163:]

  #left_train  = [filepath+left_fold+img for img in train]
  #right_train = [filepath+right_fold+img for img in train]
  #disp_train = [filepath+disp_occ+img for img in train]


  #left_val  = [filepath+left_fold+img for img in val]
  #right_val = [filepath+right_fold+img for img in val]
  #disp_val = [filepath+disp_occ+img for img in val]



  left_fold_train  = 'training/colored_0/'
  right_fold_train = 'training/colored_1/'
  disp_occ_train   = 'training/disp_occ/'

  left_fold_test = 'testing/colored_0/'
  right_fold_test = 'testing/colored_1/'
  disp_occ_test = 'testing/disp_occ/'

  image = [img for img in os.listdir(filepath+left_fold_train) if img.find('_10') > -1]

  train = image[:]
  val   = image[:]

  left_train  = [filepath+left_fold_train+img for img in train] + [filepath+left_fold_test+img for img in train]
  right_train = [filepath+right_fold_train+img for img in train] + [filepath+right_fold_test+img for img in train]
  disp_train = [filepath+disp_occ_train+img for img in train] + [filepath+disp_occ_test+img for img in train]
  #disp_train_R = [filepath+disp_R+img for img in train]

  left_val  = [filepath+left_fold_train+img for img in val]
  right_val = [filepath+right_fold_train+img for img in val]
  disp_val = [filepath+disp_occ_train+img for img in val]



  return left_train, right_train, disp_train, left_val, right_val, disp_val
