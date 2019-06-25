# packages
#-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.

import os

from random import randint, shuffle 
from toolz import compose, reduce, partial

import numpy as np
import nibabel as nib 
import sys 
import tensorflow as tf 
import random

from dltk.io.augmentation import *
from dltk.io.preprocessing import * 

sys.path.append('..')
from config import *
from .utils import get_data,\
                  get_img_label,\
                  _omit_labeless_slices,\
                  _gaussian_noise,\
                  _set_size,\
                  _flip,\
                  _normalise,\
                  _random_crop_image
# functions 
#-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
def _preprocessor_numpy(img_dir, label_dir, is_train = True):
        
    # functions 
    my_omit_slice     = partial(_omit_labeless_slices,condition = config.condition)
    my_gaussian_noise = partial(_gaussian_noise, sigma = config.sigma_gaussian)
    my_crop           = partial(_random_crop_image, patch_size = config.patch_size)
    my_normalise      = partial(_normalise, means = config.means, stds = config.stds)
    my_flip           = partial(_flip, axis = random.randint(0,2))
    
    if is_train == True:
        # wrap functions
        composed = compose(my_flip,
                           my_crop,
                           my_gaussian_noise,
                           my_normalise,
                           my_omit_slice,
                           get_img_label)
    
    if is_train == False:
        # wrap functions
        composed = compose(my_gaussian_noise,
                           my_normalise,
                           get_img_label)
        
    # apply
    img_processed, label_processed = composed(((img_dir, label_dir)))
    
    # handle type 
    img_processed, label_processed = tf.cast(img_processed, tf.float32), tf.cast(label_processed, tf.int32)    
    
    return img_processed, label_processed
    
    
    
def _preprocessor_tensor(img, label, is_train = True):
        
    # functions
    my_set_size = _set_size
    
    # wrap 
    composed = compose(my_set_size)
    
    # apply
    img_processed, label_processed = composed((img,label))
    
    # handle type 
    img_processed, label_processed = tf.cast(img_processed, tf.float32), tf.cast(label_processed, tf.int32)
    
    return img_processed,  label_processed

def get_data_pipeline(imgs_path, labels_path, epoch, batch_size = 1, prefetch = 4, cpu_n = 10, is_train = True):
    
    preprocessor_numpy  = partial(_preprocessor_numpy, is_train = is_train)
    preprocessor_tensor = partial(_preprocessor_tensor, is_train = is_train)
    
    return tf.data.Dataset.from_tensor_slices((list(imgs_path), list(labels_path))).\
        repeat(epoch).\
        map(lambda img_path, label_path: 
            tuple(tf.py_func(preprocessor_numpy, [img_path, label_path], [tf.float32, tf.int32])), 
                num_parallel_calls = cpu_n).\
        map(preprocessor_tensor, num_parallel_calls = cpu_n).\
        batch(batch_size).\
        prefetch(prefetch)


if __name__ == '__main__':
    
    import os
    import tensorflow as tf
    os.environ["CUDA_VISIBLE_DEVICES"]="3"
    tf.enable_eager_execution()

    import sys 
    sys.path.append('..')
    from dataset.utils import get_path
    
    img_loc = "/home/jacob/Intel/3D_CNN/dataset/Task01_BrainTumour/imagesTr/"    
    lab_loc = "/home/jacob/Intel/3D_CNN/dataset/Task01_BrainTumour/labelsTr/"

    # get img and label array 
    img_path = get_path(img_loc)
    label_path = get_path(lab_loc)

    dataset = get_data_pipeline(img_path, label_path, 
                                  batch_size = config.batch_size, 
                                  prefetch   = config.prefetch,
                                  cpu_n      = config.cpu_n,
                                  epoch      = config.epoch,
                                  is_train   = True)
    
    generator = dataset.make_one_shot_iterator()
    img, lab = generator.get_next()
    
    
    
    
    