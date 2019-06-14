# packages
#-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.

import os

from glob import glob
from random import shuffle
from functools import reduce, partial
from random import randint, shuffle 
from toolz import compose

import numpy as np
import nibabel as nib 
import sys 
import tensorflow as tf 

from .preprocess_utils import _omit_labeless_slices, _flip, _resize, _gaussian_noise, _normalise
sys.path.append('..')
from config import *
# functions 
#-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
get_path = lambda data_path : glob(os.path.join(data_path , "*"))

get_data = lambda img_file : nib.load(img_file.decode()).get_fdata().astype(np.float32)

get_img_label = lambda img, label : [get_data(img), get_data(label)]

def split_list(imgs_labels_path, ratio = 0.8):
    
    shuffle(imgs_labels_path)
    imgs_labels_path_train = imgs_labels_path[:int(len(imgs_labels_path) * ratio)]
    imgs_labels_path_test  = list(set(imgs_labels_path) - set(imgs_labels_path_train))
    
    return imgs_labels_path_train, imgs_labels_path_test

def preprocess(img, label):
    
    # import config from config dict
    means, stds = config.means, config.stds
    sigma_gaussian    = config.sigma_gaussian
    resize_constant_1 = config.resize_constant_1 
    resize_constant_2 = config.resize_constant_2 
    condition         = config.condition
    #alpha_distort, sigma_distort = config.alpha_distort, config.sigma_distort
    
    # feed in config  info to preprocessing functions
    
    my_flip       = partial(_flip, axis = randint(0,1)) 
    my_omit_slice = partial(_omit_labeless_slices, condition = condition)
    my_resize_1   = partial(_resize, resize_constant = resize_constant_1)
    my_resize_2   = partial(_resize, resize_constant = resize_constant_2)
    my_gaussian   = partial(_gaussian_noise, sigma   = sigma_gaussian)
    my_normalise  = partial(_normalise, means = means, stds = stds)
    #my_distort = partial(_distort, alpha = alpha, sigma = sigma) # this function is expensive, watch out.
    
    # wrap functions 
    composed = compose(my_normalise, 
                       my_gaussian, 
                       my_resize_2, 
                       my_resize_1, 
                       my_omit_slice,
                       my_flip)
    
    # do preprocessing
    img_processed, label_processed = composed( (img, label) )
    
    # hancle type 
    img_processed = img_processed.astype(np.float32)
    label_processed = label_processed.astype(np.int32)
    
    return img_processed,  label_processed

def get_data_pipeline(imgs_path, labels_path, epoch, batch_size = 1, prefetch = 4, cpu_n = 10):

    return tf.data.Dataset.from_tensor_slices((list(imgs_path), list(labels_path))).\
        repeat(epoch).\
        map(lambda img_path, label_path: 
            tuple(tf.py_func(get_img_label, [img_path, label_path], [tf.float32, tf.float32])),
            num_parallel_calls = cpu_n).\
        map(lambda img, label: 
            tuple(tf.py_func(preprocess, [img, label], [tf.float32, tf.int32])),
            num_parallel_calls = cpu_n).\
        batch(batch_size).\
        prefetch(prefetch)



if __name__ == '__main__':
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="3"

    tf.enable_eager_execution()
    img_loc = "/home/jacob/Intel/3D_CNN/dataset/Task01_BrainTumour/imagesTr/"    
    lab_loc = "/home/jacob/Intel/3D_CNN/dataset/Task01_BrainTumour/labelsTr/"

    # get img and label array 
    imgs_path = get_path(img_loc)
    labels_path = get_path(lab_loc)

    # split train & test
    train_img_labels_path, test_img_labels_path = split_list(list(zip(imgs_path, labels_path)), ratio = 0.8)

    # establish data_pipeline for trainset 
    train_img_path, train_label_path = zip(*train_img_labels_path)
    test_img_path, test_label_path = zip(*test_img_labels_path)

    dataset_train = get_data_pipeline(train_img_path, train_label_path)
    dataset_test = get_data_pipeline(test_img_path, test_label_path)
    
    train_generator = dataset_train.make_one_shot_iterator()
    img, lab = train_generator.get_next()
    print(img.get_shape())
    print(lab.get_shape())