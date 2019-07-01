from dltk.io.augmentation import *
from dltk.io.preprocessing import * 

import numpy as np
import tensorflow as tf
import nibabel as nib 
from random import shuffle
import os 
from glob import glob
from functools import reduce, partial

import sys
sys.path.append('..')
from config import *

"""
FUNCTIONS FOR NUMPY ARRAY 
"""
get_path = lambda data_path : glob(os.path.join(data_path , "*"))

get_data = lambda img_file : nib.load(img_file.decode()).get_fdata()

def get_img_label(img_label_dir):
    return (get_data(img_label_dir[0]), get_data(img_label_dir[1]))

def split_list(imgs_labels_path, ratio = 0.8):
    
    shuffle(imgs_labels_path)
    imgs_labels_path_train = imgs_labels_path[:int(len(imgs_labels_path) * ratio)]
    imgs_labels_path_test  = list(set(imgs_labels_path) - set(imgs_labels_path_train))
    
    return imgs_labels_path_train, imgs_labels_path_test


def _mean_std_shift(img_label, mean_shift_range, std_shift_range):
    mean_shift = np.random.uniform(*mean_shift_range)
    std_shift = np.random.uniform(*std_shift_range)
    return  (img_label[0] + mean_shift) * std_shift, img_label[1]


def _elastic_transform(img_label, alpha=[10, 2e6, 2e6], sigma=[1, 25, 25]):
    
    img, label = img_label[0], np.expand_dims(img_label[1], -1)
    img_label = np.concatenate((img,label), axis = -1)
    
    elastic_img_label = elastic_transform(img_label, alpha = alpha, sigma = sigma)
    elastic_img, elastic_label = elastic_img_label[:,:,:,:-1], elastic_img_label[:,:,:,-1] 
    
    return elastic_img, elastic_label
    
    
def _gaussian_noise(img_label, sigma):
    return (add_gaussian_noise(img_label[0], sigma),
            img_label[1])     

def _random_crop_image(img_label, patch_size):
	
    img, label = img_label[0], np.expand_dims(img_label[1], -1)
    
    img_label = np.concatenate((img,label), axis = -1)

    img_dim = np.array(np.shape(img_label)[:-1]) #[H, W, D]
        
    patch_size = np.array(patch_size)
    
    # range of coord values that a center of patch can take 
    center_range = zip((img_dim//2)  - (img_dim - patch_size)//2,
                       (img_dim//2)  + (img_dim - patch_size)//2)
    
    # get a possible center from center range
    center = [np.random.randint(*r) for r in center_range]
    
	#crop
    img_label_cropped = img_label[(center[0] - patch_size[0]//2) : (center[0] + patch_size[0]//2),
                                  (center[1] - patch_size[1]//2) : (center[1] + patch_size[1]//2),
                                  (center[2] - patch_size[2]//2) : (center[2] + patch_size[2]//2),
                                  :]

    img_croped = img_label_cropped[:,:,:,:-1]
    label_croped = img_label_cropped[:,:,:,-1] 
    
    return img_croped, label_croped



def _flip(img_label, axis):
    
    img, label = img_label[0], np.expand_dims(img_label[1], -1)
    img_label = np.concatenate((img,label), axis = -1)

    # random flip function called from DLTK module
    fliped_img_label = flip(img_label, axis = axis)
    fliped_img, flipted_label = fliped_img_label[:,:,:,:-1], fliped_img_label[:,:,:,-1] 
    
    return fliped_img, flipted_label

def _normalise(img_label, means, stds): 
    return ((img_label[0] - means) / (np.asarray(stds)),
            img_label[1])    

#.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-       
"""
FUNCTIONS FOR TF ARRAY 
"""
def _set_size(img_label):
    
    return (tf.reshape(img_label[0], config.patch_size + [config.num_classes]),
            tf.reshape(img_label[1], config.patch_size))    
    
