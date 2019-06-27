"""
enviroment: 
    * Ubuntu 16.04.6
    * python 3.7
    * Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz (20 cores used for preprocessing)
    * V100 32G MEMMORY(utilized entire memmory)
    
packages installed: 
    * tf-1.13
    * numpy
    * toolz(funcitonal programmings)
    * nibabel(data-loading)
    * dltk(pre-processing)
    * easydict(config dictionary)
"""    
# packages
#-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from glob import glob
from random import shuffle
from functools import reduce, partial
from random import randint, shuffle 
from toolz import compose

import numpy as np
import nibabel as nib 
import time 

import tensorflow as tf 

tf.reset_default_graph()
# personal modules and functions
#-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
from config import *

from dataset.brats_data_loader import get_data_pipeline 
from dataset.utils import get_path, split_list

from network.models import unet3d
from network.metrics import *

#-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.

# get img and label array 
imgs_path = get_path(config.img_loc)
labels_path = get_path(config.lab_loc)

dataset_train = get_data_pipeline(imgs_path, labels_path, 
                                  batch_size = config.batch_size, 
                                  prefetch = config.prefetch,
                                  cpu_n = config.cpu_n,
                                  epoch = config.epoch,
                                  is_train  = True)


train_generator = dataset_train.make_one_shot_iterator()
                                
model = partial(unet3d, reuse = False, training = True)

img, label = train_generator.get_next()

logit = model(img)
pred = tf.nn.softmax(logit, axis = 4)

ce_loss = get_ce(label, logit, weighted = False)
acc = get_acc(label, pred)
iou, iou_op = get_iou(label, pred)
dice_loss = get_dice_loss(label, pred, 4)
label_wise_dice = get_label_wise_dice_coef(label, pred, 4)

loss = (0.2 * ce_loss) + (0.8 * dice_loss)

optimizer = config.optimizer(learning_rate = config.lr) 
train_op = optimizer.minimize(loss)

variables_to_load = [ce_loss, 
                     acc,  
                     dice_loss,
                     label_wise_dice, 
                     loss, 
                     train_op,
                     iou_op]


with tf.Session() as sess:
        
    # initialize list of variables 
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for e in range(0,config.epoch):

        for i in range(len(imgs_path)//config.batch_size):            
            _ce_loss, _acc, _dice_loss, _label_wise_dice, _loss, _, _ = sess.run(variables_to_load)                
            
            if i % 1 == 0:
                print("epoch:{} iteration:{}".format(e,i))
                print("LOSS            =", _loss)
                print("DICE Loss       =", _dice_loss)
                print("CE Loss         =", _ce_loss)
                print("ACC             =", _acc)
                print("IOU             =", sess.run(iou))
                print("DICE label wise =", _label_wise_dice)
                print("\n")