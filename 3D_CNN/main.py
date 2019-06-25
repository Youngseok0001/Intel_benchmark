"""
enviroment: 
    * Ubuntu 16.04.6
    * python 3.7
    * Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz (10 cores used for preprocessing)
    * TITAN XP 12G MEMMORY(utilized entire memmory)
    
pacakges installed: 
    * tf-1.13
    * numpy
    * toolz(funcitonal programmings)
    * nibabel(data-loadding)
    * dltk(pre-processing)
    * easydict(config dictionary)
"""    
# packages
#-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.

import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

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
from network.metrics import get_acc, get_ce, get_mean_se, get_dice

#-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.

# get img and label array 
imgs_path = get_path(config.img_loc)
labels_path = get_path(config.lab_loc)

# split train & test
train_img_labels_path, test_img_labels_path = split_list(list(zip(imgs_path, labels_path)),
                                                         ratio = config.split_ratio)

# establish data_pipeline for trainset 
train_img, train_label = zip(*train_img_labels_path)
test_img, test_label = zip(*test_img_labels_path)

dataset_train = get_data_pipeline(train_img, train_label, 
                                  batch_size = config.batch_size, 
                                  prefetch = config.prefetch,
                                  cpu_n = config.cpu_n,
                                  epoch = config.epoch,
                                  is_train  = True)

dataset_test = get_data_pipeline(test_img, test_label,
                                 batch_size = config.batch_size, 
                                 prefetch = config.prefetch,
                                 cpu_n = config.cpu_n,
                                 epoch = config.epoch,
                                 is_train  = False)
                                

train_model = partial(unet3d, reuse = False, training = True)
test_model = partial(unet3d, reuse = True, training = False)

def run(model_train_fn, model_test_fn, input_train_fn, input_test_fn):
    
    train_generator = input_train_fn.make_one_shot_iterator()
    test_generator = input_test_fn.make_one_shot_iterator()

    
    def train(model_train_fn, train_generator):
        
        optimizer = config.optimizer(learning_rate = config.lr) 
        
        img, label = train_generator.get_next()

        logit = model_train_fn(img)
        
        loss = get_ce(logit, label) 
        acc = get_acc(logit, label)
        
        train_op = optimizer.minimize(loss)

        return loss, acc, train_op, img, label

    

    def test(model_test_fn, test_generator):
        
        img, label = test_generator.get_next()
                
        logit = model_test_fn(img)
        
        acc = get_acc(logit, label)
        
        return acc, img, label
    
    train_loss, train_acc, train_op, train_img, train_label = train(model_train_fn, train_generator)
    print("\n")
    test_acc, test_img, test_label = test(model_test_fn, test_generator)

 
    with tf.Session() as sess:
        
        # initialize list of variables 
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # start train 
            
        times = []

        for e in range(0,config.epoch):

            start_time = time.time()

            for i in range(len(train_img_labels_path)//config.batch_size):            
                _, loss_realised, acc_realised, img_realised, gt_realised = sess.run([train_op,
                                                                                  train_loss,
                                                                                  train_acc,
                                                                                  train_img,
                                                                                  train_label])
                if i % 10 == 0:
                    print("epoch: ",e)
                    print("train loss at {}th iteration is {}".format(i, loss_realised))
                    print("train acc at {}th iteration is {}".format(i,acc_realised))
                    print("\n")
                    
                    print("predicted segmentation")
                    
                    slice_vis = toolz.compose(vis, get_slice)
                    map(slice_vis,[img,label])
                    plt.show()
                    

            end_time = time.time()
            time_diff = end_time - start_time 
            times.append(time_diff)

        mean, se = get_mean_se(times)
        print("forward+backward time: mean:{mean}[{lb}, {ub}]/epoch".format(
                                                                   mean = mean,
                                                                   lb   = mean - 2*se,
                                                                   ub   = mean + 2*se))            
        times = []

        for e in range(0,config.epoch):

            start_time = time.time()

            for i in range(len(test_img_labels_path)//config.batch_size):            
                acc_realised, img_realised, gt_realised,  = sess.run([test_acc, test_img, test_label])
                
                if i % 10 == 0:
                    print("epoch: ",e)
                    print("test acc at {}th iteration is {}".format(i,acc_realised))
                    print("\n")

            end_time = time.time()
            time_diff = end_time - start_time 
            times.append(time_diff)

        mean, se = get_mean_se(times)
        print("forward time: mean:{mean}[{lb}, {ub}]/epoch".format(
                                                                   mean = mean,
                                                                   lb   = mean - 2*se,
                                                                   ub   = mean + 2*se))            
            

            
run(train_model, test_model,dataset_train, dataset_test)