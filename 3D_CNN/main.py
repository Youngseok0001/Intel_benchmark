"""
enviroment: 
    * Ubuntu 16.04.6
    * python 3.7
    * Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz (20 cores used for preprocessing)
    * V100 32G MEMMORY(utilized entire memmory)
    
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
from IPython import display

import numpy as np
import nibabel as nib 
import time
import tensorflow as tf 
from matplotlib import pyplot as plt
tf.reset_default_graph()

# personal modules and functions
#-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
from config import *

from dataset.brats_data_loader import get_data_pipeline 
from dataset.utils import get_path, split_list
from log.utils import _print, vis_slice 

from network.models import unet3d
from network.metrics import *
from network.save_load_utils import save

#-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.

# get img and label array 
imgs_path = get_path(config.img_loc)
labels_path = get_path(config.lab_loc)

# split train & test
train_img_labels_path, test_img_labels_path = split_list(list(zip(imgs_path, labels_path)),
                                                         ratio = config.split_ratio)


# establish data_pipeline for trainset 
train_img, train_label = zip(*train_img_labels_path)
test_img, test_label   = zip(*test_img_labels_path)

# train data pipeline 
dataset_train = get_data_pipeline(train_img, train_label, 
                                  batch_size = config.batch_size, 
                                  prefetch   = config.prefetch,
                                  cpu_n      = config.cpu_n,
                                  epoch      = config.epoch,
                                  is_train   = True)


# test data pipeline
dataset_test = get_data_pipeline(test_img, test_label,
                                 batch_size  = config.batch_size, 
                                 prefetch    = config.prefetch,
                                 cpu_n       = config.cpu_n,
                                 epoch       = config.epoch,
                                 is_train    = False)
                                

# get models     
print("initializing models ... \n")
train_model = partial(unet3d, reuse = False)
test_model  = partial(unet3d, reuse = True)

def run(model_train_fn, model_test_fn, input_train_fn, input_test_fn):
    
    train_generator = input_train_fn.make_one_shot_iterator()
    test_generator  = input_test_fn.make_one_shot_iterator()
    
    def train_fn(model_train_fn, train_generator):  
        
        img, label = train_generator.get_next()
        
        logit = model_train_fn(img)
        pred  = tf.nn.softmax(logit, axis = config.depth)

        ce_loss         = get_ce(label, logit, weighted = False)
        acc             = get_acc(label, pred)
        iou, iou_op     = get_iou(label, pred)
        dice_loss       = get_dice_loss(label, pred, config.depth) 
        exp_dice_loss   = get_exp_dice_loss(label, pred, config.depth) 
        label_wise_dice = get_label_wise_dice_coef(label, pred, config.depth)
        overall_loss    = (config.loss_weights[0] * ce_loss) + \
                          (config.loss_weights[1] * exp_dice_loss) + \
                          (config.loss_weights[2] * dice_loss)
        
        optimizer = config.optimizer()
        train_op  = optimizer.minimize(overall_loss)

        values_to_load = [ce_loss, dice_loss, exp_dice_loss, overall_loss, # lossees
                          acc, iou, label_wise_dice, # evalutation metrics
                          iou_op, train_op] # update ops         

        return [img, label, pred], values_to_load

    def test_fn(model_test_fn, test_generator):
        
        img, label = test_generator.get_next()
        
        logit = model_test_fn(img)
        pred  = tf.nn.softmax(logit, axis = config.depth)

        acc             = get_acc(label, pred)
        iou, iou_op     = get_iou(label, pred)
        label_wise_dice = get_label_wise_dice_coef(label, pred, config.depth)

        values_to_load = [acc, iou, label_wise_dice, #evaluation metircs
                          iou_op] # update ops
        
        return [img, label, pred], values_to_load
    
    print("\n")
    im_lab_pred_train, values_to_load_train = train_fn(model_train_fn, train_generator)
    print("\n")
    im_lab_pred_test, values_to_load_test  = test_fn(model_test_fn, test_generator)

    with tf.Session() as sess:
        
        # initialize list of variables 
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
                
        saver = tf.train.Saver(max_to_keep = 3)
                

        # list to store test/train time for every iteration.
        times_test  = []
        times_train = []
        counter = 1
        
        for e in range(1, config.epoch + 1):
            
            """
            TRAIN 
            """
            
            txt_file_path = config.text_log_path + "log_{}.txt".format(config.model_name)
            f_txt = open(txt_file_path,"a")
            print_log = partial(_print, f = f_txt)

            #########################################################################################################

            start_time = time.time()

            for i in range(len(train_img_labels_path)//config.batch_size): 
                
                _im, _lab, _pred, _ce_loss, _dice_loss, _exp_dice_loss, _overall_loss, _acc, _iou, _label_wise_dice, *_ = sess.run(im_lab_pred_train + values_to_load_train)  
                
                if i % 1 == 0:
                    
                    print_log("epoch:{} iteration:{}".format(e,i))
                    print_log("    LOSS            = {}".format(_overall_loss))
                    print_log("    DICE Loss       = {}".format(_dice_loss))
                    print_log("    EXP Loss        = {}".format(_exp_dice_loss))                    
                    print_log("    CE Loss         = {}".format(_ce_loss))
                    print_log("    ACC             = {}".format(_acc))
                    print_log("    IOU             = {}".format(_iou))
                    print_log("    DICE label wise = {}".format(_label_wise_dice))
                    print_log("\n\n")                    
                    
                if (i % 100 == 0):
                    
                    vis_slice(_im, _lab, _pred, 32, config.visual_log_path +"train_{}epoch_{}iter.png".format(e,i))
        
            end_time = time.time()
            time_diff = end_time - start_time 
            times_train.append(time_diff)
            counter = counter + 1
                  
            """
            TEST 
            """
            #########################################################################################################
            # list to store evaluation per test batch
            acc_stack             = []
            iou_stack             = []
            label_wise_dice_stack = []
                
            start_time = time.time()
            
            for i in range(len(test_img_labels_path)//config.batch_size):    
                
                _acc, _iou, _label_wise_dice, *_ = sess.run(values_to_load_test)   
                                
                acc_stack.append(_acc)   
                iou_stack.append(_iou)
                label_wise_dice_stack.append(_label_wise_dice)

            
            print_log("epoch:{}".format(e))
            print_log("    test ACC  = {}".format(np.mean(acc_stack)))
            print_log("    test IOU  = {}".format(np.mean(iou_stack)))
            print_log("    test DICE label wise = {}".format(np.mean(label_wise_dice_stack, axis = 0)))
            print_log("\n")                                             

            end_time = time.time()
            time_diff = end_time - start_time 
                        
            times_test.append(time_diff)
            
            print_log("Saving model ....")
            save(sess, saver, config.model_save_path, config.model_name, counter)
            
            f_txt.close() 
            
        print("the training has ended \n")
        
        mean, se = get_mean_se(times_train)
        print("forward+backward time: mean:{mean}[{lb}, {ub}]/epoch".format(
                                                                mean = mean,
                                                                lb   = mean - 2*se,
                                                                ub   = mean + 2*se))    
                
        mean, se = get_mean_se(times_test)
        print("forward time: mean:{mean}[{lb}, {ub}]/epoch".format(
                                                                mean = mean,
                                                                lb   = mean - 2*se,
                                                                ub   = mean + 2*se))    
           
run(train_model, test_model,dataset_train, dataset_test)
