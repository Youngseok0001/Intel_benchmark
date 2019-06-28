import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import sys
from toolz import compose, partial

sys.path.append('..')
from config import *

#################################

def get_mean_se(xs):

    mean = np.mean(xs)
    se = np.std(xs)/np.sqrt(len(xs))
    
    return mean, se
    

def get_ce(gt, logit, weighted = True):
    
    epsilon = tf.constant(value=1e-10) 
    num_classes = config.num_classes
    
    gt = tf.reshape( tf.cast(gt, tf.int32), (-1,))
    logit = tf.reshape(logit, (-1,config.num_classes))   
    pred = tf.nn.softmax(logit, axis = 1)

    if weighted:
        
        gt_one_hot = tf.one_hot(gt, depth = config.num_classes)

        class_freq = tf.reduce_sum(gt_one_hot, axis = 0)
        total_freq = tf.reduce_sum(class_freq)
        weight = total_freq / (class_freq + epsilon)
                
        weighted_gt = weight * gt_one_hot
        
        _ce = weighted_gt * ((-tf.math.log(pred + epsilon))**0.3) 
        
    else: 
        _ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = gt,
                                                             logits = logit)
    
    return tf.reduce_mean(_ce) + epsilon

def get_acc(gt, pred):
    
    gt = tf.reshape( tf.cast(gt, tf.int32), (-1,))
    pred = tf.reshape(pred, (-1,config.num_classes))
    pred = tf.arg_max(pred, dimension = 1, output_type=tf.int32)
    
    acc = tf.reduce_mean(tf.cast(tf.equal(gt, pred), tf.float32)) 
    
    return acc * 100


def get_iou(gt, pred):
    
    gt = tf.reshape(gt, (-1,))
    pred = tf.argmax(tf.reshape(pred, (-1,config.num_classes)), axis = 1)
    
    mean_iou = tf.metrics.mean_iou(
        labels = tf.cast(gt, tf.float32),
        predictions = tf.cast(pred, tf.float32),
        num_classes = config.num_classes)

    return mean_iou 


def get_dice_coef(y_true, y_pred, smooth= 0.0000001):
    
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def get_label_wise_dice_loss(y_true, y_pred):
    return 1 - get_dice_coef(y_true, y_pred)


def get_label_wise_dice_loss(y_true, y_pred):
    return 1 - get_dice_coef(y_true, y_pred)

def get_label_wise_dice_coef(y_true, y_pred, max_label):
    y_true = tf.one_hot(y_true, depth = max_label)
    return [get_dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index]) 
                 for index
                 in range(max_label)]  
    
    
def get_dice_loss(*args, **kwargs):
    return 1- tf.reduce_sum(get_label_wise_dice_coef(*args, **kwargs))/args[-1]


def get_exp_dice_loss(*args, **kwargs):
    
    return tf.reduce_sum(
            (-tf.math.log(
                get_label_wise_dice_coef(*args, **kwargs))) ** 0.3) / args[-1]
    
if __name__ == '__main__':

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
    
    # bad case_1
    #img = np.random.normal(10, size = (6,64,64,64,4))
    #gt = tf.cast(tf.constant(np.random.randint(0,4, size = (6,64,64,64))), tf.int32)
    #pred = tf.nn.softmax(
    #    tf.constant(
    #        np.random.normal(10, size = (6,64,64,64,4)), dtype = tf.float32), axis = -1)
    
    
    # bad case_2
    gt = tf.cast(tf.constant(np.random.randint(0,4, size = (1,64,64,64))), tf.int32)
    pred = tf.constant(np.zeros(shape = (1,64,64,64,4)), dtype = tf.float32)


    #good case
    #gt = tf.cast(tf.constant(np.random.randint(0,4, size = (1,64,64,64))), tf.int32)
    #pred = tf.cast(tf.one_hot(gt, config.num_classes), dtype = tf.float32)
    
    ce = get_ce(gt, pred, weighted = False)
    acc = get_acc(gt, pred)
    iou, iou_op = get_iou(gt, pred)
    dice_loss = get_dice_loss(gt, pred, 4)
    exp_dice_loss = get_exp_dice_loss(gt, pred, 4)
    label_wise_dice = get_label_wise_dice_coef(gt, pred, 4)

    to_call = [iou_op, acc, ce, dice_loss, exp_dice_loss, label_wise_dice]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        [_, _acc, _ce, _dice_loss, _exp_dice_loss,_label_wise_dice] = sess.run(to_call)
        print("ce                   :",_ce)
        print("acc                  :",_acc) 
        print("dice loss            :",_dice_loss)
        print("exp dice loss        :",_exp_dice_loss)        
        print("dice coef label wise :",_label_wise_dice)
        print("iou                  :",sess.run(iou) * 100)
