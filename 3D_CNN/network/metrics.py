import tensorflow as tf
import tensorflow.keras as K
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
    

def get_ce(logit, gt):
    
    gt = tf.reshape( tf.cast(gt, tf.int32), (-1,))
    logit = tf.reshape(logit, (-1,config.num_classes))

    _ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = gt, logits = logit)
    ce = tf.reduce_mean(_ce) 
    
    return ce 

def get_acc(logit, gt):
    
    gt = tf.reshape( tf.cast(gt, tf.int32), (-1,))
    logit = tf.reshape(logit, (-1,config.num_classes))
    logit = tf.arg_max(logit, dimension = 1, output_type=tf.int32)
    
    acc = tf.reduce_mean(tf.cast(tf.equal(gt, logit), tf.float32)) 
    
    return acc 

def get_dice(logit, gt):

    smooth = 0.0000001
    
    logit = tf.reshape(logit, (-1,config.num_classes))
    logit = tf.nn.softmax(logit, axis = 1)
    
    gt = tf.cast(tf.reshape(gt, (-1,)), tf.int32)
    gt = tf.one_hot(gt, config.num_classes, dtype=tf.float32)
    
    weights = 1.0 / (tf.reduce_sum(gt, axis= 0) ** 2)
    
    numerator = tf.reduce_sum(gt * logit, axis = 0)
    numerator = tf.reduce_sum(weights * numerator)

    denominator = tf.reduce_sum(gt + logit, axis = 0)
    denominator = tf.reduce_sum(weights * denominator)
    
    loss = 1.0 - 2.0*(numerator + smooth)/(denominator + smooth)
    
    return loss

if __name__ == '__main__':

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
    tf.enable_eager_execution()
    
    logit = tf.random.normal(shape = (1,100,100,50,4))
    #gt = tf.arg_max(logit, dimension = -1)
    #logit = tf.one_hot(gt, 4, axis = -1)
    #logit = tf.clip_by_value(logit,1,4)
    gt = tf.random.uniform(minval=0, maxval=4, shape = (1,100,100,50), dtype=tf.int32)
    
    ce = get_ce(logit, gt)
    acc = get_acc(logit, gt)
    dice_1 = get_dice(logit, gt)
    
    print(ce)
    print(acc)
    print(dice_1)
    