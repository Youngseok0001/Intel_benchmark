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
    

def get_ce(gt, logit, weighted = True):
    
    epsilon = tf.constant(value=1e-10) 
    num_classes = config.num_classes
    
    gt = tf.reshape( tf.cast(gt, tf.int32), (-1,))
    logit = tf.reshape(logit, (-1,config.num_classes))   

    if weighted:
        
        gt_one_hot = tf.one_hot(gt, depth = config.num_classes)

        class_freq = tf.reduce_sum(gt_one_hot, axis = 0)
        total_freq = tf.reduce_sum(class_freq)
        weight = total_freq / class_freq
                
        weighted_gt = weight * gt_one_hot
        
        _ce = weighted_gt * ((-tf.math.log(tf.math.softmax(logit) + epsilon ))**0.3) 
        
    else: 
        _ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = gt,
                                                             logits = logit)
    print("get_ce done")
    return tf.reduce_mean(_ce)  

def get_tversky_loss(gt, logit):

    alpha = 0.5
    beta  = 0.5
    num_classes = config.num_classes

    y_pred = tf.math.softmax(logit)
    y_true = tf.one_hot(gt,depth = num_classes)
    
    p0 = y_pred      
    p1 = 1 - y_pred 
    g0 = y_true
    g1 = 1 - y_true
    
    num = tf.reduce_sum(p0 * g0, (0,1,2,3))
    
    den = num + \
          alpha * tf.reduce_sum(p0 * g1,(0,1,2,3)) + \
          beta * tf.reduce_sum(p1 * g0,(0,1,2,3)) 
    
    T = tf.reduce_sum(num/den) 
    
    Ncl = tf.cast(y_true.get_shape()[-1], tf.float32)
    print("get_tversky_loss done")
    return Ncl - T

def get_acc(gt, logit):
    
    gt = tf.reshape( tf.cast(gt, tf.int32), (-1,))
    logit = tf.reshape(logit, (-1,config.num_classes))
    logit = tf.arg_max(logit, dimension = 1, output_type=tf.int32)
    
    acc = tf.reduce_mean(tf.cast(tf.equal(gt, logit), tf.float32)) 
    print("get_acc done")
    return acc * 100


def get_iou(gt, logit):
    
    gt = tf.reshape(tf.cast(gt, tf.float32), (-1,))
    pred = tf.cast(tf.argmax(tf.reshape(logit, (-1,config.num_classes)), axis = 1), tf.float32)

    mean_iou = tf.metrics.mean_iou(
        labels = gt,
        predictions = pred,
        num_classes = config.num_classes)
    print("get_iou done")
    return mean_iou 

# def get_dice(gt, logit):

#     smooth = 0.0000001
    
#     logit = tf.reshape(logit, (-1,config.num_classes))
#     logit = tf.nn.softmax(logit, axis = 1)
    
#     gt = tf.cast(tf.reshape(gt, (-1,)), tf.int32)
#     gt = tf.one_hot(gt, config.num_classes, dtype=tf.float32)
    
#     weights = 1.0 / (tf.reduce_sum(gt, axis= 0) ** 2)
    
#     numerator = tf.reduce_sum(gt * logit, axis = 0)
#     numerator = tf.reduce_sum(weights * numerator)

#     denominator = tf.reduce_sum(gt + logit, axis = 0)
#     denominator = tf.reduce_sum(weights * denominator)
    
#     loss = 1.0 - (2.0*(numerator + smooth)/(denominator + smooth))
    
#     return loss

if __name__ == '__main__':

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
    
    ## bad case
    #gt = tf.constant(np.random.randint(0,4, size = (1,64,64,64)))
    #slogit = tf.constant(np.random.normal(10, size = (1,64,64,64,4)), dtype = tf.float32)

    #good case
    gt = tf.constant(np.random.randint(0,4, size = (1,64,64,64)))
    logit = tf.cast(tf.one_hot(gt, config.num_classes), dtype = tf.float32)

    ce = get_ce(gt, logit, weighted = True)
    acc = get_acc(gt, logit)
    iou, iou_op = get_iou(gt, logit)
    tversky_loss = get_tversky_loss(gt, logit)

    to_call = [iou_op, tversky_loss, ce, acc]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        [_, _tversky_loss, _ce, _acc] = sess.run(to_call)
        print("ce:",_ce)
        print("acc:",_acc) 
        print("dice:",_tversky_loss)
        print("iou:",sess.run(iou) * 100)