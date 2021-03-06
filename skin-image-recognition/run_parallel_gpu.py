#Make tf.slim to work with tf.data in parallel.

#working environment
#   tensorlow 1.13.1
#   python 3.5 
#   dataset
#   ISIC 2018: skin cancer images

#models
#   resnet v4, vgg 19, densenet 

#STEPS
#* create data pipeline
#* load models(Inception V4, Vgg19 and Dense121)
#* load ckpt(only weights)
#* run single gpu model
#   * get losses of each tower
#   * get grads of each tower
#   * get average of grads
#   * update grads with whatever optimizer you want

# STEP 0: LOAD PACKAGES
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import csv, os, functools, itertools, collections, six, time, re

from glob import glob
from collections import deque
from datasets.utils import get_imgdir_label, label_to_index, parser, preprocessors
from models import nets_factory


# STEP 1: SET_config
# model config
num_gpu = 2
os.environ["CUDA_VISIBLE_DEVICES"]= "1,2"
lr = 0.001 #learning_rate
batch_size = 32 # batch_size
epoch = 3 
weight_decay = 0.0 # regularizer constant 
model_name = ["densenet161","inception_v4","vgg_19"][0] # model selection 
variable_to_exclude = {
                      "densenet161":['densenet161/logits'],
                      "inception_v4":['InceptionV4/AuxLogits','InceptionV4/Logits'],
                      "vgg_19":['vgg_19/fc8']
                        }
variable_to_exclude = variable_to_exclude[model_name]
num_classes = 7 # number of classes
weights_loc = "./weights/" # weights locations 
pretrained_dir = glob(weights_loc + "pretrained/" + "*{}*.ckpt*".format(model_name))[0]
pretrained_dir = re.search(".*ckpt",pretrained_dir)[0]
#pretrained weights directory
#trained_dir =  glob(weights_loc + "trained/" + "*{}*.ckpt".format(model_name))[0]  #post trained weights save location

# data config
dataset_dir = "./datasets/ISIC_2018/" # direct to your dataset location
class_list = ["MEL","NV","BCC","AKIEC","BKL","DF","VASC"] # total of 7 classes in the dataset 
class_dict = {c:i for i,c in enumerate(class_list)} # to map str::class -> int::class for train/test
class_frequency = {c:None for c in class_list} # prevenlance of each class in the dataset. Will used to adjust loss.
ratio = 0.8
img_size = [224,224]
channel_n = 3

# 1. create datapipeline
img_lab = get_imgdir_label(dataset_dir) # get list of tuples [(img_dir_i, str::label_i) for i in range(len(dataset))]
img_lab = label_to_index(img_lab, class_dict) # get list of tuples [(img_dir_i, int::label_i) for i in range(len(dataset))]

img_lab_train = img_lab[:int(len(img_lab) * ratio)] # train set: 80% of img_lab 
img_lab_test = list(set(img_lab) - set(img_lab_train)) # test set 20% of img_lab

imgs_train, labels_train = zip(*img_lab_train) # transpose train_data 
imgs_test, labels_test = zip(*img_lab_test) # transpose test_data

def get_mean_se(xs):
  
    mean = np.mean(xs)
    se = np.std(xs)/np.sqrt(len(xs))
    
    return mean, se

def forward(model_fn,inputs):

    images = inputs[0]
    labels = tf.cast(inputs[1], tf.int32)
    logits,_ = model_fn(images)
    
    return logits, labels 

def forward_backward(model_fn,inputs):
    
    logits, labels = forward(model_fn, inputs)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    
    return tf.reduce_mean(loss), logits, labels

def start_training(sess, loss, train_op, acc_train, acc_test):

    # inits
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    times = []

    for e in range(epoch):

        start_time = time.time()

        for i in range(int(len(img_lab_train)/batch_size/num_gpu)):

            loss_realized, _ = sess.run([loss, train_op])

            if i % 20 == 0:
                print("train loss at {}th iteration is {}".format(i,loss_realized))
            
        end_time = time.time()
        time_diff = end_time - start_time 
        times.append(time_diff)
      
    mean, se = get_mean_se(times)
    print("forward+backward time: mean:{mean}[{lb}, {ub}]/epoch".format(mean = mean,
                                                       lb = mean - 2*se,
                                                       ub = mean + 2*se))

    times = []

    for e in range(epoch):
    
        acc_realized_cum = []
        start_time = time.time()
        
        for i in range(int(len(img_lab_test)/batch_size)):

            acc_realized = sess.run(acc_test)
            acc_realized_cum.append(acc_realized)
        
        end_time = time.time()
        time_diff = end_time - start_time
        times.append(time_diff)
    
    mean, se = get_mean_se(times)
    print("test_accuracy is:{}".format(np.mean(acc_realized_cum)))
    print("forward time: mean:{mean}[{lb}, {ub}]/epoch".format(mean = mean,
                                                       lb = mean - 2*se,
                                                       ub = mean + 2*se))
      
def assign_to_device(device, ps_device):

    def _assign(op):
        
        PS_OPS = [
        'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
        'MutableHashTableOfTensors', 'MutableDenseHashTable']
        
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        
        if node_def.op in PS_OPS:
            return ps_device
        else:
            return device
    return _assign    

def create_parallel_optimization(model_fn, input_fn, optimizer, controller="/cpu:0"):
    devices = get_available_gpus()
        
    tower_grads = []
    losses = []
    accs = []
    
    # Get the current variable scope so we can reuse all variables we need once we get
    # to the second iteration of the loop below
    with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
        for i, id in enumerate(devices):
            name = 'tower_{}'.format(i)
            # Use the assign_to_device function to ensure that variables are created on the
            # controller.
            with tf.device(assign_to_device(id, controller)), tf.name_scope(name):
                
                # Compute loss and gradients, but don't apply them yet
                loss, logits_train, labels_train = forward_backward(model_fn, input_fn())
                equality_train = tf.equal(labels_train, tf.cast(tf.argmax(logits_train, 1), tf.int32))
                acc_train = tf.reduce_mean(tf.cast(equality_train, tf.float32))

                with tf.name_scope("compute_gradients"):
                    # `compute_gradients` returns a list of (gradient, variable) pairs
                    grads = optimizer.compute_gradients(loss)

                    tower_grads.append(grads)
                    
                accs.append(acc_train)
                losses.append(loss)
            # After the first iteration, we want to reuse the variables.
            outer_scope.reuse_variables()
            
    # Apply the gradients on the controlling device
    with tf.name_scope("apply_gradients"), tf.device(controller):
        # Note that what we are doing here mathematically is equivalent to returning the
        # average loss over the towers and compute the gradients relative to that.
        # Unfortunately, this would place all gradient-computations on one device, which is
        # why we had to compute the gradients above per tower and need to average them here.
        
        # This function is defined below; it takes the list of (gradient, variable) lists
        # and turns it into a single (gradient, variables) list.
        gradients = average_gradients(tower_grads)
        global_step = tf.train.get_or_create_global_step()
        apply_gradient_op = optimizer.apply_gradients(gradients, global_step)
        avg_loss = tf.reduce_mean(losses)
        avg_acc = tf.reduce_mean(accs)
    return apply_gradient_op, avg_loss, avg_acc

# Source:
# https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
def get_available_gpus():
    """
        Returns a list of the identifiers of all visible GPUs.
    """
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# Source:
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L101
def average_gradients(tower_grads):
    gradvars = []
    with tf.name_scope('gradient_averaging'):
        all_grads = {}
        for grad, var in itertools.chain(*tower_grads):
            if grad is not None:
                all_grads.setdefault(var, []).append(grad)
        for var, grads in six.iteritems(all_grads):
            # Average gradients on the same device as the variables
            # to which they apply.
            with tf.device(var.device):
                if len(grads) == 1:
                    avg_grad = grads[0]
                else:
                    avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
            gradvars.append((avg_grad, var))
    return gradvars        

def parallel_training(model_train_fn, model_test_fn, input_train_fn, input_test_fn):
    
    geneartor_train = input_train_fn.make_one_shot_iterator()
    genearator_test = input_test_fn.make_one_shot_iterator()
    
    def input_train_fn():
        with tf.device(None):
        # remove any device specifications for the input data
            return geneartor_train.get_next()
    
    optimizer = tf.train.AdamOptimizer(learning_rate=1E-3)
    
    update_op, loss, acc_train = create_parallel_optimization(model_train_fn, input_train_fn, optimizer)
    
    logits_test, labels_test = forward(model_test_fn, genearator_test.get_next())
    equality_test = tf.equal(labels_test, tf.cast(tf.argmax(logits_test, 1), tf.int32))
    acc_test = tf.reduce_mean(tf.cast(equality_test, tf.float32))

    
    # list of parameters to load
    get_model_var_list = lambda : slim.get_model_variables()
    is_in_exclusion = lambda exclusions, x : [x for exclusion in exclusions if x.op.name.startswith(exclusion) ]
    is_in_model_exclusion = functools.partial(is_in_exclusion, variable_to_exclude)
    variable_to_include = list(set(get_model_var_list()) - set(filter(is_in_model_exclusion, get_model_var_list())))
    
    weights_loader = slim.assign_from_checkpoint_fn(
        pretrained_dir,
        variable_to_include,
        ignore_missing_vars=True)

    with tf.Session() as sess:
        weights_loader(sess)
        start_training(sess, loss, update_op, acc_train, acc_test)
                
            
tf.reset_default_graph()

# establish data_pipeline for trainset 
dataset_train = tf.data.Dataset.from_tensor_slices((list(imgs_train),list(labels_train))) 
dataset_train = dataset_train.\
           map(parser).\
           batch(batch_size).\
           repeat(epoch)

# establish data_pipeline for testset
dataset_test = tf.data.Dataset.from_tensor_slices((list(imgs_test),list(labels_test)))
dataset_test = dataset_test.\
            map(parser).\
            batch(batch_size).\
            repeat(epoch)

# initialize model function
network_train_fn = nets_factory.get_network_fn(
    name = model_name,
    weight_decay = weight_decay,
    num_classes = num_classes,
    is_training = True)

network_test_fn = nets_factory.get_network_fn(
    name = model_name,
    weight_decay = weight_decay,
    num_classes = num_classes,
    is_training = False,
    reuse = True)


parallel_training(network_train_fn, network_test_fn, dataset_train, dataset_test)        