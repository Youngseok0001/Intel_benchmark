
from glob import glob
from random import shuffle
from functools import reduce, partial
from random import randint, shuffle 
from toolz import compose

from matplotlib import pyplot as plt
from IPython import display

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
from log.utils import _print, vis_slice 

from network.models import unet3d
from network.metrics import *
from network.save_load_utils import save

#-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.

num_classes       = 4
weights_load_path = "./weights/model_3/" 
model_save_path   = "./saved_model" 

g_1 = tf.Graph()
with g_1.as_default():
    
    # dummy_input
    
    features = tf.placeholder(tf.float32, [None , None, None, None, 4], 'features')
    
    # get models     
    model = partial(unet3d, reuse = False)
    
    # run it  
    logits = model(features) 
    pred = tf.math.softmax(logits, axis = -1)
    
    saver = tf.train.Saver()
#################################################
        
    with tf.Session() as sess:

        # initialize list of variables 
        ckpt = tf.train.get_checkpoint_state(weights_load_path)
        saver.restore(sess, ckpt.model_checkpoint_path)


        inputs_dict  = {"features_data": features}
        outputs_dict = {"pred"       : pred}

        tf.saved_model.simple_save(sess, model_save_path, inputs_dict, outputs_dict)


