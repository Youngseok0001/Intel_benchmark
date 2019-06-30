    
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import sys

sys.path.append('..')
from config import *



# -*- coding: utf-8 -*-
# File: custom_ops.py
###
# Code are borrowed from tensorpack modified to support 5d input for batchnorm.
# https://github.com/tensorpack/tensorpack/blob/master/tensorpack/models/batch_norm.py
###

from tensorflow.contrib.framework import add_model_variable
from tensorpack import VariableHolder

#############################################
#low level 3D-Unet implementation 
#############################################


depth = config.depth
BASE_FILTER = config.BASE_FILTER
PADDING = config.PADDING
DEEP_SUPERVISION = config.DEEP_SUPERVISION
NUM_CLASS = config.num_classes

def unet3d(inputs, reuse):
        
    with tf.variable_scope("unet_3d", reuse = reuse) as sc:
    
        filters = []
        down_list = []
        deep_supervision = None

        layer = tf.layers.conv3d(inputs=inputs, 
                       filters=BASE_FILTER,
                       kernel_size=(3,3,3),
                       strides=1,
                       padding=PADDING,
                       activation=lambda x, name=None: BN_ReLU(x),
                       name="init_conv")

        for d in range(depth):

            num_filters = BASE_FILTER * (2**d)
            filters.append(num_filters)

            layer = Unet3dBlock('down{}'.format(d), layer, (3,3,3), num_filters, 1)
            print("{}th down feature with shape:{}".format(d + 1 ,layer.shape))

            down_list.append(layer)


            if d != depth - 1:
                layer = tf.layers.conv3d(inputs=layer, 
                                        filters=num_filters*2,
                                        kernel_size=(3,3,3),
                                        strides=(2,2,2),
                                        padding=PADDING,
                                        activation=lambda x, name=None: BN_ReLU(x),
                                        name="stride2conv{}".format(d))


        print("\n")

        for d in range(depth-2, -1, -1):

            layer = UnetUpsample(d, layer, filters[d])
            layer = tf.concat([layer, down_list[d]], axis = -1)
            layer = tf.layers.conv3d(inputs=layer, 
                                    filters=filters[d],
                                    kernel_size=(3,3,3),
                                    strides=1,
                                    padding=PADDING,
                                    activation=lambda x, name=None: BN_ReLU(x),
                                    name="lo_conv0_{}".format(d))
            layer = tf.layers.conv3d(inputs=layer, 
                                    filters=filters[d],
                                    kernel_size=(1,1,1),
                                    strides=1,
                                    padding=PADDING,
                                    activation=lambda x, name=None: BN_ReLU(x),
                                    name="lo_conv1_{}".format(d))

            print("{}th up feature with shape:{}".format(d + 1 ,layer.shape))

            if DEEP_SUPERVISION:

                if d < 3 and d > 0:
                    pred = tf.layers.conv3d(inputs = layer, 
                                        filters = NUM_CLASS,
                                        kernel_size=(1,1,1),
                                        strides=1,
                                        padding=PADDING,
                                        activation=tf.identity,
                                        name="deep_super_{}".format(d))
                    if deep_supervision is None:
                        deep_supervision = pred
                    else:
                        deep_supervision = deep_supervision + pred
                    deep_supervision = Upsample3D(d, deep_supervision)

        layer = tf.layers.conv3d(layer, 
                                filters = NUM_CLASS,
                                kernel_size=(1,1,1),
                                padding=PADDING,
                                activation=tf.identity,
                                name="final")

        if DEEP_SUPERVISION:
            layer = layer + deep_supervision

        print("final", layer.shape) # [3, num_class, d, h, w]
        return layer


def Upsample3D(prefix, l, scale=2):
    l = tf.keras.layers.UpSampling3D(size=(2,2,2))(l)
    return l

def UnetUpsample(prefix, l, num_filters):
    
    l = Upsample3D('', l)
    l = tf.layers.conv3d(inputs=l, 
                        filters=num_filters,
                        kernel_size=(3,3,3),
                        strides = 1,
                        padding=PADDING,
                        activation=lambda x, name=None: BN_ReLU(x),
                        name="up_conv1_{}".format(prefix))
    return l


def InstanceNorm5d(x, epsilon=1e-5, use_affine=True, gamma_init=None, data_format='channels_last'):

    shape = x.get_shape().as_list()
    # assert len(shape) == 4, "Input of InstanceNorm has to be 4D!"
    
    axis = [1, 2, 3]
    ch = shape[4]
    new_shape = [1, 1, 1, 1, ch]

    mean, var = tf.nn.moments(x, axis, keep_dims=True)

    if not use_affine:
        return tf.divide(x - mean, tf.sqrt(var + epsilon), name='output')

    beta = tf.get_variable('beta', [ch], initializer=tf.constant_initializer())
    beta = tf.reshape(beta, new_shape)
    if gamma_init is None:
        gamma_init = tf.constant_initializer(1.0)
    gamma = tf.get_variable('gamma', [ch], initializer=gamma_init)
    gamma = tf.reshape(gamma, new_shape)
    ret = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon, name='output')

    vh = ret.variables = VariableHolder()
    if use_affine:
        vh.gamma = gamma
        vh.beta = beta
    return ret



def BN_ReLU(inputs):

    l = InstanceNorm5d(inputs)
    
    return tf.nn.relu6(l)


def Unet3dBlock(prefix, l, kernels, n_feat, s):
    
    l_in = l

    
    for i in range(2):
        l = tf.layers.conv3d(inputs=l, 
                   filters=n_feat,
                   kernel_size=kernels,
                   strides=1,
                   padding=PADDING,
                   activation=lambda x, name=None: BN_ReLU(x),
                   name="{}_conv_{}".format(prefix, i))

    return l_in + l




if __name__ == '__main__':
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="3"

    tf.enable_eager_execution()
    x = tf.random.normal(shape = (1,256,256,128,4))
    y = unet3d(x, reuse = True)
    y = unet3d(x, reuse = False)


        

    
#############################################
#KERAS 3D-Unet
#############################################

"""
KERAS MODEL IS STILL UNDER DEVEPMENT
YOU CAN IGNORE IT FOR NOW
"""

# down_block: conv.conv.pool 
class down_block(K.Model):
    
    def __init__(self, filter_n, name):
        super(down_block, self).__init__(name)
        
        self.convolve_1 = K.layers.Conv3D(filter_n, 
                                          (3, 3, 3),
                                          activation='relu', 
                                          padding='same')
        
        self.convolve_2 = K.layers.Conv3D(filter_n, 
                                          (3, 3, 3), 
                                          activation='relu',
                                          padding='same')
        
        self.pool       = K.layers.MaxPooling3D(pool_size=(2, 2, 2), 
                                                padding='same')
        
    def call(self, x):
        
        conv = self.convolve_1(x)
        conv = self.convolve_2(conv)
        conc = tf.concat([x, conv], axis=-1)
        pool = self.pool(conc)
        
        return conv, conc, pool
    
# waist_block: conv.conv
class waist_block(K.Model):
    
    def __init__(self, filter_n, name):
        super(waist_block, self).__init__(name)
        
        self.convolve_1 = K.layers.Conv3D(filter_n, 
                                          (3, 3, 3), 
                                          activation='relu', 
                                          padding='same')
        
        self.convolve_2 = K.layers.Conv3D(filter_n, 
                                          (3, 3, 3), 
                                          activation='relu', 
                                          padding='same')
        
    def call(self, x):
        
        conv = self.convolve_1(x)
        conv = self.convolve_2(conv)
        conc = tf.concat([x, conv], axis=-1)
        
        return conc     

# up_block: t_conv,conv,conv
class up_block(K.Model):

    def __init__(self, filter_n, name):
        super(up_block, self).__init__(name)
        
        self.transposed_convolve =  K.layers.Conv3DTranspose(filter_n, 
                                                             (2, 2, 2),
                                                             strides=(2, 2, 2),
                                                             padding='same')
        
        self.convolve_1          =  K.layers.Conv3D(filter_n,
                                                    (3, 3, 3),
                                                    activation='relu',
                                                    padding='same')
        
        self.convolve_2          =  K.layers.Conv3D(filter_n,
                                                    (3, 3, 3),
                                                    activation='relu', 
                                                    padding='same')
        
    def call(self, conc, conv):
         
        conc = tf.concat([self.transposed_convolve(conc), conv], axis =-1)
        conv = self.convolve_1(conc)
        conv = self.convolve_2(conv)
        conc = tf.concat([conc,conv], axis=-1)
        
        return conc    
              
        

# Unet with few skip connections in each block
class unet_3d(K.Model):
    
    def __init__(self,name):
        super(unet_3d, self).__init__(name = name)
        
        self.down_1 = down_block(16,"down_1")
        self.down_2 = down_block(16 * 2**1, "down_2")
        self.down_3 = down_block(16 * 2**2, "down_3")
        self.down_4 = down_block(16 * 2**3, "down_4")
        
        self.waist = waist_block(16 * 2**3,"waist")
        
        self.up_1 = up_block(16 * 2**3,"up_1")
        self.up_2 = up_block(16 * 2**2,"up_2")
        self.up_3 = up_block(16 * 2**1,"up_3")
        self.up_4 = up_block(16,"up_4")
    
    
    def forward(self, x, n_classes):
        
        conv_1, conc_1, pool_1 = self.down_1(x)
        conv_2, conc_2, pool_2 = self.down_2(pool_1)
        conv_3, conc_3, pool_3 = self.down_3(pool_2)
        conv_4, conc_4, pool_4 = self.down_4(pool_3)
        
        waist = self.waist(pool_4)
        
        conc_1 = self.up_1(waist, conv_4)
        conc_2 = self.up_2(conc_1, conv_3)
        conc_3 = self.up_3(conc_2, conv_2)
        conc_4 = self.up_4(conc_3, conv_1)
                
        logit = K.layers.Conv3D(n_classes,
                                (3,3,3), 
                                padding='same', 
                                activation = "sigmoid")(conc_4)
        
        return logit
    
    