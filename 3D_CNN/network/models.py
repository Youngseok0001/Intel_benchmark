import sys
sys.path.append('..')
from config import *

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models


#############################################
#basic 3D-Unet implementation 
#############################################

    
def unet3d_keras():

    """
    3D U-Net
    """
    def ConvolutionBlock(x, name, fms, params):

        x = layers.Conv3D(filters=fms, **params, name=name+"_conv0")(x)
        x = layers.BatchNormalization(name=name+"_bn0")(x)
        x = layers.Activation("relu", name=name+"_relu0")(x)

        x = layers.Conv3D(filters=fms, **params, name=name+"_conv1")(x)
        x = layers.BatchNormalization(name=name+"_bn1")(x)
        x = layers.Activation("relu", name=name)(x)

        return x

    inputs = layers.Input(shape = config.patch_size + [4], name = "MRImages")

    params = dict(kernel_size = (3, 3, 3),
                  activation = None,
                  padding = "same",
                  data_format = "channels_last",
                  kernel_initializer="he_uniform")

    # Transposed convolution parameters
    params_trans = dict(data_format= "channels_last",
                        kernel_size=(2, 2, 2),
                        strides=(2, 2, 2),
                        padding="same")


    # BEGIN - Encoding path
    encodeA = ConvolutionBlock(inputs, "encodeA", config.BASE_FILTER, params)
    poolA = layers.MaxPooling3D(name="poolA", pool_size=(2, 2, 2))(encodeA)

    encodeB = ConvolutionBlock(poolA, "encodeB", config.BASE_FILTER*2, params)
    poolB = layers.MaxPooling3D(name="poolB", pool_size=(2, 2, 2))(encodeB)

    encodeC = ConvolutionBlock(poolB, "encodeC", config.BASE_FILTER*4, params)
    poolC = layers.MaxPooling3D(name="poolC", pool_size=(2, 2, 2))(encodeC)

    encodeD = ConvolutionBlock(poolC, "encodeD", config.BASE_FILTER*8, params)
    poolD = layers.MaxPooling3D(name="poolD", pool_size=(2, 2, 2))(encodeD)

    encodeE = ConvolutionBlock(poolD, "encodeE", config.BASE_FILTER*16, params)
    # END - Encoding path

    # BEGIN - Decoding path
    up = layers.UpSampling3D(name="upE", size=(2, 2, 2))(encodeE)

    concatD = layers.concatenate([up, encodeD], axis = -1, name="concatD")
    decodeC = ConvolutionBlock(concatD, "decodeC", config.BASE_FILTER*8, params)

        
    up = layers.UpSampling3D(name="upC", size=(2, 2, 2))(decodeC)
    concatC = layers.concatenate([up, encodeC], axis = -1, name="concatC")
    decodeB = ConvolutionBlock(concatC, "decodeB", config.BASE_FILTER*4, params)
    
    up = layers.UpSampling3D(name="upB", size=(2, 2, 2))(decodeB)
    concatB = layers.concatenate([up, encodeB], axis = -1, name="concatB")
    decodeA = ConvolutionBlock(concatB, "decodeA", config.BASE_FILTER*2, params)

    up = layers.UpSampling3D(name="upA", size=(2, 2, 2))(decodeA)
    concatA = layers.concatenate([up, encodeA], axis = -1, name="concatA")

    # END - Decoding path
    convOut = ConvolutionBlock(concatA, "convOut", config.BASE_FILTER, params)
    logits = layers.Conv3D(name="PredictionMask",filters = config.num_classes, kernel_size=(1, 1, 1), data_format="channels_last")(convOut)

    model = models.Model(inputs=[inputs], outputs=[logits])
    
    print(model.summary())

    return model


    
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import sys
from tensorpack import VariableHolder

#############################################
#Complicated 3D-Unet implementation 
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


if __name__ == '__main__':
    
    from functools import partial 
    from metrics import *
    
    
    x = tf.random_normal(shape = (1,256,256,128,4))
    gt = tf.argmax(tf.random_normal(shape = (1,256,256,128,4)), axis = -1)
    
    model = partial(unet3d, reuse = False)
    logit = model(x)
    pred = tf.nn.softmax(logit, axis = -1)

    ce = get_ce(gt, pred, weighted = False)
    acc = get_acc(gt, pred)
    iou, iou_op = get_iou(gt, pred)
    dice_loss = get_dice_loss(gt, pred, 4)
    exp_dice_loss = get_exp_dice_loss(gt, pred, 4)
    label_wise_dice = get_label_wise_dice_coef(gt, pred, 4)

    
    optimizer = tf.train.AdadeltaOptimizer()

    train_op  = optimizer.minimize(dice_loss + ce + exp_dice_loss)
    
    to_call = [acc, ce, dice_loss, exp_dice_loss, label_wise_dice, iou_op, train_op]

    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        
        for _ in range(10):
            
            [_acc, _ce, _dice_loss, _exp_dice_loss,_label_wise_dice, *_] = sess.run(to_call)
            
            print("ce                   :",_ce)
            print("acc                  :",_acc) 
            print("dice loss            :",_dice_loss)
            print("exp dice loss        :",_exp_dice_loss)        
            print("dice coef label wise :",_label_wise_dice)
            print("iou                  :",sess.run(iou) * 100)
