from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.layer_utils import get_source_inputs
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils.data_utils import get_file
from model_utils import *
from subpixel import *

WEIGHTS_PATH_X = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
                        
def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = tf.keras.layers.ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = tf.keras.layers.BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = tf.keras.layers.BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = tf.keras.layers.Activation('relu')(x)

    return x


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return tf.keras.layers.Conv2D(filters,
                                      kernel_size= kernel_size,
                                      strides= stride,
                                      padding='same', use_bias=False,
                                      dilation_rate= rate,
                                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = tf.keras.layers.ZeroPadding2D((pad_beg, pad_end))(x)
        return tf.keras.layers.Conv2D(filters,
                                      kernel_size=kernel_size,
                                      strides=stride,
                                      padding='valid', use_bias=False,
                                      dilation_rate=rate,
                                      name=prefix)(x)


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            return_skip: flag to return additional tensor after 2 SepConvs for decoder
            """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = tf.keras.layers.BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = tf.keras.layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = tf.keras.layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def relu6(x):
    return tf.keras.activations.relu(x, max_value=6)


def Deeplabv3(metrics, loss, lr, weights='pascal_voc', input_tensor=None, input_shape=(224, 224, 3),
              classes=1, backbone='xception', OS=16, alpha=1., activation= 'sigmoid', num_gpu = 0):
    """ Instantiates the Deeplabv3+ architecture
    Optionally loads weights pre-trained
    on PASCAL VOC or Cityscapes. This model is available for TensorFlow only.
    # Arguments
        weights: one of 'pascal_voc' (pre-trained on pascal voc), 
            'cityscapes' (pre-trained on cityscape) or None (random initialization)
        input_tensor: optional Keras tensor (i.e. output of `tf.keras.layers.Input()`)
            to use as image input for the model.
        input_shape: shape of input image. format HxWxC
            PASCAL VOC model was trained on (512,512,3) images. None is allowed as shape/width
        classes: number of desired classes. PASCAL VOC has 21 classes, Cityscapes has 19 classes.
            If number of classes not aligned with the weights used, last layer is initialized randomly
        backbone: backbone to use. one of {'xception','mobilenetv2'}
        activation: optional activation to add to the top of the network. 
            One of 'softmax', 'sigmoid' or None
        OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.
            Used only for xception backbone.
        alpha: controls the width of the MobileNetV2 network. This is known as the
            width multiplier in the MobileNetV2 paper.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                    are used at each layer.
            Used only for mobilenetv2 backbone. Pretrained is only available for alpha=1.
    # Returns
        A Keras model instance.
    # Raises
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
        ValueError: in case of invalid argument for `weights` or `backbone`
    """

    if not (weights in {'pascal_voc', None}):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `pascal_voc`'
                         '(pre-trained on PASCAL VOC)')

    if not (backbone in {'xception'}):
        raise ValueError('Current version only supports'
                         '`xception`')

    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape=input_shape)
    else:
        img_input = input_tensor
        
    
    if OS == 8:
        entry_block3_stride = 1
        middle_block_rate = 2  # ! Not mentioned in paper, but required
        exit_block_rates = (2, 4)
        atrous_rates = (12, 24, 36)
    else:
        entry_block3_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)
        atrous_rates = (6, 12, 18)

    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2),
                               name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
    x = tf.keras.layers.BatchNormalization(name='entry_flow_conv1_1_BN')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
    x = tf.keras.layers.BatchNormalization(name='entry_flow_conv1_2_BN')(x)
    x = tf.keras.layers.Activation('relu')(x)


    x = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
                        skip_connection_type='conv', stride=2,
                        depth_activation=False)
    x, skip1 = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
                               skip_connection_type='conv', stride=2,
                               depth_activation=False, return_skip=True)

    x = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
                        skip_connection_type='conv', stride=entry_block3_stride,
                        depth_activation=False)
    for i in range(16):
        x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                            skip_connection_type='sum', stride=1, rate=middle_block_rate,
                            depth_activation=False)

    x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                        skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                        depth_activation=False)
    x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                        skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                        depth_activation=True)

    # end of feature extractor
    # branching for Atrous Spatial Pyramid Pooling
    # Image Feature branch
    b4 = tf.keras.layers.AveragePooling2D(pool_size=(int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(x)
    b4 = tf.keras.layers.Conv2D(256, (1, 1), padding='same',
                                use_bias=False, name='image_pooling')(b4)
    b4 = tf.keras.layers.BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = tf.keras.layers.Activation('relu')(b4)
    b4 = tf.keras.layers.Lambda(lambda xx: tf.image.resize_bilinear(xx, size= (int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))), align_corners=True))(b4)
    

    # simple 1x1
    b0 = tf.keras.layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = tf.keras.layers.BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = tf.keras.layers.Activation('relu', name='aspp0_activation')(b0)

    # there are only 2 branches in mobilenetV2. not sure why
    # rate = 6 (12)
    b1 = SepConv_BN(x, 256, 'aspp1',
                    rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
    # rate = 12 (24)
    b2 = SepConv_BN(x, 256, 'aspp2',
                    rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
    # rate = 18 (36)
    b3 = SepConv_BN(x, 256, 'aspp3',
                    rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

    # concatenate ASPP branches & project
    x = tf.keras.layers.Concatenate()([b4, b0, b1, b2, b3])
    x = tf.keras.layers.Conv2D(256, (1, 1),
                               padding='same',
                               use_bias=False,
                               name='concat_projection')(x)
    x = tf.keras.layers.BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    # DeepLab v.3+ decoder

    # Feature projection
    # x4 (x2) block
    x = tf.keras.layers.Lambda(lambda xx: tf.image.resize_bilinear(xx, size=(int(np.ceil(input_shape[0]/4)), int(np.ceil(input_shape[1]/4))),
                                                                   align_corners=True))(x) 
    dec_skip1 = tf.keras.layers.Conv2D(48, (1, 1), padding='same',
                                       use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = tf.keras.layers.BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = tf.keras.layers.Activation('relu')(dec_skip1)
    x = tf.keras.layers.Concatenate()([x, dec_skip1])
    x = SepConv_BN(x, 256, 'decoder_conv0',
                   depth_activation=True, epsilon=1e-5)
    x = SepConv_BN(x, 256, 'decoder_conv1',
                   depth_activation=True, epsilon=1e-5)

    # you can use it with arbitary number of classes
    if (weights == 'pascal_voc' and classes == 21):
        last_layer_name = 'logits_semantic'
    else:
        last_layer_name = 'custom_logits_semantic'

    x = tf.keras.layers.Lambda(lambda xx: tf.image.resize_bilinear(xx, tf.shape(img_input)[1:3], align_corners=True))(x) 
    x = tf.keras.layers.Conv2D(classes, (1, 1), padding='same', name=last_layer_name)(x)
    if classes == 1:
        output = tf.keras.layers.Activation('sigmoid')(x)
    else:
        output = tf.keras.layers.Activation('softmax')(x)
    
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
        
    model = tf.keras.models.Model(inputs=inputs, outputs=output, name='deeplabv3plus')
    
    #extend model to train in multigpu
    model = tf.keras.utils.multi_gpu_model(model, gpus= num_gpu) if num_gpu else model

    # load weights
    if weights == 'pascal_voc':
        weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
                                WEIGHTS_PATH_X,
                                cache_subdir='models')
        model.load_weights(weights_path, by_name=True)

    return model

def deeplab_subpixel_method(metrics, loss, lr,
                            use_num_gpu = 0,
                            weights='pascal_voc', 
                            input_tensor=None,
                            input_shape=(224, 224, 3),
                            classes=1, 
                            backbone='xception',
                            OS=8, alpha=1., 
                            activation= 'sigmoid'):
    
    build_model =  Deeplabv3(metrics = metrics, 
                             loss = loss, 
                             lr = lr,
                             weights=weights, 
                             input_tensor=input_tensor,
                             input_shape= input_shape,
                             classes=classes, 
                             backbone= backbone,
                             OS=OS, alpha=alpha, 
                             activation= activation, 
                             num_gpu = 0)
    
    base_model = tf.keras.models.Model(build_model.input, build_model.layers[-5].output)
    scale = 4
    x = Subpixel(classes, 1, scale, padding = 'same')(base_model.output)
    if classes == 1:
        x = tf.keras.layers.Activation('sigmoid', name= 'pred_mask')(x)
    else:
        x = tf.keras.layers.Activation('softmax', name= 'pred_mask')(x)
    model = tf.keras.models.Model(base_model.input, x, name= 'deeplabv3p_subpixel')
    for layer in model.layers:
        if type(layer) == Subpixel:
            c, b = layer.get_weights()
            w = icnr_weights(scale = scale, shape=c.shape)
            layer.set_weights([w, b])

    if use_num_gpu:
        model = tf.keras.utils.multi_gpu_model(model, gpus = use_num_gpu)
    
    return model


def preprocess_input(x):
    """Preprocesses a numpy array encoding a batch of images.
    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].
    # Returns
        Input array scaled to [-1.,1.]
    """
    return imagenet_utils.preprocess_input(x, mode='tf')
