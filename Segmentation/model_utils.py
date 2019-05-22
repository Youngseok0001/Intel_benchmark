import tensorflow as tf
from tf_metrics import *

# Bilinear upsampling
def UpSampling2DBilinear(size):
    return tf.keras.layers.Lambda(lambda x: tf.image.resize_bilinear(x, size, align_corners=True))

# Channel wise attention
def squeeze_excite_block(x, ratio=16):
    ''' Create a channel-wise squeeze-excite block
    Args:
        x: input tensor
        filters: number of output filters
    Returns: a keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    '''
    init = x
    channel_axis = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1
    filters = int(init.shape[channel_axis])
    assert isinstance(ratio, int) and (filters // ratio > 0)
    se_shape = (1, 1, filters)

    se = tf.keras.layers.GlobalAveragePooling2D()(init)
    se = tf.keras.layers.Reshape(se_shape)(se)
    se = tf.keras.layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if channel_axis == 1:
        se = tf.keras.layers.Permute((3, 1, 2))(se)

    x = tf.keras.layers.multiply([init, se])
    return x

# Grid attention 
def grid_attention_block_2D(x, g, out_filter, dim=2, sub_sample_factor=(2,2), bn= True):    
    """ Create attention map
    Args:
        x: Input features before downsampling
        g: Gating signal from the coarser features
        out_filter: Number of output filters
    Returns:
        encoder block * attention
        Attention weight
    """
    assert isinstance(sub_sample_factor, tuple) 
    assert isinstance(dim, int)
    # preserve the filters in input x
    if tf.keras.backend.image_data_format() == 'channels_last':
        num_filter = int(x.shape[3])
    else:
        num_filter = int(x.shape[1])

    # linear transformation using 1x1 convolution (factor scaling)
    gating_signal = tf.keras.layers.Conv2D(filters=num_filter,
                                           kernel_size=(1,1),
                                           strides=1,
                                           use_bias=True,
                                           padding='same',
                                           kernel_initializer='he_normal')(g)
    sub_sample_kernel_size = sub_sample_factor
    # linear transformation using 2x2 convolution, strides = 2 (downsampled by 2) 
    input_feature = tf.keras.layers.Conv2D(filters=num_filter,
                                           kernel_size=sub_sample_kernel_size,
                                           strides=2,
                                           use_bias=False,
                                           padding='same',
                                           kernel_initializer='he_normal')(x)
    # Obtain input feature dimension
    input_feature_size = (int(input_feature.shape[1]),) * dim                                                                    
    gating_signal = UpSampling2DBilinear(input_feature_size)(gating_signal)
    # Addition of two gating signal and encoder channels
    additive_attention = tf.keras.layers.Add()([input_feature, gating_signal])
    # RELU
    relu = tf.keras.layers.Activation('relu')(additive_attention)
    # Sigmoid (attention weight)
    conv_attention = tf.keras.layers.Conv2D(filters = num_filter,
                                            kernel_size=(1,1),
                                            strides=1,
                                            use_bias=True,
                                            padding='same',
                                            kernel_initializer='he_normal',
                                            activation='sigmoid')(relu)

    upsample_size = (int(x.shape[1]),) * dim                                    
    upsample_attention_weight = UpSampling2DBilinear(upsample_size)(conv_attention) # Upsampling of attention weight to fit encoder dim
    
    # Squeeze and excited (Channel wise attention)
    se_upsample_att_weight = squeeze_excite_block(upsample_attention_weight)
    
    # Channel mean to obtain single attention (add epsilon to prevent killing of information)
    avg_att = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=3) + 0.01)(se_upsample_att_weight)
    avg_att = tf.keras.layers.Reshape((upsample_size[0], upsample_size[0], 1))(avg_att)
    # Multiplying attention map to encoder features
    element_wise_multiplication = tf.keras.layers.Multiply()([avg_att, x])
    
    conv_out = tf.keras.layers.Conv2D(filters= out_filter,
                                      kernel_size=(1,1),
                                      strides=1,
                                      padding='same',
                                      kernel_initializer='he_normal')(element_wise_multiplication)
    conv_out = tf.keras.layers.BatchNormalization()(conv_out) if bn else conv_out
    return conv_out, upsample_attention_weight

# compiling model
def compile_model(model, num_classes, metrics, loss, lr):
    from tf_metrics import dice_coeff, jaccard_index, class_jaccard_index
    from tf_metrics import pixelwise_precision, pixelwise_sensitivity, pixelwise_specificity, pixelwise_recall
    if isinstance(loss, str):
        if loss in {'ce', 'crossentropy'}:
            if num_classes == 1:
                loss = tf.keras.losses.binary_crossentropy
            else:
                loss = tf.keras.losses.categorical_crossentropy
        else:
            raise ValueError('unknown loss %s' % loss)

    if isinstance(metrics, str):
        metrics = [metrics, ]

    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric == 'acc':
            metrics[i] = tf.keras.metrics.binary_accuracy if num_classes == 1 else tf.keras.metrics.categorical_accuracy
        elif metric == 'jaccard_index':
            metrics[i] = jaccard_index(num_classes)
        elif metric == 'jaccard_index0':
            metrics[i] = class_jaccard_index(0)
        elif metric == 'jaccard_index1':
            metrics[i] = class_jaccard_index(1)
        elif metric == 'jaccard_index2':
            metrics[i] = class_jaccard_index(2)
        elif metric == 'jaccard_index3':
            metrics[i] = class_jaccard_index(3)
        elif metric == 'jaccard_index4':
            metrics[i] = class_jaccard_index(4)
        elif metric == 'jaccard_index5':
            metrics[i] = class_jaccard_index(5)
        elif metric == 'dice_coeff':
            metrics[i] = dice_coeff(num_classes)
        elif metric == 'pixelwise_precision':
            metrics[i] = pixelwise_precision(num_classes)
        elif metric == 'pixelwise_sensitivity':
            metrics[i] = pixelwise_sensitivity(num_classes)
        elif metric == 'pixelwise_specificity':
            metrics[i] = pixelwise_specificity(num_classes)
        elif metric == 'pixelwise_recall':
            metrics[i] = pixelwise_recall(num_classes)
        else:
            raise ValueError('metric %s not recognized' % metric)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                  loss=loss,
                  metrics=metrics)
    
    
def slice_batch(x, n_gpus, part):
    """
    Divide the input batch into [n_gpus] slices, and obtain slice number [part].
    i.e. if len(x)=10, then slice_batch(x, 2, 1) will return x[5:].
    """
    sh = tf.keras.backend.shape(x)
    L = sh[0] // n_gpus
    if part == n_gpus - 1:
        return x[part*L:]
    return x[part*L:(part+1)*L]


def to_multi_gpu(model, n_gpus=2):
    """
    Given a keras [model], return an equivalent model which parallelizes
    the computation over [n_gpus] GPUs.

    Each GPU gets a slice of the input batch, applies the model on that slice
    and later the outputs of the models are concatenated to a single tensor, 
    hence the user sees a model that behaves the same as the original.
    """
    
    if n_gpus == 1:
        return model
    
    with tf.device('/cpu:0'):
        x = tf.keras.layers.Input(model.input_shape[1:], name=model.input_names[0])

    towers = []
    for g in range(n_gpus):
        with tf.device('/gpu:' + str(g)):
            slice_g = tf.keras.layers.Lambda(slice_batch, 
                                             lambda shape: shape, 
                                             arguments={'n_gpus':n_gpus, 'part':g})(x)
            towers.append(model(slice_g))

    with tf.device('/cpu:0'):
        merged = tf.keras.layers.Concatenate(axis=0)(towers)

    return tf.keras.models.Model(inputs=x, outputs=merged)





    