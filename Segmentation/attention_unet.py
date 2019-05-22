import tensorflow as tf
from model_utils import *
from tf_metrics import *

def attention_gating_unet2D(num_classes, metrics, loss, lr,
                            num_gpu = 0,
                            pretrained_weights=None,
                            input_size=(224, 224, 3),
                            bn = True):
    inputs = tf.keras.Input(input_size)
    
    #Encoder-Block 1 (224x224x3)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = tf.keras.layers.BatchNormalization()(conv1) if bn else conv1
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = tf.keras.layers.BatchNormalization()(conv1) if bn else conv1
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    #Encoder-Block 2 (112x112x64)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2) if bn else conv2
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = tf.keras.layers.BatchNormalization()(conv2) if bn else conv2
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    #Encoder-Block 3 (64x64x128)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = tf.keras.layers.BatchNormalization()(conv3) if bn else conv3
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = tf.keras.layers.BatchNormalization()(conv3) if bn else conv3
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    #Encoder-Block 4 (32x32x256)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = tf.keras.layers.BatchNormalization()(conv4) if bn else conv4
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = tf.keras.layers.BatchNormalization()(conv4) if bn else conv4 
    drop4 = tf.keras.layers.Dropout(0.2)(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    #Center (16x16x512)
    conv5 = tf.keras.layers.Conv2D(1024, 2, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = tf.keras.layers.BatchNormalization()(conv5) if bn else conv5
    conv5 = tf.keras.layers.Conv2D(1024, 2, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = tf.keras.layers.BatchNormalization()(conv5) if bn else conv5
    drop5 = tf.keras.layers.Dropout(0.2)(conv5)
    #Gating (16x16x1024)
    gating = tf.keras.layers.Conv2D(1024, 1, activation='relu', padding='same', kernel_initializer='he_normal')(drop5)

    #Decoder-Block 6 (16x16x1024), Gating (16x16x512)
    attention6, att_weight6 = grid_attention_block_2D(conv4, gating, 512) #Make attention map for block 4
    upsampling6 = tf.keras.layers.Conv2DTranspose(filters=512,
                                                  kernel_size=2,
                                                  strides=2,
                                                  activation='relu',
                                                  padding='same',
                                                  kernel_initializer='he_normal')(conv5)
    merge6 = tf.keras.layers.concatenate([attention6, upsampling6], axis=3)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = tf.keras.layers.BatchNormalization()(conv6) if bn else conv6
    conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = tf.keras.layers.BatchNormalization()(conv6) if bn else conv6

    #Decoder-Block 7 (32x32x512), Gating (16x16x512)
    attention7, att_weight7 = grid_attention_block_2D(conv3, gating, 256) #Make attention map for block 3
    upsampling7 = tf.keras.layers.Conv2DTranspose(filters=256,
                                                  kernel_size=3,
                                                  strides=2,
                                                  activation='relu',
                                                  padding='same',
                                                  kernel_initializer='he_normal')(conv6)
    merge7 = tf.keras.layers.concatenate([attention7, upsampling7], axis=3)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = tf.keras.layers.BatchNormalization()(conv7) if bn else conv7
    conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = tf.keras.layers.BatchNormalization()(conv7) if bn else conv7

    #Decoder-Block 8 (64x64x256), Gating (16x16x512)
    attention8, att_weight8 = grid_attention_block_2D(conv2, gating, 128) #Make attention map for block 2
    upsampling8 = tf.keras.layers.Conv2DTranspose(filters=128,
                                                  kernel_size=3,
                                                  strides=2,
                                                  activation='relu',
                                                  padding='same',
                                                  kernel_initializer='he_normal')(conv7)
    merge8 = tf.keras.layers.concatenate([attention8, upsampling8], axis=3)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = tf.keras.layers.BatchNormalization()(conv8) if bn else conv8
    conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = tf.keras.layers.BatchNormalization()(conv8) if bn else conv8

    #Decoder-Block 9 (32x32x128), Gating (16x16x512)
    attention9, att_weight9 = grid_attention_block_2D(conv1, gating, 64) #Make attention map for block `
    upsampling9 = tf.keras.layers.Conv2DTranspose(filters=64,
                                                  kernel_size=3,
                                                  strides=2,
                                                  activation='relu',
                                                  padding='same',
                                                  kernel_initializer='he_normal')(conv8)
    merge9 = tf.keras.layers.concatenate([attention9, upsampling9], axis=3)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = tf.keras.layers.BatchNormalization()(conv9) if bn else conv9
    conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = tf.keras.layers.BatchNormalization()(conv9) if bn else conv9
    conv9 = tf.keras.layers.Conv2D(3, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = tf.keras.layers.BatchNormalization()(conv9) if bn else conv9

    output = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv9)
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    # Running on parallel data training
    model = tf.keras.utils.multi_gpu_model(model, gpus= num_gpu) if num_gpu else model
    compile_model(model, num_classes, metrics, loss, lr)
    
    model.summary() # print model

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model