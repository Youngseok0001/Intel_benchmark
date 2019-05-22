from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import csv
import re
from glob import glob
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import random
import os as os
import inspect
import cv2


image_filename = 'ISIC2018_Task1-2_Training_Input'
label_filename = 'ISIC2018_Task1_Training_GroundTruth'
dataset_dir = '/data/volume01/ISIC_2018/'

# Create image filename ID list
img_dir = os.path.join(dataset_dir, image_filename)
image_ids = list()
if os.path.isdir(img_dir):
    image_ids = [fname.rsplit('.', maxsplit=1)[0] for fname in os.listdir(img_dir)
                 if fname.startswith('ISIC') and fname.lower().endswith('.jpg')]
    image_ids.sort()

def get_img_lab(dataset_dir,image_filename, label_filename):
        
    img_dirs = glob(dataset_dir + image_filename + "/*" + "jpg")
    img_names = [img_dir.split("/")[-1] for img_dir in img_dirs]

    lab_names = [re.sub(".jpg","_segmentation.png",img_name) for img_name in img_names]
    lab_dirs = [dataset_dir + label_filename + "/" + lab_name  for lab_name in lab_names] 
    
    imgs_labs = list(zip(img_dirs,lab_dirs))
    return imgs_labs

# parser wrapper reads img,label dir and convert to training image.
def parser(img_dir,label_dir, im_input_dim = [224,224]):
    
    img = tf.read_file(img_dir)
    img = tf.image.decode_jpeg(img, channels = 3)
    
    lab = tf.read_file(label_dir)
    lab = tf.image.decode_png(lab, channels = 1)
    
    # Image Augmentation
    img, lab = preprocessors(img, lab, resize_dim= im_input_dim)

    
    return img, lab

def preprocessors(image, label,
                  random_flip = True,
                  random_brightness = True,
                  random_rot = True,
                  central_crop = True,
                  random_contrast = True,
                  brightness= 0.1,
                  rotation = 180,
                  resize_dim = [224,224],
                  seed = 609):
    
    tf.random.set_random_seed(seed)
    np.random.seed(seed)
    
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
            raise Exception('Image and label must have the same dimensions!')
    
    
    def rand_crop_img(image, label):

        central_frac = np.random.uniform(0.7, 1)
        image = tf.image.central_crop(image, central_fraction=central_frac)
        label = tf.image.central_crop(label, central_fraction=central_frac)
        
        return image, label
    
    
    def rand_contrast(image):
        return tf.image.random_contrast(image, 0.6, 1)  
    
    
    def rand_flip(image, label):
        
        # Horizontal flip
        if random.randint(0,1):
            image = tf.image.flip_left_right(image)
            label = tf.image.flip_left_right(label)
        
        # Vertical flip
        if random.randint(0,1):
            image = tf.image.flip_up_down(image)
            label = tf.image.flip_up_down(label)
            
        return image, label
    
    
    def rand_brightness(image, brightness):
        return tf.image.random_brightness(image, brightness)
    
    
    def rotation_zoom(image, label, rotation):

        angle = random.randint(0, rotation)
        image = tf.contrib.image.rotate(image, angle * np.pi / 180, interpolation='BILINEAR')
        label = tf.contrib.image.rotate(label, angle * np.pi / 180, interpolation='BILINEAR')

        return image, label
    
    
    def normalise(image):
        # resize image
        means, stds = tf.constant([181.5,154.8,141.9]), tf.constant([11.3,16.9,19.7])
        return (image - means)/stds

    
    image = tf.cast(image, tf.float32)
    #image = normalise(image)
    
    if random_flip:
        image, label = rand_flip(image, label)
    
    if central_crop:
        image, label = rand_crop_img(image, label)
        
    if random_brightness:
        image = rand_brightness(image, brightness)
    
    if random_rot:
        image, label = rotation_zoom(image, label, rotation)
    
    if random_contrast:
        image = rand_contrast(image)
    
    # resize image
    image = tf.image.resize_images(image, resize_dim)
    label = tf.image.resize_images(label, resize_dim)

    return image, label


def split_datalist(dataset_dir= dataset_dir,
                   image_filename= image_filename,
                   label_filename= label_filename,
                   train_test_ratio = 0.8):
    
    imgs_labs = get_img_lab(dataset_dir,image_filename, label_filename)

    imgs_labs_train = imgs_labs[:int(len(imgs_labs) * train_test_ratio)]
    imgs_labs_test =  list(set(imgs_labs) - set(imgs_labs_train))

    imgs_train, label_train  = zip(*imgs_labs_train)
    imgs_test, label_test  = zip(*imgs_labs_test)
    
    return imgs_train, label_train, imgs_test, label_test

imgs_train, label_train, imgs_test, label_test = split_datalist()


def make_image_generator(imgs_train = imgs_train,
                         label_train = label_train,
                         imgs_test = imgs_test,
                         label_test = label_test,
                         batch_size= 32,
                         num_parallel_calls= 24,
                         test_batch_size = False):

    # Training data input function
    D_train = tf.data.Dataset.from_tensor_slices((list(imgs_train), list(label_train )))
    D_train = D_train.shuffle(buffer_size= len(list(imgs_train)))
    D_train = D_train.map(map_func= parser, num_parallel_calls=num_parallel_calls)
    D_train = D_train.batch(batch_size= batch_size)
    D_train = D_train.prefetch(buffer_size= 10*batch_size)


    # Training data input function
    D_test = tf.data.Dataset.from_tensor_slices((list(imgs_test), list(label_test)))    
    D_test = D_test.shuffle(buffer_size= len(list(imgs_test)))
    D_test = D_test.map(map_func=parser, num_parallel_calls= num_parallel_calls)
    D_test = D_test.batch(batch_size= len(list(imgs_test)))
    D_test = D_test.prefetch(buffer_size= 10*batch_size)
        

    return D_train, D_test


## D_train, D_test = make_image_generator()
## D_train = D_train.repeat(20)
## train_batch_gen = D_train.make_one_shot_iterator()


# Data pipeline for tf.keras.model
def create_generators(batch_size = 16, image_path=imgs_train, label_path=label_train,
                      resize_shape=(224,224), crop_shape = False,
                      do_ahisteq = True, n_classes = 1, horizontal_flip = True,
                      vertical_flip = True, blur = False, brightness=0.1, rotation=5.0,
                      zoom=0.1, seed = 610):
            
    generator = SegmentationGenerator(image_path=image_path, label_path=label_path, n_classes = n_classes,
                                      do_ahisteq = do_ahisteq, batch_size=batch_size, resize_shape=resize_shape,
                                      crop_shape=crop_shape, horizontal_flip=horizontal_flip,
                                      vertical_flip=vertical_flip, blur = blur,
                                      brightness=brightness, rotation=rotation, zoom=zoom,
                                      seed = seed)
            
    return generator

        
def train_with_generator(model, epochs, train_generator, valid_generator, callbacks, workers =12, mp = False):
    steps = len(train_generator)
    h = model.fit_generator(generator=train_generator,
                            steps_per_epoch=steps, 
                            epochs = epochs, verbose=1, 
                            callbacks = callbacks, 
                            validation_data=valid_generator,
                            validation_steps=len(valid_generator),
                            max_queue_size=10,
                            workers=workers, use_multiprocessing=mp)
    return h


def train_without_generator(model, X, y,epochs, val_data, tf_board = False, plot_train_process = True):
    h = model.fit(X, y, validation_data = val_data, verbose=1, 
                  batch_size = self.batch_size, epochs = epochs, 
                  callbacks = self.build_callbacks(tf_board = tf_board, plot_process = plot_train_process))
    return h
    

 
def load_resize_data(image_path, label_path, resize_dim=(224,224)):

    imgs_length = len(image_path)
    labels_length = len(label_path)
    imgs = np.zeros((imgs_length, resize_dim[0], resize_dim[1], 3), dtype='uint8')
    labs = np.zeros((labels_length, resize_dim[0], resize_dim[1], 1), dtype='float32')

    assert imgs_length == labels_length

    for n in tqdm(range(imgs_length)):
        image = cv2.imread(image_path[n], 1)
        image = cv2.resize(image, resize_dim)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imgs[n] = image.astype(np.uint8)

        label = cv2.imread(label_path[n], 0)
        label = cv2.resize(label, resize_dim, interpolation = cv2.INTER_NEAREST)
        label[label>0] = 1
        labs[n] = np.expand_dims(label, -1)

    return imgs, labs

    
    
    
#Remember to set fold name
class SegmentationGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, image_path = imgs_train, label_path = label_train,
                 n_classes=1, batch_size=16, resize_shape=None, 
                 seed = 610, crop_shape=(224, 224), horizontal_flip=True, blur = 0,
                 vertical_flip=True, brightness=0.1, rotation=5.0, zoom=0.1, do_ahisteq = True):
        
        self.image_path_env = image_path
        self.label_path_env = label_path
        self.blur = blur
        self.histeq = do_ahisteq
        self.image_path_list = tuple(list(self.image_path_env).copy())
        self.label_path_list = tuple(list(self.label_path_env).copy())

        np.random.seed(seed)
        x = np.random.permutation(len(self.image_path_list))
            
        self.image_path_list = [self.image_path_list[j] for j in x]
        self.label_path_list = [self.label_path_list[j] for j in x]
        
        
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.brightness = brightness
        self.rotation = rotation
        self.zoom = zoom

        # Preallocate memory
        if self.crop_shape:
            self.X = np.zeros((batch_size, crop_shape[1], crop_shape[0], 3), dtype='float32')
            self.Y = np.zeros((batch_size, crop_shape[1], crop_shape[0], 1), dtype='float32')

        elif self.resize_shape:
            self.X = np.zeros((batch_size, resize_shape[1], resize_shape[0], 3), dtype='float32')
            self.Y = np.zeros((batch_size, resize_shape[1], resize_shape[0], 1), dtype='float32')

        else:
            raise Exception('No image dimensions specified!')
        
    def __len__(self):
        return len(self.image_path_list) // self.batch_size
        
    def __getitem__(self, i):
        
        for n, (image_path, label_path) in enumerate(zip(self.image_path_list[i*self.batch_size:(i+1)*self.batch_size], 
                                                        self.label_path_list[i*self.batch_size:(i+1)*self.batch_size])):
            
            image = cv2.imread(image_path, 1)
            label = cv2.imread(label_path, 0)
            labels = np.unique(label)
            
            if self.blur and random.randint(0,1):
                image = cv2.GaussianBlur(image, (self.blur, self.blur), 0)

            if self.resize_shape and not self.crop_shape:
                image = cv2.resize(image, self.resize_shape)
                label = cv2.resize(label, self.resize_shape, interpolation = cv2.INTER_NEAREST)
        
            if self.crop_shape:
                image, label = _random_crop(image, label, self.crop_shape)
                
            # Do augmentation
            if self.horizontal_flip and random.randint(0,1):
                image = cv2.flip(image, 1)
                label = cv2.flip(label, 1)
            if self.vertical_flip and random.randint(0,1):
                image = cv2.flip(image, 0)
                label = cv2.flip(label, 0)
            if self.brightness:
                factor = 1.0 + random.gauss(mu=0.0, sigma=self.brightness)
                if random.randint(0,1):
                    factor = 1.0/factor
                table = np.array([((i / 255.0) ** factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
                image = cv2.LUT(image, table)
            if self.rotation:
                angle = random.gauss(mu=0.0, sigma=self.rotation)
            else:
                angle = 0.0
            if self.zoom:
                scale = random.gauss(mu=1.0, sigma=self.zoom)
            else:
                scale = 1.0
            if self.rotation or self.zoom:
                M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, scale)
                image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                label = cv2.warpAffine(label, M, (label.shape[1], label.shape[0]))

            if self.histeq: # and convert to RGB
                img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
                image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB) # to RGB
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR to RGB
                 
            label = label.astype('int32')
            for j in np.setxor1d(np.unique(label), labels):
                label[label==j] = self.n_classes
            
            # Make label
            y = label
            y[y>0] = 1              
            self.Y[n]  = np.expand_dims(y, -1)

            # Make image
            self.X[n] = image
        

        return self.X, self.Y
        
    def on_epoch_end(self):
        # Shuffle dataset for next epoch
        c = list(zip(self.image_path_env, self.label_path_env))
        random.shuffle(c)
        self.image_path_env, self.label_path_env = zip(*c)
                
    
def _random_crop(image, label, crop_shape):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')
        
    if (crop_shape[0] < image.shape[1]) and (crop_shape[1] < image.shape[0]):
        x = random.randrange(image.shape[1]-crop_shape[0])
        y = random.randrange(image.shape[0]-crop_shape[1])
        
        return image[y:y+crop_shape[1], x:x+crop_shape[0], :], label[y:y+crop_shape[1], x:x+crop_shape[0]]
    else:
        image = cv2.resize(image, crop_shape)
        label = cv2.resize(label, crop_shape, interpolation = cv2.INTER_NEAREST)
        return image, label
