import csv
from glob import glob
import tensorflow as tf
import numpy as np



def get_imgdir_label(dataset_dir):
    
    label_dir = glob(dataset_dir + "*Task3*Train*GroundTruth*/*.csv")[0]

    imagedir_label = []

    with open(label_dir,"r") as f:
        
        lines = csv.reader(f, delimiter=',') 
        header_info = next(lines)
        
        for line in lines:
            
            image_dir =  dataset_dir + \
                         "ISIC2018_Task3_Training_Input/" + \
                          line[0] + ".jpg"

            label_index = line[1:].index("1.0")
            
            label = header_info[1:][label_index]
            
            imagedir_label.append((image_dir,label))
            
        return imagedir_label
      
      
      
def label_to_index(imagedir_label,class_dict):
    return [(img_lab[0],class_dict[img_lab[1] ])
            for img_lab 
            in imagedir_label]

  
def parser(img_dir,label):
    img = tf.read_file(img_dir)
    img = tf.image.decode_jpeg(img, channels = 3)
    
    return preprocessors(img), label

def preprocessors(img):

    def flip_left_right(img):
        return tf.image.random_flip_left_right(img)
    def random_contrast(img):
        return tf.image.random_contrast(img,0.6,1)        
    def random_brightness(img):
        return tf.image.random_brightness(img,0.1)        
    def central_crop(img):
        return tf.image.central_crop(img, central_fraction=np.random.uniform(0.7,1))
    def normalise(img):
        means, stds = tf.constant([181.5,154.8,141.9]), tf.constant([11.3,16.9,19.7] )
        return (img - means)/stds 

    img = tf.cast(img, tf.float32)
    img = central_crop(img)
    img = normalise(img)
    img = flip_left_right(img)
    img = random_contrast(img)
    img = random_brightness(img)
    img = tf.image.resize_images(img,[224,224])
    
    return img