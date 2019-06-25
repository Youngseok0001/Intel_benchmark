from matplotlib import pyplot as plt
from toolz import *
import numpy as np

import sys
sys.path.append('..')




# given 4D tensor from tf, visualise it

# from the first(T1)channel

_get_slice = lambda slice_n, x : x[0, :,:, slice_n, 0]
_vis = lambda alpha, x : plt.imshow(x, "gray", alpha = alpha)

if __name__ == '__main__':

    from dataset.brats_data_loader import *
    from dataset.utils import *
    from random import randint
    import tensorflow as tf
    import os
    
    %matplotlib inline
    
    os.environ["CUDA_VISIBLE_DEVICES"]="3"
    
    img_loc = "/home/jacob/Intel/3D_CNN/dataset/Task01_BrainTumour/imagesTr/"    
    lab_loc = "/home/jacob/Intel/3D_CNN/dataset/Task01_BrainTumour/labelsTr/"

    # get img and label array 
    imgs_path = get_path(img_loc)
    labels_path = get_path(lab_loc)

    dataset = get_data_pipeline(imgs_path, labels_path, 
                                  batch_size = 1, 
                                  prefetch = 4,
                                  cpu_n = 10,
                                  epoch = 1)
    
    
    generator = dataset.make_one_shot_iterator()
    img, lab = generator.get_next()
    
    with tf.Session() as sess:
        
        for _ in range(10):
        
            img_realised, label_realised = sess.run([img,lab])
            label_realised = label_realised[:,:,:,:,np.newaxis]
            
            slice_n = randint(0, np.shape(img_realised)[3] -1)
            slice_vis_img = compose(partial(_vis, 1), partial(_get_slice,slice_n))
            slice_vis_label = compose(partial(_vis, 1), partial(_get_slice,slice_n))
        
            slice_vis_img(img_realised)
            plt.show()
            slice_vis_label(label_realised)
            plt.show()