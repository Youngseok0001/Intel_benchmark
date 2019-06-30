import numpy as np
import tensorflow as tf
import tensorlayer as tl
import sys
from matplotlib import pyplot as plt
from IPython import display
sys.path.append('..')
from config import config

def get_mean_se(xs):
  
    mean = np.mean(xs)
    se = np.std(xs)/np.sqrt(len(xs))
    
    return mean, se
                
def _print(txt,f):
    print(txt)
    print(txt, file = f)

def normalise(x):
    val_max = 255
    val_min = 0
    return ( x - np.min(x) ) * (val_max - val_min) / ( np.max(x) - np.min(x) ) + val_min

def vis_slice(img, gt, pred, slice_n, path):
    
    # GET FIRST BATCH
    img_slice  = img[0,:,:,slice_n,0]
    gt_slice   = gt[0,:,:,slice_n]
    pred_slice = pred[0,:,:,slice_n,:]       
    pred_slice = np.argmax(pred_slice, axis = -1)   
        
    img_slice = img_slice.astype("float32")
    gt_slice = gt_slice.astype("uint8")
    pred_slice = pred_slice.astype("uint8")
    
    # argmax to get predicted label
   
    
    # colour list
    COLORS = np.array(config.colours, dtype = "uint8")
    
    # apply clour
    pred_slice = COLORS[pred_slice]
    gt_slice   = COLORS[gt_slice]
    # minor processing
    img_slice  = np.stack((img_slice,)*3, axis=-1)
    img_slice_normalised = normalise(img_slice)
    img_slice_normalised = img_slice_normalised.astype("uint8")
    
    plt.imshow(img_slice_normalised)
    plt.show()
    plt.imshow(gt_slice)
    plt.show()
    plt.imshow(pred_slice)
    plt.show()
        
    tl.vis.save_images(np.asarray([img_slice_normalised, gt_slice, pred_slice]), size=(1, 3),
        image_path=path)

if __name__ == '__main__':

    import os
    import tensorflow as tf
    
    import sys 
    sys.path.append('..')

    from dataset.brats_data_loader import *
    from dataset.utils import *
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
    tf.enable_eager_execution()

    
    img_loc = "../dataset/Task01_BrainTumour/imagesTr/"    
    lab_loc = "../dataset/Task01_BrainTumour/labelsTr/"

    # get img and label array 
    img_path = get_path(img_loc)
    label_path = get_path(lab_loc)

    dataset = get_data_pipeline(img_path, label_path, 
                                  batch_size = config.batch_size, 
                                  prefetch   = config.prefetch,
                                  cpu_n      = config.cpu_n,
                                  epoch      = config.epoch,
                                  is_train   = True)
    
    generator = dataset.make_one_shot_iterator()
    _img, _lab = generator.get_next()
    _pred = tf.one_hot(_lab, depth = config.num_classes, axis = -1) 
    
    _img = _img.numpy()
    _lab = _lab.numpy()
    _pred = _pred.numpy()
    
    vis_slice(_img, _lab, _pred ,32, "./temp.png")
