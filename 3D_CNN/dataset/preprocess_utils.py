from dltk.io.augmentation import *
from dltk.io.preprocessing import *  

import numpy as np
import nibabel as nib 

from functools import reduce, partial

_distort        = lambda img_label, alpha, sigma:\
                    (elastic_transform(img_label[0], alpha, sigma),
                     elastic_transform(img_label[1], alpha, sigma))

_flip           = lambda img_label, axis:\
                    (flip(img_label[0], axis),
                    flip(img_label[1], axis))
    
_resize         = lambda img_label, resize_constant : \
                    (resize_image_with_crop_or_pad(img_label[0], resize_constant + [4], mode='symmetric'),
                    resize_image_with_crop_or_pad(img_label[1], resize_constant, mode='symmetric'))
            
_normalise      = lambda img_label, means, stds :\
                    ((img_label[0] - means) / stds, img_label[1])

_gaussian_noise = lambda img_label, sigma:\
                    (add_gaussian_noise(img_label[0], sigma), img_label[1])

def _omit_labeless_slices(img_lab, condition = None):
        
    img = np.transpose(img_lab[0],(2,0,1,3))
    lab = np.transpose(img_lab[1],(2,0,1))
        
    img, lab = list(map(np.asarray,zip(*(filter(condition, zip(img, lab))))))
        
    img = np.transpose(img,(1,2,0,3))
    lab = np.transpose(lab,(1,2,0))

    return img,lab

if __name__ == '__main__':

    from glob import glob 
    
    img_file = glob("/home/jacob/Intel/3D_CNN/dataset/Task01_BrainTumour/imagesTr/*")[0]
    lab_file = "/home/jacob/Intel/3D_CNN/dataset/Task01_BrainTumour/labelsTr/" + img_file.split("/")[-1]
    
    img = nib.load(img_file).get_fdata().astype(np.float32)
    label = nib.load(lab_file).get_fdata().astype(np.float32)
    
    img_lab = (img, label)

    print(np.shape(img))
    print(np.shape(label))
    
    condition = lambda img_lab : np.sum(img_lab[1]) != 0
    
    omit_labeless_slices = partial(_omit_labeless_slices, condition = condition)
    img, label = omit_labeless_slices(img_lab)
    
    
    
    print(np.shape(img))
    print(np.shape(label))

    
    