# packages
#-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.

import os

from glob import glob
from functools import reduce

import numpy as np
import nibabel as nib 
# functions
#-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.

fst = lambda x: x[0]
snd = lambda x: x[1]


get_path = lambda data_path : glob(os.path.join(data_path , "*"))
get_data = lambda img_file : nib.load(img_file).get_fdata()

cal_mean = lambda img : np.mean(img, axis = (0,1,2)) 
cal_std = lambda img : np.std(img, axis = (0,1,2)) 
cal_mean_std = lambda img : [cal_mean(img),cal_std(img)]




# execute
#-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
#img_loc = "/home/jacob/Intel/3D_CNN/dataset/Task01_BrainTumour/imagesTr/" 
#img_loc = "/home/jacob/Intel/3D_CNN/dataset/Task01_BrainTumour/labelsTr/" 
img_loc = "/home/nuhs/Task01_BrainTumour/imagesTr/" 
lab_loc = "/home/nuhs/Task01_BrainTumour/labelsTr/" 

from itertools import islice

img_path = get_path(img_loc)


img = map(get_data, img_path)

def _omit_labeless_slices(lab, condition = None):
    
    # bring 3rd axis forward
    lab = np.transpose(lab,(2,0,1))
    # apply condition to ith sice of tuple(img,label) and convert back to tensor
    lab = np.asarray(list(filter(condition, lab)))
    
    print(len(lab))
    
    return len(lab)
    
from itertools import islice

lab_path = get_path(lab_loc)
labs = map(get_data, img_path)
condition = lambda lab : np.sum(lab) > 0
lens = (_omit_labeless_slices(lab, condition = condition) for lab in labs)

out = list(lens)
print(min(out))
print(max(out))

# mean_std = map(cal_mean_std, img)
# _mean = map(fst,mean_std)
# _std = map(snd,mean_std)
# mean = reduce(lambda a, b : a + b ,_mean) / len(img_label_path)
# std = reduce(lambda a, b : a + b ,_std) / len(img_label_path)  

# results
#means = np.array([73.7, 97.7, 97.2, 77.7])
#stds = np.array([179.7, 231.4, 231.3 , 192,3])
