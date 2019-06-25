from easydict import EasyDict 
import numpy as np
from random import randint
import tensorflow as tf

config = EasyDict()

data_pipeline_configs = \
    { "img_loc"           : "./dataset/Task01_BrainTumour/imagesTr/",    
      "lab_loc"           : "./dataset/Task01_BrainTumour/labelsTr/",
      "batch_size"        : 1,
      "epoch"             : 3,
      "cpu_n"             : 10,
      "prefetch"          : 4,
      "split_ratio"       : 0.8
    }

data_proprocessing_configs = \
    {
      "patch_size"        : [128, 128, 64],
      "means"             : [73.7, 97.7, 97.2, 77.7],
      "stds"              : [179.7, 231.4, 231.3, 192.3],
      "sigma_gaussian"    : 0.01,
      "condition"         : lambda img_lab : np.sum(img_lab[1]) > 50
    }

model_configs = \
    {
    "depth"               : 4, 
    "BASE_FILTER"         : 16,
    "PADDING"             : "same",
    "DEEP_SUPERVISION"    : True,
    "num_classes"         : 4,
    "optimizer"           : tf.train.AdadeltaOptimizer,
    "lr"                  : 0.001
    }


config.update(data_pipeline_configs)
config.update(data_proprocessing_configs)
config.update(model_configs)

