from easydict import EasyDict 
import numpy as np
from random import randint
import tensorflow as tf

config = EasyDict()

data_pipeline_configs = \
    { "img_loc"           : "./dataset/Task01_BrainTumour/imagesTr/",    
      "lab_loc"           : "./dataset/Task01_BrainTumour/labelsTr/",
      "batch_size"        : 1,
      "epoch"             : 100,
      "cpu_n"             : 20,
      "prefetch"          : 10,
      "split_ratio"       : 0.8
    }

data_proprocessing_configs = \
    {
      "patch_size"        : [160, 192, 128],
      "means"             : [73.7, 97.7, 97.2, 77.7],
      "stds"              : [179.7, 231.4, 231.3, 192.3],
      "mean_std_shift"    : [[-0.1,0.1], [0.9, 1.1]],
      "elas_alpha"        : [10, 2e6, 2e6],
      "elas_sigma"        : [1, 25, 25],
      "sigma_gaussian"    : 0.005,
      "condition"         : lambda img_lab : np.sum(img_lab[0]) != 0
    }

model_configs = \
    {
    "depth"               : 4, 
    "BASE_FILTER"         : 32,
    "PADDING"             : "same",
    "DEEP_SUPERVISION"    : True,
    "num_classes"         : 4,
    "optimizer"           : tf.train.AdamOptimizer,
    "lr"                  : 0.0001,
    "loss_weights"        : [0.2,0.5,0.3]
    }


train_configs = \
    {
    "max_to_keep"         : 3,
    "model_name"          : "model_5_exp_ce",
    "visualise"           : True,
    "model_save_path"     : "./weights/model_2_save/",
    "text_log_path"       : "./log/text/model_5/",
    "visual_log_path"     : "./log//img/model_5/",
    "colours"             : [[0,0,0],
                             [255,0,0],
                             [255,178,102],
                             [255,255,51]]
    }


config.update(data_pipeline_configs)
config.update(data_proprocessing_configs)
config.update(model_configs)
config.update(train_configs)
