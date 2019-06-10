# STEP 0: LOAD PACKAGES
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import csv, os, functools, itertools, collections, six, time, re

from glob import glob
from datasets.utils import get_imgdir_label, label_to_index, parser, preprocessors
from models import nets_factory

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# STEP 1: SET_config
# model config
lr = 0.001 #learning_rate
batch_size_per_gpu = 32 # batch_size
num_gpu = 3
batch_size = batch_size_per_gpu * num_gpu
epoch = 5 
weight_decay = 0.0 # regularizer constant 
model_name = ["densenet161","inception_v4","vgg_19"][1] # model selection 
variable_to_exclude = {
                      "densenet161":['densenet161/logits'],
                      "inception_v4":['InceptionV4/AuxLogits','InceptionV4/Logits'],
                      "vgg_19":['vgg_19/fc8']
                        }
variable_to_exclude = variable_to_exclude[model_name]
num_classes = 7 # number of classes
weights_loc = "./weights/" # weights locations 
pretrained_dir = glob(weights_loc + "pretrained/" + "*{}*.ckpt*".format(model_name))[0]
pretrained_dir = re.search(".*ckpt",pretrained_dir)[0]
#pretrained weights directory
#trained_dir =  glob(weights_loc + "trained/" + "*{}*.ckpt".format(model_name))[0]  #post trained weights save location


# data config
dataset_dir = "./datasets/ISIC_2018/" # direct to your dataset location
class_list = ["MEL","NV","BCC","AKIEC","BKL","DF","VASC"] # total of 7 classes in the dataset 
class_dict = {c:i for i,c in enumerate(class_list)} # to map str::class -> int::class for train/test
ratio = 0.8
img_size = [224,224]
channel_n = 3

# STEP 2: CREATE DATA-PIPELINE
img_lab = get_imgdir_label(dataset_dir) # get list of tuples [(img_dir_i, str::label_i) for i in range(len(dataset))]
img_lab = label_to_index(img_lab, class_dict) # get list of tuples [(img_dir_i, int::label_i) for i in range(len(dataset))]

img_lab_train = img_lab[:int(len(img_lab) * ratio)] # train set: 80% of img_lab 
img_lab_test = list(set(img_lab) - set(img_lab_train)) # test set 20% of img_lab

imgs_train, labels_train = zip(*img_lab_train) # transpose train_data 
imgs_test, labels_test = zip(*img_lab_test) # transpose test_data

mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])

# reset existing graphs
tf.reset_default_graph()

# input function that 
def input_fn(img_dirs, labels, batch_size, epoch):
    
    dataset = tf.data.Dataset.from_tensor_slices((list(img_dirs),list(labels))).\
        shuffle(10000).\
        repeat(epoch).\
        map(parser, num_parallel_calls = 10).\
        batch(batch_size).\
        prefetch(4)
    
    return dataset
    
with mirrored_strategy.scope():
    
    dataset_train = mirrored_strategy.make_dataset_iterator(input_fn(imgs_train, labels_train, batch_size, epoch))
    dataset_test = mirrored_strategy.make_dataset_iterator(input_fn(imgs_test, labels_test, batch_size, epoch))

    
# STEP 3: CREATE MODEL FOR TEST AND TRAIN. 
with mirrored_strategy.scope():
    
    # initialize train_model function
    network_train_fn = nets_factory.get_network_fn(
        name = model_name,
        weight_decay = weight_decay,
        num_classes = num_classes,
        is_training = True)
    
    # initialize test_model function
    network_test_fn = nets_factory.get_network_fn(
        name = model_name,
        weight_decay = weight_decay,
        num_classes = num_classes,
        is_training = False,
        reuse = True)

    
    
# STEP 4: DEFINE STEP_FN FOR TRAIN AND TEST AND CREATE TRAINNING SCHEDULE.
def run(model_train_fn, model_test_fn, input_train_fn, input_test_fn):
    
    # data generator
    #generator_train = input_train_fn.make_initializable_iterator()
    #generator_test = input_test_fn.make_initializable_iterator()
    
    
    # create keras metric functions 
    training_loss = tf.keras.metrics.Mean("training_loss", dtype=tf.float32)
    training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy("training_accuracy", dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean("test_loss", dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy("test_accuracy", dtype=tf.float32)

    # define optimizer
    optimizer = tf.train.AdadeltaOptimizer(lr)
    
    
    # get mean and 2*se of time estimate(E[time_avg] +- 2*SE[time_avg]).
    def get_mean_se(xs):
  
        mean = np.mean(xs)
        se = np.std(xs)/np.sqrt(len(xs))
    
        return mean, se

    def compute_loss(logits, labels):
    
        loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))
  
        return loss * (1. / batch_size)

    
    def train_step(inputs):
        
        images = inputs[0]
        labels = tf.cast(inputs[1], tf.int32)

        with tf.GradientTape() as tape:

            logits,_ = network_train_fn(images)
            #logits = tf.squeeze(logits)
            loss = compute_loss(logits, labels)

        grads = tape.gradient(loss, slim.get_model_variables())

        update_vars = optimizer.apply_gradients(zip(grads, slim.get_model_variables()))
        update_loss = training_loss.update_state(loss)
        update_accuracy = training_accuracy.update_state(tf.expand_dims(labels, axis = -1), logits)
        

        with tf.control_dependencies([update_vars, update_loss, update_accuracy]):
            return tf.identity(loss)
    

    def test_step(inputs):

        images = inputs[0]
        labels = tf.cast(inputs[1], tf.int32)
        logits,_ = network_test_fn(images)
        #logits = tf.squeeze(logits)
        loss = compute_loss(logits, labels)

        update_loss = test_loss.update_state(loss)
        update_accuracy = test_accuracy.update_state(tf.expand_dims(labels, axis = -1), logits)

        with tf.control_dependencies([update_loss, update_accuracy]):
            return tf.identity(loss)

    # op to run a model
    train_op = mirrored_strategy.unwrap(mirrored_strategy.experimental_run(train_step, input_train_fn))
    test_op = mirrored_strategy.unwrap(mirrored_strategy.experimental_run(test_step, input_test_fn))
    
    # op to print metric results during sees.run
    training_loss_result = training_loss.result()
    training_accuracy_result = training_accuracy.result()
    test_loss_result = test_loss.result()
    test_accuracy_result = test_accuracy.result()
    
    
    # initialise data iterators
    train_iterator_init = input_train_fn.initialize()
    test_iterator_init = input_test_fn.initialize()
    
    # variables to initialize 
    all_variables = (
        tf.global_variables() +
        training_loss.variables +
        training_accuracy.variables +
        test_loss.variables +
        test_accuracy.variables)

    with mirrored_strategy.scope():
        # list of parameters to load
        get_model_var_list = lambda : slim.get_model_variables()
        is_in_exclusion = lambda exclusions, x : [x for exclusion in exclusions if x.op.name.startswith(exclusion) ]
        is_in_model_exclusion = functools.partial(is_in_exclusion, variable_to_exclude)
        variable_to_include = list(set(get_model_var_list()) - set(filter(is_in_model_exclusion, get_model_var_list())))

        weights_loader = slim.assign_from_checkpoint_fn(
            pretrained_dir,
            variable_to_include,
            ignore_missing_vars=True)
    
    with mirrored_strategy.scope():
        # open a session
        with tf.Session() as sess:
            # initialize list of variables 
            sess.run([v.initializer for v in all_variables])
            # load pretrained weights
            weights_loader(sess)
            # start train 
            
            times = []
            sess.run(train_iterator_init)        
            
            for e in range(0,epoch):
                
                start_time = time.time()
                
                for i in range(len(img_lab_train)//batch_size):            
                    sess.run(train_op)
                    if i % 20 == 0:
                        print("epoch: ",e)
                        print("train loss at {}th iteration is {}".format(i,sess.run(training_loss_result)))
                        print("train acc at {}th iteration is {}".format(i,sess.run(training_accuracy_result)))
                        print("\n")

                        training_loss.reset_states()
                        training_accuracy.reset_states()
                            
                end_time = time.time()
                time_diff = end_time - start_time 
                times.append(time_diff)

            mean, se = get_mean_se(times)
            print("forward+backward time: mean:{mean}[{lb}, {ub}]/epoch".format(
                                                               mean = mean,
                                                               lb = mean - 2*se,
                                                               ub = mean + 2*se))
            
             
            # start test
            sess.run(test_iterator_init)
            
            times = []

            for e in range(epoch):

                start_time = time.time()

                for i in range(len(img_lab_test)//batch_size):
                    sess.run(test_op)
                    
                end_time = time.time()
                time_diff = end_time - start_time
                times.append(time_diff)

            mean, se = get_mean_se(times)
            print("test_accuracy is:{}".format(sess.run(test_accuracy_result)))
            print("forward time: mean:{mean}[{lb}, {ub}]/epoch".format(mean = mean,
                                                               lb = mean - 2*se,
                                                               ub = mean + 2*se))
            test_loss.reset_states()
            test_accuracy.reset_states()

    
# STEP 4: RUN MODEL.
run(network_train_fn, network_test_fn, dataset_train, dataset_test)        


