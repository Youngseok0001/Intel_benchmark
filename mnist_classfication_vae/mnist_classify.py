## Environments
# 1. python 3.52
# 2. intel-tensorflow 1.13.1  or
# 3. tensorflow-gpu 1.13.1 

#packages installed.
# 1. intel-tensorflow=1.13
# 2. tensorflow-gpu=1.13

# network and dataset 
# 1. net_info = 3 * (conv + batchnorm + relu + pool) + 2 * (FC) 
# 2. mnist dataset
# 3. batch_size = 256
# 4. epoch = 3

# what we are measuring
# 1. train time/iteration +-se
# 2. inference time/iteration +- 2*se 


# STEP 0: load packages

from collections import deque
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import time

tf.enable_eager_execution()

# STEP 0: SET CONFIGURATION

TRAIN_BUF = 60000
TEST_BUF = 10000
BATCH_SIZE = 256
EPOCH = 3


# STEP 1: CREATE DATA_GENERATOR
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')/255.0
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')/255.0

  # Normalize & format
train_images /= 255.
test_images /= 255.
train_labels = train_labels.astype('int32')
test_labels = test_labels.astype('int32')

  # tf.data generator
train_dataset = tf.data.Dataset.from_tensor_slices((train_images,train_labels)).\
                  shuffle(TRAIN_BUF).\
                  batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images,test_labels)).\
                  shuffle(TEST_BUF).\
                  batch(BATCH_SIZE)


# STEP 2: DEFINE MODEL
  # create convolution block layer 
class conv_block(tf.keras.Model):
  # conv_block composed of conv + bn + relu + pool 
  
  def __init__(self, kernel_size, filter_num): 
    super(conv_block, self).__init__(name='conv_block')
    
    self.conv = tf.keras.layers.Conv2D(filter_num, kernel_size, padding = 'same')
    self.bn = tf.keras.layers.BatchNormalization()
    self.relu = tf.keras.layers.Activation("relu")
    
  def call(self, input_tensor, training=False):
    x = self.conv(input_tensor)
    x = self.bn(x, training=False)
    x = self.relu(x)
    x = tf.layers.max_pooling2d(x,(2,2), strides = (2,2))
    return x    
  
  
  # create fc block layer 
class fc_block(tf.keras.Model):
  # fc_block composed FC1 + bn + FC2 
  
  def __init__(self): 
    super(fc_block, self).__init__(name='fc_block')
    
    self.flatten = tf.keras.layers.Flatten()
    self.fc1 = tf.keras.layers.Dense(128, activation = 'relu')
    self.bn = tf.keras.layers.BatchNormalization()
    self.fc2 = tf.keras.layers.Dense(10)
    
  def call(self, x, training=False):
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.bn(x, training)
    x = self.fc2(x)
    return x      
  
  # using the two building blocks to create the final layer
class simple_model(tf.keras.Model):
  def __init__(self):
    super(simple_model, self).__init__(name='simple_model')
  
    self.conv_block_1 = conv_block((3,3), 32)
    self.conv_block_2 = conv_block((3,3), 64)
    self.conv_block_3 = conv_block((3,3), 128)
    self.fc_block_1 = fc_block()

  def forward(self, x, is_train):
    x = self.conv_block_1(x, is_train)
    x = self.conv_block_2(x, is_train)
    x = self.conv_block_3(x, is_train)
    x = self.fc_block_1(x, is_train)
    logit = x 
    return logit 
  


# STEP 3: RUN MODEL
  # cross entropy loss 
def get_loss(logits, labels):
  losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
  return tf.reduce_mean(losses) 

  # gradient operation
def get_gradients(tape, loss, model):
  return tape.gradient(loss, model.trainable_variables)

  # weights update method
def apply_gradients(optimizer, gradients, variables):
  optimizer.apply_gradients(zip(gradients, variables))
  
  
  # get mean and 2*se of time estimate(E[time_avg] +- 2*SE[time_avg]).
def get_mean_se(xs):
  mean = np.mean(xs)
  se = np.std(xs)/np.sqrt(len(xs))
  return mean, 2*se
  
  # build model 
model = simple_model()


  # execute trainining 
for e in range(EPOCH):
    
    times = deque(maxlen = 10)
    
  for i, batch in enumerate(train_dataset):    
    img_batch, label_batch = batch
    with tf.GradientTape() as tape:
      start_time = time.time()
      
      logits = model.forward(img_batch, is_train = True)
      loss = get_loss(logits, label_batch)
      grads = get_gradients(tape, loss, model)
      #optimizer = tf.keras.optimizers.Adam(1e-4)
      optimizer = tf.train.AdamOptimizer(learning_rate=1E-3)
      apply_gradients(optimizer,grads,model.trainable_variables)
      
      end_time = time.time()
      
      diff_time = end_time - start_time
      times.extend([diff_time])
          
    if i % 10 == 0:
      mean, sse = get_mean_se(times)
      print("train loss at {}th iteration is {}".format(i,loss))
      print("forward+backward time:{0:.2f} +-{1:.2f}/iteration".format(mean, sse))
    
  accs = []
  times = deque(maxlen = 10)
  
  for img, label in test_dataset:
    start_time = time.time()
    
    logit =  model.forward(img, is_train = True)
    match = tf.equal(label, tf.cast(tf.argmax(logit, 1), tf.int32))
    match = tf.cast(match, tf.float32).numpy()
    acc = np.mean(match)x
    accs.append(acc)
    end_time = time.time()
    diff_time = end_time - start_time
    times.extend([diff_time])
    
  print("test_accuracy:{}".format(np.mean(accs)))
  print("forward time:{0:.2f}/iteration".format(end_time - start_time))

