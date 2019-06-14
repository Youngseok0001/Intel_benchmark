## Environments
# 1. python 3.52
# 2. intel-tensorflow 1.13.1  or
# 3. tensorflow-gpu 1.13.1 

#packages installed.
# 1. intel-tensorflow=1.13
# 2. tensorflow-gpu=1.13
# 3. matplotlib
# 4. IPython

# network and dataset 
# 1. net_info = VAE:: 2 * ConvBlocks -> x_i ~ normal(nu_i,std_i) -> 2*DeconvBlock)
# 2. mnist
# 3. batch_size = 100
# 4. epoch = 10

# what we are measuring
# 1. train time/epoch (mean, 95%CI)
# 2. inference time/epoch (mean, 95%CI)


import os
clear = lambda: os.system('cls')
clear()

# STEP 0: load packages

import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import time
from matplotlib import pyplot as plt
from IPython import display
tf.enable_eager_execution()

# STEP 0: SET CONFIGURATION
TRAIN_BUF = 60000
TEST_BUF = 10000
BATCH_SIZE = 256
EPOCH = 3
latent_dim = 50
num_examples_to_generate = 16
prefetch_buffer_size = 10
lr = 1E-4


# STEP 1: CREATE DATA_GENERATOR
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

  # 1.[N, W * H * C] to [N, W, H, C]
  # 2.Simple nomalisation = ./255
train_images = train_images.reshape(-1, 28, 28, 1).astype('float32')/255.
test_images = test_images.reshape(-1, 28, 28, 1).astype('float32')/255.

  # Threshold 
train_images = np.where(train_images>0.5, 1., 0.).astype('float32')
test_images = np.where(test_images>0.5, 1., 0.).astype('float32')

 # Convert to tensor
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).\
                  shuffle(TRAIN_BUF).\
                  batch(BATCH_SIZE).\
                  prefetch(buffer_size = prefetch_buffer_size)

test_dataset = tf.data.Dataset.from_tensor_slices(test_images).\
                  shuffle(TEST_BUF).\
                  batch(BATCH_SIZE).\
                  prefetch(buffer_size = prefetch_buffer_size)



# STEP 1: CREATE MODEL
class VAE(K.Model):
  def __init__(self, latent_dim):
    super(VAE, self).__init__()
    
    self.latent_dim = latent_dim

    self.inference_net = K.Sequential(
      [
          K.layers.InputLayer(input_shape = (28, 28, 1)),
          K.layers.Conv2D(filters = 32, kernel_size = 3, strides = (2, 2), activation = 'relu'),
          K.layers.Conv2D(filters = 64, kernel_size = 3, strides = (2, 2), activation = 'relu'),
          K.layers.Flatten(),
          K.layers.Dense(2 * latent_dim)
      ]
    )

    self.generative_net = K.Sequential(
        [
          K.layers.InputLayer(input_shape =  (latent_dim,)),
          K.layers.Dense(units = 7 * 7 * 32, activation = tf.nn.relu),
          K.layers.Reshape(target_shape = (7, 7, 32)),
          K.layers.Conv2DTranspose(
              filters = 64,
              kernel_size = 3,
              strides = (2, 2),
              padding = "SAME",
              activation = 'relu'),
          K.layers.Conv2DTranspose(
              filters = 32,
              kernel_size = 3,
              strides = (2, 2),
              padding = "SAME",
              activation = 'relu'),
          # No activation
          K.layers.Conv2DTranspose(
              filters = 1, kernel_size = 3, strides = (1, 1), padding = "SAME"),
        ]
    )

  def sample(self, eps = None):
    if eps is None:
      eps = tf.random.normal(shape = (100, self.latent_dim))
    return self.decode(eps, apply_sigmoid = True)

  def encode(self, x):
    mean, logvar = tf.split(self.inference_net(x), num_or_size_splits = 2, axis = 1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape = mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid = False):
    logits = self.generative_net(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs

    return logits
  
  
def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis = raxis)

def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)

  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

def compute_gradients(model, x):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  return tape.gradient(loss, model.trainable_variables), loss

def apply_gradients(optimizer, gradients, variables):
  optimizer.apply_gradients(zip(gradients, variables))  
  
def generate_and_save_images(model, epoch, test_input):
  predictions = model.sample(test_input)
  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0], cmap='gray')
      plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
  
  
  
  # get mean and 2*se of time estimate(E[time_avg] +- 2*SE[time_avg]).
def get_mean_se(xs):
  mean = np.mean(xs)
  se = np.std(xs)/np.sqrt(len(xs))
  return mean, se



# STEP 2: Run model
optimizer = tf.train.AdamOptimizer(learning_rate = lr)
  
random_vector_for_generation = tf.random.normal( shape = [num_examples_to_generate, latent_dim])

model = VAE(latent_dim)

  
#generate_and_save_images(model, 0, random_vector_for_generation)

times = []

for e in range(EPOCH):
  
  start_time = time.time()

  for i, train_x in enumerate(train_dataset):
    gradients, loss = compute_gradients(model, train_x)
    apply_gradients(optimizer, gradients, model.trainable_variables)
    
    if i % 20 == 0:
      print("train loss at {}th iteration is {}".format(i,loss))
      #generate_and_save_images(model, 0, random_vector_for_generation)

  end_time = time.time()
  time_diff = end_time - start_time
  times.append(time_diff)
  
mean, se = get_mean_se(times)
print("time/epoch : {mean},[{lb},{up}]".format(mean = mean , 
                                               lb   = mean - 2*se,
                                               up   = mean + 2*se))
  
times = []

for e in range(EPOCH):

  start_time = time.time()
  losses = []
  
  for test_x in test_dataset:
    loss = compute_loss(model, test_x)
    losses.append(loss.numpy())

  end_time = time.time()
  time_diff = end_time - start_time
  times.append(time_diff)
  
  
mean, se = get_mean_se(times)
print("test loss:{}".format(np.mean(losses)))
print("infernce time/epoch : {mean},[{lb},{up}]".format(mean = mean , 
                                               lb   = mean - 2*se,
                                               up   = mean + 2*se))

#generate_and_save_images(model, EPOCH, random_vector_for_generation)