# Copyright 2019, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training a CNN on MNIST with Keras and the DP SGD optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
sys.path.append("..")

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras_vectorized_gradient_compress import VectorizedDPKerasSGDOptimizer
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Input, BatchNormalization, Dropout, Activation
from tensorflow.keras import Model, regularizers, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 0.45,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 10, 'Clipping norm')
flags.DEFINE_integer('batch_size', 50, 'Batch size')
flags.DEFINE_integer('lot_size', 500, 'Lot size')
flags.DEFINE_integer('epochs', 100, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 50, 'Number of microbatches '
    '(must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')

FLAGS = flags.FLAGS


def compute_epsilon(steps):
  """Computes epsilon value for given hyperparameters."""
  if FLAGS.noise_multiplier == 0.0:
    return float('inf')
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  sampling_probability = FLAGS.batch_size / 60000
  rdp = compute_rdp(q=sampling_probability,
                    noise_multiplier=FLAGS.noise_multiplier,
                    steps=steps,
                    orders=orders)
  # Delta is set to 1e-5 because MNIST has 60000 training points.
  return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]


def load_cifar():
  """Loads MNIST and preprocesses to combine training and validation data."""
  train, test = tf.keras.datasets.cifar10.load_data()
  train_data, train_labels = train
  test_data, test_labels = test

  train_data = np.array(train_data, dtype=np.float32) / 255
  test_data = np.array(test_data, dtype=np.float32) / 255

  # train_data = train_data.reshape((train_data.shape[0], 32, 32, 3))
  # test_data = test_data.reshape((test_data.shape[0], 32, 32, 3))

  train_labels = np.array(train_labels, dtype=np.int32)
  test_labels = np.array(test_labels, dtype=np.int32)

  train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
  test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)
  print(len(train_labels),'****************************')
  assert train_data.min() == 0.
  assert train_data.max() == 1.
  assert test_data.min() == 0.
  assert test_data.max() == 1.

  return train_data, train_labels, test_data, test_labels


class MobileNetv2(Model):
  def __init__(self,batch_per_lot):
      super(MobileNetv2, self).__init__()
      self.batch_per_lot = batch_per_lot
      self.apply_flag = tf.Variable(False, dtype=tf.bool, trainable=False)
      self.weight_decay = 0.0005

      self.base_model=tf.keras.applications.mobilenet_v2.MobileNetV2(alpha=1,include_top=False,input_shape=(32,32,3),weights=None)
      # self.base_model.trainable = False,,weights=None
      self.flatten = Flatten()
      self.dense = Dense(10, activation='softmax', name='fc')

  def call(self, x):
      x=self.base_model(x)
      x = self.flatten(x)
      x = self.dense(x)
      return x

  def build_model(self):
      x = Input(shape=(32,32,3))
      model = Model(inputs=[x], outputs=self.call(x))
      self.accumulated_grads = [tf.Variable(tf.zeros_like(var), trainable=False) for var in self.trainable_variables]
      self.back_grads = [tf.Variable(tf.zeros_like(
        var), trainable=False) for var in self.trainable_variables]
      return model
    
  def apply(self):
    gradients = [g / self.batch_per_lot for g in self.accumulated_grads]

    ''' compression'''
    compressed_grads = [] 
    for grad, back_grad in zip(gradients, self.back_grads):
      
      threshold = tfp.stats.percentile(tf.abs(grad),0.01)#tf.contrib.distributions.percentile(tf.abs(grad),90.0,interpolation='higher')
      #tf.print('threshold:',threshold,'max grad:',tf.reduce_max(grad))
      grad= tf.add(grad,back_grad)

      # backed up grad that are less than threshold to use in next iteration
      bool_mask_less = tf.math.less(abs(grad), threshold)
      float_mask_less = tf.cast(bool_mask_less, grad.dtype)
      #tf.print('var grad compressed num:',tf.reduce_sum(float_mask_less))
      back_grad.assign(tf.multiply(grad, float_mask_less))

      #Add noise to summed gradients.
      noise_stddev = FLAGS.l2_norm_clip * FLAGS.noise_multiplier
      noise = tf.truediv(tf.random.normal(
          tf.shape(input=grad), stddev=noise_stddev),FLAGS.lot_size)
      grad =tf.add(grad,noise)

      compressed_grads.append(tf.multiply(grad, tf.ones(float_mask_less.shape)- float_mask_less))
      

    self.optimizer.apply_gradients(zip(compressed_grads, self.trainable_variables))
    for g in self.accumulated_grads:
      g.assign(tf.zeros_like(g))
    return

  def not_apply(self):
    return

  def train_step(self, data):
    images, labels = data
    with tf.GradientTape() as tape:
      predictions = self(images, training=True)
      loss = self.compiled_loss(labels, predictions)
      # add the regularization
      regularization = sum(self.losses)
      loss += regularization

    noised_gradients = list(zip(*self.optimizer._compute_gradients(
        loss, self.trainable_variables, tape=tape)))[0]

    for g, new_grad in zip(self.accumulated_grads, noised_gradients):
      g.assign_add(new_grad)

    tf.cond(self.apply_flag, lambda: self.apply(), lambda: self.not_apply())

    # self.optimizer.apply_gradients(zip(noised_gradients, self.trainable_variables))
    self.compiled_metrics.update_state(labels, predictions)

    return {m.name: m.result() for m in self.metrics}

  def test_step(self, data):
    images, labels = data
    predictions = self(images, training=False)
    self.compiled_metrics.update_state(labels, predictions)

    return {m.name: m.result() for m in self.metrics}

class VGG16_v2(Model):
  def __init__(self, batch_per_lot):
    super(VGG16_v2, self).__init__()
    self.batch_per_lot = batch_per_lot
    self.apply_flag = tf.Variable(False, dtype=tf.bool, trainable=False)
    self.weight_decay = 0.005

    self.base_model=tf.keras.applications.VGG16(
            include_top=False, weights='imagenet', input_tensor=None,
            input_shape=(32,32,3), pooling=None, classes=10,
            classifier_activation='softmax'
        )
    self.flatten = Flatten()
    self.d1 = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(
        self.weight_decay), name='fc1')
    self.bn1 = BatchNormalization()
    self.drop10 = Dropout(0.2)
    self.d2 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(
        self.weight_decay), name='fc2')
    self.drop11 = Dropout(0.4)
    self.d3 = Dense(10, activation='softmax', name='fc3')

  def call(self, x):
    x=self.base_model(x)
    x = self.flatten(x)
    x = self.d1(x)
    x = self.bn1(x)
    x = self.drop10(x)
    x = self.d2(x)
    x = self.drop11(x)
    return self.d3(x)

  def build_model(self):
    x = Input(shape=(32,32,3))
    model = Model(inputs=[x], outputs=self.call(x))
    self.accumulated_grads = [tf.Variable(tf.zeros_like(var), trainable=False) for var in self.trainable_variables]
    self.back_grads = [tf.Variable(tf.zeros_like(
      var), trainable=False) for var in self.trainable_variables]
    
    return model


  def apply(self):
    gradients = [g / self.batch_per_lot for g in self.accumulated_grads]

    ''' compression'''
    compressed_grads = [] 
    for grad, back_grad in zip(gradients, self.back_grads):
      
      threshold = tfp.stats.percentile(tf.abs(grad),99)#tf.contrib.distributions.percentile(tf.abs(grad),90.0,interpolation='higher')
      #tf.print('threshold:',threshold,'max grad:',tf.reduce_max(grad))
      grad= tf.add(grad,back_grad)

      # backed up grad that are less than threshold to use in next iteration
      bool_mask_less = tf.math.less(abs(grad), threshold) #x <y
      float_mask_less = tf.cast(bool_mask_less, grad.dtype)
      #tf.print('var grad compressed num:',tf.reduce_sum(float_mask_less))
      back_grad.assign(tf.multiply(grad, float_mask_less))

      #Add noise to summed gradients.
      noise_stddev = FLAGS.l2_norm_clip * FLAGS.noise_multiplier
      noise = tf.truediv(tf.random.normal(
          tf.shape(input=grad), stddev=noise_stddev),FLAGS.lot_size)
          
      grad =tf.add(grad,noise)
      
      # tf.print('compress grad rate :',tf.reduce_sum(float_mask_less)/tf.reduce_sum(tf.ones(float_mask_less.shape)))
      # tf.print('noise before compress',tf.reduce_sum(input_tensor=tf.square(noise)),'noise after compress',tf.reduce_sum(tf.square(tf.multiply(noise, tf.ones(float_mask_less.shape)- float_mask_less))))
      compressed_grads.append(tf.multiply(grad, tf.ones(float_mask_less.shape)- float_mask_less))
      

    self.optimizer.apply_gradients(zip(compressed_grads, self.trainable_variables))
    for g in self.accumulated_grads:
      g.assign(tf.zeros_like(g))
    return

  def not_apply(self):
    return

  def train_step(self, data):
    images, labels = data
    with tf.GradientTape() as tape:
      predictions = self(images, training=True)
      loss = self.compiled_loss(labels, predictions)
      # add the regularization
      regularization = sum(self.losses)
      loss += regularization

    noised_gradients = list(zip(*self.optimizer._compute_gradients(
        loss, self.trainable_variables, tape=tape)))[0]

    for g, new_grad in zip(self.accumulated_grads, noised_gradients):
      g.assign_add(new_grad)

    tf.cond(self.apply_flag, lambda: self.apply(), lambda: self.not_apply())

    # self.optimizer.apply_gradients(zip(noised_gradients, self.trainable_variables))
    self.compiled_metrics.update_state(labels, predictions)

    return {m.name: m.result() for m in self.metrics}

  def test_step(self, data):
    images, labels = data
    predictions = self(images, training=False)
    self.compiled_metrics.update_state(labels, predictions)

    return {m.name: m.result() for m in self.metrics}




class TestCallback(tf.keras.callbacks.Callback):
  def on_train_batch_begin(self, batch, logs=None):
    if (batch + 1) % self.model.batch_per_lot == 0:
      self.model.apply_flag.assign(True)
    else:
      self.model.apply_flag.assign(False)
    # print('\nStep: {}, Apply Flag: {}\n'.format(batch, self.model.apply_flag))

  # def on_epoch_begin(self, epoch, logs={}):
  #     if epoch > 1:
  #         for l in self.model.layers:
  #             if 'conl' in l.name:
  #                 l.trainable = False
  #             if 'conv' in l.name:
  #                 l.trainable = True


def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
    raise ValueError('Number of microbatches should divide evenly batch_size')

  # Load training and test data.
  train_data, train_labels, test_data, test_labels = load_cifar()
  datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

  # Define a sequential Keras model
  batch_per_lot = int(FLAGS.lot_size / FLAGS.batch_size)
  model = VGG16_v2(batch_per_lot)
  m = model.build_model()        
  m.summary()
  model(test_data[:1])
  for l in model.layers:
    print(l.name, l.trainable)

  # # Loading pretrain model
  # initial_weights = [layer.get_weights() for layer in model.layers]
  # #model.load_weights('pretrained_vgg16.h5', by_name=True)
  # for layer, initial in zip(model.layers, initial_weights):
  #   weights = layer.get_weights()
  #   if weights and all(tf.nest.map_structure(np.array_equal, weights, initial)):
  #     print(f'Checkpoint contained no weights for layer {layer.name}!')
  #   else:
  #     print(f'Loading weights for layer {layer.name}!')

  if FLAGS.dpsgd:
    optimizer = VectorizedDPKerasSGDOptimizer(
        l2_norm_clip=FLAGS.l2_norm_clip,
        noise_multiplier=FLAGS.noise_multiplier,
        num_microbatches=FLAGS.microbatches,
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(0.1, 4000, 0.5, staircase=True))
    # Compute vector of per-example loss rather than its mean over a minibatch.
    loss = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.losses.Reduction.NONE)
  else:
    optimizer = tf.keras.optimizers.SGD(learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(0.1, 4000, 0.5, staircase=True))
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

  # Compile model with Keras
  model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

  # Compute the privacy budget expended.
  if FLAGS.dpsgd:
    eps = compute_epsilon(FLAGS.epochs * 60000 // FLAGS.batch_size)
    print('For delta=1e-5, the current epsilon is: %.2f' % eps)


  else:
    print('Trained with vanilla non-private SGD optimizer')

  # Train model with Keras
  model.fit(datagen.flow(train_data, train_labels, batch_size=FLAGS.batch_size),
            steps_per_epoch=(50000/FLAGS.batch_size),
            epochs=FLAGS.epochs,
            validation_data=(test_data, test_labels),
            callbacks=[TestCallback()])



if __name__ == '__main__':
  app.run(main)
