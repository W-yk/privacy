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
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys 
sys.path.append("..") 

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras_vectorized_adaclip import VectorizedDPKerasSGDOptimizer
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Input, BatchNormalization, Dropout, Activation
from tensorflow.keras import Model, regularizers, Sequential

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 0.1,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 5.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 50, 'Batch size')
flags.DEFINE_integer('lot_size', 50, 'Lot size')
flags.DEFINE_integer('epochs', 100, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 50, 'Number of microbatches '
    '(must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')
flags.DEFINE_string('device', '1', 'device')

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

  assert train_data.min() == 0.
  assert train_data.max() == 1.
  assert test_data.min() == 0.
  assert test_data.max() == 1.

  return train_data, train_labels, test_data, test_labels


class VGG16(Model):
    def __init__(self, batch_per_lot):
        super(VGG16, self).__init__()
        self.batch_per_lot = batch_per_lot
        self.apply_flag = tf.Variable(False, dtype=tf.bool, trainable=False)
        self.weight_decay = 0.0005

        # Block 1
        self.conv1_1 = Conv2D(64, 3, activation='relu', padding='same', input_shape=[
                              32, 32, 3], kernel_regularizer=regularizers.l2(self.weight_decay), name='block1_conv1')
        self.bn1_1 = BatchNormalization()
        self.drop1 = Dropout(0.3)
        self.conv1_2 = Conv2D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(
            self.weight_decay), name='block1_conv2')
        self.bn1_2 = BatchNormalization()
        self.pool1 = MaxPool2D(pool_size=(2, 2))

        # Block 2
        self.conv2_1 = Conv2D(128, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(self.weight_decay), name='block2_conv1')
        self.bn2_1 = BatchNormalization()
        self.drop2 = Dropout(0.4)
        self.conv2_2 = Conv2D(128, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(self.weight_decay), name='block2_conv2')
        self.bn2_2 = BatchNormalization()
        self.pool2 = MaxPool2D(pool_size=(2, 2))

        # Block 3
        self.conv3_1 = Conv2D(256, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(self.weight_decay), name='block3_conv1')
        self.bn3_1 = BatchNormalization()
        self.drop3 = Dropout(0.4)
        self.conv3_2 = Conv2D(256, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(self.weight_decay), name='block3_conv2')
        self.bn3_2 = BatchNormalization()
        self.drop4 = Dropout(0.4)
        self.conv3_3 = Conv2D(256, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(self.weight_decay), name='block3_conv3')
        self.bn3_3 = BatchNormalization()
        self.pool3 = MaxPool2D(pool_size=(2, 2))

        # Block 4
        self.conv4_1 = Conv2D(512, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(self.weight_decay), name='block4_conv1')
        self.bn4_1 = BatchNormalization()
        self.drop5 = Dropout(0.4)
        self.conv4_2 = Conv2D(512, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(self.weight_decay), name='block4_conv2')
        self.bn4_2 = BatchNormalization()
        self.drop6 = Dropout(0.4)
        self.conv4_3 = Conv2D(512, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(self.weight_decay), name='block4_conv3')
        self.bn4_3 = BatchNormalization()
        self.pool4 = MaxPool2D(pool_size=(2, 2))

        # Block 5
        self.conv5_1 = Conv2D(512, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(self.weight_decay), name='block5_conv1')
        self.bn5_1 = BatchNormalization()
        self.drop7 = Dropout(0.4)
        self.conv5_2 = Conv2D(512, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(self.weight_decay), name='block5_conv2')
        self.bn5_2 = BatchNormalization()
        self.drop8 = Dropout(0.4)
        self.conv5_3 = Conv2D(512, 3, activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(self.weight_decay), name='block5_conv3')
        self.bn5_3 = BatchNormalization()
        self.pool5 = MaxPool2D(pool_size=(2, 2))
        self.drop9 = Dropout(0.5)

        self.flatten = Flatten()
        self.d1 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(
            self.weight_decay), name='fc1')
        self.bn1 = BatchNormalization()
        self.drop10 = Dropout(0.5)
        self.d2 = Dense(10, activation='softmax', name='fc2')

    def call(self, x):
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.drop1(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.drop2(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.drop3(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.drop4(x)
        x = self.conv3_3(x)
        x = self.bn3_3(x)
        x = self.pool3(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.drop5(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.drop6(x)
        x = self.conv4_3(x)
        x = self.bn4_3(x)
        x = self.pool4(x)
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.drop7(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.drop8(x)
        x = self.conv5_3(x)
        x = self.bn5_3(x)
        x = self.pool5(x)
        x = self.drop9(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.bn1(x)
        x = self.drop10(x)
        return self.d2(x)

    def build_model(self):
        x = Input(shape=(32, 32, 3))
        model = Model(inputs=[x], outputs=self.call(x))
        self.accumulated_grads = [tf.Variable(tf.zeros_like(var), trainable=False) for var in self.trainable_variables]
        return

    def apply(self):
        gradients = [g / self.batch_per_lot for g in self.accumulated_grads]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
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

        noised_gradients = self.optimizer._compute_gradients(loss, self.trainable_variables, tape=tape)
        

        grads_flat = tf.nest.flatten(noised_gradients)
        squared_l2_norms = [
            tf.reduce_sum(input_tensor=tf.square(g)) for g in grads_flat
        ]
        
        for g, new_grad in zip(self.accumulated_grads, noised_gradients):
            g.assign_add(new_grad)
        
        tf.cond(self.apply_flag, lambda: self.apply(), lambda: self.not_apply())

        self.optimizer.apply_gradients(zip(noised_gradients, self.trainable_variables))
        self.compiled_metrics.update_state(labels, predictions)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, labels = data
        predictions = self(images, training=False)
        self.compiled_metrics.update_state(labels, predictions)

        return {m.name: m.result() for m in self.metrics}


class TestCallback(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        if (batch+1) % self.model.batch_per_lot == 0:
            self.model.apply_flag.assign(True)
        else:
            self.model.apply_flag.assign(False)

def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.device

  if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
    raise ValueError('Number of microbatches should divide evenly batch_size')

  # Load training and test data.
  train_data, train_labels, test_data, test_labels = load_cifar()

  batch_per_lot = int(FLAGS.lot_size / FLAGS.batch_size)
  model = model = VGG16(batch_per_lot)
  model.build_model()

  if FLAGS.dpsgd:
    optimizer = VectorizedDPKerasSGDOptimizer(
        l2_norm_clip=FLAGS.l2_norm_clip,
        noise_multiplier=FLAGS.noise_multiplier,
        num_microbatches=FLAGS.microbatches,
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(0.1, 4000, 0.5, staircase=True))
    # Compute vector of per-example loss rather than its mean over a minibatch.
    loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.losses.Reduction.NONE)
  else:
    optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

  # Compile model with Keras
  model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'],run_eagerly=False)#True)

  # Compute the privacy budget expended.
  if FLAGS.dpsgd:
    eps = compute_epsilon(FLAGS.epochs * 60000 // FLAGS.batch_size)
    print('For delta=1e-5, the current epsilon is: %.2f' % eps)
  else:
    print('Trained with vanilla non-private SGD optimizer')

  # Train model with Keras
  model.fit(train_data, train_labels,
            epochs=FLAGS.epochs,
            validation_data=(test_data, test_labels),
            batch_size=FLAGS.batch_size,
            callbacks=[TestCallback()])



if __name__ == '__main__':
  app.run(main)
