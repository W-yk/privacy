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

from tensorflow.python.ops.gen_math_ops import TruncateDiv
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append("..")

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras_vectorized_gradient_compress  import VectorizedDPKerasSGDOptimizer
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Input, BatchNormalization, Dropout, Activation,MaxPooling2D
from tensorflow.keras import Model, regularizers, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 10,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1, 'Clipping norm')
flags.DEFINE_integer('batch_size', 50, 'Batch size')
flags.DEFINE_integer('lot_size', 500, 'Lot size')
flags.DEFINE_integer('epochs', 25, 'Number of epochs')
flags.DEFINE_integer(
    'microbatches', 50, 'Number of microbatches '
    '(must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')

FLAGS = flags.FLAGS


### Adult dataset prepare data
def find_means_for_continuous_types(X):
    means = []
    for col in range(len(X[0])):
        summ = 0
        count = 0.000000000000000000001
        for value in X[:, col]:
            if isFloat(value): 
                summ += float(value)
                count +=1
        means.append(summ/count)
    return means

def isFloat(string):
    # credits: http://stackoverflow.com/questions/2356925/how-to-check-whether-string-might-be-type-cast-to-float-in-python
    try:
        float(string)
        return True
    except ValueError:
        return False
# Adult dataset prepare
def prepare_data(raw_data, means):
    inputs = (
        ("age", ("continuous",)), 
        ("workclass", ("Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked")), 
        ("fnlwgt", ("continuous",)), 
        ("education", ("Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool")), 
        ("education-num", ("continuous",)), 
        ("marital-status", ("Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse")), 
        ("occupation", ("Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces")), 
        ("relationship", ("Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried")), 
        ("race", ("White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black")), 
        ("sex", ("Female", "Male")), 
        ("capital-gain", ("continuous",)), 
        ("capital-loss", ("continuous",)), 
        ("hours-per-week", ("continuous",)), 
        ("native-country", ("United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"))
    )
    input_shape = []
    for i in inputs:
        count = len(i[1])
        input_shape.append(count)
    input_dim = sum(input_shape)
    
    X = raw_data[:, :-1]
    y = raw_data[:, -1:]
    
    # X:
    def flatten_persons_inputs_for_model(person_inputs):
        # global means
        float_inputs = []

        for i in range(len(input_shape)):
            features_of_this_type = input_shape[i]
            is_feature_continuous = features_of_this_type == 1

            if is_feature_continuous:
                mean = means[i]
                if isFloat(person_inputs[i]):
                    scale_factor = 1/(2*mean)  # we prefer inputs mainly scaled from -1 to 1. 
                    float_inputs.append(float(person_inputs[i])*scale_factor)
                else:
                    float_inputs.append(1/2)
            else:
                for j in range(features_of_this_type):
                    feature_name = inputs[i][1][j]

                    if feature_name == person_inputs[i]:
                        float_inputs.append(1.)
                    else:
                        float_inputs.append(0)
        return float_inputs
    
    new_X = []
    for person in range(len(X)):
        formatted_X = flatten_persons_inputs_for_model(X[person])
        new_X.append(formatted_X)
    new_X = np.array(new_X)
    
    # y:
    new_y = []
    for i in range(len(y)):
        if y[i] == ">50k":
            new_y.append(1)
        else:  # y[i] == "<=50k":
            new_y.append(0)
    new_y = np.array(new_y)
    
    return (new_X, new_y)
def load_adult():
  """Loads adult and preprocesses to combine training and validation data."""
  training_data = np.genfromtxt('/home/lxiang_stu6/data/adult/adult.data.txt', delimiter=', ', dtype=str, autostrip=True)
  test_data = np.genfromtxt('/home/lxiang_stu6/data/adult/adult.test.txt', delimiter=', ', dtype=str, autostrip=True)

  means = find_means_for_continuous_types(np.concatenate((training_data, test_data), 0))

  train_data, train_labels = prepare_data(training_data, means)
  test_data, test_labels = prepare_data(test_data, means)
  
  print(train_data.shape, train_labels.shape)
  # train_data = train_data.reshape((train_data.shape[0], 28, 28, 1))
  # test_data = test_data.reshape((test_data.shape[0], 28, 28, 1))

  train_labels = np.array(train_labels, dtype=np.int32)
  test_labels = np.array(test_labels, dtype=np.int32)

  train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=2)
  test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=2)
  print(test_labels[0:20])
  # assert train_data.min() == 0.
  # assert train_data.max() == 1.
  # assert test_data.min() == 0.
  # assert test_data.max() == 1.

  return train_data, train_labels, test_data, test_labels


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




class MLP(Model):
  def __init__(self, batch_per_lot):
    """
    An modified LeNet for FL with noise on middle feature
    """
    super(MLP, self).__init__()
    self.batch_per_lot = batch_per_lot
    self.apply_flag = tf.Variable(False, dtype=tf.bool, trainable=False)
    self.weight_decay = 0.005

    self.d1 = Dense(20, activation='relu') 
    self.d2 = Dense(2, activation='softmax', name='fc2')

  def call(self, x):
    x = self.d1(x)
    return self.d2(x)


  def build_model(self):
    x = Input(shape=(105,))
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
  train_data, train_labels, test_data, test_labels = load_adult()

  # Define a sequential Keras model
  batch_per_lot = int(FLAGS.lot_size / FLAGS.batch_size)
  model = MLP(batch_per_lot)
  m = model.build_model()
  m.summary()
  model(test_data)
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
    optimizer = tf.keras.optimizers.SGD(learning_rate=FLAGS.learning_rate)
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
  model.fit(train_data, train_labels,
    epochs=FLAGS.epochs,
    validation_data=(test_data, test_labels),
    batch_size=FLAGS.batch_size,
    callbacks=[TestCallback()])



if __name__ == '__main__':
  app.run(main)
