# Copyright 2018, The TensorFlow Authors.
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
"""Implements DPQuery interface for Gaussian sum queries."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import distutils

import tensorflow.compat.v1 as tf

from tensorflow_privacy.privacy.dp_query import dp_query


class GaussianSumQuery(dp_query.SumAggregationDPQuery):
  """Implements DPQuery interface for Gaussian sum queries.

  Accumulates clipped vectors, then adds Gaussian noise to the sum.
  """

  # pylint: disable=invalid-name
  _GlobalState = collections.namedtuple('_GlobalState',
                                        ['l2_norm_clip', 'stddev'])

  def __init__(self, l2_norm_clip, stddev):
    """Initializes the GaussianSumQuery.

    Args:
      l2_norm_clip: The clipping norm to apply to the global norm of each
        record.
      stddev: The stddev of the noise added to the sum.
    """
    self._l2_norm_clip = l2_norm_clip
    self._stddev = stddev
    self._ledger = None

  def set_ledger(self, ledger):
    self._ledger = ledger

  def make_global_state(self, l2_norm_clip, stddev):
    """Creates a global state from the given parameters."""
    return self._GlobalState(
        tf.cast(l2_norm_clip, tf.float32), tf.cast(stddev, tf.float32))

  def initial_global_state(self):
    return self.make_global_state(self._l2_norm_clip, self._stddev)

  def derive_sample_params(self, global_state):
    return global_state.l2_norm_clip

  def preprocess_record_impl(self, params, record):
    """Clips the l2 norm, returning the clipped record and the l2 norm.

    Args:
      params: The parameters for the sample.
      record: The record to be processed.

    Returns:
      A tuple (preprocessed_records, l2_norm) where `preprocessed_records` is
        the structure of preprocessed tensors, and l2_norm is the total l2 norm
        before clipping.
    """
    l2_norm_clip = params
    record_as_list = tf.nest.flatten(record)
    clipped_as_list, norm = tf.clip_by_global_norm(record_as_list, l2_norm_clip)
    return tf.nest.pack_sequence_as(record, clipped_as_list), norm

  def preprocess_record(self, params, record):
    preprocessed_record, _ = self.preprocess_record_impl(params, record)
    return preprocessed_record

  def get_noised_result(self, sample_state, global_state):
    """See base class."""
    if distutils.version.LooseVersion(
        tf.__version__) < distutils.version.LooseVersion('2.0.0'):

      def add_noise(v):
        return v + tf.random.normal(
            tf.shape(input=v), stddev=global_state.stddev, dtype=v.dtype)
    else:
      random_normal = tf.random_normal_initializer(stddev=global_state.stddev)

      def add_noise(v):
        return v + tf.cast(random_normal(tf.shape(input=v)), dtype=v.dtype)

    if self._ledger:
      dependencies = [
          self._ledger.record_sum_query(global_state.l2_norm_clip,
                                        global_state.stddev)
      ]
    else:
      dependencies = []
    with tf.control_dependencies(dependencies):
      return tf.nest.map_structure(add_noise, sample_state), global_state
