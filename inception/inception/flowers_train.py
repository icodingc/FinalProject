# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A binary to train Inception on the flowers data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import tensorflow as tf

from inception import inception_train
from inception.flowers_data import FlowersData
from inception.util import tripletnet as util

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('xent', 'xent',
                           """xent_train & triplet_train""")


def main(_):
#  assert dataset.data_files()
  if tf.gfile.Exists(FLAGS.train_dir):
    print('Already exists!')
    sys.exit()
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  if FLAGS.xent=='xent':
    dataset = FlowersData(subset=FLAGS.subset)
    inception_train.xent_train(dataset)
  else:
    dataset = util.get_dataset('~/workshops/data/cifar10/train')
    inception_train.triplet_train(dataset)


if __name__ == '__main__':
  tf.app.run()
